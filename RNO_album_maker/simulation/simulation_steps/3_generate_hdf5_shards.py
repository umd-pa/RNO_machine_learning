"""
Step 3: Event Image Builder
---------------------------
Reads NuRadioMC simulation files (.nur), processes the raw waveforms into 
pixelated 'images', and saves them to an HDF5 file for Machine Learning.

Logic:
1. Loads detector configuration to map Station IDs to fixed tensor indices.
2. Iterates through simulated events.
3. Aligns waveforms in time (relative to the first trigger in the event).
4. Adds artificial noise to simulated traces (up to Nyquist limit).
5. Downsamples waveforms into fixed time bins (Mean/Max pooling).
6. Saves the resulting tensor (Channels x Bins x Stations) and labels.

Author: Santiago Sued
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import time
import numpy as np
import h5py

# NuRadioReco Imports
from NuRadioReco.framework import parameters
from NuRadioReco.utilities.logging import _setup_logger
from NuRadioReco.utilities import units
from album_simulation_utils import get_unique_events
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.detector.generic_detector import GenericDetector

# ==============================================================================
#   MAIN EXECUTION
# ==============================================================================

def main():
    start_time = time.time()

    # --------------------------------------------------------------------------
    # 1. SETUP & LOGGING
    # --------------------------------------------------------------------------
    logger = _setup_logger(name="AlbumBuilder")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Step 3: Event Image Builder')

    # Input/Output Arguments
    parser.add_argument('--input_file', 
                        type=str, 
                        default='../simulation_data/simulated_events.nur',
                        help='Path to the input .nur simulation file')
    
    parser.add_argument('--output_file', 
                        type=str,
                        required=True,
                        help='Path to the output .hdf5 file')
    
    parser.add_argument('--detector_file', 
                        type=str, 
                        default='../simulation_data/RNO_four_stations.json',
                        help='Path to the detector JSON file')

    # Image Configuration Arguments
    parser.add_argument('--time_window', 
                        type=float, 
                        default=8192,
                        help='Total time window in nanoseconds (default: 8192)')
    
    parser.add_argument('--time_bins', 
                        type=int, 
                        default=1024,
                        help='Number of time bins in the output image (default: 1024)')
    
    parser.add_argument('--bin_mode', 
                        type=str, 
                        default='MEAN', 
                        choices=['MEAN', 'MAX'],
                        help='Downsampling mode: MEAN or MAX (default: MEAN)')
    
    parser.add_argument('--noise_level',
                        type=float,
                        default=10.,
                        help='RMS of the noise to be added, in mV.')
    
    parser.add_argument('--clamp_v',
                        type=float,
                        default=1.,
                        help='Voltage to clamp. Any voltage above this value will be set to clamp_v')

    args = parser.parse_args()

    # Log Initialization
    logger.info("Starting Image Builder")
    logger.info(f"Input: {args.input_file}")
    logger.info(f"Output: {args.output_file}")
    logger.info(f"Config: {args.time_bins} bins over {args.time_window} ns using {args.bin_mode} pooling")


    # --------------------------------------------------------------------------
    # 2. DETECTOR CONFIGURATION & NOISE SETUP
    # --------------------------------------------------------------------------
    
    # Initialize noise_adder
    # Noise is added artificially AFTER the simulation has run.
    # This prevents noise-induced triggers during the simulation step.
    noise_adder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
    noise_level = args.noise_level * units.mV

    # Load Detector Configuration
    # We map dynamic Station IDs (e.g., 11, 24) to fixed array indices (0, 1).
    with open(args.detector_file, 'r') as f:
        detector_json = json.load(f)

    # Instantiate GenericDetector (Required for noise_adder physics)
    det = GenericDetector(json_filename=args.detector_file, antenna_by_depth=False)

    # Sort station IDs numerically to ensure consistent ordering in the output tensor
    STATION_IDS = sorted([s['station_id'] for s in detector_json['stations'].values()])
    N_STATIONS = len(STATION_IDS)
    N_CHANNELS = len(detector_json['channels']) # Assumes uniform channel counts

    logger.info(f"Detector Configuration: {N_STATIONS} Stations, {N_CHANNELS} Channels")
    logger.info(f"Station IDs: {STATION_IDS}")

    # Pre-calculate bin edges for digitization
    # Will return time_bins+1 edges from (0,time_window) 
    time_edges = np.linspace(0, args.time_window, args.time_bins + 1)


    # --------------------------------------------------------------------------
    # 3. EVENT PROCESSING LOOP
    # --------------------------------------------------------------------------

    # Get unique events (handles event duplicates of different stations. (See log 02/04/2026)
    logger.info("Filtering Unique Events...")
    events_unique = get_unique_events(args.input_file)
    N_EVENTS = len(events_unique)
    logger.info(f"Found {N_EVENTS} unique events. Starting processing...")

    # Data containers
    album = []                    # List of image tensors
    vertices = []                 # List of vertex coordinates (labels)
    n_detecting_stations_arr = [] # List of station hit counts

    for i_event, event in enumerate(events_unique):
        
        # --- A. Identify Stations ---
        stations = list(event.get_stations())
        n_detecting_stations = len(stations)
        station_ids_in_event = [s.get_id() for s in stations]

        logger.info(f"Event {i_event}: Detected by {n_detecting_stations} stations {station_ids_in_event}")

        # --- B. Time Alignment & Noise Injection ---
        # 1. Inject noise first so we do not trigger on noise!
        # 2. Find the earliest start time across ALL channels of ALL stations to align event to t=0.
        
        start_times = []
        
        for station in stations:
            # DYNAMIC CHECK: Get real sampling rate from the first channel
            # This prevents the "max_freq > Nyquist" warning.
            temp_channel = next(station.iter_channels())
            real_nyquist = temp_channel.get_sampling_rate() / 2.

            # Inject Noise (Clamped to actual Nyquist limit)
            noise_adder.run(
                event, 
                station, 
                det, 
                amplitude=noise_level, 
                type='rayleigh', 
                max_freq=real_nyquist
            )

            # Collect start times for alignment
            for channel in station.iter_channels():
                # channel.get_times() returns array; take [0]
                start_times.append(channel.get_times()[0])

        if not start_times:
            logger.warning(f"Event {i_event} has no channel data. Skipping.")
            continue

        # Global t=0 for this event
        min_time = min(start_times)
        logger.debug(f"Event {i_event} will be aligned to global start time: {min_time} ns")


        # --- C. Build Image Tensor ---
        # Shape: (Channels, Bins, Stations)
        # Type: float32 (Standard for ML input)
        image = np.zeros((N_CHANNELS, args.time_bins, N_STATIONS), dtype=np.float32)

        for station in stations:
            # Map Station ID to Tensor Index (0 to N_STATIONS-1)
            try:
                station_idx = STATION_IDS.index(station.get_id())
            except ValueError:
                logger.error(f"Station {station.get_id()} not in detector config.")
                return # Exit if unkown stations are found!
            
            # --- D. Channel Digitization ---
            for i_channel, channel in enumerate(station.iter_channels()):
                
                # 1. Get Data & Shift Time
                times = channel.get_times() - min_time 
                
                # 2. Get Magnitude (Hilbert Envelope)
                hilbert = np.abs(channel.get_hilbert_envelope())

                # 3. Binning (Digitize)
                # np.digitize returns indices 1..N. We subtract 1 to get 0-based indices.
                bin_indices = np.digitize(times, time_edges) - 1

                # 4. Pooling (Downsampling)
                # We only care about bins that actually fall within our 0 -> time_bins range
                valid_mask = (bin_indices >= 0) & (bin_indices < args.time_bins)
                
                # Further explanation of 4: Digitize will return 0 or len(time_edges) if the hilbert values are below or
                # above the time window. Since we subtract 1, it will return -1 or len(time_edges)-1
                # Thus, by doing bin_indices >= 0 and bin_indices < time_bins = time_deges-1, we make sure to
                # exclude these voltage readings

                # OPTIMIZATION: Loop only over unique bins that have data
                unique_valid_bins = np.unique(bin_indices[valid_mask])

                for b_idx in unique_valid_bins:
                    # Extract all signal values that fell into this single time bin
                    # This says, from the hilber trace, set values_in_bin equal to all of the traces
                    #  which the digitizer has assigned to b_idx.
                    values_in_bin = hilbert[bin_indices == b_idx]
                    
                    # Clamp voltage
                    CLAMP_V = args.clamp_v
                    values_in_bin_clamped = np.where(values_in_bin<=CLAMP_V,values_in_bin,CLAMP_V)

                    if len(values_in_bin_clamped) > 0:
                        if args.bin_mode == 'MEAN':
                            val = np.mean(values_in_bin_clamped)
                        elif args.bin_mode == 'MAX':
                            val = np.max(values_in_bin_clamped)
                        else:
                            val = 0
                            
                        # Assign to tensor
                        image[i_channel, b_idx, station_idx] = val

            # --- E. Validation ---
            # If a station's trace was entirely outside the time window, the image slice 
            # for that station will be all zeros. We should adjust the hit count.
            if not np.any(image[:, :, station_idx]): # If there are no values other than 0 in this station's traces:
                logger.warning(f'Station {station.get_id()} trace is outside time window. Removing from hit count.')
                n_detecting_stations -= 1 # Reduce the hit count
                logger.info(f"Event {i_event}: Adjusted Hit Count -> {n_detecting_stations}")


        # --- F. Save Image, Labels, Hit Count ---
        # Get Primary Vertex (Truth Label)
        primary = event.get_primary()
        vertex = primary.get_parameter(parameters.particleParameters.vertex)

        album.append(image)
        vertices.append(vertex)
        n_detecting_stations_arr.append(n_detecting_stations)


    # --------------------------------------------------------------------------
    # 4. WRITE OUTPUT (HDF5)
    # --------------------------------------------------------------------------
    logger.info("Creating H5py File")
    
    with h5py.File(args.output_file, 'w') as hf:
        # Metadata attributes
        hf.attrs['n_channels'] = N_CHANNELS
        hf.attrs['n_bins'] = args.time_bins
        hf.attrs['n_stations'] = N_STATIONS

        # 1. Image Data (The "Album")
        hf.create_dataset('album',
                          data=np.array(album, dtype='float32'),
                          chunks=(1,(N_CHANNELS,args.time_bins,N_STATIONS)),       # 1 chunk = 1 image!
                          shuffle=True)      # Improve compression
        
        # 2. Vertex Labels (XYZ coordinates)
        hf.create_dataset('vertices',
                          data=np.array(vertices, dtype='float32'),
                          chunks=True)
        
        # 3. Hit Counts (Auxiliary info)
        hf.create_dataset('station_hit_count',
                          data=np.array(n_detecting_stations_arr, dtype='int32').reshape(-1, 1), # Reshape to make it a column
                          chunks=True)

    # Summary
    logger.info(f"{len(album)} events saved to {args.output_file}")
    
    end_time = time.time()
    logger.info("Saving complete.")
    logger.info(f"Runtime: {end_time - start_time:.2f} s")

if __name__ == "__main__":
    main()