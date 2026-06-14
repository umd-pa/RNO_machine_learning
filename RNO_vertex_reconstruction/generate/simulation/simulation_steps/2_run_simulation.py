"""
Step 2: Detector Simulation
---------------------------
This script simulates the detector response for the generated neutrino events.
It applies hardware responses (filters, amps) and trigger logic to the signals.

Original framework: NuRadioMC
Adapted from: https://github.com/nu-radio/NuRadioMC/tree/vertex_reco_merge/NuRadioReco/examples/RNO_energy_reconstruction
Adapted by: Santiago Sued
"""

from __future__ import absolute_import, division, print_function
import argparse
import logging
import time
import os

# --- NuRadioMC / NuRadioReco Imports ---
from NuRadioReco.utilities import units
from NuRadioReco.utilities.logging import _setup_logger
from NuRadioMC.simulation import simulation

# Detector Components
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator

def get_abs_path(rel_path):
    """
    Converts a relative path to an absolute path based on 
    the location of THIS script (create_dagman.py).
    """
    # This gets the folder where create_dagman.py lives (e.g., .../jobs/)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, rel_path))

def main():
    # --- Argument Parsing ---
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Step 2: Detector Simulation')

    parser.add_argument('--input_file',
                        type=str,
                        default='get_abs_path("../simulation_data/neutrinos.hdf5")',
                        help='Path to the HDF5 file containing generated neutrino events to be simulated.')
    
    parser.add_argument('--output_file',
                        type=str,
                        default='get_abs_path("../simulation_data/simulated_events.nur")',
                        help='Name of the .nur file the simulated events will be written into.')
    
    parser.add_argument('--detector_file',
                        type=str,
                        default='get_abs_path("../simulation_data/RNO_four_stations.json")',
                        help='Path to the JSON file containing the detector description.')
    
    parser.add_argument('--config_file',
                        type=str,
                        default='get_abs_path("../simulation_data/config.yaml")',
                        help='Path to the .yaml file containing the simulation configuration.')
    
    parser.add_argument('--noise_level',
                        type=float,
                        default=10.,
                        help='Root mean square (in millivolt) of the noise to be simulated. Note that this noise should include the amplifier response.')

    args = parser.parse_args()

    # --- Logger Setup ---
    logger = _setup_logger(name="DetSim")
    logger.setLevel(logging.INFO)

    # --- Initialize Detector Modules ---
    # These modules define how the hardware modifies the signal.

    # Im not sure we use this (❓)
    # channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

    # 1. Trigger Simulator: Simple high/low threshold check
    highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()

    # 2. Hardware Response: Applies antennas, amps, and cable effects (RNO-G specific)
    hardware_response = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()

    # Define noise in proper units
    noise_level = args.noise_level * units.mV

    # --- Define Custom Simulation Class ---
    class mySimulation(simulation.simulation):

        def _detector_simulation_filter_amp(self, evt, station, det):
            # Simulate the hardware response
            hardware_response.run(evt, station, det, sim_to_data=True)

        def _detector_simulation_trigger(self, evt, station, det):
            # Simulate the trigger
            highLowThreshold.run(evt, station, det,
                                        threshold_high=2. * noise_level,
                                        threshold_low=-2. * noise_level,
                                        triggered_channels=[0, 1],
                                        number_concidences=2,  # 2/4 majority logic
                                        trigger_name='main_trigger'
                                )

    # --- Run Simulation ---
    logger.info("Starting detector simulation...")

    sim = mySimulation(
        inputfilename=args.input_file,
        outputfilename=get_abs_path('../simulation_data/output.hdf5'), # Good file for debugging but usually unnecessary
        detectorfile=args.detector_file,
        outputfilenameNuRadioReco=args.output_file, # Actual .nur output
        config_file=args.config_file,
        file_overwrite=True,
        write_detector=False,
        trigger_channels=[0,1] # Added to only simulate e-fields for trigger channels (speed up simulation)
    )
    sim.run()
    
    # --- Wrap up ---
    end_time = time.time()
    logger.info("Simulation complete.")
    logger.info(f"Runtime: {end_time - start_time:.2f} s")

if __name__ == "__main__":
    main()