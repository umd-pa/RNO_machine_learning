"""
Step 1: Neutrino Event Generator
--------------------------------
This script generates a list of neutrino events within a specified cylinder volume 
using the NuRadioMC framework. The neutrinos are generated using the GZK-2 + IceCube 2022 spectrum
to stay as close as possible to Critoph Welling's paper. It serves as the initial input for the simulation pipeline.

Original framework: NuRadioMC
Adapted by: Santiago Sued
"""

from __future__ import absolute_import, division, print_function
import argparse
import time

# NuRadioMC imports
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
from NuRadioReco.utilities.logging import _setup_logger

def main():
    start_time = time.time()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Step 1: Neutrino Generator')
    
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='../simulation_data/neutrinos.hdf5',
        help='Full path to the output .hdf5 file'
    )
    
    parser.add_argument(
        '--n_nu', 
        type=int, 
        default=1_000, 
        help='Number of neutrinos to generate (Higher is better for Shards!)'
    )

    args = parser.parse_args()

    # --- Logger Setup ---
    # Setting up the logger to print info to console/logs
    logger = _setup_logger(name="Generator")
    
    # --- Simulation Volume Definition ---
    # Defining a cylinder volume for event generation.
    # Note: Volume is kept artificially small/close to ensure trigger efficiency for testing.
    volume = {
        'fiducial_zmin': -2.7 * units.km,  # Depth of ice sheet at South Pole
        'fiducial_zmax': 0 * units.km,     # Surface
        'fiducial_rmin': 0 * units.km,     # Center
        'fiducial_rmax': 3.9 * units.km    # Radius
    }

    logger.info(f"Initializing generation of {args.n_nu} neutrino events...")

    # --- Event Generation ---
    # Generating events using the GZK-2 + IceCube 2022 spectrum
    # Energy range: 50 PeV to 10 EeV
    generate_eventlist_cylinder(
        filename=args.output_file, 
        n_events=args.n_nu, 
        Emin=5e16 * units.eV, 
        Emax=1e19 * units.eV, 
        volume=volume, 
        spectrum='GZK-2+IceCube-nu-2022'
    )

    # --- Wrap up ---
    end_time = time.time()
    logger.info("Generation complete.")
    logger.info(f"Runtime: {end_time - start_time:.2f} s")

if __name__ == "__main__":
    main()