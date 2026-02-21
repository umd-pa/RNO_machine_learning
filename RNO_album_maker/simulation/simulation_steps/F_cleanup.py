"""
Step F: Clean up
----------------
This script runs after a dagman and empties out all intermediate files used in simulation.

Author: Santiago Sued
"""
import os
import glob
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', required=True, help="Directory to clean")
    args = parser.parse_args()

    # Safety Check: Ensure we are targeting the right kind of folder
    if "simulation_data" not in args.target_dir:
        print(f"SAFETY ABORT: Target directory {args.target_dir} does not look like a staging folder.")
        sys.exit(1)

    # Find files
    nur_files = glob.glob(os.path.join(args.target_dir, "*.nur"))
    hdf5_files = glob.glob(os.path.join(args.target_dir, "*.hdf5"))
    
    all_files = nur_files + hdf5_files

    if not all_files:
        print("Folder is already clean.")
        return

    print(f"Deleting {len(nur_files)} intermediate .nur files and {len(hdf5_files)} intermediate .hdf5 files in {args.target_dir}...")
    print('-'*20)
    
    # Loop through the combined list and delete
    for f in all_files:
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error deleting {f}: {e}")

    print("Cleanup complete.")

if __name__ == "__main__":
    main()