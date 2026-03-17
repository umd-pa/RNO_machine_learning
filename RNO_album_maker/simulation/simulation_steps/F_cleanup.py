"""
Step F: Clean Up
----------------
Deletes all intermediate simulation files (.nur, .hdf5) from a target
staging directory after a DAGMan run completes.

.nur files are always preserved before deletion — either to a
user-specified directory (protected from overwrite) or to the default
latest_sim_nurs staging directory (cleared and replaced each run).

Usage:
    python cleanup.py
    python cleanup.py --target_dir /path/to/simulation_data
    python cleanup.py --store_nurs                          # save to default latest_sim_nurs
    python cleanup.py --store_nurs /custom/path/to/nurs    # save to custom directory

Author: Santiago Sued
"""
import os
import glob
import sys
import shutil
import argparse
from tqdm import tqdm


def get_abs_path(rel_path):
    """
    Resolves a path relative to this script's location.
    Robust to being called from any working directory.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, rel_path))


def main():

    default_nur_dir = get_abs_path('../simulation_data/latest_sim_nurs')

    parser = argparse.ArgumentParser(
        description="Clean up intermediate simulation files after a DAGMan run."
    )
    parser.add_argument('--target_dir',
                        default=get_abs_path('../simulation_data'),
                        help="Directory to clean. Default: ../simulation_data")
    parser.add_argument('--store_nurs',
                        nargs='?',
                        const=default_nur_dir,
                        default=default_nur_dir,
                        metavar='DIR',
                        help="Destination for .nur files before deletion. "
                             "If DIR is omitted, uses ../simulation_data/latest_sim_nurs. "
                             "User-specified directories are protected from overwrite.")
    args = parser.parse_args()

    # -------------------------------------------------------
    # SAFETY: Only clean directories that look like staging
    # -------------------------------------------------------
    if "simulation_data" not in args.target_dir:
        print(f"SAFETY ABORT: {args.target_dir} does not look like a staging folder.")
        sys.exit(1)

    # Discover intermediate files
    nur_files  = glob.glob(os.path.join(args.target_dir, "*.nur"))
    hdf5_files = glob.glob(os.path.join(args.target_dir, "*.hdf5"))

    print(f"Found {len(nur_files)} .nur and {len(hdf5_files)} .hdf5 files in {args.target_dir}")
    print('-' * 60)

    # -------------------------------------------------------
    # NUR HANDLING: copy before any deletion occurs
    # User-specified dirs are protected; default dir is replaced
    # -------------------------------------------------------
    dir_to_nurs    = args.store_nurs
    user_specified = dir_to_nurs != default_nur_dir

    if os.path.exists(dir_to_nurs):
        if user_specified:
            print(f"SAFETY ABORT: {dir_to_nurs} already exists. "
                  f"Refusing to overwrite an explicitly specified directory.")
            sys.exit(1)

        # Default directory — clear stale .nur files from previous run
        print(f"Clearing existing .nur files from {dir_to_nurs}...")
        for old_nur in tqdm(glob.glob(os.path.join(dir_to_nurs, "*.nur")),
                            desc="Clearing stale .nur files", unit="file"):
            try:
                os.remove(old_nur)
            except OSError as e:
                raise RuntimeError(f"Failed to remove {old_nur}: {e}")
    else:
        os.makedirs(dir_to_nurs, exist_ok=True)

    print(f"Copying {len(nur_files)} .nur files to {dir_to_nurs}...")
    for nur_file in tqdm(nur_files, desc="Copying .nur files", unit="file"):
        try:
            shutil.copy2(nur_file, dir_to_nurs)
        except shutil.Error as e:
            raise RuntimeError(f"Failed to copy {nur_file}: {e}")
    print(f"Done! Copied {len(nur_files)} .nur files to {dir_to_nurs}.")
    print('-' * 60)

    # -------------------------------------------------------
    # CLEANUP: Delete all intermediate files from target_dir
    # .nur files are already safely copied above
    # -------------------------------------------------------
    all_files = nur_files + hdf5_files

    if not all_files:
        print("Folder is already clean.")
        return

    print(f"Deleting {len(nur_files)} .nur and {len(hdf5_files)} .hdf5 files from {args.target_dir}...")
    for f in tqdm(all_files, desc="Cleaning up", unit="file"):
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error deleting {f}: {e}")

    print("Cleanup complete.")


if __name__ == "__main__":
    main()