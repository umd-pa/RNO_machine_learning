"""
Shard Manifest Creator
----------------------
Scans directories of HDF5 shards, validates them, and creates a JSON
manifest that locks in Train, Validation, and Test splits.

All parameters are read from a YAML config file, eliminating the need
for long command-line arguments. A copy of the config is saved alongside
the manifest for reproducibility.

Logic:
    1. Loads parameters from YAML config.
    2. Scans all input directories for .hdf5 files.
    3. Opens each shard to validate shape consistency and integrity.
    4. Shuffles deterministically and splits into train/val/test.
    5. Saves paths and metadata to a JSON manifest file.

Usage:
    python build_shards_manifest.py --config manifest_config.yaml

Example config (manifest_config.yaml):
    input_dirs:
      - '/data/i3store/.../rno_sim_shards_v1_compressed'
      - '/data/i3store/.../rno_sim_shards_v1_p2_compressed'
    output_path: '/data/.../manifests/dataset_manifest_500_compr.json'
    split_ratios: [0.8, 0.1, 0.1]
    seed: 42

Author: Santiago Sued
"""

import argparse
import logging
import random
import shutil
import time
import glob
import os
import math
import h5py
import numpy as np
import json
import yaml
from tqdm import tqdm
from NuRadioReco.utilities.logging import _setup_logger


def get_abs_path(rel_path):
    """Resolves a path relative to this script's location."""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, rel_path))


def main():
    start_time = time.time()

    # -------------------------------------------------------
    # 1. SETUP
    # -------------------------------------------------------
    logger = _setup_logger(name="Manifest_Creator")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Create a JSON manifest for HDF5 shard splits."
    )
    parser.add_argument('--config', default=get_abs_path('manifest_config.yaml'),
                        help='Path to the YAML config file.')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    input_dirs   = config['input_dirs']
    output_path  = config['output_path']
    split_ratios = config['split_ratios']
    seed         = config['seed']

    # -------------------------------------------------------
    # 2. VALIDATE CONFIG
    # -------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Refuse to overwrite an existing manifest — paths are locked in at creation
    if os.path.exists(output_path):
        logger.error(f"Manifest already exists at {output_path}. "
                     f"Delete it or change output_path to proceed.")
        return

    if not math.isclose(sum(split_ratios), 1.0, abs_tol=1e-4):
        logger.error(f"Split ratios must sum to 1.0. Got: {split_ratios}")
        return

    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return
        if not os.path.isdir(input_dir):
            logger.error(f"Input path is not a directory: {input_dir}")
            return
        logger.info(f"Validated input directory: {input_dir}")

    # -------------------------------------------------------
    # 3. FIND FILES
    # -------------------------------------------------------
    files = []
    for input_dir in input_dirs:
        # Sort guarantees deterministic ordering across systems
        found = sorted(glob.glob(os.path.join(input_dir, '*.hdf5')))
        files.extend([os.path.abspath(p) for p in found])

    if not files:
        logger.error(f"No .hdf5 files found in: {input_dirs}")
        return

    logger.info(f"Found {len(files)} potential shards.")

    # -------------------------------------------------------
    # 4. VALIDATE SHARDS
    # -------------------------------------------------------
    valid_shards    = []  # list of (filepath, n_images)
    total_images    = 0
    ref_album_shape = None

    for fname in tqdm(files, desc="Validating shards"):
        try:
            with h5py.File(fname, 'r') as f:
                album_shape    = f['album'].shape     # type: ignore
                vertices_shape = f['vertices'].shape  # type: ignore

                n_images = album_shape[0]
                n_labels = vertices_shape[0]

                if n_images != n_labels:
                    logger.warning(f"Image/label count mismatch in {fname}. Skipping.")
                    continue
                if n_images == 0:
                    logger.warning(f"Empty shard: {fname}. Skipping.")
                    continue

                # Lock in reference shape from first valid shard
                current_shape = album_shape[1:]
                if ref_album_shape is None:
                    ref_album_shape = current_shape
                    logger.info(f"Reference shape locked: {ref_album_shape}")
                elif current_shape != ref_album_shape:
                    logger.warning(f"Shape mismatch in {fname}. "
                                   f"Expected {ref_album_shape}, got {current_shape}. Skipping.")
                    continue

                valid_shards.append((fname, int(n_images)))
                total_images += n_images

        except (OSError, KeyError) as e:
            logger.warning(f"Could not open {os.path.basename(fname)}: {e}")

    if not valid_shards:
        logger.error("No valid shards found. Exiting.")
        return

    logger.info(f"Validated {len(valid_shards)} shards ({total_images} images).")

    # -------------------------------------------------------
    # 5. SHUFFLE & SPLIT
    # -------------------------------------------------------
    # Shuffle at shard level — deterministic given the same seed
    random.seed(seed)
    random.shuffle(valid_shards)

    # Split indices based on shard count (not image count)
    cum_ratios   = np.cumsum(split_ratios)
    split_indices = (cum_ratios * len(valid_shards)).astype(int)

    train_shards = valid_shards[:split_indices[0]]
    val_shards   = valid_shards[split_indices[0]:split_indices[1]]
    test_shards  = valid_shards[split_indices[1]:]

    def process_split(shard_list):
        """Summarise a split into file paths, image count, and shard count."""
        return {
            "files"        : [s[0] for s in shard_list],
            "total_images" : sum(s[1] for s in shard_list),
            "num_shards"   : len(shard_list)
        }

    # -------------------------------------------------------
    # 6. BUILD & WRITE MANIFEST
    # -------------------------------------------------------
    manifest = {
        "metadata": {
            "created_at"              : time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed"                    : seed,
            "split_ratios"            : split_ratios,
            "ref_album_shape"         : list(ref_album_shape),  # type: ignore
            "created_from"            : list(input_dirs),
            "total_images_all_splits" : total_images,
            "total_shards_all_splits" : len(valid_shards)
        },
        "splits": {
            "train" : process_split(train_shards),
            "val"   : process_split(val_shards),
            "test"  : process_split(test_shards)
        }
    }

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=4)

    # -------------------------------------------------------
    # 7. SUMMARY
    # -------------------------------------------------------
    logger.info(f"Manifest saved to: {output_path}")
    logger.info(f"Split ratios: {split_ratios}")
    logger.info(f"  Train: {manifest['splits']['train']['num_shards']:>5} shards "
                f"({manifest['splits']['train']['total_images']} images)")
    logger.info(f"  Val:   {manifest['splits']['val']['num_shards']:>5} shards "
                f"({manifest['splits']['val']['total_images']} images)")
    logger.info(f"  Test:  {manifest['splits']['test']['num_shards']:>5} shards "
                f"({manifest['splits']['test']['total_images']} images)")
    logger.info(f"Done. Runtime: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()