"""
Virtual Dataset Creator
-------------------------------
Scans one or more directories of HDF5 shards (from Step 3) and creates 
Train, Validation, and Test Virtual Datasets (VDS) that map them all together.

Logic:
1. Scans all provided input folders for .hdf5 files.
2. Opens each file to read 'n_images' and shape attributes.
3. Verifies all files match the configuration of the first file.
4. Shuffles and splits the list of valid files into Train/Val/Test.
5. Creates Virtual Layouts for each split.

Usage Note:
If multiple input directories are provided via --input_dirs, an explicit 
--output_dir must be specified to save the resulting .vds files.

Author: Santiago Sued
"""

import argparse
import logging
import random
import time
import glob
import os
import math
import h5py
import numpy as np
from tqdm import tqdm
from NuRadioReco.utilities.logging import _setup_logger

def create_vds_split(shard_list, output_path, ref_album_shape, logger):
    """
    Creates a single Virtual HDF5 dataset from a list of shard files.
    """
    if not shard_list:
        logger.warning(f"No shards provided for {output_path}. Skipping.")
        return

    # 1. Calculate total images for THIS specific split
    total_images = sum(n_images for _, n_images in shard_list)
    
    # 2. Define shapes for the layouts
    album_shape = (total_images,) + ref_album_shape
    vertices_shape = (total_images, 3)
    hit_counts_shape = (total_images, 1)
    
    # 3. Initialize Layouts
    layout_album = h5py.VirtualLayout(shape=album_shape, dtype='float32')
    layout_vertices = h5py.VirtualLayout(shape=vertices_shape, dtype='float32')
    layout_hit_counts = h5py.VirtualLayout(shape=hit_counts_shape, dtype='float32') 
    
    # 4. Map Shards
    cursor = 0
    for fname, n_images in shard_list:
        source_album_shape = (n_images,) + ref_album_shape
        
        # Create sources (reading from the original simulation names)
        vsource_album = h5py.VirtualSource(fname, 'album', shape=source_album_shape)
        vsource_vertices = h5py.VirtualSource(fname, 'vertices', shape=(n_images, 3))
        vsource_hit_count = h5py.VirtualSource(fname, 'station_hit_count', shape=(n_images, 1))

        # Map into the master layout using the cursor
        layout_album[cursor : cursor + n_images] = vsource_album
        layout_vertices[cursor : cursor + n_images] = vsource_vertices
        layout_hit_counts[cursor : cursor + n_images] = vsource_hit_count
        
        cursor += n_images

    # 5. Write Result
    logger.info(f"Writing {total_images} images to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        f.create_virtual_dataset('album', layout_album)
        f.create_virtual_dataset('vertices', layout_vertices)
        f.create_virtual_dataset('station_hit_count', layout_hit_counts)
        
        # 6. Copy Metadata Attributes safely
        first_file = shard_list[0][0]
        with h5py.File(first_file, 'r') as ref:
            f.attrs['n_images'] = total_images 
            for attr_name in ['n_channels', 'n_bins', 'n_stations']:
                if attr_name in ref.attrs:
                    f.attrs[attr_name] = ref.attrs[attr_name]

def main():
    start_time = time.time()
    
    # 1. Setup
    logger = _setup_logger(name="VDS_Creator")
    logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser(description='Create Virtual Dataset Splits')
    parser.add_argument('--input_dirs', nargs='+', required=True, help='Directories containing .hdf5 shards')
    parser.add_argument('--output_dir', default=None, help='Directory to save .vds files. Defaults to input_dirs.')
    parser.add_argument('--split_ratios', nargs=3, type=float, default=[0.8, 0.1, 0.1], help='Train/Test/Val ratios. Default: 0.8 0.1 0.1')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible shuffling.')
    args = parser.parse_args()

    # Require an output directory if multiple input_dirs are provided
    if len(args.input_dirs) > 1 and args.output_dir is None:
        logger.error(f"If more than one input directories are provided, the user MUST specify an output directory to save the .vds files!")
        return

    # Validate ratios add up to ~1.0
    if not math.isclose(sum(args.split_ratios), 1.0, abs_tol=1e-4):
        logger.error(f"Split ratios must sum to 1.0! Provided: {args.split_ratios} (Sum: {sum(args.split_ratios)})")
        return

    # Determine output directory
    out_dir = args.output_dir if args.output_dir else args.input_dirs[0]
    os.makedirs(out_dir, exist_ok=True)

    # 2. Find Files
    files = []
    for input_dir in args.input_dirs:
        search_path = os.path.join(input_dir, '*.hdf5')
        # Extend makes sure to grab the iterable and put every item individually into the files array!
        files.extend([os.path.abspath(p) for p in sorted(glob.glob(search_path))]) # Use full path to files, to make sure they are easily found!

    if not files:
        logger.error(f"No HDF5 files found in {args.input_dirs}")
        return

    logger.info(f"Found {len(files)} potential shards.")

    # 3. Scan Files & Validate Shapes
    valid_shards = []  # Tuples of (filename, n_images)
    total_images = 0
    ref_album_shape = None
    
    for fname in tqdm(files, desc="Validating Shards"):
        try:
            with h5py.File(fname, 'r') as f:
                # 1. Read the ground-truth shapes directly from the datasets
                album_shape_full = f['album'].shape  #type: ignore     # e.g., (100, 24, 1024, 4)
                vertices_shape_full = f['vertices'].shape #type: ignore # e.g., (100, 3)

                n_images = album_shape_full[0]
                n_labels = vertices_shape_full[0]

                # 2. Safety Check: Do inputs and targets match?
                if n_images != n_labels:
                    logger.warning(f"Corrupted Shard {fname}: {n_images} albums but {n_labels} labels! Skipping.")
                    continue

                if n_images == 0:
                    logger.warning(f"Shard {fname} is empty! Skipping.")
                    continue 

                # 3. Extract just the feature dimensions: (Channels, Bins, Stations)
                # We slice from index 1 to the end to ignore the batch/event dimension
                current_shape = album_shape_full[1:] 
                
                # 4. Compare against reference
                if ref_album_shape is None:
                    ref_album_shape = current_shape
                    logger.info(f"Reference Shape locked in: {ref_album_shape} (Channels, Bins, Stations)")
                elif current_shape != ref_album_shape:
                    logger.warning(f"Shape Mismatch in {fname}! Expected {ref_album_shape}, got {current_shape}. Skipping.")
                    continue
                
                valid_shards.append((fname, n_images))
                total_images += n_images
                
        except (OSError, KeyError) as e:
            logger.warning(f"Could not open {os.path.basename(fname)} or missing datasets: {e}")

    if ref_album_shape is None:
        logger.error("No valid reference shape found. Exiting.")
        return

    if total_images == 0:
        logger.error("No valid images found across all shards. Exiting.")
        return

    logger.info(f"Validated {len(valid_shards)} shards. Total images: {total_images}")

    # 4. Shuffle shards
    random.seed(args.seed)
    random.shuffle(valid_shards)

    # 5. Calculate Split Indices
    cum_ratios = np.cumsum(args.split_ratios)
    split_indices = (cum_ratios * len(valid_shards)).astype(int)

    train_shards = valid_shards[:split_indices[0]]                 
    val_shards   = valid_shards[split_indices[0]:split_indices[1]] 
    test_shards  = valid_shards[split_indices[1]:]

    logger.info(f"Splits: Train({len(train_shards)}), Val({len(val_shards)}), Test({len(test_shards)})")

    # 6. Build Virtual Datasets
    create_vds_split(train_shards, os.path.join(out_dir, 'train.vds'), ref_album_shape, logger)
    create_vds_split(val_shards, os.path.join(out_dir, 'val.vds'), ref_album_shape, logger)
    create_vds_split(test_shards, os.path.join(out_dir, 'test.vds'), ref_album_shape, logger)

    logger.info(f"Done. Runtime: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()