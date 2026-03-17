from h5py import File
from tqdm import tqdm
import h5py
import numpy as np
import glob
import os
import shutil
import time
import sys

def consolidate_shards(input_dir, output_dir, images_per_shard):
    """
    Merges many small shards into fewer large contiguous ones.
    Reads input shards sequentially, buffers images until the target
    shard size is reached, then writes a single contiguous HDF5 file.

    Args:
        input_dir:        directory containing your current .hdf5 shards
        output_dir:       directory to write consolidated shards
        images_per_shard: target number of images per output shard
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    files = sorted(glob.glob(os.path.join(input_dir, "*.hdf5")))
    print(f"Found {len(files)} input shards in {input_dir}")
    print(f"Target: ~{images_per_shard} images per output shard")
    print(f"Expected output shards: ~{len(files) * 500 // images_per_shard}")

    buffer_imgs   = []  # accumulates numpy arrays from input shards
    buffer_lbls   = []
    buffer_counts = []  # station_hit_count
    buffer_size   = 0   # total images currently in buffer
    shard_out_idx = 0   # output shard counter

    def write_shard(imgs, lbls, counts, idx):
        """Concatenates buffer and writes a single contiguous HDF5 shard."""
        out_path = os.path.join(output_dir, f'shard_{idx:04d}.hdf5')
        combined_imgs   = np.concatenate(imgs,   axis=0)
        combined_lbls   = np.concatenate(lbls,   axis=0)
        combined_counts = np.concatenate(counts, axis=0)
        with h5py.File(out_path, 'w') as f:
            # Apparently chunking is faster
            f.create_dataset('album', chunks=(1, 24, 1024, 4),data=combined_imgs)
            f.create_dataset('vertices',          data=combined_lbls)
            f.create_dataset('station_hit_count', data=combined_counts)
        tqdm.write(f"  --> Written shard_{idx:04d}.hdf5 "
                   f"({len(combined_imgs)} images, "
                   f"{os.path.getsize(out_path)/1e6:.1f} MB)")
        return idx + 1

    for file_path in tqdm(files, desc="Consolidating shards", unit="shard"):
        try:
            with h5py.File(file_path, 'r', locking=False) as f:
                buffer_imgs.append(f['album'][:])
                buffer_lbls.append(f['vertices'][:])
                buffer_counts.append(f['station_hit_count'][:])
                buffer_size += f['album'].shape[0]
        except Exception as e:
            tqdm.write(f"ERROR reading {os.path.basename(file_path)}: {e}")
            continue

        # Once buffer is full, flush to disk
        if buffer_size >= images_per_shard:
            shard_out_idx = write_shard(buffer_imgs, buffer_lbls, buffer_counts, shard_out_idx)
            buffer_imgs   = []
            buffer_lbls   = []
            buffer_counts = []
            buffer_size   = 0

    # Write any remaining images that didn't fill a complete shard
    if buffer_imgs:
        print(f'Flushing remaining images {len(buffer_imgs)}, to final shard shard_{shard_out_idx:04d}.hdf5...')
        shard_out_idx = write_shard(buffer_imgs, buffer_lbls, buffer_counts, shard_out_idx)

    print(f"\nDone! Written {shard_out_idx} consolidated shards to {output_dir}")

def recompress_shards(input_dir, output_dir):
    """
    Copies all HDF5 shards from input_dir to output_dir, preserving all
    attributes, keys, shapes, and dtypes from the original, but rewriting
    datasets with gzip level 1 compression, original chunk layout, and no
    byte shuffle filter.

    Args:
        input_dir  (str): Directory containing original .hdf5 shards.
        output_dir (str): Directory to write recompressed shards.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    files = sorted(glob.glob(os.path.join(input_dir, "*.hdf5")))
    print(f"Found {len(files)} shards in {input_dir}")

    for file_path in tqdm(files, desc="Recompressing shards", unit="shard"):
        file_name   = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name)

        if os.path.exists(output_path):
            tqdm.write(f"Skipping {file_name} (already exists)")
            continue

        try:
            with h5py.File(file_path, 'r') as f_in:
                with h5py.File(output_path, 'w') as f_out:

                    # Preserve all file-level attributes exactly
                    for attr_name, attr_value in f_in.attrs.items():
                        f_out.attrs[attr_name] = attr_value

                    for key in f_in.keys():
                        data   = f_in[key][:]
                        chunks = f_in[key].chunks  # preserve original chunk layout
                        assert chunks is not None, f"Expected chunked album dataset in {file_name}"
                        if key == 'album':
                            f_out.create_dataset(key,
                                                data=data,
                                                dtype=f_in[key].dtype,  # preserve original dtype
                                                chunks=chunks,
                                                compression="gzip",
                                                compression_opts=1,
                                                shuffle=False)
                        else:
                            f_out.create_dataset(key,
                                                data=data)

        except Exception as e:
            tqdm.write(f"ERROR on {file_name}: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            continue

    print(f"Done! Recompressed {len(files)} shards to {output_dir}")

def inspect_hdf5_layout(file_path):
    print(f"Inspecting: {file_path}")
    print("-" * 50)
    
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                print(f"Dataset: {key}")
                print(f"  - Shape:       {item.shape}")
                print(f"  - Dtype:       {item.dtype}")
                print(f"  - Chunk Size:  {item.chunks}")
                print(f"  - Compression: {item.compression}")
                
                if item.chunks:
                    # Calculate how many chunks exist in the file
                    # This helps identify if there are too many small files
                    n_chunks = 1
                    for s, c in zip(item.shape, item.chunks):
                        n_chunks *= (s // c + (1 if s % c != 0 else 0))
                    print(f"  - Total Chunks: {int(n_chunks):,}")
                else:
                    print("  - [!] Warning: Dataset is NOT chunked (Contiguous layout)")
                print("-" * 50)

def rechunk_shards(input_dir, output_dir):
    # 1. Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Get list of all .hdf5 files
    files = sorted(glob.glob(os.path.join(input_dir, "*.hdf5")))
    print(f"Found {len(files)} shards to process.")

    # Optimized Chunk Shape: (1 sample, 24 bins, 1024 stations, 4 channels)
    NEW_CHUNKS = (1, 24, 1024, 4)

    for file_path in tqdm(files, desc="Rechunking Shards"):
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name)

        with h5py.File(file_path, 'r') as f_in:
            with h5py.File(output_path, 'w') as f_out:
                # Copy attributes (metadata)
                for attr_name, attr_value in f_in.attrs.items():
                    f_out.attrs[attr_name] = attr_value

                # Process each dataset
                for key in f_in.keys():
                    data = f_in[key]
                    
                    if key == 'album':
                        # Apply the new chunking to the heavy image data
                        f_out.create_dataset(
                            key, 
                            data=data, 
                            chunks=NEW_CHUNKS,
                            shuffle=True,      # Keeps it fast for the 4090
                            compression=None   # High-speed I/O
                        )
                    else:
                        # For small datasets (vertices, etc), 
                        # just copy them as they are
                        f_out.create_dataset(key, data=data, chunks=True)

def copy_event(album_source, album_dest_path, event_idx):
    """
    Copies a specific event group from one HDF5 file to another.
    Creates the destination file if it doesn't exist.
    """
    
    # Ensure the destination directory exists
    dest_dir = os.path.dirname(album_dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    # Use 'a' (append) for dest. 'w' would overwrite/delete the existing file!
    with File(album_source, 'r') as source_album, \
         File(album_dest_path, 'a') as dest_album:

        event_key = f'event{event_idx}'

        # 1. Check if the event actually exists in source
        if event_key not in source_album.keys():
            print(f"Error: {event_key} not found in {album_source}")
            return

        # 2. Check if event exists in destination to prevent collision errors
        if event_key in dest_album:
            print(f"Warning: {event_key} already exists in destination. Overwriting.")
            del dest_album[event_key]

        # 3. Perform the copy
        # h5py's .copy() method handles recursive copying of groups and datasets
        source_album.copy(event_key, dest_album)
        
        print(f"Successfully copied {event_key} to {album_dest_path}")

def trainTest_split(album_dir,album_name,train_ratio = 0.8, seed = 42, backup= True):

    base_name = os.path.splitext(album_name)[0]
    with File(os.path.join(album_dir,album_name), 'r') as album, \
        File(os.path.join(album_dir,f'{base_name}_train.hdf5'), 'w') as train_album, \
        File(os.path.join(album_dir,f'{base_name}_test.hdf5'), 'w') as test_album:

        if backup:
            backup_path = os.path.join(album_dir, f"backup_{album_name}")
            print(f'Backing up album to {backup_path}')
            if not os.path.exists(backup_path):
                shutil.copy2(os.path.join(album_dir, album_name), backup_path)

        # Get all event keys and create random split
        all_keys = list(album.keys())
        total_size = len(all_keys)
        split_index = int(total_size * train_ratio)

        np.random.seed(seed)
        shuffled_indices = np.random.permutation(total_size)

        train_indices = shuffled_indices[:split_index]
        test_indices = shuffled_indices[split_index:]

        # Copy training data
        print('\nCopying training data...')
        for i, orig_idx in enumerate(train_indices):
            print(f'Copying training sample {i+1}/{len(train_indices)} (original index {orig_idx})',end='\r',flush=True)
            orig_key = all_keys[orig_idx]
            new_key = f'event{i+1}'  # Reindex starting from 1
            
            train_album.create_group(new_key)
            train_album[new_key]['image'] = album[orig_key]['image'][:]
            train_album[new_key]['label'] = album[orig_key]['label'][:]
        
        # Copy test data
        print('\nCopying training data...')
        for i, orig_idx in enumerate(test_indices):
            print(f'Copying testing sample {i+1}/{len(test_indices)} (original index {orig_idx})',end='\r',flush=True)
            orig_key = all_keys[orig_idx]
            new_key = f'event{i+1}'  # Reindex starting from 1
            
            test_album.create_group(new_key)
            test_album[new_key]['image'] = album[orig_key]['image'][:]
            test_album[new_key]['label'] = album[orig_key]['label'][:]
        
        print(f"Split complete: {len(train_indices)} training, {len(test_indices)} test samples")

def swap_phi_theta(album_path):
    with File(album_path, 'r+') as album:
        num_events = len(album.keys())
        for idx in range(num_events):
            print(f'\rSwapping phis and thetas... ({idx+1}/{num_events})', end='',flush=True)
            event_key = f'event{idx+1}'
            
            # Read the data
            label = album[event_key]['label'][:]
            r, theta, phi = label
            
            # Modify in place
            album[event_key]['label'][:] = [r, phi, theta]
        print('\nDone!')

def copy_with_progress(src, dst, buffer_size=1024*1024):
    """
    Copies a file from src to dst with a simple text progress bar.
    buffer_size: 1MB by default.
    """
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source file {src} not found")

    total_size = os.path.getsize(src)
    copied = 0
    start_time = time.time()

    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        while True:
            buf = fsrc.read(buffer_size)
            if not buf:
                break
            fdst.write(buf)
            copied += len(buf)
            
            # Calculate progress
            percent = (copied / total_size) * 100
            elapsed = time.time() - start_time
            speed = (copied / (1024 * 1024)) / (elapsed + 1e-9) # MB/s
            
            # Print status line (overwriting previous line)
            sys.stdout.write(f"\rCopying: {percent:.1f}% | {copied/1e9:.2f}/{total_size/1e9:.2f} GB | {speed:.2f} MB/s")
            sys.stdout.flush()
    
    print() # Newline after done
    
    # Preserve metadata (timestamps) like copy2 does
    shutil.copystat(src, dst)
    print(f"Backup complete: {dst}")