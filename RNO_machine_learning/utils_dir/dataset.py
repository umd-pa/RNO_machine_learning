from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import threading
import torch
import json
import os

def last_two_path_names(path):
    """Return the last two names of a path."""
    parts = os.path.normpath(path).split(os.sep)
    return os.sep.join(parts[-2:])

import hashlib

def is_valid_hdf5(path):
    try:
        with h5py.File(path, 'r') as f:
            return 'album' in f and 'vertices' in f
    except Exception:
        return False

def _stable_dest_name(src):
    """Creates a unique filename by hashing the source path."""
    path_hash = hashlib.md5(src.encode()).hexdigest()[:8]
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)
    return f"{name}_{path_hash}{ext}"  # e.g. shard_0000001_a3f2b1c4.hdf5

def stage_manifest_to_scratch(manifest_path, cache_dir, force=False):
    """
    Stages all HDF5 shards referenced in a dataset manifest from network
    storage to local scratch, and returns an updated manifest dict with
    all file paths pointing to their cached scratch locations.

    All splits (train/val/test) are staged together from a single manifest,
    ensuring stale cleanup operates across the full dataset and the returned
    manifest is a drop-in replacement for the original.

    Behavior:
    - Files already cached and valid are skipped (instant startup)
    - Corrupt/partial files are flagged for overwrite
    - Cached files not referenced by any split in the manifest are deleted
    - Missing files are copied sequentially to avoid HDD head thrashing
    - Falls back to original paths gracefully if scratch is unavailable

    Args:
        manifest_path  (str): Path to the dataset manifest JSON file.
        cache_dir_name (str): Subdirectory name inside scratch_base for
                              this dataset's cache. Use a unique name per
                              manifest to avoid cross-contamination.
        force         (bool): If True, overwrites all cached files even
                              if valid. Use when shards have been rebuilt
                              at the same source path. Default: False

    Returns:
        dict: Updated manifest with all file paths pointing to scratch
              locations. Same structure and keys as the original manifest.

    Example:
        manifest = stage_manifest_to_scratch(
            manifest_path  = '/data/.../dataset_manifest_25k.json',
            cache_dir_name = 'rno_cache_25k',
        )
        train_paths = manifest['splits']['train']['files']
        test_paths  = manifest['splits']['test']['files']

        # After rebuilding shards at the same source path:
        manifest = stage_manifest_to_scratch(
            manifest_path  = '/data/.../dataset_manifest_25k.json',
            cache_dir_name = 'rno_cache_25k',
            force          = True
        )
    """
    # Load manifest from disk
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Failsafe: if scratch unavailable, return manifest unchanged
    if not os.path.exists(cache_dir):
        print(f"⚠️ Warning: {cache_dir} not found. Falling back to original network paths.")
        return manifest

    os.makedirs(cache_dir, exist_ok=True)

    print(f"🔍 Checking local cache at {cache_dir}...")

    # -------------------------------------------------------
    # COLLECT: Gather all file paths across ALL splits.
    # Staging all splits together ensures stale cleanup removes
    # files not referenced by any split, not just the current one.
    # -------------------------------------------------------
    all_src_paths = []
    for split_name, split_data in manifest['splits'].items():
        all_src_paths.extend(split_data['files'])

    # Build set of expected cache filenames across all splits.
    # _stable_dest_name hashes the full source path — unique even
    # if two directories contain files with the same basename.
    expected_files = {_stable_dest_name(src) for src in all_src_paths}

    # -------------------------------------------------------
    # CLEANUP: Remove cached files not in current manifest.
    # Keeps scratch lean — stale shards from old datasets or
    # previous consolidation runs are deleted automatically.
    # -------------------------------------------------------
    existing_files = set(os.listdir(cache_dir))
    stale_files    = existing_files - expected_files

    if stale_files:
        print(f"🧹 Removing {len(stale_files)} stale cached files...")
        for stale in tqdm(stale_files, desc="Cleaning stale cache", unit="file"):
            stale_path = os.path.join(cache_dir, stale)
            try:
                os.remove(stale_path)
            except OSError as e:
                print(f"⚠️ Could not remove {stale}: {e}")

    # -------------------------------------------------------
    # VALIDATION: Check each expected file against the cache.
    # Three outcomes:
    #   1. Valid HDF5 + not forced → skip (cache hit)
    #   2. Corrupt/partial         → flag for overwrite
    #   3. Missing                 → flag for copy
    # -------------------------------------------------------
    files_to_copy = []
    src_to_dest   = {}  # maps original source path → scratch destination path

    for src in all_src_paths:
        file_name = _stable_dest_name(src)
        dest      = os.path.join(cache_dir, file_name)
        src_to_dest[src] = dest

        if os.path.exists(dest) and not force:
            if is_valid_hdf5(dest):
                continue  # cache hit — skip
            else:
                print(f"⚠️ Partial file detected for {file_name}. Flagging for overwrite.")
        elif os.path.exists(dest) and force:
            print(f"🔄 Force overwrite: {file_name}")

        files_to_copy.append((src, dest))

    # Fast path: everything already cached and valid
    if not files_to_copy:
        print("⚡ All files found in cache! Skipping copy phase.\n")
    else:
        # -------------------------------------------------------
        # COPY: Sequential copies only.
        # Parallel threads cause HDD head thrashing on spinning disk,
        # reducing throughput from 258MB/s to ~90MB/s per thread.
        # NFS source is additionally bottlenecked at ~67MB/s.
        # -------------------------------------------------------
        print(f"🚀 Caching {len(files_to_copy)} new/updated files to local scratch...")
        start_time = time.time()

        for src, dest in tqdm(files_to_copy, desc="Caching to /scratch",
                              unit="file", ncols=100):
            shutil.copy2(src, dest)

        print(f"✅ Cache update complete in {time.time() - start_time:.1f} seconds.\n")

    # -------------------------------------------------------
    # REWRITE MANIFEST: Replace all source paths with scratch
    # paths. Deep copy preserves the original manifest in memory
    # and leaves the JSON file on disk untouched.
    # -------------------------------------------------------
    import copy
    cached_manifest = copy.deepcopy(manifest)
    for split_name, split_data in cached_manifest['splits'].items():
        split_data['files'] = [src_to_dest[p] for p in split_data['files']]

    return cached_manifest

def spher_to_cart(label):
    r,theta,phi = label

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    cartesian_label = label = torch.tensor([x, y, z], dtype=torch.float32)

    return cartesian_label

class AlbumDataset(Dataset):

    """
    Custom PyTorch Dataset for loading particle physics data from HDF5 files.
    
    This dataset handles large HDF5 files containing particle detector images and 
    corresponding vertex position labels. Uses thread-local storage to maintain
    separate file handles per worker thread for efficient multi-threaded loading.
    
    Structure expected in HDF5 file:
    - Root level: event groups (e.g., 'event1', 'event2', ...)
    - Each event group contains:
        - 'image': 2D detector image data
        - 'label': 3D vertex coordinates [x, y, z]
    """
    
    def __init__(self, album_path, transform=None, target_transform=None, preload_keys=True,normalize_labels=True,normalization_factors = None):
        """
        Initialize the AlbumDataset.
        
        Args:
            album_path (str): Path to the HDF5 file containing the dataset
            transform (callable, optional): Transform to apply to images
            target_transform (callable, optional): Transform to apply to labels
            preload_keys (bool): Whether to preload all event keys into memory. If True improves performance but uses more memory.
            normalize_labels (bool): Will calculate mean and std. of labels and normalize utilizing the z-number.
        """
        # Store configuration
        print('entered init!')
        self.path = album_path
        self.transform = transform
        self.target_transform = target_transform
        self.normalize_labels = normalize_labels

        if self.target_transform is not None:
            # Check for Cartesian
            if self.target_transform.__name__ == 'spher_to_cart':
                # Since we converted Spherical -> Cartesian, the labels are now x, y, z
                self.label_names = ('x', 'y', 'z')
                print('Transforming labels to Cartesian coordinates.')
            else:
                # Fallback if transform exists but isn't spher_to_cart
                self.label_names = ('dim1', 'dim2', 'dim3')
                print(f'Unknown transform: {self.target_transform.__name__}')
        else:
            # Fallback if no transform exists
            self.label_names = ('x','y','z')
            print('Leaving labels AS IS.')
        
        # Validate file exists
        if not os.path.exists(album_path):
            raise FileNotFoundError(f"Album file not found: {album_path}")

        # Calculate and store file size for monitoring
        self.space_GB = f'Size of file: {os.path.getsize(album_path)*1e-9:.4f} GB'
        print(self.space_GB)

        # Preload all event keys for faster access (optional optimization)
        self.preload_keys = preload_keys
        with h5py.File(self.path, 'r') as file:
            if self.preload_keys:
                self.event_keys = list(file.keys())
            else:
                self.event_keys = None
            self.num_images = len(file.keys())

        # Save local data for thread.
        self._local = threading.local()

        # Validation - ensure we have data
        if self.num_images == 0:
            raise ValueError(f"No events found in {album_path}")
        
                # Compute normalization statistics if requested
        if self.normalize_labels:
            print('Normalizing labels...')
            if normalization_factors is None:
                print("Computing normalization statistics...")
                self._compute_normalization_stats()
                print("Normalization stats computed:")
                print(f"  {self.label_names[0]}: mean={self.x1_mean:.4f}, std={self.x1_std:.4f}")
                print(f"  {self.label_names[1]}: mean={self.x2_mean:.4f}, std={self.x2_std:.4f}")
                print(f"  {self.label_names[2]}: mean={self.x3_mean:.4f}, std={self.x3_std:.4f}")
                print(f'[{self.x1_mean},{self.x1_std},{self.x2_mean},{self.x2_std},{self.x3_mean},{self.x3_std}]')
            else:
                print("Utilizing inputted normalization statistics")
                self.x1_mean, self.x1_std, self.x2_mean, self.x2_std, \
                self.x3_mean, self.x3_std = normalization_factors
                print(f"  {self.label_names[0]}: mean={self.x1_mean:.4f}, std={self.x1_std:.4f}")
                print(f"  {self.label_names[1]}: mean={self.x2_mean:.4f}, std={self.x2_std:.4f}")
                print(f"  {self.label_names[2]}: mean={self.x3_mean:.4f}, std={self.x3_std:.4f}")
                print(f'[{self.x1_mean},{self.x1_std},{self.x2_mean},{self.x2_std},{self.x3_mean},{self.x3_std}]')
        else:
            # Set to None to indicate no normalization
            self.x1_mean = None
            self.x1_std = None
            self.x2_mean = None
            self.x2_std = None
            self.x3_mean = None
            self.x3_std = None

    def _compute_normalization_stats(self):
        """
        Compute mean and std for x1, x2, x3 across entire dataset.
        This is called once during __init__ if normalize_labels=True.
        """
        x1_values = []
        x2_values = []
        x3_values = []
        
        with h5py.File(self.path, 'r') as f:
            for idx in range(self.num_images):
                print(f'\rCompounding statistics... ({idx}/{self.num_images})',end='',flush=True)
                _, label = self.__getitem__(idx,to_normalize=True)
                x1, x2, x3 = label
                
                x1_values.append(x1)
                x2_values.append(x2)
                x3_values.append(x3)
        
        # Convert to numpy arrays for efficient computation
        x1_values = np.array(x1_values)
        x2_values = np.array(x2_values)
        x3_values = np.array(x3_values)
        
        # Compute statistics
        self.x1_mean = float(np.mean(x1_values))
        self.x1_std = float(np.std(x1_values))
        
        self.x2_mean = float(np.mean(x2_values))
        self.x2_std = float(np.std(x2_values))
        
        self.x3_mean = float(np.mean(x3_values))
        self.x3_std = float(np.std(x3_values))
        
        # Avoid division by zero
        if self.x1_std < 1e-8:
            raise ZeroDivisionError
        if self.x2_std < 1e-8:
            raise ZeroDivisionError
        if self.x3_std < 1e-8:
            raise ZeroDivisionError

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.num_images

    def __getitem__(self, idx,to_normalize=False):
        """
        Fetch a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            tuple: (image, label) where:
                - image: torch.Tensor of shape (1, H, W) - detector image with channel dim
                - label: torch.Tensor of shape (3,) - vertex coordinates [x, y, z]
        """
        # Validate index bounds
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_images}")
        
        # Get or create thread-local file handle
        # This ensures each DataLoader worker has its own file handle
        if not hasattr(self._local, 'file_handle') or self._local.file_handle is None:
            try:
                self._local.file_handle = h5py.File(self.path, 'r')
            except Exception as e:
                raise RuntimeError(f"Failed to open HDF5 file: {e}")

        # Determine event key - use preloaded keys if available, otherwise construct
        if self.event_keys:
            event_key = self.event_keys[idx]
        else:
            event_key = f'event{idx+1}'  # Assumes 1-indexed event naming

        file_handle = self._local.file_handle

        try:
            # Load image data
            # Convert numpy array to PyTorch tensor with float32 dtype
            image = torch.from_numpy(np.array(file_handle[event_key]['image'])).float()
            
            # Load label data (vertex coordinates)
            label = torch.from_numpy(np.array(file_handle[event_key]['label'])).float()
            
        except KeyError as e:
            raise KeyError(f"Event key '{event_key}' not found in dataset or missing 'image'/'label': {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading data for {event_key}: {e}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Add channel dimension (H, W) -> (1, H, W)
        # Most CNN architectures expect a channel dimension
        image = torch.unsqueeze(image, 0)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        # Apply normalization if enabled
        if self.normalize_labels and not to_normalize:
            x1, x2, x3 = label
            
            # Z-score normalization using precomputed statistics
            x1_normalized = (x1 - self.x1_mean) / self.x1_std
            x2_normalized = (x2 - self.x2_mean) / self.x2_std
            x3_normalized = (x3 - self.x3_mean) / self.x3_std
            
            label = torch.tensor([x1_normalized, x2_normalized, x3_normalized], dtype=torch.float32)

        return image, label

    def denormalize_label(self, normalized_label: torch.Tensor):
            """
            Convert normalized label back to original units.
            
            Args:
                normalized_label: torch.Tensor of shape (3,) or (batch, 3) with normalized [x1, x2, x3]
            
            Returns:
                torch.Tensor with denormalized values
            """
            if not self.normalize_labels:
                return normalized_label
            
            if not isinstance(normalized_label, torch.Tensor):
                normalized_label = torch.from_numpy(normalized_label)

            if normalized_label.dim() == 1:
                x1_norm, x2_norm, x3_norm = normalized_label
                x1 = x1_norm * self.x1_std + self.x1_mean
                x2 = x2_norm * self.x2_std + self.x2_mean
                x3 = x3_norm * self.x3_std + self.x3_mean
                return torch.tensor([x1, x2, x3], dtype=torch.float32)
            else:
                # Batch of labels
                x1_norm = normalized_label[:, 0] * self.x1_std + self.x1_mean
                x2_norm = normalized_label[:, 1] * self.x2_std + self.x2_mean
                x3_norm = normalized_label[:, 2] * self.x3_std + self.x3_mean
                return torch.stack([x1_norm, x2_norm, x3_norm], dim=1)

    def get_normalization_factors(self):
        normalization_factors = [self.x1_mean, self.x1_std, self.x2_mean, self.x2_std, \
                                 self.x3_mean, self.x3_std]
        return normalization_factors

    def close(self):
        """
        Close any open file handles.
        
        Should be called when dataset is no longer needed to free resources.
        """
        if hasattr(self._local, 'file_handle') and self._local.file_handle is not None:
            try:
                self._local.file_handle.close()
                self._local.file_handle = None
            except:
                pass  # Ignore errors during cleanup

    def __del__(self):
        """Clean up file handles"""
        self.close()

class ShardAlbumDataset(Dataset):
    """
    Custom PyTorch Dataset for voltage trace data from Virtual HDF5 files.
    
    Structure expected in HDF5 file:
    - 'album': Dataset of shape (num_events, channels, timebins, stations)
    - 'vertices': Dataset of shape (num_events, 3) containing [x, y, z]
    """
    
    def __init__(self, album_path, is_train=False, label_mean=None, label_std=None):
            print('\nInitializing ShardAlbumDataset...')
            self.path = album_path
            
            # FIX 2: Define this FIRST so __del__ never crashes if __init__ fails!
            self.file_handle = None 
            
            if not os.path.exists(album_path):
                raise FileNotFoundError(f"Album file not found: {album_path}")

            self.space_GB = f'Size of file: {os.path.getsize(album_path) * 1e-9:.4f} GB'
            print(self.space_GB)

            # Open file briefly to get metadata and compute stats, then close it!
            with h5py.File(self.path, 'r') as f:
                if 'album' not in f or 'vertices' not in f:
                    raise KeyError("HDF5 file must contain 'album' and 'vertices' datasets at the root level.")
                
                self.num_images = f['album'].shape[0] #type: ignore

                if self.num_images == 0:
                    raise ValueError(f"No events found in {album_path}")
                
                # ========================================================
                # NORMALIZATION LOGIC
                # ========================================================
                if is_train:
                    print(f"--> MODE: Auto-computing normalization stats from {self.path}")
                    
                    # FIX 1: Extract the object, cast it to silence the linter, THEN slice it!
                    import typing
                    vertices_obj = typing.cast(h5py.Dataset, f['vertices'])
                    labels = vertices_obj[:]  # Now it safely becomes a numpy array
                    
                    # Compute stats directly on the Cartesian (N,3) array
                    self.label_mean = torch.tensor(np.mean(labels, axis=0), dtype=torch.float32)
                    self.label_std = torch.tensor(np.std(labels, axis=0), dtype=torch.float32)
                    
                    print("Normalization statistics computed:")
                    print(f"  x = {self.label_mean[0]:.4f} ± {self.label_std[0]:.4f}")
                    print(f"  y = {self.label_mean[1]:.4f} ± {self.label_std[1]:.4f}")
                    print(f"  z = {self.label_mean[2]:.4f} ± {self.label_std[2]:.4f}")
                else:
                    self.label_mean = label_mean
                    self.label_std = label_std
                    
                    # ========================================================
                    # EXPLICIT RAW LABEL CHECK
                    # ========================================================
                    if self.label_mean is None or self.label_std is None:
                        print("--> MODE: No stats provided. Dataset will yield RAW, UNNORMALIZED labels.")
                    else:
                        print("--> MODE: Stats received. Dataset will yield NORMALIZED labels.")
                        print("Normalization statistics received:")
                        print(f"  x = {self.label_mean[0]:.4f} ± {self.label_std[0]:.4f}")
                        print(f"  y = {self.label_mean[1]:.4f} ± {self.label_std[1]:.4f}")
                        print(f"  z = {self.label_mean[2]:.4f} ± {self.label_std[2]:.4f}")

            print(f"Dataset successfully initialized with {self.num_images} events.")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.num_images

    def __getitem__(self, idx):
        """Fetch a single sample directly from the dataset arrays."""
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_images}")
        
        # LAZY LOADING: Open file only when the first batch is requested by a worker process.
        # This prevents HDF5 deadlocks when num_workers > 0 in DataLoader.
        if self.file_handle is None:
            self.file_handle = h5py.File(self.path, libver='latest', mode='r')

            # 1. Extract the objects ONCE and cache them to avoid dictionary lookup overhead
            self.album_ds = self.file_handle['album']
            self.vertices_ds = self.file_handle['vertices']

            # 2. Tell the linter (and Python) these are explicitly Datasets!
            if not isinstance(self.album_ds, h5py.Dataset) or not isinstance(self.vertices_ds, h5py.Dataset):
                raise TypeError("HDF5 keys must point to h5py Datasets.")

        try:
            # DIRECT INDEXING: Grab numpy arrays directly from the HDF5 file (O(1) speed)
            image_np = self.album_ds[idx] #type: ignore
            label_np = self.vertices_ds[idx] #type: ignore
            
            # Convert to PyTorch tensors
            image = torch.from_numpy(image_np).float()
            label = torch.from_numpy(label_np).float()
            
            image = image.unsqueeze(0) # Add channel dimensions: [>>1<<,channels,bin_times,stations]

        except Exception as e:
            raise RuntimeError(f"Error loading data at index {idx}: {e}")

        # ========================================================
        # NORMALIZATION SWITCH
        # ========================================================
        # If we have stats, apply Z-score normalization. 
        # If we don't, this block is skipped, returning the raw physical labels!
        if self.label_mean is not None and self.label_std is not None:
            label = (label - self.label_mean) / (self.label_std + 1e-8)

        return image, label

    def close(self):
        """Close any open file handles."""
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

    def __del__(self):
        self.close()


class PreloadShardIterableDataset(IterableDataset):
    """
    An IterableDataset that streams pre-built batches directly from HDF5 shards.

    Key design decisions:
    - Inherits from IterableDataset (not Dataset), so __iter__ drives everything.
    - Yields full [batch_size, 1, H, W] tensors, bypassing DataLoader's collation.
    - Must be used with DataLoader(batch_size=None) to pass batches straight through.
    - Shards are distributed across workers at the shard level (not sample level),
      so each worker owns a disjoint subset of shards — zero cross-worker coordination.
    - Leftovers are kept as numpy arrays and concatenated at the numpy level before
      tensor conversion, avoiding a costly torch.cat (~2GB allocation) every shard.
    """

    def __init__(self, shard_file_list, manifest_path, batch_size,
                 is_train=True, label_mean=None, label_std=None, debug=False):
        """
        Args:
            shard_file_list (list): Absolute paths to .hdf5 shard files.
            manifest_path (str):    Path to the dataset manifest JSON.
            batch_size (int):       Number of samples per yielded batch.
            is_train (bool):        If True, shuffles shard order and sample indices each epoch.
            label_mean (Tensor):    Per-coordinate mean for label normalization.
            label_std (Tensor):     Per-coordinate std for label normalization.
            debug (bool):           If True, prints per-shard timing info. Disable for production
                                    to avoid per-shard syscalls from flush=True prints.
        """
        print('\nInitializing PreloadShardIterableDataset...')
        self.shard_files = shard_file_list
        self.path        = manifest_path
        self.is_train    = is_train
        self.batch_size  = batch_size
        self.debug       = debug

        # ============================================================
        # PASS 1: Validate shard paths and count total samples.
        # Opens each file just to read shape metadata — no image data
        # is loaded into RAM here.
        # ============================================================
        total_bytes      = 0
        self.num_images  = 0

        print("--> Scanning shards for metadata (size and exact image counts)...")
        for path in tqdm(self.shard_files, desc="Scanning Shards"):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Shard file not found: {path}")
            total_bytes += os.path.getsize(path)
            try:
                with h5py.File(path, 'r') as f:
                    self.num_images += f['album'].shape[0]
            except Exception as e:
                print(f"Error reading shape from {path}: {e}")

        self.space_GB = f'Size of all shards combined: {total_bytes * 1e-9:.4f} GB'
        print(self.space_GB)

        # ============================================================
        # PASS 2: Normalization stats.
        # Train set computes global mean/std from all labels.
        # Test set receives train stats — never fit stats on test set.
        # Only labels are read here, not images — much cheaper.
        # ============================================================
        if is_train and (label_mean is None or label_std is None):
            print("--> MODE: Auto-computing normalization stats from all shards...")
            print("          (Reading labels only — no images loaded)")

            all_labels = []
            for path in tqdm(self.shard_files, desc="Aggregating Labels"):
                try:
                    with h5py.File(path, 'r') as f:
                        all_labels.append(f['vertices'][:])
                except Exception as e:
                    print(f"Error reading labels from {path}: {e}")

            if not all_labels:
                raise ValueError("Could not read any labels from the provided shards.")

            all_labels_np  = np.concatenate(all_labels, axis=0)
            self.label_mean = torch.tensor(np.mean(all_labels_np, axis=0), dtype=torch.float32)
            self.label_std  = torch.tensor(np.std(all_labels_np,  axis=0), dtype=torch.float32)

            print("Normalization statistics computed:")
            print(f"  x = {self.label_mean[0]:.4f} ± {self.label_std[0]:.4f}")
            print(f"  y = {self.label_mean[1]:.4f} ± {self.label_std[1]:.4f}")
            print(f"  z = {self.label_mean[2]:.4f} ± {self.label_std[2]:.4f}")

        else:
            self.label_mean = label_mean
            self.label_std  = label_std

            if self.label_mean is None or self.label_std is None:
                print("--> MODE: No stats provided. Yielding RAW, UNNORMALIZED labels.")
            else:
                print("--> MODE: Stats received. Yielding NORMALIZED labels.")
                print(f"  x = {self.label_mean[0]:.4f} ± {self.label_std[0]:.4f}")
                print(f"  y = {self.label_mean[1]:.4f} ± {self.label_std[1]:.4f}")
                print(f"  z = {self.label_mean[2]:.4f} ± {self.label_std[2]:.4f}")

        print(f"Dataset initialized with {self.num_images} events.")

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id   = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        my_shards = self.shard_files[worker_id::num_workers]
        if self.is_train:
            my_shards = my_shards.copy()
            random.shuffle(my_shards)

        # KEY OPTIMIZATION: Keep leftovers as numpy arrays, not tensors.
        # This avoids a torch.cat(~2GB) allocation every shard boundary.
        # Instead we do a cheap np.concatenate on the small leftover slice,
        # then do a single torch.from_numpy() conversion on the combined array.
        leftover_imgs_np = None
        leftover_lbls_np = None

        for shard_idx, shard_path in enumerate(my_shards):

            try:
                t0 = time.perf_counter()
                with h5py.File(shard_path, 'r', locking=False) as f:
                    images_np = f['album'][:]    # shape: [N, H, W, C] float32
                    labels_np = f['vertices'][:]  # shape: [N, 3]       float32
                t_load = time.perf_counter() - t0
            except Exception as e:
                print(f"[Worker {worker_id}] FAILED TO READ {shard_path}: {e}")
                continue

            # Combine with leftovers at numpy level — cheap since leftovers
            # are always < batch_size samples (tiny compared to full shard)
            if leftover_imgs_np is not None:
                images_np = np.concatenate([leftover_imgs_np, images_np], axis=0)
                labels_np = np.concatenate([leftover_lbls_np, labels_np], axis=0)

            t1 = time.perf_counter()

            # Single tensor conversion after combining — torch.from_numpy shares
            # memory with the numpy array (zero copy), .unsqueeze(1) is a view
            combined_imgs = torch.from_numpy(images_np).unsqueeze(1)  # [N, 1, H, W]
            combined_lbls = torch.from_numpy(labels_np)               # [N, 3]

            # Normalize labels in-place — avoids allocating new tensors
            if self.label_mean is not None and self.label_std is not None:
                combined_lbls = combined_lbls.float()
                combined_lbls.sub_(self.label_mean).div_(self.label_std + 1e-8)

            t_convert = time.perf_counter() - t1

            # Shuffle indices only — avoids shuffling the full ~2GB tensor,
            # just permutes a small integer index array instead
            if self.is_train:
                indices = torch.randperm(len(combined_imgs))
            else:
                indices = torch.arange(len(combined_imgs))

            last_idx       = 0
            batches_yielded = 0

            t2 = time.perf_counter()
            for start_idx in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                batch_indices = indices[start_idx : start_idx + self.batch_size]
                # .float() here is cheap — just dtype cast on the batch slice,
                # not the full shard. bfloat16 autocast in train_step handles
                # the rest, so we only need float32 here
                yield (combined_imgs[batch_indices].float(),
                       combined_lbls[batch_indices])
                last_idx        = start_idx + self.batch_size
                batches_yielded += 1
            t_yield = time.perf_counter() - t2

            if self.debug:
                print(f"[Worker {worker_id}] Shard {shard_idx}: "
                      f"load={t_load:.2f}s | "
                      f"convert={t_convert:.2f}s | "
                      f"yield={t_yield:.2f}s | "
                      f"batches={batches_yielded} | "
                      f"file={os.path.basename(shard_path)}", flush=True)

            # Preserve leftovers as numpy for cheap concatenation next shard.
            # Convert back from tensor indices to numpy slice.
            remaining_indices = indices[last_idx:].numpy()
            leftover_imgs_np  = images_np[remaining_indices]
            leftover_lbls_np  = labels_np[remaining_indices]

        # Test set only: yield final ragged batch so no samples are dropped
        if not self.is_train and leftover_imgs_np is not None and len(leftover_imgs_np) > 0:
            yield (torch.from_numpy(leftover_imgs_np).float().unsqueeze(1),
                   torch.from_numpy(leftover_lbls_np).float())

    def __len__(self):
        """Total number of full batches across all shards."""
        return self.num_images // self.batch_size


class ShardStreamIterableDataset(IterableDataset):
    """
    An IterableDataset that streams pre-built batches directly from HDF5 shards,
    using a single background thread to load the next shard while the current one
    is being consumed by the GPU — hiding HDD latency behind GPU compute.

    Key design decisions:
    - Single background thread only — more threads = HDD head thrashing on spinning disk.
      Benchmarked: 1 thread @ 258MB/s vs 2 threads @ 92+75MB/s each. 1 wins.
    - Leftovers kept as numpy arrays — avoids costly torch.cat on full ~2GB shard tensors.
      np.concatenate on a small leftover slice (<batch_size samples) is orders of magnitude cheaper.
    - torch.from_numpy() is zero-copy — shares memory with numpy array.
    - unsqueeze(1) and index slicing are views — no allocation.
    - .float() applied only to batch slice (512 images), not full shard (5000 images).
    - Normalization done in-place on CPU tensors before yielding.
    - Must be used with DataLoader(batch_size=None) — batches are pre-assembled here.
    - debug=False in production — flush=True prints are syscalls on every shard.
    """

    def __init__(self, shard_file_list, manifest_path, batch_size,
                 is_train=True, label_mean=None, label_std=None, debug=False):
        """
        Args:
            shard_file_list (list): Absolute paths to .hdf5 shard files.
            manifest_path   (str):  Path to the dataset manifest JSON.
            batch_size      (int):  Number of samples per yielded batch.
            is_train        (bool): If True, shuffles shard order and indices each epoch.
            label_mean   (Tensor):  Per-coordinate mean for label normalization.
            label_std    (Tensor):  Per-coordinate std for label normalization.
            debug           (bool): If True, prints per-shard timing. Keep False in production —
                                    flush=True is a syscall on every shard boundary.
        """
        print('\nInitializing ShardStreamIterableDataset...')
        self.shard_files = shard_file_list
        self.path        = manifest_path
        self.is_train    = is_train
        self.batch_size  = batch_size
        self.debug       = debug

        # ============================================================
        # PASS 1: Validate shard paths and count total samples.
        # Opens each file just to read shape metadata — no image data
        # loaded into RAM here.
        # ============================================================
        total_bytes     = 0
        self.num_images = 0

        print("--> Scanning shards for metadata...")
        for path in tqdm(self.shard_files, desc="Scanning Shards"):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Shard file not found: {path}")
            total_bytes += os.path.getsize(path)
            try:
                with h5py.File(path, 'r') as f:
                    self.num_images += f['album'].shape[0]
            except Exception as e:
                print(f"Error reading shape from {path}: {e}")

        print(f'Size of all shards combined: {total_bytes * 1e-9:.4f} GB')

        # ============================================================
        # PASS 2: Normalization stats.
        # Train set computes global mean/std from labels only — not images.
        # Test set receives train stats — never fit stats on test set.
        # ============================================================
        if is_train and (label_mean is None or label_std is None):
            print("--> MODE: Auto-computing normalization stats (labels only)...")

            all_labels = []
            for path in tqdm(self.shard_files, desc="Aggregating Labels"):
                try:
                    with h5py.File(path, 'r') as f:
                        all_labels.append(f['vertices'][:])
                except Exception as e:
                    print(f"Error reading labels from {path}: {e}")

            if not all_labels:
                raise ValueError("Could not read any labels from the provided shards.")

            all_labels_np   = np.concatenate(all_labels, axis=0)
            self.label_mean = torch.tensor(np.mean(all_labels_np, axis=0), dtype=torch.float32)
            self.label_std  = torch.tensor(np.std(all_labels_np,  axis=0), dtype=torch.float32)

            print("Normalization statistics computed:")
            print(f"  x = {self.label_mean[0]:.4f} ± {self.label_std[0]:.4f}")
            print(f"  y = {self.label_mean[1]:.4f} ± {self.label_std[1]:.4f}")
            print(f"  z = {self.label_mean[2]:.4f} ± {self.label_std[2]:.4f}")

        else:
            self.label_mean = label_mean
            self.label_std  = label_std

            if self.label_mean is None or self.label_std is None:
                print("--> MODE: No stats provided. Yielding RAW, UNNORMALIZED labels.")
            else:
                print("--> MODE: Stats received. Yielding NORMALIZED labels.")
                print(f"  x = {self.label_mean[0]:.4f} ± {self.label_std[0]:.4f}")
                print(f"  y = {self.label_mean[1]:.4f} ± {self.label_std[1]:.4f}")
                print(f"  z = {self.label_mean[2]:.4f} ± {self.label_std[2]:.4f}")

        print(f"Dataset initialized with {self.num_images} events.")

    @staticmethod
    def _load_shard(shard_path):
        """
        Loads images and labels from a single HDF5 shard.

        Static method — must be picklable for ThreadPoolExecutor.
        locking=False is safe here since we only read and each worker
        owns a disjoint set of shards (zero cross-worker file access).
        """
        with h5py.File(shard_path, 'r', locking=False) as f:
            return f['album'][:], f['vertices'][:]

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id   = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        my_shards = self.shard_files[worker_id::num_workers]
        if self.is_train:
            my_shards = my_shards.copy()
            random.shuffle(my_shards)

        if len(my_shards) == 0:
            return

        # Leftovers kept as numpy — np.concatenate on a tiny leftover slice
        # (<batch_size rows) is much cheaper than torch.cat on a full shard tensor
        leftover_imgs_np = None
        leftover_lbls_np = None

        # Single background thread — NEVER increase this on spinning HDD.
        # Benchmarked: 1 thread = 258MB/s, 2 threads = 92+75MB/s each.
        # More threads = HDD head thrashing = slower total throughput.
        with ThreadPoolExecutor(max_workers=1) as executor:

            # Submit first shard load before loop starts — gives maximum
            # lead time for the background thread to get ahead
            future = executor.submit(self._load_shard, my_shards[0])

            for shard_idx, shard_path in enumerate(my_shards):

                # Block until current shard is ready.
                # If async is working well: t_wait ≈ 0 (loaded during yield)
                # If shard too small to hide load: t_wait > 0 (partial overlap)
                # If shard tiny vs load time: t_wait ≈ load_time (no benefit)
                t0 = time.perf_counter()
                try:
                    images_np, labels_np = future.result()
                except Exception as e:
                    print(f"[Worker {worker_id}] FAILED TO READ {shard_path}: {e}")
                    # Still submit next shard before skipping current
                    if shard_idx + 1 < len(my_shards):
                        future = executor.submit(self._load_shard, my_shards[shard_idx + 1])
                    continue
                t_wait = time.perf_counter() - t0

                # -------------------------------------------------------
                # Submit NEXT shard load immediately after current is ready
                # — before any tensor conversion or processing.
                # This gives the background thread maximum overlap time
                # while we convert tensors and yield batches to the GPU.
                # Single thread only — no parallel HDD reads.
                # -------------------------------------------------------
                if shard_idx + 1 < len(my_shards):
                    future = executor.submit(self._load_shard, my_shards[shard_idx + 1])

                # Combine with numpy leftovers before tensor conversion.
                # Leftover is always < batch_size rows — tiny concat vs full shard.
                if leftover_imgs_np is not None:
                    images_np = np.concatenate([leftover_imgs_np, images_np], axis=0)
                    labels_np = np.concatenate([leftover_lbls_np, labels_np], axis=0)

                t1 = time.perf_counter()

                # torch.from_numpy() is zero-copy — shares memory with numpy array.
                # unsqueeze(1) is a view — no allocation.
                combined_imgs = torch.from_numpy(images_np).unsqueeze(1)  # [N, 1, H, W]
                combined_lbls = torch.from_numpy(labels_np).float()       # [N, 3]

                # In-place normalization — no new tensor allocation
                if self.label_mean is not None and self.label_std is not None:
                    combined_lbls.sub_(self.label_mean).div_(self.label_std + 1e-8)

                t_convert = time.perf_counter() - t1

                # Shuffle index array only — permutes small int tensor,
                # not the full ~2GB image tensor
                if self.is_train:
                    indices = torch.randperm(len(combined_imgs))
                else:
                    indices = torch.arange(len(combined_imgs))

                last_idx        = 0
                batches_yielded = 0

                t2 = time.perf_counter()
                for start_idx in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                    batch_indices = indices[start_idx : start_idx + self.batch_size]
                    # .float() on batch slice only (512 imgs), not full shard (5000 imgs)
                    yield (combined_imgs[batch_indices].float(),
                           combined_lbls[batch_indices])
                    last_idx        = start_idx + self.batch_size
                    batches_yielded += 1
                t_yield = time.perf_counter() - t2

                if self.debug:
                    # t_wait ≈ 0 = async working, GPU kept busy
                    # t_wait > 0 = shard too small, increase images_per_shard
                    print(f"[Worker {worker_id}] Shard {shard_idx}: "
                          f"wait={t_wait:.2f}s | "
                          f"convert={t_convert:.2f}s | "
                          f"yield={t_yield:.2f}s | "
                          f"batches={batches_yielded} | "
                          f"file={last_two_path_names(shard_path)}", flush=True)

                # Preserve leftovers as numpy for cheap concat next iteration.
                # Convert index tensor to numpy for array indexing.
                remaining_indices = indices[last_idx:].numpy()
                leftover_imgs_np  = images_np[remaining_indices]
                leftover_lbls_np  = labels_np[remaining_indices]

        # Test set only: yield final ragged batch so no samples are dropped
        if not self.is_train and leftover_imgs_np is not None and len(leftover_imgs_np) > 0:
            yield (torch.from_numpy(leftover_imgs_np).float().unsqueeze(1),
                   torch.from_numpy(leftover_lbls_np).float())

    def __len__(self):
        """Total number of full batches across all shards."""
        if self.is_train:
            # Leftovers combine across shard boundaries, almost all images
            # eventually form full batches — floor division is correct
            return self.num_images // self.batch_size
        else:
            # Test set yields a final ragged batch for leftover samples
            full_batches  = self.num_images // self.batch_size
            has_remainder = (self.num_images % self.batch_size) > 0
            return full_batches + (1 if has_remainder else 0)