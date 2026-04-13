from torch.utils.data import IterableDataset, get_worker_info
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import hashlib
import h5py
import random
import shutil
import time
import torch
import json
import os

def last_two_path_names(path):
    """Return the last two names of a path."""
    parts = os.path.normpath(path).split(os.sep)
    return os.sep.join(parts[-2:])

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

def stage_manifest_to_scratch(manifest_path, cache_dir=None, force=False):
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
    if cache_dir is None:
        print(f"No cache_dir provided. Falling back to original network paths.")
        return manifest
        
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
                    self.num_images += f['album'].shape[0] # type: ignore
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
                        all_labels.append(f['vertices'][:]) # type: ignore
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
    - Leftovers kept as numpy arrays — avoids costly torch.cat on full shard tensors.
      np.concatenate on a small leftover slice (<batch_size samples) is much cheaper.
    - torch.from_numpy() is zero-copy — shares memory with the numpy array.
    - unsqueeze(1) and index slicing are views — no new tensor allocation.
    - .float() applied only to the batch slice, not the full shard.
    - Normalization done in-place on CPU tensors before yielding.
    - station_hit_count filtering applied immediately after decompression —
      reduces downstream tensor operations when filtering is active.
    - Must be used with DataLoader(batch_size=None) — batches are pre-assembled here.
    - debug=False in production — flush=True prints are syscalls on every shard boundary.
    """

    def __init__(self, shard_file_list, manifest_path, batch_size,
                 is_train=True, label_mean=None, label_std=None,
                 debug=False, min_station_hits=1):
        """
        Args:
            shard_file_list  (list): Absolute paths to .hdf5 shard files.
            manifest_path    (str):  Path to the dataset manifest JSON.
            batch_size       (int):  Number of samples per yielded batch.
            is_train         (bool): If True, shuffles shard order and indices each epoch.
            label_mean    (Tensor):  Per-coordinate mean for label normalization.
                                     If None and is_train=True, computed automatically.
            label_std     (Tensor):  Per-coordinate std for label normalization.
                                     If None and is_train=True, computed automatically.
            debug            (bool): If True, prints per-shard timing breakdown.
                                     Keep False in production — flush=True is a syscall
                                     on every shard boundary.
            min_station_hits  (int): Minimum number of station hits required to include
                                     an event. Default=1 includes all events.
                                     Use 2+ to restrict to multi-station events only,
                                     which carry TDoA triangulation information.
        """
        print('\nInitializing ShardStreamIterableDataset...')
        self.shard_files      = shard_file_list
        self.path             = manifest_path
        self.is_train         = is_train
        self.batch_size       = batch_size
        self.debug            = debug
        self.min_station_hits = min_station_hits

        # ============================================================
        # PASS 1: Validate shard paths and count total samples.
        # Opens each file just to read shape metadata — no image data
        # loaded into RAM here. If min_station_hits > 1, reads hit counts
        # to get the true filtered image count for accurate __len__.
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
                    if min_station_hits > 1:
                        # Read hit counts to get accurate filtered event count
                        hits = f['station_hit_count'][:].squeeze()
                        self.num_images += int(np.sum(hits >= min_station_hits))
                    else:
                        self.num_images += f['album'].shape[0]
            except Exception as e:
                print(f"Error reading shape from {path}: {e}")

        print(f'Size of all shards combined: {total_bytes * 1e-9:.4f} GB')
        if min_station_hits > 1:
            print(f'Filtering to events with >= {min_station_hits} station hits.')

        # ============================================================
        # PASS 2: Normalization stats.
        # Train set computes global mean/std from labels only — not images.
        # Test set receives train stats — never fit stats on test set.
        # If filtering is active, stats are computed on filtered labels only
        # so normalization reflects the actual training distribution.
        # ============================================================
        if is_train and (label_mean is None or label_std is None):
            print("--> MODE: Auto-computing normalization stats (labels only)...")

            all_labels = []
            for path in tqdm(self.shard_files, desc="Aggregating Labels"):
                try:
                    with h5py.File(path, 'r') as f:
                        labels = f['vertices'][:]
                        if min_station_hits > 1:
                            hits   = f['station_hit_count'][:].squeeze()
                            labels = labels[hits >= min_station_hits]
                        all_labels.append(labels)
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
        Loads images, labels, and station hit counts from a single HDF5 shard.

        Returns all three arrays so the caller can apply hit count filtering
        without a second file open. squeeze() on station_hit_count returns a
        view (no copy) — faster than flatten() which always allocates.

        Static method — must be picklable for ThreadPoolExecutor.
        locking=False is safe here since we only read and each worker
        owns a disjoint set of shards (zero cross-worker file access).

        Returns:
            tuple: (images_np, labels_np, hits_np) as numpy arrays.
        """
        with h5py.File(shard_path, 'r', locking=False) as f:
            return f['album'][:], f['vertices'][:], f['station_hit_count'][:].squeeze()

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id   = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Each worker owns a disjoint slice of shards — no cross-worker contention
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
                t0 = time.perf_counter()
                try:
                    images_np, labels_np, hits_np = future.result()

                    # Apply station hit filter immediately after load —
                    # reduces all downstream operations when filtering is active
                    if self.min_station_hits > 1:
                        mask      = hits_np >= self.min_station_hits
                        images_np = images_np[mask]
                        labels_np = labels_np[mask]

                except Exception as e:
                    print(f"[Worker {worker_id}] FAILED TO READ {shard_path}: {e}")
                    # Still submit next shard before skipping current
                    if shard_idx + 1 < len(my_shards):
                        future = executor.submit(self._load_shard, my_shards[shard_idx + 1])
                    continue
                t_wait = time.perf_counter() - t0

                # Submit next shard immediately — before filtering or tensor conversion
                # so the background thread has maximum time to load while we
                # process the current shard
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
                # not the full image tensor
                if self.is_train:
                    indices = torch.randperm(len(combined_imgs))
                else:
                    indices = torch.arange(len(combined_imgs))

                last_idx        = 0
                batches_yielded = 0

                t2 = time.perf_counter()
                for start_idx in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                    batch_indices = indices[start_idx : start_idx + self.batch_size]
                    # .float() on batch slice only, not full shard
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
        """
        Total number of batches this dataset will yield per epoch.

        For train: leftovers combine across shard boundaries so almost all
        images eventually form full batches — floor division is correct.

        For test: a final ragged batch is yielded for any remaining samples
        so no events are dropped during evaluation.
        """
        if self.is_train:
            return self.num_images // self.batch_size
        else:
            full_batches  = self.num_images // self.batch_size
            has_remainder = (self.num_images % self.batch_size) > 0
            return full_batches + (1 if has_remainder else 0)
