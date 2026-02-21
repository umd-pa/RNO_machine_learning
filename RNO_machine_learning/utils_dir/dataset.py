from decimal import DivisionByZero
from torch.utils.data import Dataset
import h5py
import numpy as np
import threading
import torch
import os

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
            self.file_handle = h5py.File(self.path, 'r')

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