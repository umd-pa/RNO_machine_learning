from torch.utils.data import Dataset
from h5py import File
import numpy as np
import threading
import torch
import os

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
    
    def __init__(self, album_path, transform=None, target_transform=None, preload_keys=True):
        """
        Initialize the AlbumDataset.
        
        Args:
            album_path (str): Path to the HDF5 file containing the dataset
            transform (callable, optional): Transform to apply to images
            target_transform (callable, optional): Transform to apply to labels
            preload_keys (bool): Whether to preload all event keys into memory.
            If True improves performance but uses more memory.
        """
        # Store configuration
        self.path = album_path
        self.transform = transform
        self.target_transform = target_transform
        
        # Validate file exists
        if not os.path.exists(album_path):
            raise FileNotFoundError(f"Album file not found: {album_path}")

        # Calculate and store file size for monitoring
        self.size = f'Size of file: {os.path.getsize(album_path)*1e-9:.4f} GB'

        # Preload all event keys for faster access (optional optimization)
        self.preload_keys = preload_keys
        with File(self.path, 'r') as file:
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
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.num_images

    def __getitem__(self, idx):
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
                self._local.file_handle = File(self.path, 'r')
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
        
        return image, label

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