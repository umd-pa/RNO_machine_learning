import os
import h5py
import pickle
import argparse
import numpy as np
from six import iteritems
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from tqdm import tqdm
from radiotools import helper as hp
from radiotools import plthelpers as php

from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium, plotting
from NuRadioReco.utilities.constants import density_ice, density_water

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

""""
Last stage:
Generate trigger rate of model vs lowest channel SNR graph.
Generate effective volume from classified neutrinos vs neutrino energy graph.

Uses code from nuradio/NuRadioMC/NuRadioMC/simulation/scripts/T05visualize_sim_output.py
"""

###########################
# Models, Dataset and helper functions
###########################

def load_and_print_checkpoint_info(model: torch.nn.Module, checkpoint_path: str, device):
    latest_checkpoint = torch.load(checkpoint_path, pickle_module = pickle)

    model_dict  = latest_checkpoint['model_state']
    epoch       = latest_checkpoint['epoch']
    val_loss    = latest_checkpoint['val_loss']
    val_acc     = latest_checkpoint['val_acc']
    config      = latest_checkpoint['config']
    sig_val_idx = latest_checkpoint['sig_val_idx']
    noi_val_idx = latest_checkpoint['noi_val_idx']
    
    model.load_state_dict(model_dict)

    print(f"\n{'='*70}")
    print(f"Successfully loaded model: {model._get_name()} in device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}")
    print(f"Epoch:                      {epoch}")
    print(f"Validation Loss:            {val_loss:.6f}")
    print(f"Validation Accuracy:        {val_acc:.4f}")
    print(f"Signal validation samples:  {len(sig_val_idx)}")
    print(f"Noise validation samples:   {len(noi_val_idx)}")
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")



class WaveformDataset(Dataset):
    """
    Lazy HDF5 reader.  Each item is (waveform, label).
    waveform : float32 tensor (4, T) — optionally centre-cropped and normalised.
    label    : float32 scalar 0 or 1.
    """

    def __init__(self, h5_path: str, indices: np.ndarray,
                 crop_samples: int | None):
        self.h5_path      = h5_path
        self.indices      = indices
        self.crop_samples = crop_samples
        self._f           = None   # opened lazily (safe across worker processes)

    def _open(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self._open()
        idx = int(self.indices[i])
        wav = self._f["waveforms"][idx]   # type: ignore (4, T) float32 
        lbl = float(self._f["labels"][idx]) # type: ignore check that its float32

        # Centre-crop
        if self.crop_samples is not None:
            T = wav.shape[1] # type: ignore
            if self.crop_samples <= T:
                start = (T - self.crop_samples) // 2
                wav = wav[:, start: start + self.crop_samples] # type: ignore
            else:
                pad = np.zeros((4, self.crop_samples - T), dtype=np.float32)
                wav = np.concatenate([wav, pad], axis=1) # type: ignore

        # Per-sample peak normalisation (DO NOT THINK WE NEED TO NORMALIZE FOR THIS DATASET!)
        # peak = np.max(np.abs(wav))
        # if peak > 0:
        #     wav = wav / peak

        return torch.from_numpy(wav), torch.tensor(lbl, dtype=torch.float32)

    def __del__(self):
        if self._f is not None:
            self._f.close()



class ResBlock1d_plus(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, padding: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels_in, channels_out, kernel_size=kernel_size,
                               padding=padding, stride=stride, bias=False)
        self.conv2 = nn.Conv1d(channels_out, channels_out, kernel_size=kernel_size,
                               padding=padding, bias=False)
        self.act   = nn.ReLU()

        self.downsample = None
        if stride != 1 or channels_out != channels_in:
            self.downsample = nn.Sequential(
                nn.Conv1d(channels_in, channels_out, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        identity = x
        out = self.conv2(self.act(self.conv1(x)))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)



class RNO_resnet_plus(nn.Module):
    """
    Improved resnet architecture inspired by "Detection of Radar Pulse Signals Based on Deep Learning" by Fengyang Gu, et al.
    Input:  (batch, 4, 1024)  ← Same as FpgaCNN!
    Output: (batch, 1)        ← Binary classification
    Params: 10,657
    Size: 829.00 MBs
    """
    def __init__(self,
                 n_channels: int = 4,
                 hidden_units: int = 32,
                 output_shape: int = 1):

        super().__init__()

        self.pre_process = nn.Sequential(
            nn.Conv1d(n_channels, hidden_units, kernel_size = 5, stride = 2, padding = 2),
            nn.MaxPool1d(kernel_size = 2)
        )

        self.res_block1 = ResBlock1d_plus(hidden_units, hidden_units, kernel_size = 3, padding = 1)
        self.res_block2 = ResBlock1d_plus(hidden_units, hidden_units // 2, kernel_size = 3, padding = 1)
        avg_pool_out = 64
        self.avg_pool = nn.AdaptiveAvgPool1d(avg_pool_out)
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(in_features = (hidden_units // 2) * avg_pool_out, out_features = output_shape)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.pre_process(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear_layer(x)
        x = self.softmax(x)
        return x.squeeze(1)

###########################
# Load classifier 
###########################

import argparse

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--gpu", default = 'False', help="Signal HDF5 from extract.py")
args = p.parse_args()

print('\n###########################')
print('Load Model:')
print('###########################')

device = (torch.device("cuda") if args.gpu is True else
          torch.device("cpu"))
model = RNO_resnet_plus()
checkpoint_path = '/data/condor_shared/users/nmanic/RNO_vertex_reconstruction_ml_nmanic/RNO_machine_learning/RNO_classifier/runs/exp_09_real/best_model.pt'

load_and_print_checkpoint_info(model, checkpoint_path, device)  

# # CNN output gives us probabilities that an event is a neutrino or not. With CNN we determine number of detected neutrino
# # A single simulation ran at a fixed energy has one effective volume. Simulation needs to be repeated across multiple energy bins
# # V eff (E_i) = V_cylinder * (N_triggered / N_simulated) * 4 pi
# weights = # the sum of weights of each trigger the classifier made.
# n_events =

# @torch.no_grad()
# def get_scores(model, loader, device):
#     model.eval()
#     scores, labels = [], []
#     for wav, lbl in loader:
#         scores.append(torch.sigmoid(model(wav.to(device))).cpu().numpy())
#         labels.append(lbl.numpy())
#     return np.concatenate(scores), np.concatenate(labels)





###########################
# calculate effective volume
###########################

# fin refers to the detector file. we don't really need it, we just need the size of the detector to solve for volume.

# n_triggered = np.sum(weights)
# print('fraction of triggered events = {:.0f}/{:.0f} = {:.3f}'.format(n_triggered, n_events, n_triggered / n_events))

# V = None
# if('xmax' in fin.attrs):
#     dX = fin.attrs['xmax'] - fin.attrs['xmin']
#     dY = fin.attrs['ymax'] - fin.attrs['ymin']
#     dZ = fin.attrs['zmax'] - fin.attrs['zmin']
#     V = dX * dY * dZ
# elif('rmin' in fin.attrs):
#     rmin = fin.attrs['rmin']
#     rmax = fin.attrs['rmax']
#     dZ = fin.attrs['zmax'] - fin.attrs['zmin']
#     V = np.pi * (rmax ** 2 - rmin ** 2) * dZ
# Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights) / n_events
# print("Veff = {:.6g} km^3 sr".format(Veff / units.km ** 3))


###########################
# calculate trigger efficiency
###########################

print('###########################')
print('Trigger Efficiency:')
print('###########################\n')

# Load eval dataset (For now its just testing dataset)

test_dataset_path = '/data/condor_shared/users/ssued/RNO_machine_learning/RNO_machine_learning/RNO_classifier/data/extracted_nu.hdf5'

with h5py.File(test_dataset_path, 'r') as h5f:
    snr_arr = []
    for snrs in tqdm(h5f['snr'][:], desc='Extracting maximum SNRs for each sample', unit='samples'): # type: ignore
        snr_arr.append(np.max(snrs))  # type: ignore
test_dataset_idx = np.arange(0,len(snr_arr)) # type: ignore
print(f'SNR array loaded succesfully from {test_dataset_path} with {len(snr_arr)} samples')

test_ds = WaveformDataset(test_dataset_path, test_dataset_idx, None) # type: ignore
print(f'Test dataset loaded successfully from {test_dataset_path} with {len(test_ds)} samples.')

# For SNR distribution
plt.style.use('seaborn-v0_8-whitegrid') # Built-in matplotlib style sheet
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)

# Plot histogram
ax.hist(snr_arr, bins=100, color='#1f77b4', edgecolor='w', alpha=0.85, log=True)
ax.set(title='SNR Distribution of Test Dataset', xlabel='Max SNR', ylabel='Frequency (Log)')
ax.legend()
fig.savefig('RNO_machine_learning/RNO_classifier/eval/hist_corrected.png', bbox_inches='tight')


n_workers = min(8, os.cpu_count() or 4)
prefetch = 4
loader_kw = dict(num_workers=n_workers, pin_memory=args.gpu,
                    persistent_workers=True, prefetch_factor=prefetch)
test_loader   = DataLoader(test_ds,   batch_size=32,
                              shuffle=False, )
print(f'Test dataloader loaded successfully with {len(test_loader)} batches, n_workers: {n_workers} and prefetch factor: {prefetch} and pin_memory = {args.gpu}.')
