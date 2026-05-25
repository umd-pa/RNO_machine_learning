import os
import h5py
import argparse
import numpy as np
from six import iteritems
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from radiotools import helper as hp
from radiotools import plthelpers as php

from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium, plotting
from NuRadioReco.utilities.constants import density_ice, density_water

import torch
from torch.utils.data import ConcatDataset, DataLoader
""""
Last stage:
Generate trigger rate of model vs lowest channel SNR graph.
Generate effective volume from classified neutrinos vs neutrino energy graph.

Uses code from nuradio/NuRadioMC/NuRadioMC/simulation/scripts/T05visualize_sim_output.py
"""

# CNN output gives us probabilities that an event is a neutrino or not. With CNN we determine number of detected neutrino
# A single simulation ran at a fixed energy has one effective volume. Simulation needs to be repeated across multiple energy bins
# V eff (E_i) = V_cylinder * (N_triggered / N_simulated) * 4 pi
weights = # the sum of weights of each trigger the classifier made.
n_events =

@torch.no_grad()
def get_scores(model, loader, device):
    model.eval()
    scores, labels = [], []
    for wav, lbl in loader:
        scores.append(torch.sigmoid(model(wav.to(device))).cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(scores), np.concatenate(labels)





###########################
# calculate effective volume
###########################

# fin refers to the detector file. we don't really need it, we just need the size of the detector to solve for volume.

n_triggered = np.sum(weights)
print('fraction of triggered events = {:.0f}/{:.0f} = {:.3f}'.format(n_triggered, n_events, n_triggered / n_events))

V = None
if('xmax' in fin.attrs):
    dX = fin.attrs['xmax'] - fin.attrs['xmin']
    dY = fin.attrs['ymax'] - fin.attrs['ymin']
    dZ = fin.attrs['zmax'] - fin.attrs['zmin']
    V = dX * dY * dZ
elif('rmin' in fin.attrs):
    rmin = fin.attrs['rmin']
    rmax = fin.attrs['rmax']
    dZ = fin.attrs['zmax'] - fin.attrs['zmin']
    V = np.pi * (rmax ** 2 - rmin ** 2) * dZ
Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights) / n_events
print("Veff = {:.6g} km^3 sr".format(Veff / units.km ** 3))


###########################
# calculate trigger efficiency
###########################