"""
Print the output shape of every layer in RNO_resnet_plus, using a real
batch pulled from your signal/noise HDF5 files.

Usage:
    python print_layer_shapes.py --signal signal.h5 --noise noise.h5 [--batch_size 8]

Assumes this file sits next to train.py (or step2_train.py) so it can
import WaveformDataset and the model classes directly — no code duplication,
so it always matches whatever your train.py currently does.
"""

import argparse
import torch
from torch.utils.data import DataLoader

from train_benchmark import WaveformDataset, RNO_resnet_plus  # adjust module name if your file is step2_train.py


def register_shape_hooks(model):
    """Attach a forward hook to every leaf submodule that prints its output shape."""
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            print(f"{name:35s} {tuple(out.shape)}")
        return hook

    for name, module in model.named_modules():
        # only leaf modules (no children) so we don't double-print containers
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(make_hook(name)))

    return handles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signal", required=True, help="Signal HDF5 path")
    p.add_argument("--noise", required=True, help="Noise HDF5 path")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--crop_samples", type=int, default=None)
    args = p.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cpu"))

    # pull a single real batch (signal events only, just to get correct shape/dtype)
    import h5py
    import numpy as np
    with h5py.File(args.signal, "r") as f:
        n_events = f.attrs["n_events"]
    idx = np.arange(min(args.batch_size, n_events))

    ds = WaveformDataset(args.signal, idx, args.crop_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    wav, lbl = next(iter(loader))
    wav = wav.to(device)

    print(f"Input batch shape: {tuple(wav.shape)}  (batch, channels, time)\n")

    model = RNO_resnet_plus(n_channels=wav.shape[1]).to(device)
    model.eval()

    handles = register_shape_hooks(model)
    with torch.no_grad():
        out = model(wav)
    for h in handles:
        h.remove()

    print(f"\nFinal output shape: {tuple(out.shape)}")


if __name__ == "__main__":
    main()