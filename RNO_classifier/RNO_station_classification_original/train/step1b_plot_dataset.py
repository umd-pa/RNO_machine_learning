"""
Step 1b: Plot summary histograms of the extracted signal dataset.

Usage:
    python step1b_plot_dataset.py --signal signal.h5 --out plots/

Produces:
    snr_histogram.png       — weighted histogram of max-channel SNR
    energy_histogram.png    — weighted histogram of neutrino energy
    vertex_xy.png           — weighted 2D histogram of vertex X vs Y
    vertex_rz.png           — weighted 2D histogram of vertex R vs Z
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_signal(h5_path):
    with h5py.File(h5_path, "r") as f:
        snr    = f["snr"][:]      # (N, 4)
        energy = f["energy"][:]   # (N,)
        vertex = f["vertex"][:]   # (N, 3)
        weight = f["weight"][:]   # (N,)
    return snr, energy, vertex, weight


def clean_weights(weight):
    """Replace NaN/inf weights with 1.0 and normalise to sum=1."""
    w = weight.copy()
    bad = ~np.isfinite(w) | (w <= 0)
    if bad.any():
        print(f"  WARNING: {bad.sum()} events have invalid weights — setting to 1.0")
        w[bad] = 1.0
    return w / w.sum()


def plot_snr(snr, weight, out_path):
    max_snr = np.nanmax(snr, axis=1)   # max over 4 channels per event
    valid   = np.isfinite(max_snr) & (max_snr > 0)
    x, w    = max_snr[valid], weight[valid]
    w       = w / w.sum()

    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x, bins=bins, weights=w, histtype="step", lw=2, color="tab:blue")
    ax.set(xlabel="Max-channel SNR  (V$_{p2p}$ / 2σ$_{noise}$)",
           ylabel="Weighted fraction", title="Signal SNR Distribution",
           xscale="log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_energy(energy, weight, out_path):
    valid = np.isfinite(energy) & (energy > 0)
    x, w  = energy[valid], weight[valid]
    w     = w / w.sum()

    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x, bins=bins, weights=w, histtype="step", lw=2, color="tab:orange")
    ax.set(xlabel="Neutrino Energy [eV]",
           ylabel="Weighted fraction", title="Signal Energy Distribution",
           xscale="log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_vertex_xy(vertex, weight, out_path):
    valid = np.isfinite(vertex).all(axis=1)
    x, y, w = vertex[valid, 0], vertex[valid, 1], weight[valid]

    fig, ax = plt.subplots(figsize=(6, 5))
    h, xedges, yedges = np.histogram2d(x, y, bins=60, weights=w)
    im = ax.pcolormesh(xedges, yedges, h.T, cmap="viridis")
    plt.colorbar(im, ax=ax, label="Weighted fraction")
    ax.set(xlabel="X [m]", ylabel="Y [m]", title="Vertex X vs Y")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_vertex_rz(vertex, weight, out_path):
    valid = np.isfinite(vertex).all(axis=1)
    x, y, z, w = vertex[valid, 0], vertex[valid, 1], vertex[valid, 2], weight[valid]
    r = np.sqrt(x**2 + y**2)

    fig, ax = plt.subplots(figsize=(6, 5))
    h, xedges, yedges = np.histogram2d(r, z, bins=60, weights=w)
    im = ax.pcolormesh(xedges, yedges, h.T, cmap="viridis")
    plt.colorbar(im, ax=ax, label="Weighted fraction")
    ax.set(xlabel="R [m]", ylabel="Z [m]", title="Vertex R vs Z")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signal", required=True, help="Signal HDF5 from step1_extract.py")
    p.add_argument("--out",    default="plots", help="Output directory (default: plots/)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.signal} …")
    snr, energy, vertex, weight = load_signal(args.signal)
    print(f"  Events: {len(snr)}")

    # weight = clean_weights(weight)

    plot_snr(snr,       weight, out_dir / "snr_histogram.png")
    plot_energy(energy, weight, out_dir / "energy_histogram.png")
    plot_vertex_xy(vertex, weight, out_dir / "vertex_xy.png")
    plot_vertex_rz(vertex, weight, out_dir / "vertex_rz.png")

    print("Done.")


if __name__ == "__main__":
    main()
