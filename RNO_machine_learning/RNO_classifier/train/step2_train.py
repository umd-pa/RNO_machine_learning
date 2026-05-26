"""
Stage 2: Train a multi-channel 1D CNN on signal and noise HDF5 datasets.

Usage:
    python step2_train.py \
        --signal signal.h5 \
        --noise  noise.h5  \
        --out    runs/exp01 \
        [--arch  tiny]          \
        [--crop-samples 512]    \
        [--epochs 50]           \
        [--batch-size 256]      \
        [--lr 1e-3]             \
        [--dropout 0.2]         \
        [--weight-decay 1e-4]   \
        [--val-frac 0.15]       \
        [--seed 42]

Outputs (all inside --out):
    best_model.pt   — checkpoint with lowest validation loss
    last_model.pt   — checkpoint after final epoch
    metrics.csv     — per-epoch train/val loss and accuracy
    loss_curve.png  — updated each epoch
    config.json     — all hyperparameters, for reproducibility

Architectures (select with --arch):

  baseline  —  ~74k params, ~10M MACs @ 512 samples
    Input (B, 4, T) → early channel fusion
    Conv1d(4→32,  k=15) → ReLU → MaxPool(2)
    Conv1d(32→64, k=9)  → ReLU → MaxPool(2)
    Conv1d(64→64, k=5)  → ReLU → MaxPool(2)
    Conv1d(64→128,k=3)  → ReLU → AdaptiveAvgPool → (B,128)
    Linear(128→64) → ReLU → Dropout → Linear(64→1)

  tiny  —  ~1.4k params, ~4k MACs @ 512 samples; HLS4ML-synthesisable
    Input (B, 4, T) → early channel fusion
    Conv1d(4→8,  k=5) → ReLU → MaxPool(4)
    Conv1d(8→8,  k=3) → ReLU → MaxPool(4)
    Conv1d(8→8,  k=3) → ReLU → AdaptiveAvgPool(1) → (B,8)
    Linear(8→1)

To add a new architecture, define a class and add it to ARCH_REGISTRY.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ── Dataset ───────────────────────────────────────────────────────────────────

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
        wav = self._f["waveforms"][idx]   # (4, T) float32
        lbl = float(self._f["labels"][idx]) # check that its float32

        # Centre-crop
        if self.crop_samples is not None:
            T = wav.shape[1]
            if self.crop_samples <= T:
                start = (T - self.crop_samples) // 2
                wav = wav[:, start: start + self.crop_samples]
            else:
                pad = np.zeros((4, self.crop_samples - T), dtype=np.float32)
                wav = np.concatenate([wav, pad], axis=1)

        # Per-sample peak normalisation (DO NOT THINK WE NEED TO NORMALIZE FOR THIS DATASET!)
        # peak = np.max(np.abs(wav))
        # if peak > 0:
        #     wav = wav / peak

        return torch.from_numpy(wav), torch.tensor(lbl, dtype=torch.float32)

    def __del__(self):
        if self._f is not None:
            self._f.close()


# ── Architecture ──────────────────────────────────────────────────────────────

class BaselineCNN(nn.Module):
    """4-block 1D CNN with early channel fusion. Input: (B, 4, T).
    ~74k params, ~10M MACs @ 512 samples. Good for ARM/GPU inference."""

    def __init__(self, n_channels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32,  kernel_size=15, padding=7),
            nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(32, 64,  kernel_size=9, padding=4),
            nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(64, 64,  kernel_size=5, padding=2),
            nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)   # (B,)


class TinyCNN(nn.Module):
    """FPGA-deployable 1D CNN. Input: (B, 4, T).
    ~577 params, ~4k MACs @ 512 samples.
    Probably too small to learn well — use FpgaCNN instead."""

    def __init__(self, n_channels: int = 4, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 8, kernel_size=5, padding=2),
            nn.ReLU(), nn.MaxPool1d(4),        # 512 → 128

            nn.Conv1d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool1d(4),        # 128 → 32

            nn.Conv1d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)   # (B,)


class FpgaCNN(nn.Module):
    """FPGA-target 1D CNN. Input: (B, 4, T).
    ~8k params, ~200k MACs @ 512 samples.
    Consistent with published HLS4ML deployments on Ultrascale+-class FPGAs
    at sub-microsecond latency (reuse factor 4-8). Defensible for NSF proposal.
    No BatchNorm (fuse into weights for fixed-point quantisation).
    """

    def __init__(self, n_channels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(), nn.MaxPool1d(2),        # 512 → 256

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(), nn.MaxPool1d(2),        # 256 → 128

            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool1d(2),        # 128 → 64

            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1),  # 64 → 1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)   # (B,)

class ResBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, padding: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                               padding=padding, stride=stride, bias=False)
        self.bn1   = nn.BatchNorm1d(channels, momentum=0.01)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                               padding=padding, bias=False)
        self.bn2   = nn.BatchNorm1d(channels, momentum=0.01)
        self.act   = nn.ReLU()

        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channels, momentum=0.01)
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)

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

class ResBlock1d_bottleneck(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, 
                 padding: int, stride: int = 1):
        super().__init__()
        
        # Bottleneck width is channels_in (stay narrow for the spatial conv)
        bottleneck = channels_in

        self.conv1 = nn.Conv1d(channels_in,  bottleneck,    kernel_size=1, bias=False)  # pointwise squeeze
        self.conv2 = nn.Conv1d(bottleneck,   bottleneck,    kernel_size=kernel_size,    # spatial (cheap, narrow)
                               padding=padding, stride=stride, bias=False)
        self.conv3 = nn.Conv1d(bottleneck,   channels_out,  kernel_size=1, bias=False)  # pointwise expand
        self.act   = nn.ReLU()

        self.downsample = None
        if stride != 1 or channels_in != channels_out:
            self.downsample = nn.Conv1d(channels_in, channels_out, kernel_size=1, 
                                        stride=stride, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)

class RNO_resnet(nn.Module):
    """
    1D ResNet for signal/noise classification.
    Input:  (batch, 4, 1024)  ← Same as FpgaCNN!
    Output: (batch, 1)        ← Binary classification
    """

    def __init__(self,
                 n_channels: int = 4,
                 hidden_units: int = 32,
                 output_shape: int = 1,
                 dropout: float = 0.1):

        super().__init__()

        self.temporal_resnet = nn.Sequential(
            nn.BatchNorm1d(n_channels, momentum=0.01),
            
            # Channel lifting
            nn.Conv1d(n_channels, hidden_units, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_units, momentum=0.01),
            nn.ReLU(),
            
            ResBlock1d(hidden_units, kernel_size=5, padding=2, stride=4),   # 1024 → 256
            ResBlock1d(hidden_units, kernel_size=5, padding=2, stride=4),   # 256 → 64
            
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_units, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, output_shape),
        )

    def forward(self, x):
        x = self.temporal_resnet(x)  # (B, hidden_units, 64)
        x = self.head(x)               # (B, 1)
        return x.squeeze(1)            # (B,)

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
                 output_shape: int = 1,
                 dropout: float = 0.1):

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

class ResBlock1d_plusplus(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, padding: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels_in, channels_out, kernel_size=kernel_size,
                               padding=padding, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(channels_out, momentum=.01)
        self.conv2 = nn.Conv1d(channels_out, channels_out, kernel_size=kernel_size,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels_out, momentum=.01)
        self.act   = nn.ReLU()

        self.downsample = None
        if stride != 1 or channels_out != channels_in:
            self.downsample = nn.Sequential(
                nn.Conv1d(channels_in, channels_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channels_out, momentum=.01)
            )

    def forward(self, x):
        identity = x
        out = self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)


class RNO_resnet_plusplus(nn.Module):
    """
    Improved resnet+ architecture with batchnorm and dropout layer
    Input:  (batch, 4, 1024)  ← Same as FpgaCNN!
    Output: (batch, 1)        ← Binary classification
    Params: 10,657
    Size: 829.00 MBs
    """
    def __init__(self,
                 n_channels: int = 4,
                 hidden_units: int = 32,
                 output_shape: int = 1,
                 dropout: float = 0.1):

        super().__init__()

        self.pre_process = nn.Sequential(
            nn.BatchNorm1d(n_channels,momentum=.01),
            nn.Conv1d(n_channels, hidden_units, kernel_size = 5, stride = 2, padding = 2),
            nn.MaxPool1d(kernel_size = 2)
        )

        self.res_block1 = ResBlock1d_plusplus(hidden_units, hidden_units, kernel_size = 3, padding = 1)
        self.res_block2 = ResBlock1d_plusplus(hidden_units, hidden_units // 2, kernel_size = 3, padding = 1)
        avg_pool_out = 64
        self.avg_pool = nn.AdaptiveAvgPool1d(avg_pool_out)
        self.flatten = nn.Flatten()
        self.linear_layer1 = nn.Linear(in_features = (hidden_units // 2) * avg_pool_out, out_features = (hidden_units // 2) * avg_pool_out)
        self.relu = nn.Relu()
        self.dropout = nn.Dropout(dropout)
        self.linear_layer2 = nn.Linear(in_features = (hidden_units // 2) * avg_pool_out, out_features = output_shape)

    def forward(self, x):
        x = self.pre_process(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear_layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_layer2(x)
        return x.squeeze(1)  

ARCH_REGISTRY = {
    "baseline": BaselineCNN,
    "tiny":     TinyCNN,
    "fpga":     FpgaCNN,
    "resnet":  RNO_resnet,
    "resnet+": RNO_resnet_plus,
    "resnet++": RNO_resnet_plusplus
}


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total_loss = total_correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for wav, lbl in loader:
            wav, lbl = wav.to(device), lbl.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(wav)
            loss   = criterion(logits, lbl)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_loss    += loss.item() * len(lbl)
            total_correct += (preds == lbl).sum().item()
            total         += len(lbl)
    return total_loss / total, total_correct / total


def update_plot(csv_path, png_path):
    if not _HAS_MPL:
        return
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return
    ep  = [int(r["epoch"])        for r in rows]
    trl = [float(r["train_loss"]) for r in rows]
    vll = [float(r["val_loss"])   for r in rows]
    tra = [float(r["train_acc"])  for r in rows]
    vla = [float(r["val_acc"])    for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ep, trl, label="train"); ax1.plot(ep, vll, label="val")
    ax1.set(xlabel="Epoch", ylabel="BCE Loss", title="Loss")
    ax1.legend(); ax1.grid(True)
    ax2.plot(ep, tra, label="train"); ax2.plot(ep, vla, label="val")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    ax2.set_ylim([-0.1,1.1])
    ax2.legend(); ax2.grid(True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signal",       required=True, help="Signal HDF5 from extract.py")
    p.add_argument("--noise",        required=True, help="Noise  HDF5 from extract.py")
    p.add_argument("--out",          default="runs/exp01", help="Output directory")
    p.add_argument("--arch",         default="baseline",
                   choices=list(ARCH_REGISTRY),
                   help=f"Model architecture: {list(ARCH_REGISTRY)}")
    p.add_argument("--crop-samples", type=int, default=None,
                   help="Centre-crop traces to N samples (default: full trace)")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--val-frac",     type=float, default=0.15)
    p.add_argument("--workers",      type=int,   default=None,
                   help="DataLoader worker processes (default: min(8, cpu_count))")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--dropout",      type=float, default=0.2,
                   help="Dropout rate (default: 0.2)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = vars(args)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── device ────────────────────────────────────────────────────────────────
    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cpu"))
    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    # ── split each file independently so val has both classes ─────────────────
    def split_indices(h5_path):
        with h5py.File(h5_path, "r") as f:
            n = f.attrs["n_events"]
        idx = rng.permutation(int(n))
        n_val = max(1, int(n * args.val_frac))
        return idx[n_val:], idx[:n_val]   # train, val

    sig_train_idx, sig_val_idx = split_indices(args.signal)
    noi_train_idx, noi_val_idx = split_indices(args.noise)

    n_sig   = len(sig_train_idx)
    n_noise = len(noi_train_idx)
    print(f"Train: {n_sig} signal + {n_noise} noise")
    print(f"Val:   {len(sig_val_idx)} signal + {len(noi_val_idx)} noise")

    crop = args.crop_samples
    train_ds = ConcatDataset([
        WaveformDataset(args.signal, sig_train_idx, crop),
        WaveformDataset(args.noise,  noi_train_idx, crop),
    ])
    val_ds = ConcatDataset([
        WaveformDataset(args.signal, sig_val_idx, crop),
        WaveformDataset(args.noise,  noi_val_idx, crop),
    ])

    n_workers = args.workers if args.workers is not None else min(8, os.cpu_count() or 4)
    print(f"DataLoader workers: {n_workers}")
    loader_kw = dict(num_workers=n_workers, pin_memory=(device.type == "cuda"),
                     persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, **loader_kw)

    # ── model ─────────────────────────────────────────────────────────────────
    model = ARCH_REGISTRY[args.arch](n_channels=4, dropout=args.dropout).to(device)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architecture: {args.arch}  |  Parameters: {n_p:,}")

    # pos_weight corrects for class imbalance automatically
    pos_weight = torch.tensor([n_noise / max(n_sig, 1)], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── logging ───────────────────────────────────────────────────────────────
    csv_path = out_dir / "metrics.csv"
    png_path = out_dir / "loss_curve.png"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,elapsed_s\n")

    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion,
                                    device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   optimizer, criterion,
                                    device, train=False)
        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"Epoch {epoch:4d}/{args.epochs}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  |  "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f}  "
              f"lr={lr_now:.2e}  {time.time()-t_ep:.1f}s")

        with open(csv_path, "a") as f:
            f.write(f"{epoch},{tr_loss:.6f},{tr_acc:.6f},"
                    f"{vl_loss:.6f},{vl_acc:.6f},{lr_now:.6e},{elapsed:.1f}\n")

        ckpt = {"epoch": epoch, "model_state": model.state_dict(),
                "val_loss": vl_loss, "val_acc": vl_acc, "config": cfg,
                "sig_val_idx": sig_val_idx, "noi_val_idx": noi_val_idx}

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(ckpt, out_dir / "best_model.pt")

        if epoch % 5 == 0 or epoch == args.epochs:
            update_plot(csv_path, png_path)

    torch.save(ckpt, out_dir / "last_model.pt")
    print(f"\nDone. Best val loss: {best_val_loss:.4f}  →  {out_dir}")


if __name__ == "__main__":
    main()
