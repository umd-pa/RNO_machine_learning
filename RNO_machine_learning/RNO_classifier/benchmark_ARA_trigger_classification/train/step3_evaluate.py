"""
Stage 3: Evaluate the best trained model — ROC, threshold calibration,
         confusion matrix, and efficiency vs SNR.

Usage:
    python evaluate.py \
        --checkpoint runs/exp01/best_model.pt \
        --signal     signal.h5 \
        --noise      noise.h5  \
        [--target-fpr 1e-5] \
        [--out        runs/exp01/eval]

The --signal / --noise files and --val-frac / --seed must match what was used
in step2_train.py (defaults are identical, so you only need to change them if you did).

Outputs (inside --out):
    roc_curve.png           — ROC with operating point marked
    confusion_matrix.png    — at chosen threshold
    efficiency_vs_snr.png   — signal efficiency vs max-channel SNR
    scores.npz              — raw scores, labels, SNR (for custom analysis)
    summary.txt             — threshold, FPR, TPR, rejection factor
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
    _HAS_SKL = True
except ImportError:
    _HAS_SKL = False
    print("WARNING: scikit-learn not found — install it for ROC/confusion matrix.")

# Reuse model definition and dataset from step2_train.py
try:
    from step2_train import ARCH_REGISTRY, WaveformDataset
except ImportError:
    sys.exit("ERROR: could not import from step2_train.py. Run from the same directory.")


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_scores(model, loader, device):
    model.eval()
    scores, labels = [], []
    for wav, lbl in loader:
        scores.append(torch.sigmoid(model(wav.to(device))).cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(scores), np.concatenate(labels)


# ── threshold calibration ────────────────────────────────────────────────────

def threshold_at_fpr(fpr_arr, tpr_arr, thresholds, target_fpr):
    mask = fpr_arr <= target_fpr
    if not np.any(mask):
        idx = np.argmin(fpr_arr)
    else:
        candidates = np.where(mask)[0]
        idx = candidates[np.argmax(tpr_arr[candidates])]
    return thresholds[idx], fpr_arr[idx], tpr_arr[idx]


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_roc(fpr, tpr, roc_auc, op_fpr, op_tpr, target_fpr, path):
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.scatter([op_fpr], [op_tpr], s=100, zorder=5,
               label=f"FPR={op_fpr:.2e}, TPR={op_tpr:.3f}")
    ax.axvline(target_fpr, ls="--", color="gray", alpha=0.7,
               label=f"Target FPR={target_fpr:.0e}")
    positive_fpr = fpr[fpr > 0]
    xmin = positive_fpr.min() * 0.5 if len(positive_fpr) > 0 else 1e-6
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve", #xscale="log",
           xlim=(max(xmin, 1e-8), 1.0), ylim=(0, 1.05))
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_score_histogram(scores, labels, threshold, path, n_bins=50):
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[labels == 0], bins=n_bins, range=(0, 1),
            alpha=0.6, color="tab:red",  label="Noise")
    ax.hist(scores[labels == 1], bins=n_bins, range=(0, 1),
            alpha=0.6, color="tab:blue", label="Signal")
    ax.axvline(threshold, ls="--", color="black", label=f"Threshold={threshold:.4f}")
    ax.set(xlabel="CNN score", ylabel="Events", title="Score Distribution")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def plot_confusion(y_true, y_pred, path):
    if not (_HAS_MPL and _HAS_SKL):
        return
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Noise", "Signal"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


# def plot_efficiency_vs_snr(scores, labels, snr_per_event, threshold, path, n_bins=20):
#     if not _HAS_MPL:
#         return
#     sig_mask   = labels == 1
#     sig_scores = scores[sig_mask]
#     sig_snr    = snr_per_event[sig_mask]
#     valid      = sig_snr > 0
#     if not np.any(valid):
#         print("WARNING: no positive signal SNR values — skipping efficiency plot.")
#         return

#     bins = np.logspace(np.log10(sig_snr[valid].min()),
#                        np.log10(sig_snr.max()), n_bins + 1)
#     effs, errs, centres, counts = [], [], [], []
#     for lo, hi in zip(bins[:-1], bins[1:]):
#         mask = (sig_snr >= lo) & (sig_snr < hi)
#         if not mask.any():
#             continue
#         n_tot  = mask.sum()
#         n_pass = (sig_scores[mask] >= threshold).sum()
#         eff    = n_pass / n_tot
#         effs.append(eff)
#         errs.append(np.sqrt(eff * (1 - eff) / n_tot))
#         centres.append(np.sqrt(lo * hi))
#         counts.append(n_tot)

#     fig, ax = plt.subplots(figsize=(7, 4))
#     ax.errorbar(centres, effs, yerr=errs, fmt="o-", capsize=4, label="Efficiency")
#     ax.axhline(0.5, ls="--", color="gray", alpha=0.7)
#     ax.set(xlabel="Max-channel SNR", ylabel="Signal Efficiency",
#            title=f"Efficiency vs SNR  (threshold={threshold:.4f})",
#            xscale="log", ylim=(-0.05, 1.05))
#     ax.grid(True, which="both", alpha=0.3)

#     ax2 = ax.twinx()
#     ax2.bar(centres, counts, width=[c * 0.3 for c in centres],
#             alpha=0.2, color="gray", label="Events per bin")
#     ax2.set_ylabel("Events per bin")
#     ax2.set_ylim(bottom=0)

#     lines1, labs1 = ax.get_legend_handles_labels()
#     lines2, labs2 = ax2.get_legend_handles_labels()
#     ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9)

#     fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
#     print(f"Saved: {path}")

def plot_efficiency_vs_snr(scores, labels, snr_per_event, threshold, path,
                           n_bins=20, snr_max=6.0):
    if not _HAS_MPL:
        return
    sig_mask   = labels == 1
    sig_scores = scores[sig_mask]
    sig_snr    = snr_per_event[sig_mask]
    if not np.any(sig_snr > 0):
        print("WARNING: no positive signal SNR values — skipping efficiency plot.")
        return

    # Linear bins up to snr_max, then one overflow bin
    bin_edges = np.linspace(0, snr_max, n_bins + 1)
    all_edges = np.append(bin_edges, np.inf)
    centres   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    centres   = np.append(centres, snr_max + 0.5 * snr_max / n_bins)  # overflow centre

    effs, errs, counts = [], [], []
    for lo, hi in zip(all_edges[:-1], all_edges[1:]):
        mask  = (sig_snr >= lo) & (sig_snr < hi)
        n_tot = mask.sum()
        if n_tot == 0:
            effs.append(np.nan); errs.append(0.0); counts.append(0)
            continue
        n_pass = (sig_scores[mask] >= threshold).sum()
        eff    = n_pass / n_tot
        effs.append(eff)
        errs.append(np.sqrt(eff * (1 - eff) / n_tot))
        counts.append(n_tot)

    effs   = np.array(effs)
    errs   = np.array(errs)
    counts = np.array(counts)
    valid  = counts > 0
    eff_vs_snr_data = {"SNR bin": np.round(centres[valid],4),
            "efficiency value": np.round(effs[valid],4),
            "efficiency error bar": np.round(errs[valid],4),
            }
    pd.DataFrame(eff_vs_snr_data).to_csv(str(path).replace(".png", ".csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(centres[valid], effs[valid], yerr=errs[valid],
                fmt="o-", capsize=4, label="Efficiency")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.7)
    ax.axvline(snr_max, ls=":", color="gray", alpha=0.5)
    ax.text(snr_max + 0.05, 0.05, "overflow", fontsize=8, color="gray")
    ax.set(xlabel="Max-channel SNR  (V$_{p2p}$ / 2σ$_{noise}$)",
           ylabel="Signal Efficiency",
           title=f"Efficiency vs SNR  (threshold={threshold:.4f})",
           xlim=(0, snr_max * 1.15), ylim=(-0.05, 1.05))
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    bin_width = snr_max / n_bins
    ax2.bar(centres[valid], counts[valid], width=bin_width * 0.8,
            alpha=0.2, color="gray", label="Events per bin")
    ax2.set_ylabel("Events per bin")
    ax2.set_ylim(bottom=0)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9)

    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def print_summary(threshold, fpr, tpr, roc_auc, target_fpr, path):
    lines = [
        "",
        "=" * 55,
        "  CNN Trigger Evaluation Summary",
        "=" * 55,
        f"  ROC AUC                     : {roc_auc:.6f}",
        f"  Target FPR                  : {target_fpr:.2e}",
        "  " + "─" * 43,
        f"  Threshold (CNN score)       : {threshold:.6f}",
        f"  False Positive Rate (FPR)   : {fpr:.4e}",
        f"  True Positive Rate (TPR)    : {tpr:.4f}",
        f"  Signal Efficiency           : {tpr * 100:.2f} %",
        f"  Background rejection (1/FPR): {1/fpr if fpr > 0 else float('inf'):.3e}",
        "=" * 55,
        "",
    ]
    text = "\n".join(lines)
    print(text)
    path.write_text(text)
    print(f"Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--signal",     required=True, help="Signal HDF5 from extract.py")
    p.add_argument("--noise",      required=True, help="Noise  HDF5 from extract.py")
    p.add_argument("--target-fpr", type=float, default=1e-3)
    p.add_argument("--out",        default=None)
    p.add_argument("--batch-size", type=int,   default=512)
    return p.parse_args()


def main():
    args     = parse_args()
    ckpt_path = Path(args.checkpoint)
    out_dir   = Path(args.out) if args.out else ckpt_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    # ── load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    print(f"Checkpoint: epoch {ckpt['epoch']}  "
          f"val_loss={ckpt['val_loss']:.4f}  val_acc={ckpt['val_acc']:.4f}")

    model = ARCH_REGISTRY[cfg.get("arch", "baseline")](n_channels=4).to(device)
    model.load_state_dict(ckpt["model_state"])

    # ── load val indices saved by step2_train.py ───────────────────────────────────
    if "sig_val_idx" not in ckpt or "noi_val_idx" not in ckpt:
        sys.exit(
            "ERROR: checkpoint does not contain val indices.\n"
            "Re-run step2_train.py to produce a new checkpoint, then evaluate again."
        )
    sig_val_idx = ckpt["sig_val_idx"]
    noi_val_idx = ckpt["noi_val_idx"]

    crop = cfg.get("crop_samples", None)
    val_ds = ConcatDataset([
        WaveformDataset(args.signal, sig_val_idx, crop),
        WaveformDataset(args.noise,  noi_val_idx, crop),
    ])
    loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, persistent_workers=True,
                        pin_memory=(device.type == "cuda"))

    print(f"Val set: {len(sig_val_idx)} signal + {len(noi_val_idx)} noise")

    # ── inference ─────────────────────────────────────────────────────────────
    print("Running inference…")
    scores, val_labels = get_scores(model, loader, device)

    # Gather SNR for signal val events (from HDF5 directly)
    with h5py.File(args.signal, "r") as f:
        sig_snr_all = f["snr"][:]          # (N_sig, 4)
    sig_snr_val = sig_snr_all[sig_val_idx]
    # Pad noise SNR with NaN so arrays align with val_labels
    noise_snr_val = np.full((len(noi_val_idx), 4), float("nan"), dtype=np.float32)
    snr_val = np.concatenate([sig_snr_val, noise_snr_val], axis=0)
    # nanmax gives NaN for all-NaN rows (noise events); replace with 0 for plotting
    snr_per_event = np.where(
        np.all(np.isnan(snr_val), axis=1),
        0.0,
        np.nanmax(snr_val, axis=1),
    )

    np.savez(out_dir / "scores.npz",
             scores=scores, labels=val_labels, snr=snr_per_event)
    print(f"Saved: {out_dir / 'scores.npz'}")

    if not _HAS_SKL:
        print("scikit-learn not available — skipping ROC. Install and rerun.")
        return

    # ── sanity check ─────────────────────────────────────────────────────────
    n_pos = int((val_labels == 1).sum())
    n_neg = int((val_labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        print(f"ERROR: val set has {n_pos} signal and {n_neg} noise events — "
              "need at least one of each for ROC. "
              "Try a larger dataset or increase --val-frac.")
        sys.exit(1)

    # ── ROC + threshold ───────────────────────────────────────────────────────
    fpr_arr, tpr_arr, thresholds = roc_curve(val_labels, scores, pos_label=1)
    roc_auc = auc(fpr_arr, tpr_arr)
    threshold, op_fpr, op_tpr = threshold_at_fpr(
        fpr_arr, tpr_arr, thresholds, args.target_fpr)

    plot_roc(fpr_arr, tpr_arr, roc_auc, op_fpr, op_tpr,
             args.target_fpr, out_dir / "roc_curve.png")
    plot_score_histogram(scores, val_labels, threshold,
                         out_dir / "score_histogram.png")
    plot_confusion(val_labels.astype(int), (scores >= threshold).astype(int),
                   out_dir / "confusion_matrix.png")
    plot_efficiency_vs_snr(scores, val_labels, snr_per_event,
                           threshold, out_dir / "efficiency_vs_snr.png")
    print_summary(threshold, op_fpr, op_tpr, roc_auc,
                  args.target_fpr, out_dir / "summary.txt")

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
