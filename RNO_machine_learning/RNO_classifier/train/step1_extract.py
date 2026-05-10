"""
Stage 1: Extract raw waveforms from one or more .nur files into a single HDF5 file.

Run once for signal, once for noise:

    python extract.py --input signal1.nur signal2.nur --label 1 --out signal.h5
    python extract.py --input noise1.nur noise2.nur   --label 0 --out noise.h5

Output HDF5 layout:
    /waveforms   float32  (N, 4, T)  — raw voltage traces, channels 0–3
    /labels      int8     (N,)       — 1=signal, 0=noise
    /snr         float32  (N, 4)     — per-channel SNR (signal only; else NaN)
    /energy      float32  (N,)       — neutrino energy in eV (signal only; else NaN)
    /vertex      float32  (N, 3)     — interaction vertex x,y,z in m (signal only)
    /weight      float32  (N,)       — MC event weight (signal only; else NaN)

Full trace length is preserved. Cropping is done at train time.
SNR = peak(noiseless) / RMS(noised – noiseless) when sim_station is available,
      otherwise peak(noised) / RMS(noised).
"""

import argparse
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

try:
    import NuRadioReco.modules.io.eventReader as _ev_reader_mod
except ImportError:
    sys.exit(
        "ERROR: NuRadioReco not found.\n"
        "  Activate the correct environment or: pip install NuRadioReco"
    )
from NuRadioReco.framework.parameters import channelParameters as cp
from NuRadioReco.framework.parameters import particleParameters as pp
from NuRadioReco.framework.parameters import stationParameters as sp

NOISED_CHANNELS = [0, 1, 2, 3]
SIM_CHANNELS    = [40, 41, 42, 43]


# ── SNR helpers ───────────────────────────────────────────────────────────────

def _snr(noised: np.ndarray, noiseless: np.ndarray | None) -> np.ndarray:
    """Per-channel SNR = V_p2p / (2 * noise_rms), shape (4,).
    Noise RMS estimated from samples outside a T/2 window centred on the signal peak,
    with wraparound so the window is always exactly T/2 samples.
    """
    snr = np.zeros(4, dtype=np.float32)
    for i in range(4):
        if noiseless is not None:
            v_p2p    = np.max(noiseless[i]) - np.min(noiseless[i])
            T        = len(noised[i])
            half     = T // 2
            peak_idx = np.argmax(np.abs(noiseless[i]))
            signal_indices = np.arange(peak_idx - half // 2,
                                       peak_idx - half // 2 + half) % T
            mask = np.ones(T, dtype=bool)
            mask[signal_indices] = False
            noise_samples = noised[i][mask]
            noise_rms = np.sqrt(np.mean(noise_samples ** 2)) if len(noise_samples) > 0 else 0.0
            snr[i] = v_p2p / (2 * noise_rms) if noise_rms > 0 else 0.0
        else:
            v_p2p     = np.max(noised[i]) - np.min(noised[i])
            noise_rms = np.sqrt(np.mean(noised[i] ** 2))
            snr[i]    = v_p2p / (2 * noise_rms) if noise_rms > 0 else 0.0
    return snr


# ── event iterator ────────────────────────────────────────────────────────────

def iter_events(nur_path: str, label: int):
    reader = _ev_reader_mod.eventReader()
    reader.begin(nur_path)

    for event in reader.run():
        stations = list(event.get_stations())
        if not stations:
            continue
        station = stations[0]

        ch_data = {ch.get_id(): ch.get_trace().astype(np.float32)
                   for ch in station.iter_channels()
                   if ch.get_id() in NOISED_CHANNELS}
        if len(ch_data) < 4:
            warnings.warn(f"Skipping event: only found channels {list(ch_data.keys())}")
            continue
        noised = np.stack([ch_data[c] for c in NOISED_CHANNELS])

        noiseless = None
        if station.has_sim_station():
            sim = station.get_sim_station()
            sim_data = {sc.get_id(): sc.get_trace().astype(np.float32)
                        for sc in sim.iter_channels()
                        if sc.get_id() in SIM_CHANNELS}
            if len(sim_data) == 4:
                noiseless = np.stack([sim_data[c] for c in SIM_CHANNELS])

        snr = _snr(noised, noiseless)

        energy = vertex = weight = None
        if label == 1:
            primary = event.get_primary()
            if primary is not None:
                energy = primary.get_parameter(pp.energy)
                weight = primary.get_parameter(pp.weight)
                vertex = np.array(primary.get_parameter(pp.vertex), dtype=np.float32)[:3]

        yield {
            "waveform": noised,
            "snr":      snr,
            "energy":   np.float32(energy if energy is not None else float("nan")),
            "vertex":   vertex if vertex is not None else np.full(3, float("nan"), np.float32),
            "weight":   np.float32(weight if weight is not None else float("nan")),
            "label":    np.int8(label),
        }


# ── pre-scan ──────────────────────────────────────────────────────────────────

def count_valid_events(nur_paths: list[str]) -> tuple[int, int]:
    """Return (total_n_valid, T) across all files."""
    total, T = 0, None
    for path in nur_paths:
        print(f"Pre-scanning {path} …")
        reader = _ev_reader_mod.eventReader()
        reader.begin(path)
        for event in reader.run():
            stations = list(event.get_stations())
            if not stations:
                continue
            station = stations[0]
            ch_ids = [ch.get_id() for ch in station.iter_channels()]
            if all(c in ch_ids for c in NOISED_CHANNELS):
                if T is None:
                    for ch in station.iter_channels():
                        if ch.get_id() == 0:
                            T = len(ch.get_trace())
                            break
                total += 1
    return total, T


# ── writer ────────────────────────────────────────────────────────────────────

def write_hdf5(nur_paths: list[str], label: int, out_path: str):
    n, T = count_valid_events(nur_paths)
    if n == 0:
        sys.exit("ERROR: no valid events found.")
    print(f"Total events: {n}   Trace length: {T} samples")

    with h5py.File(out_path, "w") as f:
        ds_wav = f.create_dataset("waveforms", shape=(n, 4, T), dtype=np.float32,
                                  chunks=(min(256, n), 4, T))
        ds_lbl = f.create_dataset("labels",  shape=(n,),   dtype=np.int8)
        ds_snr = f.create_dataset("snr",     shape=(n, 4), dtype=np.float32)
        ds_nrg = f.create_dataset("energy",  shape=(n,),   dtype=np.float32)
        ds_vtx = f.create_dataset("vertex",  shape=(n, 3), dtype=np.float32)
        ds_wgt = f.create_dataset("weight",  shape=(n,),   dtype=np.float32)

        f.attrs["label"]     = label
        f.attrs["trace_len"] = T
        f.attrs["n_events"]  = n

        i = 0
        for path in nur_paths:
            for ev in tqdm(iter_events(path, label), desc=Path(path).name):
                ds_wav[i] = ev["waveform"]
                ds_lbl[i] = ev["label"]
                ds_snr[i] = ev["snr"]
                ds_nrg[i] = ev["energy"]
                ds_vtx[i] = ev["vertex"]
                ds_wgt[i] = ev["weight"]
                i += 1

    print(f"Saved {n} events → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, nargs="+", help=".nur file(s) to extract")
    p.add_argument("--label", required=True, type=int, choices=[0, 1],
                   help="1 = signal, 0 = noise")
    p.add_argument("--out",   required=True, help="Output HDF5 path")
    args = p.parse_args()

    missing = [f for f in args.input if not Path(f).exists()]
    if missing:
        sys.exit(f"ERROR: file(s) not found: {', '.join(missing)}")

    write_hdf5(args.input, args.label, args.out)


if __name__ == "__main__":
    main()
