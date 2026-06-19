"""
Stage 1: Extract raw waveforms from one or more .nur files into a single HDF5 file.

Run once for signal, once for noise. If you want typical amplification (x1500) use:

    python extract.py --input signal1.nur signal2.nur --label 1 --hw-resp SCALE --out signal.h5
    python extract.py --input noise1.nur noise2.nur   --label 0 --hw-resp SCALE --out noise.h5

--hw-resp options are:
    APPLY: Apply full hardware response simulation (not functional with custom detectors, but will work with RNO-G single station!)
    SCALE: Simply amplify signal by 1500
    NONE: Do not apply any gain/hardware amp. It is assumed it was already applied during simulation

--trigger:
    Define which trigger to test if it triggered during simulation. Default is 'rnog_proxy_3.5sigma'. Type "None" if this is not desired

Output HDF5 layout:
    /waveforms   float32  (N, 4, T)  — raw voltage traces, channels 0–3
    /labels      int8     (N,)       — 1=signal, 0=noise
    /snr         float32  (N, 4)     — per-channel SNR (signal only; else NaN)
    /energy      float32  (N,)       — neutrino energy in eV (signal only; else NaN)
    /vertex      float32  (N, 3)     — interaction vertex x,y,z in m (signal only)
    /weight      float32  (N,)       — MC event weight (signal only; else NaN)
    /triggered   int8     (N,)       — Whether the given trigger triggered on an event or not (Optional)

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
from enum import Enum, auto
import gc

try:
    import NuRadioReco.modules.io.eventReader as _ev_reader_mod
    from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator
    from NuRadioReco.detector.RNO_G import rnog_detector
except ImportError:
    sys.exit(
        "ERROR: NuRadioReco not found.\n"
        "  Activate the correct environment or: pip install NuRadioReco"
    )

from NuRadioReco.framework.parameters import particleParameters as pp


NOISED_CHANNELS = [0, 1, 2, 3]
SIM_CHANNELS    = [40, 41, 42, 43]


# ── helpers ───────────────────────────────────────────────────────────────

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

def digitize(values, bits=14, v_min=-.5, v_max=.5):
    """Simulate digitization to N bits with given voltage range."""
    values = np.clip(values,v_min,v_max) # Clip
    # Quantize to integer levels
    levels = 2**bits - 1  # e.g., 16383 for 14-bit
    # Scale to [0, levels]
    scaled = (values - v_min) / (v_max - v_min) * levels
    # Round to nearest integer
    digitized = np.round(scaled).astype(int)
    # Convert back to voltage
    return digitized / levels * (v_max - v_min) + v_min

# ── det finder ────────────────────────────────────────────────────────────

def _path_to_det():
    """Quick helper function to find the detector file assuming the detector is inside RNO_classifier/generate/station.json

    Returns:
        Path: Path to detector
    """
    script_path = Path(__file__).resolve()
    det_path = script_path.parent.parent / 'generate' / 'station.json'
    return str(det_path)


# ── event iterator ────────────────────────────────────────────────────────────

class HardwareResponse(Enum):
    APPLY = auto()  # run hardware response + detector
    SCALE = auto()  # multiply by 1500 (no detector)
    NONE  = auto()  # pass through raw traces

    def scale_factor(self) -> float:
        return 1500.0 if self is HardwareResponse.SCALE else 1.0

    def needs_detector(self) -> bool:
        return self is HardwareResponse.APPLY


def iter_events(nur_path: str, label: int, hw_resp: HardwareResponse, trigger: str | None = None):
    reader = _ev_reader_mod.eventReader()
    reader.begin(nur_path)

    hardware_response = None
    det_RNOG = None

    if hw_resp.needs_detector():
        print('Adding hardware response...')
        hardware_response = hardwareResponseIncorporator.hardwareResponseIncorporator()
        hardware_response.begin()
        det_RNOG = rnog_detector.Detector(detector_file=_path_to_det())

    for event in reader.run():
        stations = list(event.get_stations())
        if not stations:
            continue
        station = stations[0]
        triggered = None
        if trigger:
            triggered = 1 if station.get_triggers()[trigger].has_triggered() else 0


        if hw_resp.needs_detector() and hardware_response is not None:
            hardware_response.run(event, station, det_RNOG, sim_to_data=True)

        scale = hw_resp.scale_factor()

        ch_data = {
            ch.get_id(): ch.get_trace().astype(np.float32) * scale
            for ch in station.iter_channels()
            if ch.get_id() in NOISED_CHANNELS
        }
        if len(ch_data) < 4:
            warnings.warn(f"Skipping event: only found channels {list(ch_data.keys())}")
            continue

        noised_undigitized = np.stack([ch_data[c] for c in NOISED_CHANNELS])

        noiseless_undigitized = None
        if label==1:
            sim_data = {
                c.get_id(): c.get_trace().astype(np.float32) * scale
                for c in station.iter_channels()
                if c.get_id() in SIM_CHANNELS
            }
            if len(sim_data) == 4:
                noiseless_undigitized = np.stack([sim_data[c] for c in SIM_CHANNELS])

        snr    = _snr(noised_undigitized, noiseless_undigitized)
        noised = digitize(noised_undigitized)

        energy = vertex = weight = None
        if label == 1:
            primary = event.get_primary()
            if primary is not None:
                energy = primary.get_parameter(pp.energy)
                weight = primary.get_parameter(pp.weight)
                vertex = np.array(primary.get_parameter(pp.vertex), dtype=np.float32)[:3]

        output = {
            "waveform": noised,
            "snr":      snr,
            "energy":   np.float32(energy if energy is not None else float("nan")),
            "vertex":   vertex if vertex is not None else np.full(3, float("nan"), np.float32),
            "weight":   np.float32(weight if weight is not None else float("nan")),
            "label":    np.int8(label)
        }

        if triggered is not None:
                output['triggered'] = np.int8(triggered) 
        yield output

    reader.end()
    del reader
    if hardware_response is not None:
        del hardware_response
    if det_RNOG is not None:
        del det_RNOG
    gc.collect()


# ── pre-scan ──────────────────────────────────────────────────────────────────

def count_valid_events(nur_paths: list[str], trigger: str | None) -> tuple[int, int | None]:
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
            if trigger is not None and trigger not in station.get_triggers():
                raise ValueError(
                    f"Trigger {trigger!r} completely missing from station triggers in {path}."
                    f"Available options are: {sorted(station.get_triggers().keys())}"
                )
            ch_ids = [ch.get_id() for ch in station.iter_channels()]
            if all(c in ch_ids for c in NOISED_CHANNELS):
                if T is None:
                    for ch in station.iter_channels():
                        if ch.get_id() == 0:
                            T = len(ch.get_trace())
                            break
                total += 1
        reader.end()
        del reader
        gc.collect()
    return total, T

# ── writer & appender ────────────────────────────────────────────────────────────────────

def write_hdf5(nur_paths: list[str], label: int, out_path: str, hw_resp: HardwareResponse = HardwareResponse.NONE, trigger: str | None = None):
    n, T = count_valid_events(nur_paths, trigger)
    if n == 0:
        sys.exit("ERROR: no valid events found.")
    print(f"Total events: {n}   Trace length: {T} samples")

    chunk_size = min(256, n)

    with h5py.File(out_path, "w") as f:
        ds_wav = f.create_dataset("waveforms", shape=(n, 4, T), dtype=np.float32,
                                  maxshape=(None, 4, T), chunks=(chunk_size, 4, T))
        ds_lbl = f.create_dataset("labels", shape=(n,), dtype=np.int8,
                                   maxshape=(None,), chunks=(chunk_size,))
        ds_snr = f.create_dataset("snr", shape=(n, 4), dtype=np.float32,
                                   maxshape=(None, 4), chunks=(chunk_size, 4))
        ds_nrg = f.create_dataset("energy", shape=(n,), dtype=np.float32,
                                   maxshape=(None,), chunks=(chunk_size,))
        ds_vtx = f.create_dataset("vertex", shape=(n, 3), dtype=np.float32,
                                   maxshape=(None, 3), chunks=(chunk_size, 3))
        ds_wgt = f.create_dataset("weight", shape=(n,), dtype=np.float32,
                                   maxshape=(None,), chunks=(chunk_size,))
        if trigger is not None:
            ds_trg = f.create_dataset("triggered", shape=(n,), dtype=np.int8, 
                                      maxshape=(None,), chunks=(chunk_size,))

        f.attrs["label"]     = label
        f.attrs["trace_len"] = T
        f.attrs["n_events"]  = n

        i = 0
        buffer = []
        
        for path in nur_paths:
            for ev in tqdm(iter_events(path, label, hw_resp=hw_resp, trigger=trigger), desc=Path(path).name):
                buffer.append(ev)
                
                # Write to disk only when the buffer matches our chunk size
                if len(buffer) >= chunk_size:
                    num = len(buffer)
                    ds_wav[i : i + num] = np.stack([e["waveform"] for e in buffer])
                    ds_lbl[i : i + num] = [e["label"] for e in buffer]
                    ds_snr[i : i + num] = np.stack([e["snr"] for e in buffer])
                    ds_nrg[i : i + num] = [e["energy"] for e in buffer]
                    ds_vtx[i : i + num] = np.stack([e["vertex"] for e in buffer])
                    ds_wgt[i : i + num] = [e["weight"] for e in buffer]
                    if trigger is not None:
                        ds_trg[i : i + num] = [e["triggered"] for e in buffer]
                    i += num
                    buffer.clear()
            
            # Flush remaining events for the current file if any exist
            if buffer:
                num = len(buffer)
                ds_wav[i : i + num] = np.stack([e["waveform"] for e in buffer])
                ds_lbl[i : i + num] = [e["label"] for e in buffer]
                ds_snr[i : i + num] = np.stack([e["snr"] for e in buffer])
                ds_nrg[i : i + num] = [e["energy"] for e in buffer]
                ds_vtx[i : i + num] = np.stack([e["vertex"] for e in buffer])
                ds_wgt[i : i + num] = [e["weight"] for e in buffer]
                if trigger is not None:
                    ds_trg[i : i + num] = [e["triggered"] for e in buffer]
                i += num
                buffer.clear()
                
            gc.collect()

    print(f"Saved {n} events → {out_path}")


def append_hdf5(nur_paths: list[str], label: int, out_path: str, hw_resp: HardwareResponse = HardwareResponse.NONE, trigger: str | None = None):
    n, T = count_valid_events(nur_paths, trigger)
    if n == 0:
        sys.exit("ERROR: no valid events found.")
    print(f"Total events: {n}   Trace length: {T} samples")

    with h5py.File(out_path, "a") as f:
        
        i = int(f.attrs["n_events"])
        T = int(f.attrs["trace_len"])

        f["waveforms"].resize((i + n, 4, T))
        f["labels"].resize((i + n,))
        f["snr"].resize((i + n, 4))
        f["energy"].resize((i + n,))
        f["vertex"].resize((i + n, 3))
        f["weight"].resize((i + n,))
        if trigger:
            f["triggered"].resize((i + n,))
        
        # Determine chunk size from the existing dataset configuration
        chunk_size = f["waveforms"].chunks[0] if f["waveforms"].chunks else 256
        buffer = []
        
        for path in nur_paths:
            for ev in tqdm(iter_events(path, label, hw_resp=hw_resp, trigger=trigger), desc=Path(path).name):
                buffer.append(ev)
                
                if len(buffer) >= chunk_size:
                    num = len(buffer)
                    f["waveforms"][i : i + num] = np.stack([e["waveform"] for e in buffer])
                    f["labels"][i : i + num]    = [e["label"] for e in buffer]
                    f["snr"][i : i + num]       = np.stack([e["snr"] for e in buffer])
                    f["energy"][i : i + num]    = [e["energy"] for e in buffer]
                    f["vertex"][i : i + num]    = np.stack([e["vertex"] for e in buffer])
                    f["weight"][i : i + num]    = [e["weight"] for e in buffer]
                    if trigger:
                        f["triggered"][i : i + num] = [e["triggered"] for e in buffer]
                    i += num
                    buffer.clear()
            
            # Flush remaining events for the current file if any exist
            if buffer:
                num = len(buffer)
                f["waveforms"][i : i + num] = np.stack([e["waveform"] for e in buffer])
                f["labels"][i : i + num]    = [e["label"] for e in buffer]
                f["snr"][i : i + num]       = np.stack([e["snr"] for e in buffer])
                f["energy"][i : i + num]    = [e["energy"] for e in buffer]
                f["vertex"][i : i + num]    = np.stack([e["vertex"] for e in buffer])
                f["weight"][i : i + num]    = [e["weight"] for e in buffer]
                if trigger:
                    f["triggered"][i : i + num] = [e["triggered"] for e in buffer]
                i += num
                buffer.clear()
                
            gc.collect()

        f.attrs["n_events"] = i

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, nargs="+", help=".nur file(s) to extract")
    p.add_argument("--label", required=True, type=int, choices=[0, 1],
                   help="1 = signal, 0 = noise")
    p.add_argument("--hw-resp", choices=[m.name for m in HardwareResponse], default="NONE", help="Hardware response mode (default: NONE)")
    p.add_argument("--trigger", default="rnog_proxy_3.5sigma", help="Trigger name to check if it fired during simulation (Use 'None' for no trigger firing)")
    p.add_argument("--out",   required=True, help="Output HDF5 path")
    args = p.parse_args()

    missing = [f for f in args.input if not Path(f).exists()]
    if missing:
        sys.exit(f"ERROR: file(s) not found: {', '.join(missing)}")

    if args.trigger is not None and args.trigger.strip().upper() == 'NONE':
        args.trigger = None

    hw_resp = HardwareResponse[args.hw_resp]
    print(args.trigger)
    write_hdf5([args.input[0]], args.label, args.out, hw_resp=hw_resp, trigger=args.trigger)
    if len(args.input) > 1:
        append_hdf5(args.input[1:], args.label, args.out, hw_resp=hw_resp, trigger=args.trigger)


if __name__ == "__main__":
    main()