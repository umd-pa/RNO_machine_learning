"""
Simulation utils script to analyze simulation statistics quickly
"""
import h5py
from pathlib import Path
import re
import glob
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

shards_dir = Path('/data/condor_shared/users/ssued/RNO_vertex_reconstruction_ml/RNO_album_maker/jobs/shards_debug_clamp')

def print_shards_dir_mem(shards_dir: str | Path, verbose = True):
    shards_dir = Path(shards_dir)

    total_bytes = 0
    for p in shards_dir.iterdir():
        if p.is_file() and 'shard' in p.name:
            try:
                size = p.stat().st_size
                total_bytes += size
                with h5py.File(p, 'r') as s:
                    shard_n = s['vertices'].shape[0] #type: ignore
            except Exception as e:
                if verbose:
                    print(f"Skipping {p.name}: {e}")
                continue
            if verbose:
                print(f"For {shard_n} events | file size: {size/1e6:.3f} MB ({size/1024**2:.3f} MiB) — {p.name}")

    print(f"Total size: {total_bytes/1e6:.3f} MB ({total_bytes/1024**2:.3f} MiB)")

def print_avg_step_runtime(job_logs_dir: str | Path):
    job_logs_dir = Path(job_logs_dir)

    # Patterns for extracting runtimes
    step1_pattern = re.compile(r'finished in ([\d.]+)([smh])')
    step23_pattern = re.compile(r'Runtime: ([\d.]+) s')

    step1_times = []
    step2_times = []
    step3_times = []

    # Process step1 files
    for file in tqdm(glob.glob('step1*', root_dir=job_logs_dir)):
        with open(job_logs_dir / file, 'r') as f:
            content = f.read()
            match = step1_pattern.search(content)
            if match:
                time_val = float(match.group(1))
                unit = match.group(2)
                # Convert to seconds
                if unit == 'm':
                    time_val *= 60
                elif unit == 'h':
                    time_val *= 3600
                step1_times.append(time_val)

    # Process step2 files
    for file in tqdm(glob.glob('step2*', root_dir=job_logs_dir)):
        with open(job_logs_dir / file, 'r') as f:
            content = f.read()
            match = step23_pattern.search(content)
            if match:
                step2_times.append(float(match.group(1)))

    # Process step3 files
    for file in tqdm(glob.glob('step3*', root_dir=job_logs_dir)):
        with open(job_logs_dir / file, 'r') as f:
            content = f.read()
            match = step23_pattern.search(content)
            if match:
                step3_times.append(float(match.group(1)))

    # Calculate averages
    avg_step1 = sum(step1_times) / len(step1_times) if step1_times else 0
    avg_step2 = sum(step2_times) / len(step2_times) if step2_times else 0
    avg_step3 = sum(step3_times) / len(step3_times) if step3_times else 0

    print(f"Total runtime: {(avg_step1+avg_step2+avg_step3):.2f} seconds ({(avg_step1+avg_step2+avg_step3)/60:.2f} minutes)")
    print(f"Average Step 1 runtime: {avg_step1:.2f} seconds ({avg_step1/60:.2f} minutes) | {(avg_step1/(avg_step1+avg_step2+avg_step3))*100:.2f}%")
    print(f"Average Step 2 runtime: {avg_step2:.2f} seconds ({avg_step2/60:.2f} minutes) | {(avg_step2/(avg_step1+avg_step2+avg_step3))*100:.2f}%")
    print(f"Average Step 3 runtime: {avg_step3:.2f} seconds ({avg_step3/60:.2f} minutes) | {(avg_step3/(avg_step1+avg_step2+avg_step3))*100:.2f}%")
    print(f"\nSamples: Step1={len(step1_times)}, Step2={len(step2_times)}, Step3={len(step3_times)}")

def plot_hist_events(shards_dir: str | Path, n_nu: int):

    shards_dir = Path(shards_dir)

    n_events_arr = []
    for file in os.listdir(shards_dir):
        if 'vds' not in file:
            file_path = shards_dir / file
            with h5py.File(file_path, 'r') as f:
                n_events = len(f['vertices']) #type: ignore
                n_events_arr.append(n_events)

    plt.figure()
    plt.hist(n_events_arr)
    plt.title(f'Number of events generated for {n_nu} neutrino simulations in {len(n_events_arr)} shards')
    plt.axvline(sum(n_events_arr) / len(n_events_arr), color='red', linestyle='--', label=f'Mean: {sum(n_events_arr) / len(n_events_arr):.1f}')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xlabel('Event Number in Shard')
    plt.grid(True,alpha=0.8)