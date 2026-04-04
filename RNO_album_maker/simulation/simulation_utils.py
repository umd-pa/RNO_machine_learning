"""
Simulation utils script to analyze simulation statistics quickly
"""
from random import shuffle

import h5py
from pathlib import Path
import re
import glob
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import hashlib
import numpy as np
import json



def get_max_memory_of_logs(job_logs_dir: str | Path, verbose: bool = False):
    job_logs_dir = Path(job_logs_dir)
    get_max_memory_of_step(job_logs_dir/'step1*', verbose = verbose)
    get_max_memory_of_step(job_logs_dir/'step2*', verbose = verbose)
    get_max_memory_of_step(job_logs_dir/'step3*', verbose = verbose)


def get_max_memory_of_step(job_logs_step: str | Path, verbose: bool = False):
    job_logs_step = Path(job_logs_step)
    # Find all stepn log files
    log_files = glob.glob(str(job_logs_step))
    
    if not log_files:
        print(f"No log files found matching {job_logs_step}")
        return

    results = []

    # Regex to find the number before " - MemoryUsage of job (MB)"
    # Example line: 5  -  MemoryUsage of job (MB)
    mem_pattern = re.compile(r"(\d+)\s+-\s+MemoryUsage of job \(MB\)")

    for log in log_files:
        max_in_file = 0
        with open(log, 'r') as f:
            for line in f:
                match = mem_pattern.search(line)
                if match:
                    val = int(match.group(1))
                    val = int(match.group(1))
                    if 0 < val < 128000:  # Ignore anything over 128GB as a glitch
                        if val > max_in_file:
                            max_in_file = val
        results.append((log, max_in_file))

    # Sort results by memory usage (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    if verbose:
        print(f"{'Log File':<30} | {'Max Memory (MB)':<15}")
        print("-" * 50)
    for log, mem in results:
        if verbose:
            print(f"{log:<30} | {mem:<15}")

    # Final Summary
    grand_max = results[0][1] if results else 0
    print("\n" + "="*50)
    print(f"GRAND MAXIMUM ACROSS ALL JOBS IN {job_logs_step.name}: {grand_max} MB")
    print(f"RECOMMENDED REQUEST_MEMORY:   {max(200, grand_max * 1.5)} MB")
    print("="*50)

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

def plot_hist_xyz(shards_dir: str | Path, n_nu: int, n_events: int = 1000):
    
    shards_dir = Path(shards_dir)
    shards = os.listdir(shards_dir)
    shuffle(shards)
    x_arr = []
    y_arr = []
    z_arr = []
    for i, file in enumerate(shards):
        if i >= n_events:
            break
        if 'vds' not in file:
            file_path = shards_dir / file
            with h5py.File(file_path, 'r') as f:
                vertices = f['vertices'][:] #type: ignore
                x_arr.extend(vertices[:, 0])
                y_arr.extend(vertices[:, 1])
                z_arr.extend(vertices[:, 2])

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(x_arr, bins=50, color='tab:orange')
    plt.title(f'X Distribution for {n_nu} neutrino simulations')
    plt.xlabel('X Position')
    plt.ylabel('Frequency')
    plt.grid(True,alpha=0.8)

    plt.subplot(1, 3, 2)
    plt.hist(y_arr, bins=50, color='tab:green')
    plt.title(f'Y Distribution for {n_nu} neutrino simulations')
    plt.xlabel('Y Position')
    plt.ylabel('Frequency')
    plt.grid(True,alpha=0.8)

    plt.subplot(1, 3, 3)
    plt.hist(z_arr, bins=50, color='tab:blue')
    plt.title(f'Z Distribution for {n_nu} neutrino simulations')
    plt.xlabel('Z Position')
    plt.ylabel('Frequency')
    plt.grid(True,alpha=0.8)

def check_for_duplicates(manifest_path):
    """
    Check if any two events (images + labels) are identical across all shards
    """
    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Get all shard files (train + val + test)
    all_shards = []
    for split in ['train', 'val', 'test']:
        all_shards.extend(manifest['splits'][split]['files'])
    
    print(f"Checking {len(all_shards)} shards for duplicates...")
    print(f"Total events: {manifest['metadata']['total_images_all_splits']}")
    
    # Strategy: Hash each event, look for collisions
    event_hashes = defaultdict(list)  # hash -> [(shard_file, index), ...]
    
    total_events = 0
    
    for shard_idx, shard_path in enumerate(tqdm(all_shards, desc="Hashing events")):
        try:
            with h5py.File(shard_path, 'r') as f:
                images = f['album'][:]
                labels = f['vertices'][:]
                
                num_events = images.shape[0]
                
                for event_idx in range(num_events):
                    # Combine image and label into one array for hashing
                    event_data = np.concatenate([
                        images[event_idx].flatten(),
                        labels[event_idx].flatten()
                    ])
                    
                    # Hash the event
                    event_hash = hashlib.sha256(event_data.tobytes()).hexdigest()
                    
                    # Store location
                    event_hashes[event_hash].append((shard_path, event_idx))
                    total_events += 1
                    
        except Exception as e:
            print(f"Error reading {shard_path}: {e}")
            continue
    
    print(f"\nProcessed {total_events} total events")
    print(f"Unique hashes: {len(event_hashes)}")
    
    # Find duplicates
    duplicates = {h: locs for h, locs in event_hashes.items() if len(locs) > 1}
    
    if duplicates:
        print(f"\n🚨 FOUND {len(duplicates)} DUPLICATE EVENTS!")
        print(f"   (events that appear multiple times)")
        
        # Show first few examples
        for i, (event_hash, locations) in enumerate(list(duplicates.items())[:5]):
            print(f"\nDuplicate {i+1}: Hash {event_hash[:16]}...")
            print(f"  Appears {len(locations)} times:")
            for shard, idx in locations[:10]:  # Show first 10 occurrences
                print(f"    - {shard} at index {idx}")
            if len(locations) > 10:
                print(f"    ... and {len(locations) - 10} more")
        
        # Verify one duplicate by actually comparing arrays
        print("\n🔍 Verifying first duplicate with full array comparison...")
        first_dup = list(duplicates.values())[0]
        loc1, loc2 = first_dup[0], first_dup[1]
        
        with h5py.File(loc1[0], 'r') as f1:
            img1 = f1['album'][loc1[1]]
            lbl1 = f1['vertices'][loc1[1]]
        
        with h5py.File(loc2[0], 'r') as f2:
            img2 = f2['album'][loc2[1]]
            lbl2 = f2['vertices'][loc2[1]]
        
        images_equal = np.array_equal(img1, img2)
        labels_equal = np.array_equal(lbl1, lbl2)
        
        print(f"  Images equal: {images_equal}")
        print(f"  Labels equal: {labels_equal}")
        
        if images_equal and labels_equal:
            print("  ✅ Confirmed: These are true duplicates!")
        else:
            print("  ⚠️ Hash collision (not true duplicates - very rare!)")
        
        return duplicates
        
    else:
        print("\n✅ NO DUPLICATES FOUND!")
        print("   All events are unique across the dataset.")
        return None
    
def get_channel_loc(detector_path, station_id, channel_id):
    with open(detector_path, 'r') as f:
        detector_config = json.load(f)

    station_vertex = None
    channel_vertex = None

    # 1. Find the station coordinates
    # Using .items() is much cleaner and faster for iterating through dictionaries
    for st_key, st_data in detector_config.get('stations', {}).items():
        # Note: Based on NuRadioMC's JSON structure, the key is usually 'station_id', not 'id'
        if st_data.get('station_id') == station_id:
            station_vertex = np.array([
                st_data.get('pos_easting', 0.0), 
                st_data.get('pos_northing', 0.0), 
                st_data.get('pos_altitude', 0.0)
            ])
            break # Stop looping once we find it
            
    # 2. Find the channel coordinates
    for ch_key, ch_data in detector_config.get('channels', {}).items():
        # We must match BOTH the station_id and the channel_id
        if ch_data.get('channel_id') == channel_id:
            channel_vertex = np.array([
                ch_data.get('ant_position_x', 0.0), 
                ch_data.get('ant_position_y', 0.0), 
                ch_data.get('ant_position_z', 0.0)
            ])
            break # Stop looping once we find it
            
    # 3. Handle cases where the station or channel isn't found
    if station_vertex is None:
        raise ValueError(f"Station ID {station_id} not found in {detector_path}")
    if channel_vertex is None:
        raise ValueError(f"Channel ID {channel_id} for Station {station_id} not found in {detector_path}")

    # Return the absolute position
    return station_vertex + channel_vertex