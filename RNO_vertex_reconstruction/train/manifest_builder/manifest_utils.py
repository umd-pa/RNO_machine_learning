import json
import h5py
import numpy as np
from tqdm import tqdm

def compare_2_manifests(old_manifest,new_manifest):
    for manifest_path in [old_manifest, new_manifest]:
        with open(manifest_path) as f:
            manifest = json.load(f)

            # sample a train shards
            train_shards = manifest['splits']['train']['files']
            labels = []
            for f_path in train_shards:
                with h5py.File(f_path, 'r', locking=False) as f:
                    labels.append(f['vertices'][:])

    labels = np.concatenate(labels)
    print(f"\n{manifest_path}")
    print(f"  x: mean={labels[:,0].mean():.1f}, std={labels[:,0].std():.1f}")
    print(f"  y: mean={labels[:,1].mean():.1f}, std={labels[:,1].std():.1f}")
    print(f"  z: mean={labels[:,2].mean():.1f}, std={labels[:,2].std():.1f}")

def print_hitcount_splits(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)

    train_shards = manifest['splits']['train']['files']
    test_shards  = manifest['splits']['test']['files']
    val_shards   = manifest['splits']['val']['files']

    def count_hits(shard_list):
        counts = [0, 0, 0, 0]  # index 0 = 1 station, index 3 = 4 stations
        for f_path in tqdm(shard_list, desc="Reading shards"):
            with h5py.File(f_path, 'r', locking=False) as f:
                hits = f['station_hit_count'][:].flatten() #type: ignore
                for i in range(1, 5):
                    counts[i-1] += int(np.sum(hits == i))
        return counts

    train_counts = count_hits(train_shards)
    test_counts  = count_hits(test_shards)
    val_counts = count_hits(val_shards)

    train_total = sum(train_counts)
    test_total  = sum(test_counts)
    val_total = sum(val_counts)

    print(f"\n{'':20} {'TRAIN':>15} {'TRAIN %':>15} {'TEST':>15} {'TEST %':>15} {'VAL':>15} {'VAL %':>15}")
    print('-' * 120)
    for i in range(4):
        train_pct = train_counts[i] / train_total
        test_pct  = test_counts[i] / test_total
        val_pct = val_counts[i] / val_total
        print(f"{i+1} station(s):{'':8} "
              f"{train_counts[i]:>15} "
              f"{train_pct:>15.10f} "
              f"{test_counts[i]:>15} "
              f"{test_pct:>15.10f} "
              f"{val_counts[i]:>15} "
              f"{val_pct:>15.10f}")
    print('-' * 120)
    print(f"{'Total':20} {train_total:>15} {'':>15} {test_total:>15} {'':>15} {val_total:>15}")
            