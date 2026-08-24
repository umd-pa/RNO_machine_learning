#!/usr/bin/env python3
"""
Generate sbatch job list for NuRadioMC simulations.
Walks step0 directories and writes one sbatch line per input file.

Modes: veff, nu, noise
"""

import argparse
import os
import glob
import math

BASE_DIR     = "/home/baclark/scratch/career/sims"
SLURM_SCRIPT = "NuRadioMC_sims_slurm.sh"


def get_output_path(input_path, sim_type):
    step0    = os.path.join(BASE_DIR, sim_type, "step0")
    step1    = os.path.join(BASE_DIR, sim_type, "step1")
    rel      = os.path.relpath(input_path, step0)
    basename = os.path.basename(rel)
    subdir   = os.path.dirname(rel)
    stem     = os.path.splitext(basename)[0]
    out_dir  = os.path.join(step1, subdir)
    out_hdf5 = os.path.join(out_dir, f"output_{stem}.hdf5")
    out_nur  = os.path.join(out_dir, f"output_{stem}.nur")
    return out_hdf5, out_nur


def collect_inputs(sim_type, energy=None):
    step0 = os.path.join(BASE_DIR, sim_type, "step0")
    if energy is not None:
        pattern = os.path.join(step0, f"1e{energy}", "*.hdf5")
    else:
        pattern = os.path.join(step0, "**", "*.hdf5")
    return sorted(glob.glob(pattern, recursive=True))


def build_commands(sim_type, energy=None):
    inputs = collect_inputs(sim_type, energy)
    if not inputs:
        print(f"WARNING: No input files found for mode '{sim_type}' in {BASE_DIR}/{sim_type}/step0")
        return []
    commands = []
    for inp in inputs:
        out_hdf5, out_nur = get_output_path(inp, sim_type)
        cmd = f"sbatch {SLURM_SCRIPT} {sim_type} {inp} {out_hdf5} {out_nur}"
        commands.append(cmd)
    return commands


def write_jobs(commands, output_file, split):
    total    = len(commands)
    split    = max(1, split)
    per_file = math.ceil(total / split)

    for i in range(split):
        subset = commands[i * per_file:(i + 1) * per_file]
        if split == 1:
            path = output_file
        else:
            base, ext = os.path.splitext(output_file)
            path = f"{base}_part{i+1}{ext}"
        with open(path, "w") as f:
            f.write("#!/bin/bash\n# Generated sbatch commands for NuRadioMC simulations\n\n")
            for cmd in subset:
                f.write(cmd + "\n")
        os.chmod(path, 0o755)
        print(f"Created: {path} ({len(subset)} jobs)")


def parse_args():
    p = argparse.ArgumentParser(description="Generate NuRadioMC sbatch job list.")
    p.add_argument("mode", choices=["veff", "nu", "noise"])
    p.add_argument("-o", "--output", required=True, help="Output shell script path")
    p.add_argument("--split", type=int, default=1,
                   help="Split commands across N files (for large job sets)")
    p.add_argument("--energy", type=str, default=None,
                   help="veff only: energy label to filter on, e.g. 19 or 19.5")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.energy is not None and args.mode != "veff":
        print("WARNING: --energy is ignored for non-veff modes")
    energy   = args.energy if args.mode == "veff" else None
    commands = build_commands(args.mode, energy)
    if commands:
        write_jobs(commands, args.output, args.split)
        print(f"Total jobs: {len(commands)}")
