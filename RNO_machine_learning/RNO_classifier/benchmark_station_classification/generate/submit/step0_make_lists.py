#!/usr/bin/env python3
"""
Generate NuRadioMC input event lists.

Modes:
  veff  - loop over energies (decade increments), energy-dependent volumes and file counts
  nu    - spectrum-weighted generation, single volume config
  noise - noise events (energy not important)
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
from multiprocessing import Pool

# ── Thread throttling (set before any numpy/scipy imports) ────────────────────
os.environ.setdefault("OMP_NUM_THREADS",       "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",  "1")
os.environ.setdefault("MKL_NUM_THREADS",       "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",   "1")

from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder

# ── Top-level output directory ────────────────────────────────────────────────
BASE_DIR = "/data/condor_shared/users/ssued/RNO_machine_learning/RNO_machine_learning/RNO_classifier/generate/submit/condor/sim_data"

# ── Config for veff mode ──────────────────────────────────────────────────────
VEFF_LOG10_ENERGIES = [17, 18, 19, 20]
VEFF_CONFIG = {
    16:   dict(r_km=3000,  n_files=10,  n_events=1_000),
    # 16.5: dict(r_km=4000,  n_files=10,  n_events=1_000),
    17:   dict(r_km=4000,  n_files=100, n_events=5_000),
    # 17.5: dict(r_km=5500,  n_files=10,  n_events=5_000),
    18:   dict(r_km=5500,  n_files=100, n_events=5_000),
    # 18.5: dict(r_km=7000,  n_files=20,  n_events=5_000),
    19:   dict(r_km=7000,  n_files=100, n_events=5_000),
    # 19.5: dict(r_km=10000, n_files=20,  n_events=5_000),
    20:   dict(r_km=10000, n_files=100, n_events=5_000),
    # 20.5: dict(r_km=12000, n_files=30,  n_events=5_000),
    21:   dict(r_km=12000, n_files=30,  n_events=5_000),
}

# ── Config for nu (spectrum) mode ─────────────────────────────────────────────
NU_CONFIG = dict(
    r_km        = 3.0,
    n_files     = 100,
    n_events    = 10_000,
    log10_e_min = float(16),
    log10_e_max = float(21),
    spectrum    = "GZK-1",
)

# ── Config for noise mode ─────────────────────────────────────────────────────
NOISE_CONFIG = dict(
    r_km     = 0.1,
    n_files  = 100,
    n_events = 4_000,
)

# ── Shared geometry ───────────────────────────────────────────────────────────
Z_MIN = -2.7 * units.km
Z_MAX =  0.0 * units.km


def make_volume(r_km):
    return {
        'fiducial_zmin': Z_MIN,
        'fiducial_zmax': Z_MAX,
        'fiducial_rmin': 0 * units.km,
        'fiducial_rmax': r_km * units.km,
    }


# ── Per-file worker functions ─────────────────────────────────────────────────

def _gen_veff(args):
    log10e, i, fname = args
    cfg    = VEFF_CONFIG[log10e]
    energy = 10**float(log10e) * units.eV
    volume = make_volume(cfg['r_km'])
    generate_eventlist_cylinder(fname, cfg['n_events'], energy, energy, volume, seed=i)

def _gen_nu(args):
    fname, i = args
    cfg    = NU_CONFIG
    volume = make_volume(cfg['r_km'])
    e_min  = 10**cfg['log10_e_min'] * units.eV
    e_max  = 10**cfg['log10_e_max'] * units.eV
    generate_eventlist_cylinder(fname, cfg['n_events'], e_min, e_max, volume, seed=i)

def _gen_noise(args):
    fname, i = args
    cfg    = NOISE_CONFIG
    volume = make_volume(cfg['r_km'])
    e_min  = 1e15 * units.eV
    e_max  = 1e21 * units.eV
    generate_eventlist_cylinder(fname, cfg['n_events'], e_min, e_max, volume, seed=i)


# ── Mode runners ──────────────────────────────────────────────────────────────

def run_veff(n_jobs, start_index):
    step_dir = os.path.join(BASE_DIR, "veff", "step0")
    tasks = []
    for log10e in VEFF_LOG10_ENERGIES:
        cfg     = VEFF_CONFIG[log10e]
        out_dir = os.path.join(step_dir, f"1e{log10e}")
        os.makedirs(out_dir, exist_ok=True)
        for i in range(start_index, start_index + cfg['n_files']):
            fname = os.path.join(out_dir, f"1e{log10e}_f{i:06d}.hdf5")
            tasks.append((log10e, i, fname))
    with Pool(n_jobs) as pool:
        pool.map(_gen_veff, tasks)


def run_nu(n_jobs, start_index):
    cfg     = NU_CONFIG
    out_dir = os.path.join(BASE_DIR, "nu", "step0")
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(os.path.join(out_dir, f"nu_f{i:06d}.hdf5"), i)
             for i in range(start_index, start_index + cfg['n_files'])]
    with Pool(n_jobs) as pool:
        pool.map(_gen_nu, tasks)


def run_noise(n_jobs, start_index):
    cfg     = NOISE_CONFIG
    out_dir = os.path.join(BASE_DIR, "noise", "step0")
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(os.path.join(out_dir, f"noise_f{i:06d}.hdf5"), i)
             for i in range(start_index, start_index + cfg['n_files'])]
    with Pool(n_jobs) as pool:
        pool.map(_gen_noise, tasks)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate NuRadioMC input event lists.")
    p.add_argument('mode', choices=['veff', 'nu', 'noise'])
    p.add_argument('--jobs', type=int, default=1,
                   help="Number of parallel worker processes (default: 1)")
    p.add_argument('--start-index', type=int, default=0,
                   help="Starting file index, also used as seed base (default: 0)")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    runner = {'veff': run_veff, 'nu': run_nu, 'noise': run_noise}[args.mode]
    runner(args.jobs, args.start_index)
