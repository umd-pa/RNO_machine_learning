"""
DAG Generator for NuRadioReco Simulation Pipeline
--------------------------------------------------
Generates a HTCondor DAG file (master.dag) that orchestrates a
three-step neutrino simulation pipeline across N parallel jobs:

    Step 1 (NU):  Generate neutrino events
    Step 2 (SIM): Run detector simulation
    Step 3 (SRD): Convert simulated events to HDF5 image shards
    Step F:       Clean up intermediate files after all jobs complete

All simulation parameters are read from a YAML config file.
A copy of the config is saved alongside the output shards for reproducibility.

Usage:
    python create_dagman.py --config create_dagman_config.yaml --album_dir /data/.../shards

Then submit with:
    condor_submit_dag master.dag

Author: Santiago Sued
"""

import shutil
import argparse
import yaml
import os

def get_abs_path(rel_path):
    """
    Converts a relative path to an absolute path based on 
    the location of THIS script (create_dagman.py).
    """
    # This gets the folder where create_dagman.py lives (e.g., .../jobs/)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, rel_path))

def main():
    path_to_config = get_abs_path('create_dagman_config.yaml')
    parser = argparse.ArgumentParser(description="DAG Generator for NuRadioReco")
    parser.add_argument('--config', type=str, default=f'{path_to_config}', help='Path to the YAML config file')
    parser.add_argument('--album_dir', type=str, required=True, 
                        help='The album directory where final shards are saved')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Define Constants from config
    N_JOBS = config['simulation']['n_jobs']
    N_NU = config['simulation']['n_nu']

    TIME_WINDOW = config['signal']['time_window']
    TIME_BINS = config['signal']['time_bins']
    BIN_MODE = config['signal']['bin_mode']
    NOISE_LVL = config['signal']['noise_level']
    CLAMP_V = config['signal']['clamp_v']

    STORE_NURS = config['cleanup']['store_nurs']

    # ---------------------------------------------------------
    # 1. SETUP PATHS
    # ---------------------------------------------------------
    
    # A. Locate Code & Staging Area (Relative to 'jobs/')
    # We look up one level (..) to get out of 'jobs', then into 'simulation'
    simulation_dir = get_abs_path('../simulation/simulation_steps')
    simulation_data_dir = get_abs_path('../simulation/simulation_data')

    # Script Paths
    script_1 = os.path.join(simulation_dir, '1_generate_neutrinos.py')
    script_2 = os.path.join(simulation_dir, '2_run_simulation.py')
    script_3 = os.path.join(simulation_dir, '3_generate_hdf5_shards.py')
    script_F = os.path.join(simulation_dir, 'F_cleanup.py')

    # Log path
    log_dir = get_abs_path('job_logs')

    # B. Locate Final Destination
    # We trust the user provided a valid path for the albums
    ALBUM_DIR = os.path.abspath(args.album_dir)

    # C. Create Directories if they don't exist
    os.makedirs(ALBUM_DIR, exist_ok=True)
    print(f"Directory ready: {ALBUM_DIR}")
    shutil.copy2(args.config, os.path.join(ALBUM_DIR, 'used_config.yaml'))  # Save a copy of the config in the album dir for posterity

    # D. Locate Submit Templates
    sub_dir = get_abs_path('submissions')
    sub1 = os.path.join(sub_dir, 'step1.sub')
    sub2 = os.path.join(sub_dir, 'step2.sub')
    sub3 = os.path.join(sub_dir, 'step3.sub')
    subF = os.path.join(sub_dir, 'stepF.sub')

    # ---------------------------------------------------------
    # 2. GENERATE DAG FILE
    # ---------------------------------------------------------
    dag_filename = get_abs_path('master.dag')
    print(f"Generating DAG: {dag_filename}")

    with open(dag_filename, 'w') as f:
        f.write("# HTCondor DAG - Simulation Shards Pipeline\n")

        for i in range(N_JOBS):
            jid = f"{i:07d}"
            
            # --- DEFINE FILE PATHS ---
            # Intermediate files go to simulation_data
            f_det = os.path.join(simulation_data_dir, "RNO_four_stations.json")
            f_config = os.path.join(simulation_data_dir, "config.yaml")
            f_neutrinos = os.path.join(simulation_data_dir, f"neutrinos_{jid}.hdf5")
            f_simulated_events = os.path.join(simulation_data_dir, f"simulated_events_{jid}.nur")

            
            # Final output goes to the album dir
            f_shard = os.path.join(ALBUM_DIR, f"shard_{jid}.hdf5")

            # --- JOB 1: Generate Neutrinos ---
            f.write(f"JOB NU_{jid} {sub1}\n")
            f.write(f'VARS NU_{jid} myscript="{script_1}" output_file="{f_neutrinos}" n_nu="{N_NU}" logdir="{log_dir}" job_id="{jid}"\n')

            # --- JOB 2: Simulate Detector ---
            f.write(f"JOB SIM_{jid} {sub2}\n")
            f.write(f'VARS SIM_{jid} myscript="{script_2}" input_file="{f_neutrinos}" output_file="{f_simulated_events}" f_det="{f_det}" f_config="{f_config}" noise_lvl="{NOISE_LVL}" logdir="{log_dir}" job_id="{jid}"\n')

            # --- JOB 3: Image Shards ---
            f.write(f"JOB SRD_{jid} {sub3}\n")
            f.write(f'VARS SRD_{jid} myscript="{script_3}" input_file="{f_simulated_events}" output_file="{f_shard}" f_det="{f_det}" time_window="{TIME_WINDOW}" time_bins="{TIME_BINS}" bin_mode="{BIN_MODE}" noise_lvl="{NOISE_LVL}" clamp_v="{CLAMP_V}" logdir="{log_dir}" job_id="{jid}"\n')

            # --- DEPENDENCIES ---
            f.write(f"PARENT NU_{jid} CHILD SIM_{jid}\n")
            f.write(f"PARENT SIM_{jid} CHILD SRD_{jid}\n\n")

        f.write(f"FINAL CLEANUP {subF}\n")
        if STORE_NURS:
            print(f'STORE_NURS = {STORE_NURS}, saving .nur files to {STORE_NURS}')
            f.write(f'VARS CLEANUP myscript="{script_F}" target_dir="{simulation_data_dir}" store_nurs="{STORE_NURS}" logdir="{log_dir}"')
        else:
            print(f'STORE_NURS = {STORE_NURS}, saving to latest_sim_nurs by default')
            f.write(f'VARS CLEANUP myscript="{script_F}" target_dir="{simulation_data_dir}" store_nurs=" " logdir="{log_dir}"')

    print(f"Done! Submit with: condor_submit_dag -F {dag_filename}")

if __name__ == "__main__":
    main()