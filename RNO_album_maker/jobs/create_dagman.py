import argparse
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
    parser = argparse.ArgumentParser(description="DAG Generator for NuRadioReco")
    parser.add_argument('--n_jobs', default=100, type=int, help='Number of jobs to run')
    parser.add_argument('--album_dir', type=str, required=True, 
                        help='The album directory where final shards are saved')
    parser.add_argument('--n_nu', required=True, help='Number of neutrinos to simulate')
    parser.add_argument('--time_window', 
                        type=float, 
                        default=8192,
                        help='Total time window in nanoseconds (default: 8192)')
    parser.add_argument('--time_bins', 
                        type=int, 
                        default=1024,
                        help='Number of time bins in the output image (default: 1024)')
    parser.add_argument('--bin_mode', 
                        type=str, 
                        default='MEAN', 
                        choices=['MEAN', 'MAX'],
                        help='Downsampling mode: MEAN or MAX (default: MEAN)')
    parser.add_argument('--noise_level',
                        type=float,
                        default=10.,
                        help='RMS of the noise to be added, in mV.')
    parser.add_argument('--clamp_v',
                        type=float,
                        default=1.,
                        help='Voltage to clamp. Any voltage above this value will be set to clamp_v')
    args = parser.parse_args()


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
    album_dir = os.path.abspath(args.album_dir)

    # C. Create Directories if they don't exist
    os.makedirs(album_dir, exist_ok=True)
    print(f"Directory ready: {album_dir}")

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

        for i in range(args.n_jobs):
            jid = f"{i:07d}"
            
            # --- DEFINE FILE PATHS ---
            # Intermediate files go to simulation_data
            f_det = os.path.join(simulation_data_dir, "RNO_four_stations.json")
            f_config = os.path.join(simulation_data_dir, "config.yaml")
            f_neutrinos = os.path.join(simulation_data_dir, f"neutrinos_{jid}.hdf5")
            f_simulated_events = os.path.join(simulation_data_dir, f"simulated_events_{jid}.nur")

            
            # Final output goes to the album dir
            f_shard = os.path.join(album_dir, f"shard_{jid}.hdf5")

            # --- JOB 1: Generate Neutrinos ---
            f.write(f"JOB NU_{jid} {sub1}\n")
            f.write(f'VARS NU_{jid} myscript="{script_1}" output_file="{f_neutrinos}" n_nu="{args.n_nu}" logdir="{log_dir}" job_id="{jid}"\n')

            # --- JOB 2: Simulate Detector ---
            f.write(f"JOB SIM_{jid} {sub2}\n")
            f.write(f'VARS SIM_{jid} myscript="{script_2}" input_file="{f_neutrinos}" output_file="{f_simulated_events}" f_det="{f_det}" f_config="{f_config}" noise_lvl="{args.noise_level}" logdir="{log_dir}" job_id="{jid}"\n')

            # --- JOB 3: Image Shards ---
            f.write(f"JOB SRD_{jid} {sub3}\n")
            f.write(f'VARS SRD_{jid} myscript="{script_3}" input_file="{f_simulated_events}" output_file="{f_shard}" f_det="{f_det}" time_window="{args.time_window}" time_bins="{args.time_bins}" bin_mode="{args.bin_mode}" noise_lvl="{args.noise_level}" clamp_v="{args.clamp_v}" logdir="{log_dir}" job_id="{jid}"\n')

            # --- DEPENDENCIES ---
            f.write(f"PARENT NU_{jid} CHILD SIM_{jid}\n")
            f.write(f"PARENT SIM_{jid} CHILD SRD_{jid}\n\n")

        f.write(f"FINAL CLEANUP {subF}\n")
        f.write(f'VARS CLEANUP myscript="{script_F}" target_dir="{simulation_data_dir}" logdir="{log_dir}"')

    print(f"Done! Submit with: condor_submit_dag {dag_filename}")

if __name__ == "__main__":
    main()