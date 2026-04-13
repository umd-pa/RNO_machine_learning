"""
Helper script to delete logs. Assumes this script is stored in job_logs directory. Could expand to also delete master.dag.* files, but not necessary

Author: Santiago Sued
"""
import os
import sys
import argparse
from tqdm import tqdm
from pathlib import Path

# Get the full path to current script file
script_path = Path(__file__).resolve()

parser = argparse.ArgumentParser(description='Log Deleter')
parser.add_argument('-d', '--delete_dagman_logs', action='store_true', help='If True, will also delete dagman logs in /jobs directory.')
args = parser.parse_args()

# Get parent directory of script
script_dir = script_path.parent

if 'job_logs' not in script_dir.name:
    print('ERROR: Script not in job_logs directory. Stopping...')
    sys.exit(1)

# Remove logs
for file in tqdm(os.listdir(script_dir), desc='Deleting Job Files'):
    if '.py' not in file:
        tqdm.write(f'Deleting {file}')
        os.remove(script_dir / file)

if args.delete_dagman_logs:

    # Get jobs directory
    jobs_dir = script_dir.parent

    if 'jobs' != jobs_dir.name:
        print('ERROR: Unexpected directory structure. Stopping...')
        sys.exit(1)

    # Remove dagman files
    for file in tqdm(os.listdir(jobs_dir), desc='Deleting Dagman Files'):
        if '.dag.' in file:
            tqdm.write(f'Deleting {file}')
            os.remove(jobs_dir / file)
