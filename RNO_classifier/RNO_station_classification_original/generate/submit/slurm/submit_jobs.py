#!/usr/bin/env python3
import subprocess
import fcntl
import time
from pathlib import Path
from datetime import datetime

# CONFIGURATION
JOB_LIMIT = 4500
JOB_LIST_FILE = Path("/home/baclark/ARA/FiveStation/sims/submit/slurm/to_run.sh")   # <-- UPDATE THIS
LOCK_FILE = Path("/tmp/baclark_submit_jobs.lock")

def get_current_job_count():
    try:
        result = subprocess.run(
            ["squeue", "-u", subprocess.getoutput("whoami"), "-h"],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().split("\n")
        return len([line for line in lines if line.strip()])
    except subprocess.CalledProcessError:
        return 0

def submit_jobs(n, job_list_file):
    with job_list_file.open("r") as f:
        lines = f.readlines()

    to_submit = lines[:n]
    remaining = lines[n:]

    # Rewrite the job list without submitted lines
    with job_list_file.open("w") as f:
        f.writelines(remaining)

    # Submit each job line
    for cmd in to_submit:
        cmd = cmd.strip()
        if cmd:
            print(f"[INFO] Submitting: {cmd}")
            subprocess.run(cmd, shell=True)

def main():
    with LOCK_FILE.open("w") as lock_handle:
        try:
            # Acquire exclusive lock, wait if needed
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[WARN] Another instance is already running. Exiting.")
            return

        current_jobs = get_current_job_count()
        available_slots = JOB_LIMIT - current_jobs

        if available_slots <= 0:
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Job limit reached: {current_jobs} jobs running/submitted.")
            return

        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] {current_jobs} jobs in queue. Submitting up to {available_slots} jobs.")
        submit_jobs(available_slots, JOB_LIST_FILE)

        fcntl.flock(lock_handle, fcntl.LOCK_UN)

if __name__ == "__main__":
    main()