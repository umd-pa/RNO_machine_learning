#!/bin/bash

log_dir="/home/baclark/scratch/logs/log"
err_dir="/home/baclark/scratch/logs/err"
logfile="/home/baclark/scratch/logs/deleted_logs.txt"

# Ensure logfile exists
touch "$logfile"

for out_file in "$log_dir"/NuRadioMC_sims_*.out; do
    [[ -e "$out_file" ]] || continue

    jobid=$(basename "$out_file" | sed -E 's/NuRadioMC_sims_([0-9]+)\.out/\1/')
    err_file="$err_dir/NuRadioMC_sims_${jobid}.err"

    if [[ ! -f "$err_file" ]]; then
        echo "Skipping job $jobid: .err file not found"
        continue
    fi

    # Query SLURM for job status
    status=$(sacct -j "$jobid" --format=State --noheader | awk '{print $1}' | head -n 1)

    if [[ "$status" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT)$ ]]; then
        echo "Deleting logs for job $jobid with status $status"
        echo "[$(date)] Deleted: $out_file and $err_file (status: $status)" >> "$logfile"
        rm -f "$out_file" "$err_file"
    else
        echo "Job $jobid is still running or unknown (status: $status)"
    fi
done