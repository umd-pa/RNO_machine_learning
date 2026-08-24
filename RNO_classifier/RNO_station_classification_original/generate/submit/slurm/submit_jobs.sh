#!/bin/bash

source /cvmfs/rnog.opensciencegrid.org/software/trunk/setup.sh
source /home/baclark/career/software/venv/bin/activate

python /home/baclark/career/rnog_plots/2026-career/sims/generate/submit/slurm/submit_jobs.py
