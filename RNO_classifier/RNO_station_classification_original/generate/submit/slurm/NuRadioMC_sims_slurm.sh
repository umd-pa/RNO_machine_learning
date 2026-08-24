#!/bin/bash

#SBATCH --job-name=NuRadioMC_sims
#SBATCH --output=/home/baclark/scratch/logs/log/%x_%j.out
#SBATCH --error=/home/baclark/scratch/logs/err/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --tmp=3G
#SBATCH --partition=scavenger

# ------------------------------------------------------------------
# SLURM script for running NuRadioMC simulations
# ------------------------------------------------------------------

source /cvmfs/rnog.opensciencegrid.org/software/trunk/setup.sh
source /home/baclark/career/software/venv/bin/activate

SIM_SCRIPT="/home/baclark/career/rnog_plots/2026-career/sims/generate/step2_run_sims.py"
STATION_JSON="/home/baclark/career/rnog_plots/2026-careers/sims/generate/station.json"
CONFIG_SIGNAL="/home/baclark/career/rnog_plots/2026-career/sims/generate/config_signal.yaml"
CONFIG_NOISE="/home/baclark/career/rnog_plots/2026-career/sims/generate/config_noise.yaml"
RECOVERY_FILE="/home/baclark/career/rnog_plots/2026-career/sims/generate/submit/slurm/recovery.txt"

_cleanup_done=0

cleanup_on_fail() {
    if [ $_cleanup_done -eq 1 ]; then
        return
    fi
    _cleanup_done=1
    echo "sbatch NuRadioMC_sims_slurm.sh $sim_type $input_file $output_hdf5 $output_nur" >> "$RECOVERY_FILE"
}

trap cleanup_on_fail ERR
trap cleanup_on_fail SIGTERM

# Parse arguments
sim_type=$1
input_file=$2
output_hdf5=$3
output_nur=$4

if [ -z "$sim_type" ] || [ -z "$input_file" ] || [ -z "$output_hdf5" ] || [ -z "$output_nur" ]; then
    echo "ERROR: Missing arguments."
    echo "Usage: sbatch NuRadioMC_sims_slurm.sh <sim_type> <input_file> <output_hdf5> <output_nur>"
    exit 1
fi

if [ ! -f "$input_file" ]; then
    echo "ERROR: Input file not found: $input_file"
    exit 1
fi

# Set threshold, channel, and config based on sim_type
if [ "$sim_type" == "noise" ]; then
    THRESHOLD=0.01
    TRIG_CHAN=0
    CONFIG="$CONFIG_NOISE"
elif [ "$sim_type" == "nu" ]; then
    THRESHOLD=0.5
    TRIG_CHAN=40
    CONFIG="$CONFIG_SIGNAL"
elif [ "$sim_type" == "veff" ]; then
    THRESHOLD=1.0
    TRIG_CHAN=40
    CONFIG="$CONFIG_SIGNAL"
else
    echo "ERROR: Unknown sim_type '$sim_type'. Must be one of: veff, nu, noise."
    exit 1
fi

# Make final output directory and pre-flight write check
out_dir=$(dirname "$output_hdf5")
mkdir -p "$out_dir"
test_path="${out_dir}/.touchtest_${SLURM_JOB_ID}"
touch "$test_path" || { echo "ERROR: Cannot write to $out_dir"; exit 1; }
rm "$test_path"

# Local scratch paths
LOCAL_HDF5="${TMPDIR}/$(basename "$output_hdf5")"
LOCAL_NUR="${TMPDIR}/$(basename "$output_nur")"

echo "Running NuRadioMC:"
echo "  Sim type   : $sim_type"
echo "  Input      : $input_file"
echo "  Output hdf5: $output_hdf5"
echo "  Output nur : $output_nur"
echo "  Config     : $CONFIG"
echo "  Threshold  : $THRESHOLD"
echo "  Trig chan  : $TRIG_CHAN"

python "$SIM_SCRIPT" \
    "$input_file" \
    "$STATION_JSON" \
    "$CONFIG" \
    "$LOCAL_HDF5" \
    "$LOCAL_NUR" \
    --threshold "$THRESHOLD" \
    --trig_chan "$TRIG_CHAN"

status=$?

if [ $status -eq 0 ]; then
    cp "$LOCAL_HDF5" "$output_hdf5" || { echo "ERROR: Failed to copy $LOCAL_HDF5 to $output_hdf5"; exit 1; }
    cp "$LOCAL_NUR"  "$output_nur"  || { echo "ERROR: Failed to copy $LOCAL_NUR to $output_nur"; exit 1; }
    echo "NuRadioMC completed successfully."
    echo "  Output: $output_hdf5"
    echo "  Output: $output_nur"
    trap - ERR
    trap - SIGTERM
else
    echo "NuRadioMC failed with status $status — triggering recovery."
    exit $status
fi
