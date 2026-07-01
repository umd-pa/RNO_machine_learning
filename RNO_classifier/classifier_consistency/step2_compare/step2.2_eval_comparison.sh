# Get DIRECTORY of absolute path of current bash script
DIR_NAME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Get Path to eval script from current path
EVAL_SCRIPT_PATH="${DIR_NAME}/train_benchmark.py"
# Run eval script
echo "Tip: Run this script with '--help' to see all available options."
echo "------------------------------------------------------------"
# "$@" takes ALL arguments given to this shell script and passes them to Python
python "${EVAL_SCRIPT_PATH}" "$@"
