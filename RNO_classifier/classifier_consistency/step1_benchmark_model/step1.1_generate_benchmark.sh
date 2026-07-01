# Get DIRECTORY of absolute path of current bash script
DIR_NAME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Get Path to dagman script from current path
DAGMAN_SCRIPT_PATH="${DIR_NAME}/../generate_dataset/create_dagman.py"
# Call dagman script
python "${DAGMAN_SCRIPT_PATH}"
