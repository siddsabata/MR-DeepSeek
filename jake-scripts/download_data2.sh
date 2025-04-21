#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A cis250063p
#SBATCH --gpus=v100-32:4    # Request 4 GPU (adjust as needed)
#SBATCH --time=1:00:00     # Request 1 hour of runtime (adjust as needed)
#SBATCH --ntasks=1         # Run a single task
#SBATCH --cpus-per-task=4  # Request 4 CPU cores per task (adjust as needed)
#SBATCH --mem=16G 


################################################################################
# download_data.sbatch
#
# This script:
#  1) Loads/activates your previously created conda environment
#  2) Runs 'download_data.py' to fetch a Hugging Face dataset
#  3) Saves that dataset to /ocean/projects/YOUR_GROUP/YOUR_USERNAME/data
################################################################################

# 1. Load Anaconda (on Bridges-2)
module load anaconda3

# 2. Activate the environment you created previously.
#    This might be a path like /ocean/projects/YOUR_GROUP/YOUR_USERNAME/my_env
conda activate /ocean/projects/cis250063p/jbentley/my_env

# 3. Set a variable to define where you want the data saved
#    e.g., a 'data' folder in your project space
OUTPUT_DIR="/ocean/projects/cis250063p/jbentley/dl_project/data"

# 4. Make sure the directory exists (optional; the Python script also ensures existence)
mkdir -p "$OUTPUT_DIR"

# 5. Run the Python script to download the dataset
#    We pass --outdir to specify the save location
python download_data.py --outdir "$OUTPUT_DIR"

# 6. Print a message
echo "Download job is complete. Check $OUTPUT_DIR for the dataset."
