#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A cis250063p
#SBATCH --gpus=h100-80:1     # Request 1 GPU (adjust as needed)
#SBATCH --time=24:00:00     # Request 3 hours of runtime (adjust as needed)
#SBATCH --ntasks=1         # Run a single task
#SBATCH --cpus-per-task=4  # Request 8 CPU cores per task (adjust as needed)
#SBATCH --mem=16G 


################################################################################
# train_script.sbatch
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


export HF_HOME=/ocean/projects/cis250063p/jbentley/dl_project/hf_cache
export TRANSFORMERS_CACHE=/ocean/projects/cis250063p/jbentley/dl_project/hf_cache

# Create the cache directory if it doesn't exist.
mkdir -p $HF_HOME

# 5. Run the Python script to download the dataset
#    We pass --outdir to specify the save location
python train.py

# 6. Print a message
echo "Training job is complete."
