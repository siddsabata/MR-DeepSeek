#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A cis250063p
#SBATCH --gpus=h100-80:1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G 

# Load modules (adjust as needed for your cluster)
module load anaconda3
module load cuda

# Activate conda environment
conda activate eval_env

# Run the script
python eval/load_model.py

# Deactivate environment
conda deactivate 
