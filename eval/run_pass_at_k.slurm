#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A cis250063p
#SBATCH --gpus=h100-80:1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=pass_at_k_%j.out
#SBATCH --error=pass_at_k_%j.err

# Record start time
START_TIME=$(date +%s)
echo "Pass@k evaluation started at $(date)"

# Load modules
module load anaconda3
module load cuda

# Activate environment
conda activate /ocean/projects/cis250063p/ssabata/conda_envs/py310_env

# Set parameters
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LORA_ADAPTER=""
CUSTOM_DIR=""
STRICT_PROMPT=true
EVAL_FILE="/ocean/projects/cis250063p/ssabata/MR-DeepSeek/eval/data/eval_data_subset.json"
BATCH_SIZE=8
K=3

echo "----------------------------------------"
echo "Running pass@k evaluation with:"
echo "Model: $MODEL_NAME"
echo "LoRA adapter: $LORA_ADAPTER"
echo "Custom directory: $CUSTOM_DIR"
echo "Strict prompt: $STRICT_PROMPT"
echo "Eval file: $EVAL_FILE"
echo "Batch size: $BATCH_SIZE"
echo "K: $K"
echo "----------------------------------------"

# Change to evaluation directory
cd /ocean/projects/cis250063p/ssabata/MR-DeepSeek/eval

# Run the pass_at_k script

# python pass_at_k.py --model_name "$MODEL_NAME" \
#     --lora_adapter "$LORA_ADAPTER" \
#     --custom_dir "$CUSTOM_DIR" \
#     --eval_file "$EVAL_FILE" \
#     --batch_size $BATCH_SIZE \
#     --strict_prompt \
#     --k $K

python pass_at_k.py --model_name "$MODEL_NAME" \
    --eval_file "$EVAL_FILE" \
    --batch_size $BATCH_SIZE \
    --strict_prompt \
    --k $K

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "----------------------------------------"
echo "Pass@k evaluation completed at $(date)"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "----------------------------------------"

# Deactivate environment
conda deactivate 