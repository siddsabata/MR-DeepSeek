#SBATCH -A cis250063p
#SBATCH --gpus=h100-80:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=eval_pipeline_%j.out
#SBATCH --error=eval_pipeline_%j.err

# Record start time
START_TIME=$(date +%s)
echo "Evaluation started at $(date)"

# Load modules
module load anaconda3
module load cuda

# Activate environment
conda activate /ocean/projects/cis250063p/ssabata/conda_envs/py310_env

# Set fixed parameters
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
LORA_ADAPTER="jakebentley2001/fine-tuned-llama-distilled-deepseek-not-overfit"
CUSTOM_DIR="ft2"
STRICT_PROMPT=true
EVAL_FILE="/ocean/projects/cis250063p/ssabata/MR-DeepSeek/eval/data/eval_data.json"
BATCH_SIZE=8
HF_TOKEN="YOUR TOKEN HERE"

echo "----------------------------------------"
echo "Running evaluation with:"
echo "Model: $MODEL_NAME"
echo "Strict prompt: $STRICT_PROMPT"
echo "Eval file: $EVAL_FILE"
echo "Batch size: $BATCH_SIZE"
echo "----------------------------------------"

# Run the evaluation script
cd /ocean/projects/cis250063p/ssabata/MR-DeepSeek/eval

# Build the command with LoRA support
# python evaluator.py --model_name "$MODEL_NAME" --lora_adapter "$LORA_ADAPTER" --custom_dir "$CUSTOM_DIR" --eval_file "$EVAL_FILE" --batch_size $BATCH_SIZE --strict_prompt
python evaluator.py --model_name "$MODEL_NAME" --eval_file "$EVAL_FILE" --batch_size $BATCH_SIZE --hf_token $HF_TOKEN --strict_prompt

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo "----------------------------------------"
echo "Evaluation completed at $(date)"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "----------------------------------------"

# Deactivate environment
conda deactivate 