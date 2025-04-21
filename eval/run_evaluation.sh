#!/bin/bash

# This script runs the evaluation pipeline for specified models

# Check if a model name was provided
if [ -z "$1" ]; then
  echo "Usage: ./run_evaluation.sh <model_name> [strict_prompt]"
  echo "Example: ./run_evaluation.sh deepseek-ai/DeepSeek-R1-Distill-Llama-8B true"
  exit 1
fi

MODEL_NAME=$1
STRICT_PROMPT=${2:-false}
EVAL_FILE="/ocean/projects/cis250063p/ssabata/HuatuoGPT-o1/evaluation/data/eval_data.json"
BATCH_SIZE=4
MAX_NEW_TOKENS=2000

echo "==== Starting Evaluation ===="
echo "Model: $MODEL_NAME"
echo "Eval File: $EVAL_FILE"
echo "Strict Prompt: $STRICT_PROMPT"
echo "Batch Size: $BATCH_SIZE"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "=========================="

# Build command
CMD="python evaluator.py --model_name \"$MODEL_NAME\" --eval_file \"$EVAL_FILE\" --max_new_tokens $MAX_NEW_TOKENS --batch_size $BATCH_SIZE"

# Add strict prompt if needed
if [ "$STRICT_PROMPT" = "true" ]; then
  CMD="$CMD --strict_prompt"
fi

# Run the command
echo "Running: $CMD"
eval $CMD

echo "Evaluation completed!" 