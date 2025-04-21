#!/bin/bash
# Shell script to easily upload a model to Hugging Face Hub

# Default values
CONFIG_FILE="upload_config.json"
MODEL_PATH=""
HF_REPO_NAME=""
IS_LORA=false
BASE_MODEL=""
COMMIT_MESSAGE="Upload model to Hugging Face Hub"
USE_CONFIG=true

# Display usage information
function show_usage {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --config FILE            Path to JSON config file (default: scripts/upload_config.json)"
    echo "  --no-config              Don't use config file, use only command line arguments"
    echo "  --model_path PATH        Path to the trained model (required if no config)"
    echo "  --hf_repo_name NAME      Name for HF repository (required if no config, format: username/repo_name)"
    echo "  --hf_token TOKEN         Hugging Face access token (optional)"
    echo "  --is_lora                Specify if the model is a LoRA adapter"
    echo "  --base_model NAME        Base model name for LoRA adapters (required if --is_lora is set)"
    echo "  --commit_message MSG     Commit message for the upload"
    echo "  --help                   Display this help message"
    echo ""
    echo "Examples:"
    echo "  # Upload using config file:"
    echo "  $0"
    echo ""
    echo "  # Upload with custom config:"
    echo "  $0 --config my_config.json"
    echo ""
    echo "  # Upload without config:"
    echo "  $0 --no-config --model_path /path/to/model --hf_repo_name username/my-model"
    echo ""
    echo "  # Upload a LoRA adapter without config:"
    echo "  $0 --no-config --model_path /path/to/lora --hf_repo_name username/my-lora-adapter --is_lora --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --no-config)
            USE_CONFIG=false
            shift
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --hf_repo_name)
            HF_REPO_NAME="$2"
            shift 2
            ;;
        --hf_token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --is_lora)
            IS_LORA=true
            shift
            ;;
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --commit_message)
            COMMIT_MESSAGE="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Change to the directory containing this script
cd "$(dirname "$0")"

# Prepare command
if [ "$USE_CONFIG" = true ]; then
    CMD="python upload_to_hf.py --config \"$CONFIG_FILE\""
else
    # Check required arguments when not using config
    if [ -z "$MODEL_PATH" ] || [ -z "$HF_REPO_NAME" ]; then
        echo "Error: Missing required arguments (--model_path and/or --hf_repo_name)"
        show_usage
        exit 1
    fi

    # Check if base_model is provided for LoRA adapters
    if [ "$IS_LORA" = true ] && [ -z "$BASE_MODEL" ]; then
        echo "Error: --base_model is required when --is_lora is set"
        show_usage
        exit 1
    fi

    CMD="python upload_to_hf.py --model_path \"$MODEL_PATH\" --hf_repo_name \"$HF_REPO_NAME\""

    if [ "$IS_LORA" = true ]; then
        CMD="$CMD --is_lora --base_model \"$BASE_MODEL\""
    fi
fi

# Add common arguments regardless of config/no-config mode
if [ ! -z "$HF_TOKEN" ]; then
    CMD="$CMD --hf_token \"$HF_TOKEN\""
fi

if [ ! -z "$COMMIT_MESSAGE" ] && [ "$USE_CONFIG" = false ]; then
    CMD="$CMD --commit_message \"$COMMIT_MESSAGE\""
fi

# Execute the command
echo "Executing: $CMD"
eval $CMD 