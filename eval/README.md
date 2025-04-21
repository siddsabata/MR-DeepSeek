# Model Evaluation Pipeline

A simple, elegant evaluation pipeline for medical benchmarks without sglang dependency. This pipeline is designed to produce results directly comparable to HuatuoGPT-o1.

## Components

- `evaluator.py`: Main evaluation script
- `scorer.py`: Scoring logic for evaluating model responses
- `model_manager.py`: Model loading and management

## Usage

```bash
python evaluator.py \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --eval_file "/path/to/evaluation/data/eval_data.json" \
    --max_new_tokens 2000 \
    --batch_size 4 \
    --use_chat_template True
```

## Command Line Arguments

- `--model_name`: HuggingFace model identifier (default: "meta-llama/Llama-2-7b-chat-hf")
- `--eval_file`: Path to the evaluation data JSON file (required)
- `--max_new_tokens`: Maximum new tokens to generate (default: 2000)
- `--max_tokens`: Maximum number of tokens in prompt (-1 for no limit, default: -1)
- `--use_chat_template`: Whether to use the model's chat template (default: True)
- `--strict_prompt`: Use strict prompt format that requires "The answer is X." (action flag)
- `--task`: Task identifier for naming output files (default: 'huggingface')
- `--batch_size`: Batch size for processing (default: 4)
- `--hf_token`: HuggingFace token (optional, can be set in .env file)

## Example

```bash
# Evaluate HuatuoGPT model
python evaluator.py \
    --model_name "FreedomIntelligence/HuatuoGPT-o1-8B" \
    --eval_file "/ocean/projects/cis250063p/ssabata/HuatuoGPT-o1/evaluation/data/eval_data.json" \
    --max_new_tokens 2000 \
    --strict_prompt
```

## Output

The evaluation results are saved to two files:
1. `{model}_{eval_file}_{task}.json`: Raw model outputs
2. `result_{model}_{eval_file}_{task}.json`: Scored results by category

## Feature Comparison with HuatuoGPT-o1

This pipeline matches HuatuoGPT-o1's evaluation functionality:
- Uses identical prompt templates
- Matches the same scoring logic
- Produces compatible result formats
- Supports the same evaluation data format

The main difference is that this pipeline uses direct model inference instead of sglang server. 