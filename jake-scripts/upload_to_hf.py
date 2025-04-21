#!/usr/bin/env python
# Script to upload a trained model to Hugging Face Hub

import os
import json
import argparse
from huggingface_hub import login, HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    
    # Add argument to use config file
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to JSON configuration file with default settings"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--hf_repo_name", 
        type=str, 
        default=None,
        help="Name for the Hugging Face repository (format: username/repo_name)"
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default=None,
        help="Hugging Face access token. If not provided, will use the cached token or prompt for login."
    )
    parser.add_argument(
        "--is_lora", 
        action="store_true",
        help="Whether the model is a LoRA adapter"
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        default=None,
        help="Base model name for LoRA adapters (required if --is_lora is set)"
    )
    parser.add_argument(
        "--commit_message", 
        type=str, 
        default="Upload model",
        help="Commit message for the upload"
    )
    
    args = parser.parse_args()
    
    # If config file is provided, load it and use as defaults
    if args.config:
        if not os.path.exists(args.config):
            print(f"Warning: Config file {args.config} not found. Using command line arguments only.")
        else:
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
                    
                # Set defaults from config file if not specified in command line
                if args.model_path is None and 'model_path' in config:
                    args.model_path = config['model_path']
                    
                if args.hf_repo_name is None and 'hf_repo_name' in config:
                    args.hf_repo_name = config['hf_repo_name']
                    
                if args.hf_token is None and 'hf_token' in config:
                    args.hf_token = config['hf_token']
                    
                if not args.is_lora and 'is_lora' in config and config['is_lora']:
                    args.is_lora = True
                    
                if args.base_model is None and 'base_model' in config:
                    args.base_model = config['base_model']
                    
                if args.commit_message == "Upload model" and 'commit_message' in config:
                    args.commit_message = config['commit_message']
                    
                print(f"Loaded defaults from config file: {args.config}")
            except json.JSONDecodeError:
                print(f"Error: Config file {args.config} is not valid JSON. Using command line arguments only.")
    
    # Check required arguments
    if args.model_path is None:
        parser.error("--model_path is required")
        
    if args.hf_repo_name is None:
        parser.error("--hf_repo_name is required")
        
    if args.is_lora and args.base_model is None:
        parser.error("--base_model is required when --is_lora is set")
    
    return args


def upload_model(model_path, hf_repo_name, token=None, is_lora=False, 
                 base_model=None, commit_message="Upload model"):
    """Upload a model to the Hugging Face Hub.
    
    Args:
        model_path: Path to the trained model
        hf_repo_name: Name for the Hugging Face repository
        token: Hugging Face access token
        is_lora: Whether the model is a LoRA adapter
        base_model: Base model name for LoRA adapters
        commit_message: Commit message for the upload
    """
    # Login to Hugging Face
    if token:
        login(token=token)
    else:
        print("No token provided. Using cached credentials or will prompt for login.")
    
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=hf_repo_name,
            exist_ok=True,
            private=False
        )
        print(f"Repository {hf_repo_name} created or already exists.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Upload the model
    try:
        # If it's a LoRA adapter
        if is_lora:
            if not base_model:
                raise ValueError("Base model name is required for LoRA adapters")
                
            print(f"Uploading LoRA adapter from {model_path} to {hf_repo_name}...")
            
            # For LoRA adapters, we directly upload the adapter files
            api.upload_folder(
                folder_path=model_path,
                repo_id=hf_repo_name,
                commit_message=commit_message
            )
            
            # Create a model card if it doesn't exist
            readme_content = f"""
# LoRA Adapter for {base_model}

This repository contains a LoRA adapter for fine-tuning the {base_model} model.

## Base Model
The adapter is designed to be used with [{base_model}](https://huggingface.co/{base_model}).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load the LoRA configuration
config = PeftConfig.from_pretrained("{hf_repo_name}")

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the LoRA adapter
model = PeftModel.from_pretrained(model, "{hf_repo_name}")

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
            with open("README.md", "w") as f:
                f.write(readme_content)
            
            api.upload_file(
                path_or_fileobj="README.md",
                path_in_repo="README.md",
                repo_id=hf_repo_name,
                commit_message="Add model card"
            )
            
            os.remove("README.md")  # Clean up
            
        else:
            # For full models, we load and push with the transformers API
            print(f"Uploading full model from {model_path} to {hf_repo_name}...")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            model.push_to_hub(hf_repo_name, commit_message=commit_message)
            tokenizer.push_to_hub(hf_repo_name, commit_message=commit_message)
        
        print(f"Model successfully uploaded to https://huggingface.co/{hf_repo_name}")
        
    except Exception as e:
        print(f"Error uploading model: {e}")


def main():
    """Main function to upload model to Hugging Face Hub."""
    args = parse_args()
    
    upload_model(
        model_path=args.model_path,
        hf_repo_name=args.hf_repo_name,
        token=args.hf_token,
        is_lora=args.is_lora,
        base_model=args.base_model,
        commit_message=args.commit_message
    )


if __name__ == "__main__":
    main() 