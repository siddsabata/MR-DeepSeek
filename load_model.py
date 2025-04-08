import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import numpy as np
import pandas as pd
from peft import PeftModel, PeftConfig
from huggingface_hub import snapshot_download

# Set your Hugging Face token 
HF_TOKEN = "YOUR_TOKEN"  # replace with your HF token

# Set project directory path for downloads
PROJECT_DIR = "/ocean/projects/cis250063p/ssabata/eval"    # replace with your proejct dir
CACHE_DIR = os.path.join(PROJECT_DIR, "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Download the base model
print("Downloading the base model...")
base_model_dir = os.path.join(PROJECT_DIR, "deepseek_model")
if not os.path.exists(base_model_dir):
    snapshot_download(
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        local_dir=base_model_dir,
        token=HF_TOKEN,
        cache_dir=CACHE_DIR
    )
    print(f"Model downloaded to {base_model_dir}")
else:
    print(f"Using existing model at {base_model_dir}")

# Download LoRA adapter
print("Downloading LoRA adapter...")
lora_adapter_dir = os.path.join(PROJECT_DIR, "lora_adapter")
if not os.path.exists(lora_adapter_dir):
    snapshot_download(
        repo_id="jakebentley2001/fine-tuned-llama-distilled-deepseek",
        local_dir=lora_adapter_dir,
        token=HF_TOKEN,
        cache_dir=CACHE_DIR
    )
    print(f"LoRA adapter downloaded to {lora_adapter_dir}")
else:
    print(f"Using existing LoRA adapter at {lora_adapter_dir}")

# Load the LoRA configuration
print("Loading LoRA config...")
config = PeftConfig.from_pretrained(lora_adapter_dir)

# Load the base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_dir,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

# Load the LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_adapter_dir)

# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model and tokenizer loaded successfully!")