import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from huggingface_hub import snapshot_download
import gc
import argparse
from dotenv import load_dotenv
import time

class ModelLoader:
    """Simple loader for a single AI model."""
    
    def __init__(self, hf_token=None, project_dir=None):
        self.hf_token = hf_token 
        self.project_dir = project_dir 
        self.models_dir = os.path.join(self.project_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Single model tracking
        self.model = None
        self.tokenizer = None
    
    def load_model(self, model_name, lora_adapter=None, custom_dir=None, use_cache=True):
        """
        Load a single model - with or without LoRA adapter.
        
        Args:
            model_name: Base model name or path
            lora_adapter: Optional LoRA adapter name or path
            custom_dir: Optional custom directory name
            use_cache: Whether to cache the model locally
        """
        is_lora = lora_adapter is not None
        
        # Set up directories
        dir_name = custom_dir or (f"{model_name.split('/')[-1]}_lora" if is_lora else model_name.split('/')[-1])
        model_dir = os.path.join(self.models_dir, dir_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Set up cache dir
        cache_dir = os.path.join(model_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download base model if needed
        if use_cache:
            base_dir = os.path.join(model_dir, "base_model")
            os.makedirs(base_dir, exist_ok=True)
            
            if not os.path.exists(os.path.join(base_dir, "config.json")):
                print(f"Downloading base model {model_name}...")
                snapshot_download(
                    repo_id=model_name,
                    local_dir=base_dir,
                    token=self.hf_token,
                    cache_dir=cache_dir
                )
                
            model_path = base_dir
            
            # Download adapter if needed
            if is_lora:
                adapter_dir = os.path.join(model_dir, "adapter")
                os.makedirs(adapter_dir, exist_ok=True)
                
                if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
                    print(f"Downloading LoRA adapter {lora_adapter}...")
                    snapshot_download(
                        repo_id=lora_adapter,
                        local_dir=adapter_dir,
                        token=self.hf_token,
                        cache_dir=cache_dir
                    )
                    
                adapter_path = adapter_dir
            else:
                adapter_path = None
        else:
            model_path = model_name
            adapter_path = lora_adapter
        
        # Load the tokenizer
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=self.hf_token
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the model
        print(f"Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=self.hf_token
        )
        
        # Apply LoRA adapter if needed
        if is_lora:
            print(f"Applying LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                self.model, 
                adapter_path,
                token=self.hf_token
            )
                
        # Clear cache to free memory
        print("Clearing cache to free memory...")
        gc.collect()
        torch.cuda.empty_cache()
        
        # Report memory usage after loading
        if torch.cuda.is_available():
            allocated_mem = torch.cuda.memory_allocated(0)
            print(f"GPU memory used by model: {allocated_mem/1e9:.2f}GB")
        
        print(f"Model loaded successfully!")
    
    def test_inference(self, prompt="What is your name?", max_tokens=50):
        """Run a quick inference test on the loaded model and time it."""
        if self.model is None or self.tokenizer is None:
            print("No model loaded. Call load_model() first.")
            return None
            
        print(f"\n----- Testing Model Inference -----")
        print(f"Test prompt: '{prompt}'")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response (with timing)
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
        inference_time = time.time() - start_time
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Report results
        print(f"\nModel response: '{response}'")
        print(f"Inference time: {inference_time:.4f} seconds")
        print("\n✓ Model inference test completed successfully!")
        
        return response
    
    def unload_model(self):
        """Unload the model to free up memory."""
        if self.model is None:
            print("No model loaded")
            return
            
        print(f"Unloading model...")
        
        # Clear from memory
        if hasattr(self.model, 'to'):
            self.model.to('cpu')
        
        # Delete references
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Model unloaded successfully")


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load and test a single model")
    parser.add_argument("--model", type=str, required=True, help="Base model name/path")
    parser.add_argument("--lora", type=str, help="LoRA adapter name/path (optional)")
    parser.add_argument("--dir", type=str, help="Custom directory name (optional)")
    parser.add_argument("--prompt", type=str, default="What are the symptoms of pneumonia?", 
                        help="Test prompt")
    args = parser.parse_args()
    load_dotenv()  # pip install python-dotenv first
    # Create loader
    loader = ModelLoader(hf_token=os.environ.get("HF_TOKEN"),
                         project_dir="/ocean/projects/cis250063p/ssabata/eval")
    print(f"Using device: {loader.device}")
    
    # Print header based on model type
    if args.lora:
        print(f"\n===== Testing LoRA Model: {args.model} with {args.lora} =====")
    else:
        print(f"\n===== Testing Base Model: {args.model} =====")
    
    try:
        # Load model
        loader.load_model(
            model_name=args.model,
            lora_adapter=args.lora,
            custom_dir=args.dir
        )
        
        # Test inference
        loader.test_inference(args.prompt)
        
        # Unload model
        loader.unload_model()
        
    except Exception as e:
        print(f"Error: {e}")
        
    print("\n===== Test completed =====")


# Example commands for the three model cases:
#
# 1. For DeepSeek base model:
# python eval/model_manager.py --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
#
# 2. For HuatuoGPT base model:
# python eval/model_manager.py --model "FreedomIntelligence/HuatuoGPT-o1-8B"
#
# 3. For DeepSeek with LoRA adapter:
# python eval/model_manager.py --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --lora "jakebentley2001/fine-tuned-llama-distilled-deepseek" --dir "jake-lora-ft"
#
# Additional options:
# --prompt "Your custom prompt here"  # To use a different test prompt
