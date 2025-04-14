import torch
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
import random

# Set paths
PROJECT_DIR = "/ocean/projects/cis250063p/ssabata/eval"
BASE_MODEL_DIR = os.path.join(PROJECT_DIR, "deepseek_model")
LORA_ADAPTER_DIR = os.path.join(PROJECT_DIR, "lora_adapter")
RESULTS_DIR = os.path.join(PROJECT_DIR, "benchmark_results")
CACHE_DIR = os.path.join(PROJECT_DIR, "datasets_cache")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Function to load the model from the eval/load_model.py script
def load_finetuned_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Import here to avoid GPU memory issues until needed
    from transformers import AutoModelForCausalLM
    
    print("Loading LoRA config...")
    config = PeftConfig.from_pretrained(LORA_ADAPTER_DIR)
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_DIR)
    
    return model, tokenizer

# Load MedQA dataset
def load_medqa_dataset():
    print("Loading MedQA dataset...")
    try:
        # Try to load from Hugging Face with trust_remote_code=True
        dataset = load_dataset(
            "bigbio/med_qa", 
            name="med_qa_en", 
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        return dataset["test"]  # Use test split for benchmarking
    except Exception as e:
        print(f"Error loading MedQA from Hugging Face: {e}")
        
        # Try alternate method to download
        try:
            print("Trying alternate download method...")
            # Use a local JSON file of MedQA test questions (you'll need to provide this)
            medqa_json_path = os.path.join(PROJECT_DIR, "medqa_test.json")
            
            # If the file doesn't exist, create a simple sample file
            if not os.path.exists(medqa_json_path):
                print(f"Creating sample MedQA test data at {medqa_json_path}")
                sample_data = [
                    {
                        "question": "What is the most common cause of community-acquired pneumonia?",
                        "optiona": "Staphylococcus aureus",
                        "optionb": "Streptococcus pneumoniae",
                        "optionc": "Klebsiella pneumoniae",
                        "optiond": "Pseudomonas aeruginosa",
                        "optione": "Haemophilus influenzae",
                        "answer_idx": 1  # B is correct
                    },
                    {
                        "question": "Which of the following is a beta-blocker?",
                        "optiona": "Amlodipine",
                        "optionb": "Lisinopril",
                        "optionc": "Metoprolol",
                        "optiond": "Hydrochlorothiazide",
                        "optione": "Losartan",
                        "answer_idx": 2  # C is correct
                    }
                ]
                with open(medqa_json_path, 'w') as f:
                    json.dump(sample_data, f)
            
            # Load the data
            with open(medqa_json_path, 'r') as f:
                data = json.load(f)
            return data
            
        except Exception as e2:
            print(f"Failed with alternate method too: {e2}")
            raise Exception("Could not load MedQA dataset")

# Format prompt for the model
def format_prompt(question, options):
    formatted_options = ""
    option_letters = "ABCDE"
    
    for i, option in enumerate(options):
        formatted_options += f"{option_letters[i]}. {option}\n"
    
    prompt = f"""
Question: {question}

Options:
{formatted_options}

Answer with the letter of the correct option.
"""
    return prompt

# Generate answer with the model
def generate_answer(model, tokenizer, prompt, max_new_tokens=32):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Limit response length to 32 tokens (sufficient for letter responses)
            temperature=0.1,  # Low temperature (0.1) for more deterministic outputs
                              # Higher values (0.7-1.0) would increase randomness
            top_p=0.95,       # Nucleus sampling - consider tokens with top 95% probability mass
                              # Helps avoid very low probability tokens while maintaining some diversity
            num_return_sequences=1,  # Return only one answer
            do_sample=False   # Use greedy decoding (most likely token at each step)
                              # Setting to True would enable sampling based on temperature and top_p
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract generated part (remove prompt)
    response = response[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
    
    # Extract the option letter (A, B, C, D, or E)
    for letter in "ABCDE":
        if letter in response[:10]:  # Check first 10 chars for the answer letter
            return letter
    
    # Default to first answer if no letter found
    return "A"

# Calculate accuracy
def calculate_metrics(predictions, references):
    correct = sum(p == r for p, r in zip(predictions, references))
    accuracy = correct / len(references) if len(references) > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(references)
    }

# Main evaluation function
def evaluate_medqa(num_samples=None, seed=42):
    """
    Evaluate the model on MedQA dataset.
    
    Parameters:
    - num_samples: Number of samples to evaluate. If None, use all samples.
                   Using a smaller subset allows for quicker testing of the pipeline.
    - seed: Random seed for reproducibility when subsetting.
    
    Returns:
    - metrics: Dictionary with accuracy metrics.
    
    Note on subsetting:
    - Using a subset will give you a faster evaluation but may not represent the
      true model performance across the entire dataset.
    - Small subsets (e.g., <50 samples) may give accuracy estimates with high variance.
    - Recommended subsets: 50-100 samples for quick testing, 500+ for meaningful estimates.
    """
    print("Starting MedQA evaluation...")
    if num_samples:
        print(f"Using a subset of {num_samples} samples")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load model and tokenizer
    model, tokenizer = load_finetuned_model()
    
    # Load dataset
    try:
        dataset = load_medqa_dataset()
        print(f"Loaded {len(dataset)} MedQA questions for evaluation")
        
        # Subset the dataset if requested
        if num_samples and num_samples < len(dataset):
            # Convert to list for random sampling if needed
            if not isinstance(dataset, list):
                dataset_list = list(dataset)
            else:
                dataset_list = dataset
                
            # Randomly sample from the dataset
            sampled_indices = random.sample(range(len(dataset_list)), num_samples)
            dataset = [dataset_list[i] for i in sampled_indices]
            print(f"Sampled {num_samples} questions for evaluation")
            
    except Exception as e:
        print(f"Failed to load MedQA dataset: {e}")
        return
    
    results = []
    predictions = []
    references = []
    
    # Process each question
    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            # Extract question, options and answer
            question = example["question"]
            options = [example[f"option{chr(97+j)}"] for j in range(5) if f"option{chr(97+j)}" in example]
            correct_answer_idx = example.get("answer_idx", None)
            
            if correct_answer_idx is None:
                continue
                
            correct_letter = chr(65 + correct_answer_idx)  # Convert to A, B, C, D, E
            
            # Format the prompt
            prompt = format_prompt(question, options)
            
            # Generate prediction
            predicted_letter = generate_answer(model, tokenizer, prompt)
            
            # Save results
            predictions.append(predicted_letter)
            references.append(correct_letter)
            
            results.append({
                "question": question,
                "options": options,
                "correct_answer": correct_letter,
                "predicted_answer": predicted_letter,
                "is_correct": predicted_letter == correct_letter
            })
            
            # Log progress for every 10 questions
            if (i + 1) % 10 == 0:
                current_metrics = calculate_metrics(predictions, references)
                print(f"Processed {i+1} questions. Current accuracy: {current_metrics['accuracy']:.4f}")
                
        except Exception as e:
            print(f"Error processing question {i}: {e}")
    
    # Calculate final metrics
    metrics = calculate_metrics(predictions, references)
    print(f"Final accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    
    # Add subset information to metrics
    metrics["subset_size"] = num_samples
    metrics["is_subset"] = num_samples is not None
    
    # Save results
    results_df = pd.DataFrame(results)
    subset_str = f"_subset{num_samples}" if num_samples else ""
    results_file = os.path.join(RESULTS_DIR, f"medqa_results{subset_str}.csv")
    results_df.to_csv(results_file, index=False)
    
    # Save metrics
    metrics_file = os.path.join(RESULTS_DIR, f"medqa_metrics{subset_str}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {results_file}")
    print(f"Metrics saved to {metrics_file}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark a model on MedQA")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to evaluate. If not provided, all samples are used.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run evaluation
    metrics = evaluate_medqa(num_samples=args.num_samples, seed=args.seed) 