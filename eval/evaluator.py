import argparse
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from jinja2 import Template

from model_manager import ModelLoader
from scorer import get_results

def postprocess_output(pred):
    """Clean up model output by removing special tokens and leading spaces."""
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def load_file(input_fp):
    """Load and organize benchmark data from a JSON file."""
    with open(input_fp, 'r') as f:
        data = json.load(f)
    input_data = []
    if isinstance(data, list):
        data = {'normal': data}
    for k, v in data.items():
        for da in v:
            da['source'] = k
        input_data.extend(v)
    return input_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=2000)
    parser.add_argument('--max_tokens', type=int, default=-1)
    parser.add_argument('--use_chat_template', type=bool, default=True)
    parser.add_argument('--strict_prompt', action="store_true")
    parser.add_argument('--task', type=str, default='huggingface')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--lora_adapter', type=str, default=None, help='Path to LoRA adapter')
    parser.add_argument('--custom_dir', type=str, default=None, help='Custom directory name for cached model')
    args = parser.parse_args()

    # Initialize model loader with optional HF token
    print(f"Initializing ModelLoader with model {args.model_name}")
    model_loader = ModelLoader(
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        project_dir=os.path.dirname(os.path.abspath(__file__))
    )
    
    # Load the model, now with LoRA support
    print(f"Loading model: {args.model_name}")
    if args.lora_adapter:
        print(f"With LoRA adapter: {args.lora_adapter}")
    model_loader.load_model(
        model_name=args.model_name,
        lora_adapter=args.lora_adapter,
        custom_dir=args.custom_dir
    )
    model = model_loader.model
    tokenizer = model_loader.tokenizer

    # Setup chat template if needed
    if args.use_chat_template and hasattr(tokenizer, 'chat_template'):
        template = Template(tokenizer.chat_template)
    else:
        template = None

    def call_model(prompts, max_new_tokens=2000, print_example=False):
        """Run inference on the model with the given prompts."""
        if print_example:
            print("Example prompt:")
            print(prompts[0])
        
        # Format prompts using chat template if applicable
        formatted_prompts = prompts
        if template:
            formatted_prompts = [
                template.render(
                    messages=[{"role": "user", "content": prom}],
                    bos_token=tokenizer.bos_token,
                    add_generation_prompt=True
                ) for prom in prompts
            ]
        
        # Apply token limit if specified
        if args.max_tokens > 0:
            new_prompts = []
            for prompt in formatted_prompts:
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(input_ids) > args.max_tokens:
                    input_ids = input_ids[:args.max_tokens]
                    new_prompts.append(tokenizer.decode(input_ids))
                else:
                    new_prompts.append(prompt)
            formatted_prompts = new_prompts
        
        # Process in smaller batches to avoid OOM
        all_outputs = []
        for i in range(0, len(formatted_prompts), args.batch_size):
            batch_prompts = formatted_prompts[i:i+args.batch_size]
            
            # Tokenize inputs
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(model.device)
            
            # Generate with the model
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.5,
                    top_p=0.9,
                    do_sample=True
                )
            
            # Decode outputs
            for i, output in enumerate(outputs):
                # Get only the newly generated tokens
                response = tokenizer.decode(
                    output[inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                all_outputs.append(response)
        
        # Apply postprocessing
        processed_outputs = [postprocess_output(out) for out in all_outputs]
        return processed_outputs, all_outputs

    # Load evaluation data
    print(f"Loading evaluation data from {args.eval_file}")
    input_data = load_file(args.eval_file)
    
    # Determine prompt format based on args
    if args.strict_prompt:
        query_prompt = "Please answer the following multiple-choice questions. Please answer the following multiple-choice questions, ensuring your response concludes with the correct option in the format: 'The answer is A.'.\n{question}\n{option_str}"
    else:
        query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}"
    
    # Process data in batches
    final_results = []
    for idx in tqdm(range(0, len(input_data), args.batch_size)):
        batch = input_data[idx:idx+args.batch_size]
        if len(batch) == 0:
            break
        
        # Format each example in the batch
        for item in batch:
            item['option_str'] = '\n'.join([f'{op}. {ans}' for op, ans in item['options'].items()])
            item["input_str"] = query_prompt.format_map(item)
        
        processed_batch = [item["input_str"] for item in batch]
        
        # Print an example for the first batch
        print_example = (idx == 0)
        
        # Run inference
        preds, _ = call_model(
            processed_batch, 
            max_new_tokens=args.max_new_tokens,
            print_example=print_example
        )
        
        # Store results
        for j, item in enumerate(batch):
            if j < len(preds):  # Ensure we have a prediction
                pred = preds[j]
                if len(pred) == 0:
                    continue
                item["output"] = pred
                final_results.append(item)
    
    # Generate result filename
    model_name_short = os.path.split(args.model_name)[-1]
    # Add LoRA info to filename if present
    if args.lora_adapter:
        lora_name = os.path.split(args.lora_adapter)[-1]
        model_name_short = f"{model_name_short}-lora-{lora_name}"
    eval_file_base = os.path.basename(args.eval_file).replace('.json', '')
    task_name = f"{model_name_short}_{eval_file_base}_{args.task}"
    if args.strict_prompt:
        task_name += "_strict-prompt"
    
    # Save results
    save_path = f'{task_name}.json'
    print(f"Saving results to {save_path}")
    with open(save_path, 'w') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)
    
    # Evaluate results
    print("Calculating scores...")
    get_results(save_path)
    
    # Cleanup
    model_loader.unload_model()
    print("Evaluation completed!")

if __name__ == "__main__":
    main() 