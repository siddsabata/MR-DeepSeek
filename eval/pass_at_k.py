import argparse
import os
import json
import torch
from tqdm import tqdm
from jinja2 import Template

from model_manager import ModelLoader
from scorer import match_choice


def postprocess_output(pred):
    """Clean up model output by removing special tokens and leading spaces."""
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred


def load_file(input_fp):
    """Load and flatten evaluation data from a JSON file."""
    with open(input_fp, 'r') as f:
        data = json.load(f)
    input_data = []
    # If top-level is a list, treat as 'normal' source
    if isinstance(data, list):
        data = {'normal': data}
    # Tag each example with its source
    for src, examples in data.items():
        for ex in examples:
            ex['source'] = src
            input_data.append(ex)
    return input_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate pass@k accuracy for a multiple-choice model")
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-chat-hf",
                        help="Base HuggingFace model identifier or path")
    parser.add_argument('--eval_file', type=str, required=True,
                        help="Path to evaluation JSON file")
    parser.add_argument('--k', type=int, required=True,
                        help="Number of times to repeat each question (pass@k)")
    parser.add_argument('--max_new_tokens', type=int, default=2000,
                        help="Max tokens to generate per call")
    parser.add_argument('--max_tokens', type=int, default=-1,
                        help="Max tokens in prompt (-1 for no limit)")
    parser.add_argument('--use_chat_template', type=bool, default=True,
                        help="Whether to format with tokenizer.chat_template if available")
    parser.add_argument('--strict_prompt', action='store_true',
                        help="Use strict prompt that ends with 'The answer is X.'")
    parser.add_argument('--task', type=str, default='huggingface',
                        help="Task identifier for naming output files")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for model.generate calls")
    parser.add_argument('--hf_token', type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN in env)")
    parser.add_argument('--lora_adapter', type=str, default=None,
                        help="Optional LoRA adapter name or path")
    parser.add_argument('--custom_dir', type=str, default=None,
                        help="Optional custom cache directory name")
    args = parser.parse_args()

    # Initialize model loader
    print(f"Initializing ModelLoader with model {args.model_name}")
    loader = ModelLoader(
        hf_token=args.hf_token or os.environ.get('HF_TOKEN'),
        project_dir=os.path.dirname(os.path.abspath(__file__))
    )

    # Load the model (with optional LoRA)
    print(f"Loading model: {args.model_name}")
    if args.lora_adapter:
        print(f"With LoRA adapter: {args.lora_adapter}")
    loader.load_model(
        model_name=args.model_name,
        lora_adapter=args.lora_adapter,
        custom_dir=args.custom_dir,
        use_cache=True
    )
    model = loader.model
    tokenizer = loader.tokenizer

    # Define a helper to call the model on a list of prompts
    def call_model(prompts, max_new_tokens=2000, print_example=False):
        if print_example:
            print("Example prompt:")
            print(prompts[0])

        # Apply chat template if provided using the recommended tokenizer method
        formatted = prompts
        if args.use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            try:
                # Use tokenizer.apply_chat_template directly
                formatted = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False,  # We want the formatted string, not tokens
                        add_generation_prompt=True # Important for instruction-following models
                    ) for p in prompts
                ]
            except Exception as e:
                # Add a warning if applying the template fails for some reason
                print(f"Warning: Failed to apply chat template using tokenizer.apply_chat_template: {e}. Falling back to raw prompts.")
                # Keep 'formatted' as the original prompts if template application fails
        elif args.use_chat_template:
            # Add a warning if the user wants to use the template but the method isn't available
            print("Warning: Tokenizer does not have 'apply_chat_template' method, but use_chat_template=True. Using raw prompts.")

        # Truncate prompts if max_tokens set
        if args.max_tokens > 0:
            truncated = []
            for pr in formatted:
                ids = tokenizer.encode(pr, add_special_tokens=False)
                if len(ids) > args.max_tokens:
                    ids = ids[:args.max_tokens]
                    truncated.append(tokenizer.decode(ids))
                else:
                    truncated.append(pr)
            formatted = truncated

        # Generate in batches to avoid OOM
        all_outputs = []
        for start in range(0, len(formatted), args.batch_size):
            batch = formatted[start:start + args.batch_size]
            inputs = tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(model.device)
            with torch.no_grad():
                outs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.5,
                    top_p=0.9,
                    do_sample=True
                )
            # Decode only the newly generated tokens
            for seq in outs:
                gen = tokenizer.decode(
                    seq[inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                all_outputs.append(gen)

        # Postprocess each output and return both lists
        processed = [postprocess_output(o) for o in all_outputs]
        return processed, all_outputs

    # Load and prepare evaluation data
    print(f"Loading evaluation data from {args.eval_file}")
    items = load_file(args.eval_file)
    for itm in items:
        # Build the option string and full input prompt
        opts = itm['options']
        itm['option_str'] = '\n'.join([f"{k}. {v}" for k, v in opts.items()])
        if args.strict_prompt:
            prompt_template = (
                "Please answer the following multiple-choice questions, ensuring your response "
                "concludes with the correct option in the format: 'The answer is A.'.\n{question}\n{option_str}"
            )
        else:
            prompt_template = "Please answer the following multiple-choice question:\n{question}\n{option_str}"
        itm['input_str'] = prompt_template.format_map(itm)

    # Run pass@k evaluation
    total = len(items)
    pass_k_counts = {j: 0 for j in range(1, args.k + 1)}
    results = []

    for idx, itm in enumerate(tqdm(items, desc="Evaluating pass@k")):
        prompts = [itm['input_str']] * args.k
        preds, _ = call_model(prompts, args.max_new_tokens, print_example=(idx == 0))
        itm['outputs'] = preds

        # Extract predicted labels for each repeat
        predicted = []
        for text in preds:
            ans, _ = match_choice(text, itm['options'])
            # Use the first matched label as prediction
            predicted.append(ans[0])

        # Update counts: pass@j if any of first j predictions is correct
        correct_idx = itm['answer_idx'].lower()
        for j in range(1, args.k + 1):
            if any(p.lower() == correct_idx for p in predicted[:j]):
                pass_k_counts[j] += 1

        results.append(itm)

    # Compute final pass@k metrics
    pass_at_k = {f"pass@{j}": pass_k_counts[j] / total for j in range(1, args.k + 1)}

    # Build filenames with model, task, and k-info
    model_short = os.path.split(args.model_name)[-1]
    if args.lora_adapter:
        lora = os.path.split(args.lora_adapter)[-1]
        model_short = f"{model_short}-lora-{lora}"
    base = os.path.basename(args.eval_file).replace('.json', '')
    fname = f"{model_short}_{base}_{args.task}"
    if args.strict_prompt:
        fname += "_strict-prompt"
    raw_out = f"{fname}_passat{args.k}.json"
    score_out = f"result_{fname}_passat{args.k}.json"

    # Save raw responses
    print(f"Saving raw outputs to {raw_out}")
    with open(raw_out, 'w') as fw:
        json.dump(results, fw, ensure_ascii=False, indent=2)

    # Save pass@k results
    print("Calculating pass@k scores...")
    print(f"Saving pass@k results to {score_out}")
    with open(score_out, 'w') as fw:
        json.dump(pass_at_k, fw, ensure_ascii=False, indent=2)

    # Free resources
    loader.unload_model()
    print("Pass@k evaluation completed!")


if __name__ == '__main__':
    main() 