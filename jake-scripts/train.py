from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import get_cosine_schedule_with_warmup  
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
#import bitsandbytes as bnb
from torch.utils.data import Dataset, DataLoader
# LoraConfig: A configuration object that specifies how LoRA (Low-Rank Adapters)
# should be inserted (how many “r” ranks, which target modules, dropout, etc.).

# get_peft_model: A function that, given a base model and a LoRA config, 
# returns a wrapped version of the model with LoRA adapter layers attached.

# prepare_model_for_kbit_training: A helper function that properly configures 
# the model for k-bit training (k=4 or k=8). It makes certain layers trainable,
# merges any existing weights if needed, and sets the model up to accept LoRA layers.
import torch
from datasets import load_from_disk
from jinja2 import Template
import torch.nn.functional as F
from torch.optim import AdamW
import time

import wandb

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def print_gpu_util(name=""):
    """Utility to print out current GPU memory usage (allocated and reserved) for debugging."""
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{name}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")


class Train_dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, max_seq_len):
        self.tokenizer = tokenizer
        self.data = data
        
        newdata = []
        for da in self.data:
            newdata.append(da)
        print('Hello',len(self.data),len(newdata))
        self.data = newdata

        self.max_seq_len = max_seq_len
        self.debug = 0

        # 如果从Base LLMs训练，选择 llama3-instruct作为模版
        chat_template_llama3 = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        if not tokenizer.chat_template:
            tokenizer.chat_template = chat_template_llama3
            
        self.template = Template(tokenizer.chat_template)

    
    def __getitem__(self, index):
        return self.data[index]

    def get_response(self,da):
        temp = '## Thinking\n\n{}\n\n## Final Response\n\n{}'
        return temp.format(da['Complex_CoT'],da['Response'])
        
    def get_prompt(self,da):

        q = da['Question']
        a = self.get_response(da)
        assert q is not None and a is not None, f'q:{q} a:{a}'

        input =  self.template.render(messages=[{"role": "user", "content": q},{"role": "assistant", "content": a}],bos_token=self.tokenizer.bos_token,add_generation_prompt=False)
        #print(input)
        input_ids = self.tokenizer.encode(input,add_special_tokens= False)

        query = self.template.render(messages=[{"role": "user", "content": q}],bos_token=self.tokenizer.bos_token,add_generation_prompt=True)
        query_ids = self.tokenizer.encode(query,add_special_tokens= False)

        labels = [-100]*len(query_ids) + input_ids[len(query_ids):]
        assert len(labels) == len(input_ids)
        return {"input_ids": input_ids[-self.max_seq_len:], "labels": labels[-self.max_seq_len:]}        

    def collate_fn(self, batch):
        data = [ self.get_prompt(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        labels = [item["labels"] for item in data]
        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len,self.max_seq_len)
        input_ids = [ item[:max_len] + [self.tokenizer.eos_token_id]*(max_len-len(item)) for item in input_ids]
        labels = [ item[:max_len] + [-100]*(max_len-len(item)) for item in labels]
        if self.debug < 3:
            print("JAKKKKKKEEEEEEE")
            print('input_ids',self.tokenizer.decode(input_ids[-1]))
            print('labels',self.tokenizer.decode([0 if x == -100 else x for x in labels[-1]]))
            self.debug += 1

        return {
                "input_ids": torch.LongTensor(input_ids),
                "labels": torch.LongTensor(labels),
            }
    
    def __len__(self):
        return len(self.data)



def main():
    print("Starting training script...")
    print("login into wandb")
    wandb.login(key="00b5a9f86429598bdcc1fb4639c71b8cec609b57")
        # Initialize wandb run with a project name and configuration
    wandb.init(
        project="DeepSeek-Training",
        config={
            "learning_rate": 1e-6,
            "batch_size": 8,
            "num_epochs": 3,
            "model_name": "bs4-001-h100",
            "weight_decay": 0.1,
            "warmup_ratio": 0.05 
        }
    )

    # Load the dataset
    path = "/ocean/projects/cis250063p/jbentley/dl_project/data/"
    dataset = load_from_disk(path + "medical-o1-reasoning-SFT_dataset")

    # n = len(dataset["train"])
    # subset = 0.10
    # n = int(n * subset)
    # dataset_sub = dataset["train"].select(range(n))

    # print(dataset_sub)
    print(dataset)
    

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 



    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # means load the base model weights in 4-bit precision (instead of 16-bit or 32-bit).
        bnb_4bit_use_double_quant=True, # an additional optimization that can help reduce quantization error by applying a second quantization layer internally.
        bnb_4bit_quant_type="nf4", # uses Normal Float 4 (NF4), an advanced quantization format that typically yields better accuracy than older 4-bit methods.
        bnb_4bit_compute_dtype=torch.float16 # means that while the model weights are stored in 4-bit, the actual calculations during forward and backward passes are done in 16-bit floating-point precision (FP16
    )

    print_gpu_util("Before model load")

    # Load the base model in 4-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" # means the Accelerate library automatically places different layers of the model on available GPU (or CPU) to fit memory constraints
    )

    print_gpu_util("After model load")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Loads the tokenizer associated with the same model

    # Prepare the model for k-bit training (for inserting LoRA layers)
    model = prepare_model_for_kbit_training(model)


    lora_config = LoraConfig(
        r=16, #  the rank of the LoRA decomposition. This directly influences how many additional parameters you’re training. A higher rank means more adapter parameters but potentially greater fine-tuning capacity.
        lora_alpha=32, # a scaling factor for the LoRA updates, which can help with training stability and effectiveness.
        target_modules=["q_proj","k_proj","v_proj","o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"], # ,"mlp.gate_proj" add o_proj and mlp.up_proj, mlp.down_proj if performance is not great
        lora_dropout=0.05, # dropout rate applied to LoRA layers, to help prevent overfitting
        bias="none",
        task_type="CAUSAL_LM" # specifies the type of task (causal language modeling) for these LoRA adapters.
    )

    # Insert LoRA layers into the model
    # Takes the quantized model (prepared for 4-bit training) and the LoRA config, 
    # then creates a PEFT (LoRA)-enhanced model by injecting LoRA adapter layers 
    # into the specified modules.
    lora_model = get_peft_model(model, lora_config)


    # Prevent caching of key/value states during training to save GPU memory
    lora_model.config.use_cache = False

    # Allows the model inputs to require gradients, 
    # which can be important for fine-tuning certain adapter layers
    lora_model.enable_input_require_grads()

    # Enable gradient checkpointing: saves VRAM by re-computing 
    # forward passes during backward pass at the cost of extra compute time
    lora_model.gradient_checkpointing_enable()

    wandb.watch(lora_model, log="all")

    import gc
    gc.collect()

    train_data_object = Train_dataset(tokenizer, dataset["train"], 8192)

    print_gpu_util("Before creating the DataLoader")


    training_dataloader = DataLoader(
        train_data_object, 
        batch_size=8,
        shuffle=True, 
        drop_last=True, 
        collate_fn=train_data_object.collate_fn
    )

    print_gpu_util("After creating the DataLoader")

    gc.collect()

    # Typically we'll want to filter out LoRA parameters
    # so that we only optimize them and not the base model's
    optim = AdamW(lora_model.parameters(), lr=1e-6, weight_decay=0.1)


    num_epochs = 3
    warmup_ratio = 0.05

    num_update_steps_per_epoch = len(training_dataloader)
    total_training_steps = num_epochs * num_update_steps_per_epoch
    warmup_steps = int(warmup_ratio * total_training_steps)


    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )




    lora_model.train()  # Set model to training mode


    global_step = 0
    loss_accum = 0.0
    step_count = 0

    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        for step, batch in enumerate(training_dataloader):
            
            input_ids = batch["input_ids"].cuda()   # or .to(device)
            labels = batch["labels"].cuda()
            print_gpu_util(f"Step {step} after moving batch to GPU")
            
            #print(torch.cuda.memory_summary(device=torch.cuda.current_device()))

            
            print_gpu_util(f"Start of step {step}")

            outputs = lora_model(
                input_ids=input_ids, 
                labels=labels
            )

            print_gpu_util(f"Step {step} after forward pass")

            loss = outputs.loss

            # Backprop and optimize
            optim.zero_grad()
            loss.backward()

            print_gpu_util(f"Step {step} after backward pass")

            optim.step()

            print_gpu_util(f"Step {step} after optimizer step")

            # Step the scheduler
            scheduler.step()
            
            # Print GPU memory usage after each step (converted to MB)
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"Step {step}: Loss: {loss.item():.4f} | Allocated Memory: {allocated:.2f} MB | Reserved Memory: {reserved:.2f} MB")

            # Accumulate the loss and step count
            loss_accum += loss.item()
            step_count += 1
            global_step += 1

            # Every 5 steps, log the average loss and reset the accumulator
            if step_count == 5:
                avg_loss = loss_accum / step_count
                wandb.log({
                    "epoch": epoch,
                    "global_step": global_step,
                    "lr": scheduler.get_last_lr()[0],
                    "avg_loss": avg_loss,
                    "allocated_mem_MB": allocated,
                    "reserved_mem_MB": reserved
                }, step=global_step)
            
                loss_accum = 0.0
                step_count = 0
            # Clear unnecessary cached memory (optional; may impact performance)
            torch.cuda.empty_cache()
            
            print_gpu_util(f"Step {step} after everything")


            # if step % 10 == 0:
            #     print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Code to be timed
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Finally, save just the LoRA adapter weights

    wandb.finish()

    save_path = "/ocean/projects/cis250063p/jbentley/dl_project/model/"
    lora_model.save_pretrained(save_path + "lora-output")




if __name__ == "__main__":
    main()