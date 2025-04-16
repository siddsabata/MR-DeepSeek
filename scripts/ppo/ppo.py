from transformers import HfArgumentParser, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, PreTrainedTokenizerBase
from trl import ModelConfig, ScriptArguments

import os

from datasets import PPODataset
from trainer import PPOTrainer, PPOConfig

def get_hfargparser():
  parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))

  return parser

def main():
  script_args, training_args, model_config = get_hfargparser().parse_args_into_dataclasses()
  training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

  output_dir = training_args.output_dir
  run_name = training_args.run_name
  if run_name not in output_dir:
    output_dir = os.path.join(output_dir,run_name)
    training_args.output_dir = output_dir

  tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
  reward_model = AutoModelForSequenceClassification.from_pretrained(
    training_args.reward_model_path, attn_implementation="flash_attention_2", num_labels=2
  )
  value_model = AutoModelForSequenceClassification.from_pretrained(
    training_args.value_model_path, trust_remote_code=model_config.trust_remote_code, attn_implementation="flash_attention_2", num_labels=1
  )

  ref_policy = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path,attn_implementation="flash_attention_2")
  policy = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path,attn_implementation="flash_attention_2")

  reward_tokenizer = AutoTokenizer.from_pretrained(training_args.reward_model_path)

  if '<|eot_id|>' in tokenizer.vocab:
    assert '<|end_of_text|>' in tokenizer.vocab
    tokenizer.pad_token = '<|end_of_text|>'
    tokenizer.pad_token_id = tokenizer.encode('<|end_of_text|>',add_special_tokens=False)[0]
  assert tokenizer.pad_token_id != tokenizer.eos_token_id

  training_args.stop_token_id = tokenizer.eos_token_id

  train_data = NotImplementedError # FILL THIS IN WHEN IT IS TIME TO TRAIN
  eval_data = NotImplementedError  # FILL THIS IN WHEN IT IS TIME TO TRAIN

  train_dataset = PPODataset(train_data, tokenizer)
  eval_dataset = PPODataset(eval_data, tokenizer)

  trainer = PPOTrainer(
    config=training_args,
    processing_class=tokenizer,
    reward_processing_class = reward_tokenizer,
    policy=policy,
    ref_policy=ref_policy,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator = train_dataset.collate_fn
  )
  trainer.train()

if __name__ == "__main__":
  main()
