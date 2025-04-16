import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import argparse
import json
import os

class PPODataset(Dataset):
  def __init__(self, data, tokenizer, text_set='train', max_len=1000, verbose=False, max_collate_fn_printout=1):
    self.tokenizer = tokenizer
    self.max_len = max_len

    self.data = list()
    for data_point in data:
      assert 'Open-ended Verifiable Question' in list(data_point.keys()) and 'Ground-True Answer' in list(data_point.keys()), \
                "Missing 'Open-ended Verifiable Question' or 'Ground-True Answer' key in data point"
      if len(data_point['Open-ended Verifiable Question']) > 0 and len(data_point['Ground-True Answer']):
        self.data.append({
          'question': data_point['Open-ended Verifiable Question'],
          'answer': data_point['Ground-True Answer']
        })
    self.length = len(self.data)
    self.verbose = verbose
    self.max_collate_fn_printout = max_collate_fn_printout
    if self.verbose:
      print(f'Initialized a {text_set} set with {self.length} data points')

  def get_prompt(self, data_point):
    system_message = [{
      'role': 'user',
      'content': data_point['question']
    }]
    prompt = self.tokenizer.apply_chat_template(system_message, tokenize=False, add_generation_prompt=True)
    
    input_tokens = self.tokenizer(prompt, padding=False, truncation=False, add_special_tokens=False)
    data_point['input_ids'] = input_tokens['input_ids']

    return data_point

  def __getitem__(self, idx):
    return self.data[idx]

  def __len__(self):
    return self.length
  
  def collate_fn(self, batch):
    items = [self.get_prompt(data_points) for data_points in batch]
    input_ids, questions, answers = zip(*[(item['input_ids'], item['question'], item['answer']) for item in items])
    
    max_len_input_ids = max(len(input_id) for input_id in input_ids)
    max_len = min(self.max_len, max_len_input_ids)
    input_ids = [
      [self.tokenizer.pad_token_id] * (max_len - len(input_id)) + input_id[:max_len]
        for input_id in input_ids
    ]

    if self.verbose:
      max_printout = min(self.max_collate_fn_printout, len(input_ids))
      for input_id, question, answer in zip(input_ids[:max_printout], questions[:max_printout], answers[:max_printout]):
        print(f'[input_ids] {input_id}')
        print(f'[question] {question}')
        print(f'[answer] {answer}')

    return {
      'input_ids': torch.LongTensor(input_ids),
      'question': questions,
      'answer': answers
    }

def get_argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_data', default='data/demo_data.json')
  parser.add_argument('--max_len', type=int, default=1000)
  parser.add_argument('--max_collate_fn_printout', type=int, default=1)
  
  return parser

if __name__ == '__main__':
  os.environ["TOKENIZERS_PARALLELISM"] = "false" # set to 'false' to prevent potential deadlock

  args = get_argparser().parse_args()
  test_data = args.test_data
  max_len = args.max_len
  max_collate_fn_printout = args.max_collate_fn_printout

  print(f'Reading data from {test_data}')
  with open(test_data, 'r') as jf:
    data = json.load(jf)
  
  tokenizer = AutoTokenizer.from_pretrained('FreedomIntelligence/HuatuoGPT-o1-8B')
  dataset = PPODataset(
    data=data,
    tokenizer=tokenizer,
    text_set='train',
    max_len=max_len,
    verbose=True,
    max_collate_fn_printout=max_collate_fn_printout
  )
  dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
    collate_fn=dataset.collate_fn,
    drop_last=True
  )
  
  # test printout for PPODataset.collate_fn
  for batch in dataloader:
    pass

  os.environ["TOKENIZERS_PARALLELISM"] = "true"
