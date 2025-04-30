import argparse
import json
import os

def get_argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_outputs_dir', type=str, default='./results/model_outputs')

  return parser

def compute_average_character(model_output, dataset_name):
  len_counter = question_counter = 0
  for output in model_output:
    if output['source'] == dataset_name:
      len_counter += len(output['output'])
      question_counter += 1
  return len_counter / question_counter

def main():
  args = get_argparser().parse_args()
  base_dir = args.model_outputs_dir
  json_files = [os.path.join(base_dir, f) for f in os.listdir(args.model_outputs_dir) if f.endswith('.json')]
  
  results = {}
  for jf in json_files:
    with open(jf, 'r') as f:
      results[jf] = json.load(f)

  dataset_names = ['MedMCQA_validation', 'MedQA_USLME_test', 'PubMedQA_test']
  model_names = json_files
  for model_name in model_names:
    for dataset_name in dataset_names:
      avg_character_count = compute_average_character(results[model_name], dataset_name)
      print(f"{model_name.lstrip(args.model_outputs_dir)} on {dataset_name} is {avg_character_count:.2f} characters")

if __name__ == "__main__":
  main()
