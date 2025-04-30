import matplotlib.pyplot as plt
import numpy as np

import argparse
import csv

def get_argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--pass_at_k_results', type=str, default='./results/pass_at_k.csv')

  return parser

def main():
  args = get_argparser().parse_args()
  results = dict()
  is_header = True
  with open(args.pass_at_k_results, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      if is_header:
        is_header = False
        continue
      results[row[0]] = [float(val) for val in row[1:]]

  for model_name, pass_at_k_values in results.items():
    plt.plot(list(range(1, len(row))), pass_at_k_values, label=model_name, marker='o')
  plt.title("pass@k Accuracy")
  plt.xlabel("k")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.xticks(list(range(1, len(row))))
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()