#!/usr/bin/env python3
import os
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description='Download a dataset from Hugging Face')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset on Hugging Face')
    parser.add_argument('--subset', type=str, default=None, help='Subset of the dataset')
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to save the dataset')
    args = parser.parse_args()
    
    print(f"Downloading dataset: {args.dataset}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load the dataset
        if args.subset:
            dataset = load_dataset(args.dataset, args.subset)
            print(f"Loaded dataset {args.dataset}/{args.subset}")
        else:
            dataset = load_dataset(args.dataset)
            print(f"Loaded dataset {args.dataset}")
            
        # Print information about the dataset
        print(f"Dataset structure: {dataset}")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Save each split to CSV
        for split_name, split_data in dataset.items():
            output_path = os.path.join(
                args.output_dir, 
                f"{args.dataset.replace('/', '_')}"
            )
            if args.subset:
                output_path += f"_{args.subset}"
            output_path += f"_{split_name}.csv"
            
            # Convert to pandas and save as CSV
            df = split_data.to_pandas()
            df.to_csv(output_path, index=False)
            print(f"Saved {split_name} split to {output_path} ({len(df)} examples)")
            
        print("Download completed successfully!")
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        
if __name__ == "__main__":
    main()