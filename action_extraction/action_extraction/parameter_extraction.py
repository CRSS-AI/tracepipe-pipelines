# dummy file to test pipeline runner
import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Parameter Extraction Step: Reading from {args.input}")
    os.makedirs(args.output, exist_ok=True)

    for filename in os.listdir(args.input):
        if filename.endswith(".parquet"):
            in_path = os.path.join(args.input, filename)
            print(f"Reading {in_path}...")
            df = pd.read_parquet(in_path)
            
            out_path = os.path.join(args.output, filename)
            print(f"Writing to {out_path}...")
            df.to_parquet(out_path, index=False)

if __name__ == "__main__":
    main()