import pandas as pd
import sys
import os
import random

def shuffle_and_filter_csv(file_path, seed=None, delete_prob=0.4):
    """Shuffle rows of a CSV file and delete rows containing 'gay' with a certain probability."""
    if not os.path.isfile(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    try:
        # Read CSV
        df = pd.read_csv(file_path)

        # Drop rows containing 'gay' with a probability
        def should_keep(row):
            row_str = " ".join(map(str, row)).lower()
            if "gay" in row_str and random.random() < delete_prob:
                return False
            return True

        df = df[df.apply(should_keep, axis=1)]

        # Shuffle remaining rows
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Overwrite CSV
        df.to_csv(file_path, index=False)
        print(f"[Success] Filtered, shuffled, and saved: {file_path}")

    except Exception as e:
        print(f"[Error] Failed to process file: {e}")

# Usage: python shuffle_csv.py yourfile.csv
if __name__ == "__main__":
    
    csv_file = 'new_tuples.csv'
    shuffle_and_filter_csv(csv_file, seed=random.randint(0, 2**32 - 1))
