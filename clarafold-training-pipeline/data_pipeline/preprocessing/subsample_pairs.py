# data_pipeline/preprocessing/subsample_pairs.py

import os
import random
import shutil
from pathlib import Path

def subsample_pairs(input_dir, output_dir, num_pairs):
    """
    Randomly select a given number of .ct/.seq file pairs from input_dir
    and copy them to output_dir.

    Args:
        input_dir (str): Directory containing .ct and .seq files.
        output_dir (str): Destination directory for selected pairs.
        num_pairs (int): Number of pairs to select.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Find all .ct files
    ct_files = [f for f in os.listdir(input_dir) if f.endswith('.ct')]

    # Build list of valid (.ct, .seq) pairs
    valid_pairs = []
    for ct_file in ct_files:
        base = ct_file[:-3]  # remove '.ct'
        seq_file = base + '.seq'
        if os.path.exists(os.path.join(input_dir, seq_file)):
            valid_pairs.append((ct_file, seq_file))

    if len(valid_pairs) < num_pairs:
        raise ValueError(
            f"Requested {num_pairs} pairs but only {len(valid_pairs)} are available."
        )

    selected_pairs = random.sample(valid_pairs, num_pairs)

    # Copy selected files
    for ct_file, seq_file in selected_pairs:
        shutil.copy2(os.path.join(input_dir, ct_file), os.path.join(output_dir, ct_file))
        shutil.copy2(os.path.join(input_dir, seq_file), os.path.join(output_dir, seq_file))

    print(f"Successfully selected {num_pairs} file pairs.")

    return selected_pairs

# Standalone CLI interface (for testing or ad hoc use)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Subsample .ct/.seq file pairs.")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory containing files")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory for selected files")
    parser.add_argument("-n", "--num_pairs", type=int, required=True, help="Number of pairs to select")

    args = parser.parse_args()

    try:
        selected = subsample_pairs(args.input_dir, args.output_dir, args.num_pairs)
        print("Selected pairs:")
        for ct_file, seq_file in selected:
            print(f"- {ct_file}, {seq_file}")
    except Exception as e:
        print(f"Error: {e}")

