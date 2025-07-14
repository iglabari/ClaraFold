# data_pipeline/preprocessing/analyze_sequence_lengths.py

import os
from pathlib import Path
from collections import defaultdict

def analyze_sequence_lengths(input_dir):
    """
    Analyzes sequence lengths from .seq files in the given directory.
    Assumes sequences end with '1' and subtracts 1 from total length.

    Args:
        input_dir (str): Directory containing .seq files.

    Returns:
        dict: Summary statistics (min, max, avg, distribution).
    """

    file_lengths = []

    # Process all .seq files in input_dir
    for file_path in Path(input_dir).glob("*.seq"):
        try:
            with open(file_path, "r") as file:
                file.readline()  # Skip first line (;)
                name = file.readline().strip()  # Second line (name)
                sequence = file.readline().strip()  # Third line (sequence)

                sequence_length = len(sequence) - 1  # Remove final '1'
                file_lengths.append((file_path.name, sequence_length))
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    file_lengths.sort(key=lambda x: x[1])

    if not file_lengths:
        print("No .seq files found or processed.")
        return {}

    lengths = [length for _, length in file_lengths]
    min_length = min(lengths)
    max_length = max(lengths)
    avg_length = sum(lengths) / len(lengths)

    length_freq = defaultdict(int)
    for _, length in file_lengths:
        length_freq[length] += 1

    # Print ordered file lengths
    print("\nFiles sorted by sequence length:")
    print(f"{'Filename'.ljust(50)} Length")
    print("-" * 70)
    for filename, length in file_lengths:
        print(f"{filename.ljust(50)} {length}")

    # Print statistics
    print("\nStatistics:")
    print(f"Min length: {min_length}")
    print(f"Max length: {max_length}")
    print(f"Average length: {avg_length:.2f}")

    print("\nLength distribution:")
    print(f"{'Length'.ljust(15)} Count")
    print("-" * 40)
    for length in sorted(length_freq.keys()):
        print(f"{str(length).ljust(15)} {length_freq[length]}")

    print(f"\nTotal files analyzed: {len(file_lengths)}")

    # Return summary (useful for future extensions)
    return {
        "min": min_length,
        "max": max_length,
        "avg": avg_length,
        "total_files": len(file_lengths),
        "distribution": dict(length_freq),
    }


# Optional CLI interface (for standalone use)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze .seq file sequence lengths.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing .seq files")

    args = parser.parse_args()

    analyze_sequence_lengths(args.input_dir)

