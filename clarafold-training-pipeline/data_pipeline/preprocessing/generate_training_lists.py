# data_pipeline/preprocessing/generate_training_lists.py

import os
from pathlib import Path

def extract_encoded_sequence(file_path):
    """
    Extracts the encoded sequence under 'coding_1D+2D:' section from a .code file.
    """
    with open(file_path, 'r') as file:
        content = file.read()
        marker = 'coding_1D+2D:'
        if marker in content:
            start = content.index(marker) + len(marker)
            return content[start:].strip()
    return None

def extract_identifier(filename):
    """
    Extracts the base identifier from filenames like:
    - something_probknot.code  -> something
    - something.code           -> something
    """
    if filename.endswith('_probknot.code'):
        return filename.replace('_probknot.code', '')
    elif filename.endswith('.code'):
        return filename.replace('.code', '')
    else:
        return filename

def generate_training_lists(probknot_dir, gt_dir, output_file_probknot, output_file_gt):
    """
    Aligns and extracts encoded sequences from both predicted and ground-truth directories,
    and writes aligned lists to output files.
    """
    probknot_dir = Path(probknot_dir)
    gt_dir = Path(gt_dir)

    # Process probknot files
    probknot_sequences = {}
    for file in probknot_dir.glob("*.code"):
        identifier = extract_identifier(file.name)
        sequence = extract_encoded_sequence(file)
        if sequence:
            probknot_sequences[identifier] = sequence

    # Process ground-truth files
    gt_sequences = {}
    for file in gt_dir.glob("*.code"):
        identifier = extract_identifier(file.name)
        sequence = extract_encoded_sequence(file)
        if sequence:
            gt_sequences[identifier] = sequence

    # Find common identifiers
    common_ids = sorted(set(probknot_sequences.keys()) & set(gt_sequences.keys()))
    if not common_ids:
        print("No common identifiers found between probknot and gt sets.")
        return

    # Write aligned sequences
    with open(output_file_probknot, 'w') as out_probknot, open(output_file_gt, 'w') as out_gt:
        for identifier in common_ids:
            out_probknot.write(probknot_sequences[identifier] + "\n")
            out_gt.write(gt_sequences[identifier] + "\n")

    print(f"Extracted and aligned {len(common_ids)} sequence pairs.")
    print(f"Saved to:\n- {output_file_probknot}\n- {output_file_gt}")

    # Report any mismatches for diagnostics
    only_in_probknot = set(probknot_sequences.keys()) - set(gt_sequences.keys())
    only_in_gt = set(gt_sequences.keys()) - set(probknot_sequences.keys())

    if only_in_probknot:
        print(f"Identifiers only in probknot set: {only_in_probknot}")
    if only_in_gt:
        print(f"Identifiers only in gt set: {only_in_gt}")

# Optional standalone CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate aligned training lists from encoded .code files.")
    parser.add_argument("-p", "--probknot_dir", required=True, help="Directory containing probknot .code files.")
    parser.add_argument("-g", "--gt_dir", required=True, help="Directory containing ground truth .code files.")
    parser.add_argument("-op", "--output_probknot", required=True, help="Output file for probknot encoded sequences.")
    parser.add_argument("-og", "--output_gt", required=True, help="Output file for ground truth encoded sequences.")

    args = parser.parse_args()

    generate_training_lists(
        args.probknot_dir,
        args.gt_dir,
        args.output_probknot,
        args.output_gt
    )