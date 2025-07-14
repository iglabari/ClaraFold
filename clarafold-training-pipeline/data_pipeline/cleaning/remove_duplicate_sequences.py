# remove_duplicate_sequences.py

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def process(input_dir, output_dir):
    """
    This cleaning stage removes duplicate sequences from .seq files.
    For each group of identical sequences, only one file is retained (randomly selected).
    The selected .seq file and its corresponding .ct file (if found) are copied to output_dir.
    """

    os.makedirs(output_dir, exist_ok=True)

    sequences_dict = defaultdict(list)
    total_files = 0
    ct_files_not_found = []

    # First pass: read sequences and group by content
    print("Analyzing .seq files for duplicates...")
    for file_path in Path(input_dir).glob('*.seq'):
        total_files += 1
        try:
            with open(file_path, 'r') as file:
                semicolon = file.readline().strip()
                name = file.readline().strip()
                sequence = file.readline().strip()
                sequence_key = sequence.upper()
                sequences_dict[sequence_key].append(file_path)
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    duplicates = []
    unique_count = 0

    for sequence, file_list in sequences_dict.items():
        if len(file_list) == 1:
            seq_file = file_list[0]
            unique_count += 1
        else:
            seq_file = random.choice(file_list)
            duplicates.append({
                'files': [f.name for f in file_list],
                'selected': seq_file.name,
                'count': len(file_list)
            })

        # Copy selected .seq file
        shutil.copy2(seq_file, os.path.join(output_dir, seq_file.name))

        # Copy corresponding .ct file
        ct_file = seq_file.with_suffix('.ct')
        if ct_file.exists():
            shutil.copy2(ct_file, os.path.join(output_dir, ct_file.name))
        else:
            ct_files_not_found.append(seq_file.stem)

    # Summary report
    print("=== Remove Duplicate Sequences stage summary ===")
    print(f"Total .seq files processed: {total_files}")
    print(f"Unique sequences: {unique_count}")
    print(f"Duplicate groups: {len(duplicates)}")
    print(f".ct files not found for {len(ct_files_not_found)} sequences")