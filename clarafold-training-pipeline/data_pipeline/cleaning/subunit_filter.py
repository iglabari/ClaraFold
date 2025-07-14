# subunit_filter.py

import os
import shutil
from pathlib import Path
from collections import defaultdict

def find_subunits_with_positions(long_seq, sequences):
    """
    Find subunits inside a longer sequence, ignoring the final '1'.
    Returns subunits and their positions.
    """
    subunits = {}
    long_seq_base = long_seq[:-1]

    for seq, files in sequences.items():
        if seq != long_seq:
            seq_base = seq[:-1]
            if len(seq_base) < len(long_seq_base):
                if seq_base in long_seq_base:
                    position = long_seq_base.find(seq_base)
                    for file_path in files:
                        subunits[file_path] = position
    return subunits

def process(input_dir, output_dir):
    """
    Identifies sub-sequences (subunits) inside longer sequences.
    Creates 3 folders inside output_dir:
      - enteros/ (full sequences)
      - subunidades/ (subunits)
      - posta/ (non-subunit sequences)
    """

    # Prepare output subfolders
    enteros_dir = Path(output_dir) / 'enteros'
    subunidades_dir = Path(output_dir) / 'subunidades'
    posta_dir = Path(output_dir) / 'posta'
    for dir_path in [enteros_dir, subunidades_dir, posta_dir]:
        os.makedirs(dir_path, exist_ok=True)

    sequences_dict = defaultdict(list)
    total_files = 0
    all_files = set()

    # First pass: read files and group by sequence
    print("Analyzing .seq files...")
    for file_path in Path(input_dir).glob('*.seq'):
        total_files += 1
        all_files.add(file_path)
        try:
            with open(file_path, 'r') as file:
                file.readline()  # skip semicolon line
                file.readline()  # skip name line
                sequence = file.readline().strip()
                sequences_dict[sequence].append(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")

    # Detect subunits
    complete_sequences = []
    processed_subunits = set()
    sequences_with_subunits = []
    sorted_sequences = sorted(sequences_dict.items(), key=lambda x: len(x[0][:-1]), reverse=True)

    for long_seq, files in sorted_sequences:
        if not long_seq.endswith('1'):
            print(f"Warning: sequence in {files[0].name} doesn't end with '1'")
            continue

        if not any(files[0] in subunits for _, subunits in sequences_with_subunits):
            subunits = find_subunits_with_positions(long_seq, sequences_dict)
            if subunits:
                complete_sequences.extend(files)
                sequences_with_subunits.append((files, subunits))
                processed_subunits.update(subunits.keys())

    # Copy complete sequences
    for file_path in complete_sequences:
        shutil.copy2(file_path, enteros_dir / file_path.name)
        ct_file = file_path.with_suffix('.ct')
        if ct_file.exists():
            shutil.copy2(ct_file, enteros_dir / ct_file.name)

    # Copy subunits
    for _, subunits in sequences_with_subunits:
        sorted_subunits = sorted(subunits.items(), key=lambda x: x[1])
        for position, (file_path, _) in enumerate(sorted_subunits, 1):
            new_seq_name = f"{file_path.stem}_subunit{position}{file_path.suffix}"
            new_ct_name = f"{file_path.stem}_subunit{position}.ct"
            shutil.copy2(file_path, subunidades_dir / new_seq_name)
            ct_file = file_path.with_suffix('.ct')
            if ct_file.exists():
                shutil.copy2(ct_file, subunidades_dir / new_ct_name)

    # Copy remaining files to 'posta'
    subunit_files = {f for _, subunits in sequences_with_subunits for f in subunits.keys()}
    for file_path in all_files:
        if file_path not in subunit_files and file_path not in complete_sequences:
            shutil.copy2(file_path, posta_dir / file_path.name)
            ct_file = file_path.with_suffix('.ct')
            if ct_file.exists():
                shutil.copy2(ct_file, posta_dir / ct_file.name)

    print("=== Subunit Filter stage finished ===")
    print(f"Total .seq files processed: {total_files}")
    print(f"Full sequences detected: {len(complete_sequences)}")
    print(f"Subunits found: {sum(len(subs) for _, subs in sequences_with_subunits)}")
    print(f"Remaining 'posta' sequences: {len(all_files - subunit_files - set(complete_sequences))}")