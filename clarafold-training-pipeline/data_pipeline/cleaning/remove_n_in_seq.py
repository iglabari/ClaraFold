# remove_n_in_seq.py

import os
import shutil
from pathlib import Path

def process(input_dir, output_dir):
    """
    This cleaning stage filters out .seq files containing 'N' in the primary sequence.
    Valid files are copied to the output directory, along with their corresponding .ct files.
    """

    os.makedirs(output_dir, exist_ok=True)

    files_with_n = []
    ct_files_not_found = []
    total_files = 0

    for file_path in Path(input_dir).glob('*.seq'):
        total_files += 1
        has_n = False

        try:
            with open(file_path, 'r') as file:
                semicolon = file.readline().strip()  # First line (;)
                name = file.readline().strip()       # Second line (name)
                sequence = file.readline().strip()   # Third line (sequence)

                if 'N' in sequence:
                    has_n = True
                    files_with_n.append(file_path.name)
                else:
                    # Copy valid .seq file
                    dest_seq_path = os.path.join(output_dir, file_path.name)
                    shutil.copy2(file_path, dest_seq_path)

                    # Look for corresponding .ct file
                    ct_file = file_path.with_suffix('.ct')
                    if ct_file.exists():
                        dest_ct_path = os.path.join(output_dir, ct_file.name)
                        shutil.copy2(ct_file, dest_ct_path)
                    else:
                        ct_files_not_found.append(file_path.stem)

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    # Summary report
    print("=== Remove N in SEQ stage summary ===")
    print(f"Total .seq files processed: {total_files}")
    print(f"Files containing 'N': {len(files_with_n)}")
    print(f"Files copied (without 'N'): {total_files - len(files_with_n)}")
    if ct_files_not_found:
        print(f".ct files not found for {len(ct_files_not_found)} sequences")