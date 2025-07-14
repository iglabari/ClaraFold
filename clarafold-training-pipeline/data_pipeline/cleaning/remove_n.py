# remove_n.py

import os
import shutil

def process(input_dir, output_dir):
    """
    This cleaning stage filters out .ct files that contain ambiguous nucleotides ('N')
    in the second column. Files without 'N' are copied to the output directory,
    along with their corresponding .seq files (if available).
    """

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Counters
    total_files_processed = 0
    files_copied = 0
    files_with_n = []

    # Process each .ct file
    for filename in os.listdir(input_dir):
        if filename.endswith('.ct'):
            total_files_processed += 1
            ct_path = os.path.join(input_dir, filename)

            has_n = False

            with open(ct_path, 'r') as f:
                header = f.readline().strip()
                for line in f:
                    columns = line.strip().split()
                    if len(columns) > 1 and columns[1] == 'N':
                        has_n = True
                        break

            if not has_n:
                files_copied += 1

                # Copy .ct file
                shutil.copy(ct_path, output_dir)

                # Also copy corresponding .seq file if it exists
                seq_filename = filename.replace('.ct', '.seq')
                seq_path = os.path.join(input_dir, seq_filename)
                if os.path.exists(seq_path):
                    shutil.copy(seq_path, output_dir)
            else:
                files_with_n.append(filename)

    # Print summary (you can later replace this with logging)
    print("=== Remove N stage summary ===")
    print(f"Total .ct files processed: {total_files_processed}")
    print(f"Files copied (without 'N'): {files_copied}")
    if files_with_n:
        print(f"Files skipped due to 'N': {len(files_with_n)}")
        for f in files_with_n:
            print(f"  - {f}")