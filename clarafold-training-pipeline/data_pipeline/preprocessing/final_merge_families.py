# data_pipeline/preprocessing/final_merge_families.py

import os
from pathlib import Path

def merge_family_files(family_probknot_files, family_gt_files, output_probknot_file, output_gt_file, input_dir):
    """
    Merges multiple family-level files into final global training files.

    Args:
        family_probknot_files (list): List of probknot family filenames.
        family_gt_files (list): List of ground truth family filenames.
        output_probknot_file (str): Path to save the merged probknot file.
        output_gt_file (str): Path to save the merged ground truth file.
        input_dir (str): Directory where all per-family files are located.
    """

    def concatenate_files(file_list, output_path, base_dir):
        with open(output_path, 'w') as out_file:
            for filename in file_list:
                file_path = Path(base_dir) / filename
                if not file_path.exists():
                    print(f"Warning: File not found: {file_path}")
                    continue

                with open(file_path, 'r') as infile:
                    content = infile.read().rstrip('\n')
                    if content:
                        out_file.write(content)
                        if not content.endswith('\n'):
                            out_file.write('\n')

                print(f"File {filename} successfully concatenated.")

    # Merge probknot files
    print("Merging probknot files...")
    concatenate_files(family_probknot_files, output_probknot_file, input_dir)

    # Merge ground truth files
    print("Merging ground truth files...")
    concatenate_files(family_gt_files, output_gt_file, input_dir)

    print("\nMerging complete!")
    print(f"Output files generated:\n- {output_probknot_file}\n- {output_gt_file}")

    # Size diagnostics
    print(f"Size of {output_probknot_file}: {os.path.getsize(output_probknot_file)} bytes")
    print(f"Size of {output_gt_file}: {os.path.getsize(output_gt_file)} bytes")


# Optional standalone CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge per-family encoded files into final training dataset files.")
    parser.add_argument("-d", "--input_dir", required=True, help="Directory containing per-family files.")
    parser.add_argument("-op", "--output_probknot", required=True, help="Output file for merged probknot data.")
    parser.add_argument("-og", "--output_gt", required=True, help="Output file for merged ground truth data.")
    parser.add_argument("-fp", "--families_probknot", nargs="+", required=True, help="List of probknot family filenames.")
    parser.add_argument("-fg", "--families_gt", nargs="+", required=True, help="List of ground truth family filenames.")

    args = parser.parse_args()

    merge_family_files(
        args.families_probknot,
        args.families_gt,
        args.output_probknot,
        args.output_gt,
        args.input_dir
    )
