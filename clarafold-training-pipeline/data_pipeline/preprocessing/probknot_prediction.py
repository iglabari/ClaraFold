# data_pipeline/preprocessing/probknot_prediction.py

import os
import subprocess
from pathlib import Path

def construct_command(probknot_executable, seq_file, ct_predicted_file):
    """
    Constructs the command to run ProbKnot.

    Args:
        probknot_executable (str): Path to the ProbKnot executable.
        seq_file (str): Input .seq file.
        ct_predicted_file (str): Output .ct file.
    """
    command = [
        probknot_executable,
        seq_file,
        ct_predicted_file,
        "--sequence"
    ]
    return command

def run_probknot_on_folder(input_dir, output_dir, probknot_executable="ProbKnot"):
    """
    Runs ProbKnot predictions for all .seq files in input_dir.

    Args:
        input_dir (str): Directory containing .seq files.
        output_dir (str): Directory where output .ct files will be stored.
        probknot_executable (str): Path to the ProbKnot executable (default assumes it's in PATH).
    """
    os.makedirs(output_dir, exist_ok=True)
    input_dir = Path(input_dir)

    seq_files = list(input_dir.glob("*.seq"))
    if not seq_files:
        print("No .seq files found in input directory.")
        return

    for seq_path in seq_files:
        seq_name = seq_path.stem
        ct_output_path = Path(output_dir) / f"{seq_name}_probknot.ct"

        command = construct_command(
            probknot_executable,
            str(seq_path),
            str(ct_output_path)
        )

        print(f"Running: {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing ProbKnot for {seq_path.name}: {e}")

    print("ProbKnot predictions completed.")


# Optional standalone CLI usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ProbKnot predictions for a set of .seq files.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing .seq files.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to store output .ct files.")
    parser.add_argument("-p", "--probknot_path", default="ProbKnot", help="Path to the ProbKnot executable (default assumes it's in PATH).")

    args = parser.parse_args()

    run_probknot_on_folder(args.input_dir, args.output_dir, args.probknot_path)

