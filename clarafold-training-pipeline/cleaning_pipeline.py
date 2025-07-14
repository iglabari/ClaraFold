# cleaning_pipeline.py

import os
import argparse
from data_pipeline.cleaning import (
    remove_n,
    remove_n_in_seq,
    remove_duplicate_sequences,
    subunit_filter
)

def run_cleaning_pipeline(input_dir, output_dir):
    """
    Orchestrates the full cleaning pipeline in 4 stages.
    Each stage operates on the output of the previous stage.
    """

    # Stage 1: Remove N from .ct files
    stage1_dir = os.path.join(output_dir, "stage1_remove_n")
    os.makedirs(stage1_dir, exist_ok=True)
    remove_n.process(input_dir, stage1_dir)

    # Stage 2: Remove N from .seq files
    stage2_dir = os.path.join(output_dir, "stage2_remove_n_in_seq")
    os.makedirs(stage2_dir, exist_ok=True)
    remove_n_in_seq.process(stage1_dir, stage2_dir)

    # Stage 3: Remove duplicate sequences
    stage3_dir = os.path.join(output_dir, "stage3_remove_duplicates")
    os.makedirs(stage3_dir, exist_ok=True)
    remove_duplicate_sequences.process(stage2_dir, stage3_dir)

    # Stage 4: Subunit filtering
    stage4_dir = os.path.join(output_dir, "stage4_subunit_filter")
    os.makedirs(stage4_dir, exist_ok=True)
    subunit_filter.process(stage3_dir, stage4_dir)

    print("Full cleaning pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClaraFold Data Cleaning Pipeline")
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Path to the input dataset directory (raw data)")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Path to the output directory for cleaned data")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    run_cleaning_pipeline(input_dir, output_dir)