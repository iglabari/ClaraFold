# data_pipeline/preprocessing/validate_alignment.py

from pathlib import Path

def validate_alignment(probknot_file, gt_file):
    """
    Verifies that:
    - Both files have same number of lines.
    - Each corresponding pair of lines has same sequence length.
    Prints summary statistics after verification.

    Args:
        probknot_file (str): Path to ArchiveII_probknot.txt
        gt_file (str): Path to ArchiveII_gt.txt
    """

    # Read both files
    with open(probknot_file, 'r') as f_probknot, open(gt_file, 'r') as f_gt:
        lines_probknot = f_probknot.readlines()
        lines_gt = f_gt.readlines()

    # Check total number of lines match
    assert len(lines_probknot) == len(lines_gt), (
        f"Files have different number of lines: "
        f"probknot({len(lines_probknot)}) vs gt({len(lines_gt)})"
    )

    # Check length of each corresponding line
    for i, (line_prob, line_gt) in enumerate(zip(lines_probknot, lines_gt), 1):
        seq_prob = line_prob.strip()
        seq_gt = line_gt.strip()

        if len(seq_prob) != len(seq_gt):
            print(f"Error at line {i}: Length mismatch")
            print(f"Probknot: {len(seq_prob)} | GT: {len(seq_gt)}")
            print(f"Probknot first 50 chars: {seq_prob[:50]}")
            print(f"GT first 50 chars: {seq_gt[:50]}")
            raise ValueError(f"Mismatch found at line {i}")

    print("✅ Validation completed successfully: all lines are aligned.")

    # Statistics
    total_lines = len(lines_probknot)
    lengths = [len(line.strip()) for line in lines_probknot]
    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = sum(lengths) / total_lines

    print(f"\nAlignment statistics:")
    print(f"- Total lines: {total_lines}")
    print(f"- Min length: {min_len}")
    print(f"- Max length: {max_len}")
    print(f"- Average length: {avg_len:.2f}")


# Optional standalone CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate alignment of ArchiveII_probknot.txt and ArchiveII_gt.txt")
    parser.add_argument("-p", "--probknot_file", required=True, help="Path to ArchiveII_probknot.txt")
    parser.add_argument("-g", "--gt_file", required=True, help="Path to ArchiveII_gt.txt")

    args = parser.parse_args()

    validate_alignment(args.probknot_file, args.gt_file)