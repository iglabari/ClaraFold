# data_pipeline/preprocessing/encode_dotbracket.py

import os
from pathlib import Path

# Full mapping dictionary (combining nucleotides and structure symbols)
mapeo_original = {
    'A(': 'Q', 'C(': 'W', 'G(': 'R', 'U(': 'Y',
    'A.': 'I', 'C.': 'V', 'G.': 'P', 'U.': 'S',
    'A)': 'H', 'C)': 'J', 'G)': 'K', 'U)': 'L',
    'A<': '1', 'C<': '2', 'G<': '3', 'U<': '4',
    'A>': '5', 'C>': '6', 'G>': '7', 'U>': '8',
    'A{': 'D', 'C{': 'F', 'G{': 'Z', 'U{': 'X',
    'A}': 'B', 'C}': 'M', 'G}': 'O', 'U}': '0',
}

# Case-insensitive support (e.g. a( => Q as well)
mapeo = {**mapeo_original, **{k[0].lower() + k[1]: v for k, v in mapeo_original.items()}}

def parse_dotbracket_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    sequence_identifier = lines[0].strip()[1:]
    ct2dot_identifier = lines[1].strip()[1:]
    sequence = lines[2].strip()
    dot_bracket_result = ''.join(lines[3:]).strip()

    return sequence_identifier, ct2dot_identifier, sequence, dot_bracket_result

def concatenate_inputs(seq, dot_bracket):
    if len(seq) != len(dot_bracket):
        raise ValueError("Sequence and dot-bracket strings have different lengths.")

    return [a + b for a, b in zip(seq, dot_bracket)]

def encode_concatenation(concatenated_list):
    encoded = ""
    for elem in concatenated_list:
        encoded += mapeo.get(elem, elem)  # fallback: keep char if no mapping
    return encoded

def encode_dotbracket_folder(input_dir, output_dir):
    """
    Encodes all .dotbracket files in input_dir into extended alphabet .code files.

    Args:
        input_dir (str): Directory containing .dotbracket files.
        output_dir (str): Directory where .code files will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_dir = Path(input_dir)

    dotbracket_files = list(input_dir.glob("*.dotbracket"))
    if not dotbracket_files:
        print("No .dotbracket files found.")
        return

    for dot_path in dotbracket_files:
        (
            sequence_identifier,
            ct2dot_identifier,
            sequence,
            dot_bracket_result
        ) = parse_dotbracket_file(dot_path)

        concatenated = concatenate_inputs(sequence, dot_bracket_result)
        encoded_sequence = encode_concatenation(concatenated)

        output_path = Path(output_dir) / (dot_path.stem + ".code")
        with open(output_path, 'w') as f:
            f.write(f"sequence_identifier: {sequence_identifier}\n")
            f.write(f"ct2dot_identifier: {ct2dot_identifier}\n")
            f.write(f"length: {len(sequence)}\n")
            f.write("sequence:\n")
            f.write(sequence + "\n")
            f.write("dot_bracket_result:\n")
            f.write(dot_bracket_result + "\n")
            f.write("concatenation:\n")
            f.write(''.join(concatenated) + "\n")
            f.write("coding_1D+2D:\n")
            f.write(encoded_sequence + "\n")

        print(f"Generated: {output_path.name}")

    print("Dotbracket encoding completed.")


# Optional CLI interface for standalone usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encode .dotbracket files into .code files using extended alphabet.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing .dotbracket files.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to store .code files.")

    args = parser.parse_args()

    encode_dotbracket_folder(args.input_dir, args.output_dir)