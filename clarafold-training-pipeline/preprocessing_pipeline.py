# preprocessing_pipeline.py

import os
from pathlib import Path

# Import the pipeline modules
from data_pipeline.preprocessing import (
    subsample_pairs,
    analyze_sequence_lengths,
    probknot_prediction,
    ct2dot_conversion,
    encode_dotbracket,
    generate_training_lists,
    final_merge_families,
    validate_alignment
)


def run_preprocessing_pipeline(config):
    """
    Master pipeline function to execute all preprocessing stages.
    """

    # Stage 1 - Subsampling pairs
    print("\nStage 1: Subsampling .seq/.ct pairs...")
    subsample_pairs.subsample_pairs(
        input_dir=config['subsample_input'],
        output_dir=config['subsample_output'],
        num_pairs=config['subsample_count']
    )

    # Stage 2 - Analyze sequence lengths
    print("\nStage 2: Analyzing sequence lengths...")
    analyze_sequence_lengths.analyze_sequence_lengths(
        input_dir=config['subsample_output']
    )

    # Stage 3 - Run ProbKnot predictions
    print("\nStage 3: Running ProbKnot predictions...")
    probknot_prediction.run_probknot(
        input_dir=config['probknot_input'],
        output_dir=config['probknot_output'],
        probknot_executable=config['probknot_executable']
    )

    # Stage 4 - Convert .ct to dot-bracket
    print("\nStage 4: CT → DotBracket conversion...")
    ct2dot_conversion.convert_ct_to_dotbracket(
        input_dir=config['ct2dot_input'],
        output_dir=config['ct2dot_output'],
        ct2dot_executable=config['ct2dot_executable']
    )

    # Stage 5 - Encode dot-bracket to extended alphabet
    print("\nStage 5: Encoding dotbracket...")
    encode_dotbracket.encode_dotbracket_folder(
        input_dir=config['encode_input'],
        output_dir=config['encode_output']
    )

    # Stage 6 - Generate aligned paired lists
    print("\nStage 6: Generating aligned training lists...")
    generate_training_lists.generate_training_lists(
        probknot_dir=config['probknot_code_dir'],
        gt_dir=config['gt_code_dir'],
        output_file_probknot=config['output_probknot_list'],
        output_file_gt=config['output_gt_list']
    )

    # Stage 7 - Merge families into final training files
    print("\nStage 7: Merging families...")
    final_merge_families.merge_family_files(
        family_probknot_files=config['family_probknot_files'],
        family_gt_files=config['family_gt_files'],
        output_probknot_file=config['merged_probknot_file'],
        output_gt_file=config['merged_gt_file'],
        input_dir=config['families_dir']
    )

    # Stage 8 - Validate alignment of final files
    print("\nStage 8: Validating alignment...")
    validate_alignment.validate_alignment(
        probknot_file=config['merged_probknot_file'],
        gt_file=config['merged_gt_file']
    )


if __name__ == "__main__":

    # ⚠ Customize the config dictionary as needed:

    config = {
        # Stage 1 - Subsampling
        'subsample_input': 'data/cleaned/archiveII',
        'subsample_output': 'data/preprocessed/subsampled',
        'subsample_count': 378,

        # Stage 2 - Sequence length analysis (uses same subsample_output)

        # Stage 3 - ProbKnot predictions
        'probknot_input': 'data/preprocessed/subsampled',
        'probknot_output': 'data/preprocessed/probknot_predictions',
        'probknot_executable': 'ProbKnot',  # full path if needed

        # Stage 4 - ct2dot conversion
        'ct2dot_input': 'data/preprocessed/subsampled',
        'ct2dot_output': 'data/preprocessed/gt_dotbracket',
        'ct2dot_executable': 'ct2dot',  # full path if needed

        # Stage 5 - DotBracket encoding
        'encode_input': 'data/preprocessed/gt_dotbracket',
        'encode_output': 'data/preprocessed/gt_code',

        # Stage 6 - Generate paired lists
        'probknot_code_dir': 'data/preprocessed/probknot_code',
        'gt_code_dir': 'data/preprocessed/gt_code',
        'output_probknot_list': 'data/preprocessed/probknot_family.txt',
        'output_gt_list': 'data/preprocessed/gt_family.txt',

        # Stage 7 - Merge multiple families
        'families_dir': 'data/preprocessed/',
        'family_probknot_files': [
            'probknot_tRNA.txt',
            'probknot_5sRNA.txt',
            'probknot_RNAseP.txt',
            'probknot_tmRNA.txt'
        ],
        'family_gt_files': [
            'gt_tRNA.txt',
            'gt_5sRNA.txt',
            'gt_RNAseP.txt',
            'gt_tmRNA.txt'
        ],
        'merged_probknot_file': 'data/ArchiveII_probknot.txt',
        'merged_gt_file': 'data/ArchiveII_gt.txt',
    }

    run_preprocessing_pipeline(config)