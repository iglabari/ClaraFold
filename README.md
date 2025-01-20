# ClaraFold
A novel transformer-based approach designed to refine noisy RNA secondary structure predictions.

# Model Execution
To test the model, you need to run the Python script in the directory containing the four required *.pt files, which include the transformer models (T1 and T2) and the vocabulary files for input (src) and output (tgt).

# Running the Model
Execute the following command in your terminal:
python complete_process.py input.txt

The input file (which can have any name) must follow this specific format:

sequence
secondary_structure

For multiple molecules, use this format:

sequence1
secondary_structure1
sequence2
secondary_structure2
sequence3
secondary_structure3
...

Each sequence must be immediately followed by its corresponding secondary structure on the next line. There should be no empty lines between entries.

# Input Examples
Two sample input files are provided:
example_of_one_sequence.txt: Contains a single molecule
example_of_four_sequences.txt: Contains four molecules

# Execution Process
During execution, the program will display all ongoing operations in real-time. The process generates three output files:

1-sequence_xx.txt
*Individual output files for each processed sequence
*Generated sequentially during execution

2-test_results.txt
*Contains all predictions made by ClaraFold
*Comprehensive record of model outputs

3-for_quantify.txt
*Structured output containing: Input sequence; 2D structure ; ClaraFold prediction
*Generated for each molecule tested by the model
*Used for model evaluation and analysis

# File Structure

.
├── *.pt                           # Model and vocabulary files
├── complete_process.py            # Main execution script
├── example_of_one_sequence.txt    # Single molecule example
├── example_of_four_sequences.txt  # Multiple molecule example
└── output/
    ├── sequence_xx.txt           # Individual predictions
    ├── test_results.txt         # All predictions
    └── for_quantify.txt         # Structured output for analysis
