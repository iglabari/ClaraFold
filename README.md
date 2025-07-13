# ClaraFold: Prediction of RNA secondary structures including pseudoknots using Transformers

🧬 ClaraFold refines existing RNA secondary structure predictions — including pseudoknots — using a Transformer-based model.

---


Accurate prediction of RNA secondary structures, including pseudoknots, is crucial for understanding RNA function. However, traditional methods often fail to capture the long-range base-pairing interactions that define pseudoknots. We introduce ClaraFold, a hybrid framework that refines noisy structure predictions using a transformer-based deep learning model. Our approach encodes RNA sequence and secondary structure information into linear sequences over an extended 28-symbol alphabet, enabling the model to learn folding constraints from data. Trained on diverse structures, ClaraFold corrects errors in base pairing, recovers missing pseudoknots, and removes spurious ones. It outperforms classical thermodynamic methods and generalizes better than recent deep learning models, and scales to RNA molecules exceeding 1,500 nucleotides. We also propose a mathematical measure for comparing RNA secondary structures, particularly effective for assessing pseudoknot accuracy.



---

**ClaraFold** is a Python-based tool for predicting pseudoknots in RNA secondary structures, using a Transformer architecture. This guide provides step-by-step instructions to install and run ClaraFold locally.

---

## Requirements
- Python 3.11 (tested)
- torch==2.0.1
- torchtext==0.15.2
- torchdata==0.6.1
- [Git LFS](https://git-lfs.com/) (for downloading pre-trained model files)

---

## Installation instructions

1. Clone the repository

```bash
git clone https://github.com/iglabari/ClaraFold.git
cd ClaraFold
```

2. We recommend (but do not require) setting up a Python environment, for example:

```bash
conda create -n ClaraFoldEnv python=3.11
conda activate ClaraFoldEnv
```

3. Install the required Python packages (after activating your Conda environment):

```bash
pip install -r requirements.txt
```

### Pre-trained Models

ClaraFold's pre-trained model and vocabulary files are downloaded automatically via Git LFS and are required for prediction. These include:

- model_Transformer.pt – the trained Transformer model
- vocab.pt – the vocabulary file used for input tokenization

4. Download model files with Git LFS

ClaraFold stores large pre-trained model files using [Git Large File Storage (LFS)](https://git-lfs.com/). To retrieve these files:

- Install Git LFS:
  - macOS:

```bash
brew install git-lfs
```

  - Ubuntu/Debian:

```bash
sudo apt update
sudo apt install git-lfs
```



For other systems, see the [Git LFS installation instructions](https://git-lfs.com/).

- Enable and pull the actual model files:

```bash
git lfs install
git lfs pull
```



---

## Running ClaraFold

To test the model, you need to run the Python script in the directory containing the two required \*.pt files, which include the Transformer model and the vocabulary file.



The input file (which can have any name) must follow this format:

<sequence_1>
<secondary_structure_1>
<sequence_2>
<secondary_structure_2>
...



Each sequence and its corresponding structure should be provided on two consecutive lines.



### Example of usage

To run ClaraFold on a single RNA sequence, use the following command in your terminal:

```bash
python ClaraFold.py example_of_one_sequence.txt
```

This command:

- Processes the RNA sequence in the input file example_of_one_sequence.txt
- Refines its secondary structure using the pre-trained Transformer model
- Generates three output files



After running the above example, you will generate output similar to the example below:

```bash
>new_sequence_prediction1
GGGGGCGAUCGGUUUCGACGGUGACCUGUUCGCUGCAGGGAAGCGUGCCGAGAACGCCGGGUCCGCUCGUGGAUGACCCCGGCAAAAGAAUAAGUGCUAAAUCUAACCGCACUGAGUUCGCUCUCGCUGCCUGAUUUUUAGUCAGGAUUAACCAGUGAGCAGCCGCUCCGUUCACUCCUCCUCGUCUUUGGGAGGAUGCUGAGCGUCGUUUAGAAGACUUGCUGAUGCAGUUGAGCCUCAGGGCUGCAUCGGGACUUUAACUGCGGAUAUGCUCGCCAUCAGUUGUCUGCGACAUCGAUGGGGGCAGAAAAAUCGCCAACUUGGCCAGCAGACUACGCACGUAGAAGACUGUGGGUUCGGUCAUCGGACCGGGGUUCAAUUCCCCGCGCCUCCACCA
(((((((.(((....)))((((((((...........(..(((.((..<<<<...((((((((..>>>>.....)).)))))).........(((((...........)))))((((..(((((((((((((((((...)))))))......))))))).))).))))((((((.((((((.(<<<<<<)))))))...)))))).......>>>>>>...((((((((((..<.<<.<<..)))))))))).)))))..)>>.>>.>.(((((.(((((...(<<<<<<...)..))))))))))........((((...))))..>>>>>>(((....)))...............))))))))...(((((.......))))))))))))....

The sequence file was successfully generated and saved as sequence_1.txt
Sequence 1 processed successfully
```





## Output files

After a successful run, ClaraFold will generate:

1. sequence_xx.txt
   - Individual output files for each processed sequence
   - **xx** is a number assigned sequentially (e.g., sequence_1.txt, sequence_2.txt, ...) based on the order of input
   - Generated sequentially during execution
   
2. test_results.txt
   - Contains all predictions made by ClaraFold
   - Comprehensive record of model outputs
3. for_quantify.txt
   - Structured output containing: input sequence, 2D structure, and ClaraFold prediction
   - Generated for each molecule tested by the model
   - Used for model evaluation and analysis



## Training and evaluation (advanced use)

If you would like to train ClaraFold from scratch or evaluate it on your own datasets, please refer to our companion repository: [ClaraFold-training (GitHub)](https://github.com/iglabari/ClaraFold-training.git)

This repository includes:

- Scripts for training a Transformer model on RNA secondary structure data
- Evaluation tools to benchmark predictions against reference structures
- Sample datasets and output files for reproducibility

Please refer to the `README.md` file in the repository for detailed instructions on data preparation, training, and evaluation.

---

## Contact and citation

For questions, issues, or collaboration requests, please contact [garcialabari@cifasis-conicet.gov.ar](mailto:garcialabari@cifasis-conicet.gov.ar)

If you use ClaraFold in your work, please cite our article:

> *ClaraFold: Prediction of RNA secondary structures including pseudoknots using Transformers*. [Preprint or journal info here].
