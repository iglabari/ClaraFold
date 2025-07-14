# ClaraFold data cleaning and preprocessing pipeline

This repository provides the **data preparation pipeline** used for training ClaraFold — a transformer-based framework for RNA secondary structure prediction, including pseudoknots.

---

## Pipeline overview

The ClaraFold pipeline prepares RNA secondary structure data for training by combining two main stages:

### Cleaning Stage
Removes low-quality or redundant data:
1. **Remove N from `.ct` files**: Filters out `.ct` files where column 2 contains ambiguous nucleotides (`N`).
2. **Remove N from `.seq` files**: Filters out `.seq` files where the RNA sequence contains any `N`.
3. **Remove duplicate sequences**: Detects and removes duplicate `.seq` entries.
4. **Subunit filtering**: Identifies sequences that are subunits of longer molecules and separates them.

### Preprocessing Stage
Transforms clean RNA data into inputs suitable for ClaraFold's transformer-based model:
1. **Subsample .ct/.seq pairs** from each family for training.
2. **Analyze sequence lengths** to understand data distribution.
3. **Predict secondary structures** using the ProbKnot tool.
4. **Convert `.ct` to dot-bracket** notation.
5. **Encode sequences** into ClaraFold's 28-symbol extended alphabet.
6. **Generate aligned training lists** for model input/output.
7. **Merge families** into unified training files.
8. **Validate alignment** of input/output pairs.

> Note: Steps 3 and 4 require external tools from the RNAstructure suite (`ProbKnot` and `ct2dot`).

---

## Directory Structure

```bash
clarafold-training-pipeline/
├── data
│   ├── ArchiveII_gt.txt
│   ├── ArchiveII_probknot.txt
│   ├── cleaned/
│   └── raw/
│       ├── archiveII/
│       └── RNAStrAlign/
├── data_pipeline/
│   ├── cleaning/
│   │   ├── remove_duplicate_sequences.py
│   │   ├── remove_n_in_seq.py
│   │   ├── remove_n.py
│   │   └── subunit_filter.py
│   ├── preprocessing/
│   │   ├── subsample_pairs.py
│   │   ├── analyze_sequence_lengths.py
│   │   ├── probknot_prediction.py
│   │   ├── ct2dot_conversion.py
│   │   ├── encode_dotbracket.py
│   │   ├── generate_training_lists.py
│   │   ├── final_merge_families.py
│   │   └── validate_alignment.py
│   └── README.md
├── notebook/
│   └── ClaraFoldTraining.ipynb
├── cleaning_pipeline.py
├── preprocessing_pipeline.py
└── README.md
```

---

## Running the Cleaning Pipeline

1. **Prepare your input data**  
   Place your datasets inside the `data/raw/` directory. The pipeline expects `.seq` and `.ct` files.

2. **Install requirements**  
   Only Python standard libraries are used — no external dependencies required.

3. **Run the pipeline**
   ```bash
   python cleaning_pipeline.py -i <input_dir> -o <output_dir>
   ```
   Example:
   ```bash
   python cleaning_pipeline.py -i data/raw/archiveII -o data/cleaned/archiveII
   ```

4. **Results**  
   The cleaned data will appear under:
   ```bash
   data/cleaned/
   ```
   Intermediate folders:
   - `stage1_remove_n/`
   - `stage2_remove_n_in_seq/`
   - `stage3_remove_duplicates/`
   - `stage4_subunit_filter/`

---

## Running the Preprocessing Pipeline

1. **Customize the configuration** in `preprocessing_pipeline.py`. Update the `config` dictionary at the bottom to match your paths and preferences.

2. **Run the pipeline**
   ```bash
   python preprocessing_pipeline.py
   ```

3. **Results**
   Intermediate outputs will be stored under:
   ```bash
   data/preprocessed/
   ```

   Including:
   - Subsampled pairs
   - ProbKnot predictions
   - Dot-bracket files
   - Encoded sequences
   - Final training lists


---

## Related Repository

> The core ClaraFold inference framework is available here: https://github.com/iglabari/ClaraFold

---

## Contact

For questions, suggestions, or collaborations:

- [garcialabari@cifasis-conicet.gov.ar](mailto:garcialabari@cifasis-conicet.gov.ar)
- [alejandrofloreslamas@cinvestav.mx](mailto:alejandrofloreslamas@cinvestav.mx)
