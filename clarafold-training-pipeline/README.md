# ClaraFold: Prediction of RNA secondary structures including pseudoknots using Transformers (training pipeline)

---

## Project Description

**ClaraFold** is a transformer-based framework for accurate RNA secondary structure prediction, including pseudoknots. The method refines noisy initial predictions by integrating RNA primary sequence and secondary structure information into an expanded 28-symbol alphabet. Using a custom tokenization scheme, ClaraFold enables the transformer model to learn complex RNA folding constraints from data.

Key features include:

- Correction of erroneous base pair predictions
- Recovery of missing pseudoknots
- Removal of spurious structural artifacts
- Scalability to RNA sequences exceeding 1,500 nucleotides
- Outperformance classical thermodynamic methods and generalizes better than recent deep learning models
- Integration of a novel structural dissimilarity metric for evaluation

The full training pipeline implemented here covers data preparation, tokenization, vocabulary construction, model training (single and distributed), checkpoint management, and model evaluation.

This repository provides a fully documented, reproducible pipeline developed for both research and publication purposes.

---


## Related Repository

> The core ClaraFold inference framework is available here:
> https://github.com/iglabari/ClaraFold

This repository (current one) focuses on the full training pipeline for generating a transformer model that can be used with the ClaraFold inference system.

---

## Repository Structure

- `notebook/`
  - `ClaraFoldTraining.ipynb`   # Full Colab notebook

- `data/`   # Provided in the repository. To be located in the user's Google Drive for training when using the Colab notebook.
  - `ArchiveII_gt.txt`

  - `ArchiveII_probknot.txt`

- `README.md`

- `.gitignore`

---

## Reproducibility

- The full training pipeline is implemented in **Google Colab**.
- GPU acceleration is required (use `Runtime → Change runtime type → GPU`).
- All package installations and environment checks are included in the notebook.
- Training checkpoints and vocabularies are automatically saved to disk (or to Google Drive if mounted).

---

## Notes

- The Colab notebook can be executed end-to-end to reproduce the full experiment.
- Make sure to mount Google Drive if you wish to persist output files across sessions.

---

## Version

- **Current version:** `v1.0`

---

## Contact

For questions, suggestions, or collaborations:

- [garcialabari@cifasis-conicet.gov.ar](mailto:garcialabari@cifasis-conicet.gov.ar)
- [alejandrofloreslamas@cinvestav.mx](mailto:alejandrofloreslamas@cinvestav.mx)

