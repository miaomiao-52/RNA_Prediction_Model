# RNA_Secondary Structure_Prediction_Model

A Sparse Multi-Relational Graph Attention Network for Resolving Local RNA Conformational Ensembles from Chemical Probing Features

This repository contains thw Python analysis tools used in the study:

**"RNA-Quadraduplex and Stem-Loop Prediction Based on SMR-GAT"**

---

# 1. System Requirements

## Hardware

* Multi-core CPU recommended (10+ cores recommended for batch simulations)
* 50 GB RAM or higher recommended for large-scale simulations

---

## Software Dependencies

### Environment Setup

Tested with **Python 3.10** and **ViennaRNA >= 2.5**.

```bash
# 1. Create and activate environment
conda create -n bio_env python=3.10 -y
conda activate bio_env

# 2. Install ViennaRNA via Bioconda
conda install -c bioconda viennarna -y

# 3. Install required Python packages
pip install pandas importlib-metadata torch biopython tqdm scikit-learn numpy seaborn matplotlib
pip install rna-fm
```

---

# 2. Directory Structure

```RNA-DynamicNet/
├── data_processing/
│   ├── getHeLa.py
│   ├── make_new_fa.py
│   ├── RASP_GTF_fa_vivo_pds.py
│   ├── BERT_MLP_vivo_pds.py
│   └── BERT_MLP_whole.py
├── dataset_building/
│   ├── split_rG4_fa.py
│   ├── rG4+_SL+_label.py
│   ├── rG4-_SL+_label.py
│   ├── rG4+_SL-_label.py
│   ├── rG4-_SL-_label.py
│   └── merge_rG4_SL_neg.py
├── models/
    └── 0823_prediction_model.py
```

# 3. Core Algorithmic Workflow

The complete data-processing and model-training pipeline consists of five major stages:

1. **Reference Transcript Preparation (`getHeLa.py`, `make_new_fa.py`)**

   * Filters expressed transcripts from the whole-transcriptome TPM table
   * Generates the cell-line-specific transcript FASTA file used throughout the pipeline

2. **Transcriptome-wide Reactivity Imputation (`RASP_GTF_fa_vivo_pds.py`, `BERT_MLP_whole.py`)**

   * Maps RASP reactivity scores from genomic coordinates to transcript coordinates
   * Predicts missing nucleotide reactivity values using the RNA-FM + MLP model
   * Produces complete transcript-level reactivity matrices

3. **Genome-wide Candidate Region Extraction (`split_rG4_fa.py`)**

   * Integrates G4Atlas annotations with transcript sequences and reactivity profiles
   * Splits the transcriptome into potential rG4-positive and rG4-negative sequence sets
   * Generates candidate regions for downstream structural annotation

4. **Multi-label Dataset Construction (`rG4±_SL±_label.py`)**

   * Detects stem-loop (SL) structures using RNA secondary structure prediction
   * Assigns four structural categories:
     * rG4 + SL (Co-existing)
     * rG4 only
     * SL only
     * Neither structure
   * Merges all annotated regions into a unified four-class training dataset

5. **Deep Learning Model Training (`0823_prediction_model.py`)**

   * Combines RNA sequences, transcript-level reactivity profiles, and structural annotations
   * Trains the multi-modal neural network for local RNA structural state prediction
   * Produces the final optimized classification model (`best_model_0823.pt`)

---

# 4. File Descriptions

| File | Description |
| --------------------------- | ----------------------------------------------------------------------------------- |
| `Data_Process.sh`           | Data source and download                                                            |
| `getHeLa.py`                | Filters high-expression gene whitelist for HeLa cell line from HPA matrix           |
| `make_new_fa.py`            | Builds HeLa-specific cDNA FASTA reference sequence file                             |
| `RASP_GTF_fa_vivo_pds.py`   | Maps experimental reactivity scores to cDNA transcript relative coordinates         |
| `BERT_MLP_vivo_pds.py`      | Imputes missing reactivity scores for training datasets using RNA-FM embeddings     |
| `BERT_MLP_whole.py`         | Performs whole-transcriptome streaming reactivity score predictions (`.npy` output)|
| `split_rG4_fa.py`           | Splits transcriptome into rG4 potential and negative FASTA via Random Forest       |
| `rG4+_SL+_label.py`         | Mines rG4 and Stem-Loop co-existing candidates via PQS regex and ViennaRNA MFE      |
| `rG4-_SL+_label.py`         | Mines pure Stem-Loop structures from rG4 negative background FASTA                  |
| `rG4+_SL-_label.py`         | Decouples co-existing and pure rG4 labels using G4Atlas interval overlap            |
| `rG4-_SL-_label.py`         | Generates neutral background negative controls via greedy non-overlapping sampling  |
| `merge_rG4_SL_neg.py`       | Integrates all four sub-label CSV files into a unified dataset index                |
| `0823_prediction_model.py` | Core multi-modal classifier training script (Sparse R-GAT + BiLSTM)                |
| `Run.sh`                   | Training pipeline launcher                                                          |

---

# 5. Quick Start

## Step 1: Reference Transcript Preparation (`getHeLa.py`, `make_new_fa.py`)

- Filters expressed transcripts from the TPM table
- Generates the HeLa-specific transcript FASTA (`HeLa.fa`)

## Step 2: Transcriptome-wide Reactivity Imputation (`RASP_GTF_fa_vivo_pds.py`, `BERT_MLP_whole.py`)

- Maps RASP reactivity scores onto transcript coordinates
- Predicts missing reactivity values using RNA-FM + MLP
- Produces complete nucleotide-level reactivity matrices (`*.npy`)

## Step 3: Genome-wide Candidate Region Extraction (`split_rG4_fa.py`)

- Integrates G4Atlas annotations with transcript sequences
- Splits transcripts into rG4-positive and rG4-negative candidate sets

## Step 4: Multi-label Dataset Construction (`rG4±_SL±_label.py`)

- Predicts stem-loop structures
- Generates four structural labels:
  - rG4 + SL
  - rG4 only
  - SL only
  - Neither structure
- Merges all labeled samples into the final training dataset

## Step 5: Deep Learning Model Training (`0823_prediction_model.py`)

- Uses sequence, reactivity, and structural features as input
- Trains the four-class prediction model
- Outputs the optimized model checkpoint (`best_model_0823.pt`)

---

# 6. Reproducing Manuscript Results

## Symmetry-Breaking Analysis

After simulations:

* Analyze `output_prediction.log`

---

# 7. Typical Runtime

Typical runtime on a standard workstation:

| Task                             | Runtime               |
| -------------------------------- | --------------------- |
| Initial configuration generation | Seconds to minutes    |
| Data Processing and Manipulation | ~5 days               |
| Training Pipeline                | Several hours         |

Runtime depends on:

* Number of Epoches
* Batch Size
* CPU resources

---

# 8. Reproducibility Notes

For reproducibility:

* All simulations in the manuscript were generated using the scripts provided in this repository
* Independent replicas use different random seeds
* No manual intervention is required after launching `Run.sh`

Before re-running simulations, remove old output files to avoid mixing previous results with new trajectories.

---

# 9. License

This project is distributed under the MIT License.

---

# 10. Citation

If you use this code in your research, please cite:

> Zhiru Zheng and Jiansong Yang,
> "A Sparse Multi-Relational Graph Attention Network for Resolving Local RNA Conformational Ensembles from Chemical Probing Features" (2026).
