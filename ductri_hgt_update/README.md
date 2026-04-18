# CaiTien_HGT

This repository contains an improved AMDGT variant focused on the heterogeneous graph transformer block.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DucTri2207/CaiTien_HGT/blob/main/notebooks/Train_CaiTien_HGT_on_Colab.ipynb)

The code was prepared so it can run on:
- GPU with CUDA if `torch.cuda.is_available()` is `True`
- CPU automatically if CUDA is not available

## What Changed

- Replaced the original heterogeneous branch with `RLGHGT`
- Added relation-aware local attention
- Added explicit meta-path reasoning
- Added topological / subgraph-aware aggregation
- Added global context and layer-wise aggregation
- Added ablation flags for the HGT components
- Kept the original AMDGT similarity branch and prediction pipeline as much as possible

## Tested Environment

This project was tested with:

- Python `3.9.13`
- PyTorch `1.10.0`
- DGL `0.9.0`
- NumPy `1.23.1`
- scikit-learn `0.24.2`
- networkx `2.8.4`

The repository now supports CPU fallback, so a CUDA GPU is optional for basic reproduction.

## Install

Windows PowerShell:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-py39.txt
```

If PyTorch 1.10 raises a `pkg_resources` import error, run:

```powershell
.\.venv\Scripts\python -m pip install "setuptools<70"
```

## Google Colab

Recommended entrypoint:

- `notebooks/Train_CaiTien_HGT_on_Colab.ipynb`
- Direct link: https://colab.research.google.com/github/DucTri2207/CaiTien_HGT/blob/main/notebooks/Train_CaiTien_HGT_on_Colab.ipynb

The Colab notebook:

- clones this repository automatically
- installs a Colab-friendly dependency set
- tries to install a matching DGL build for the runtime CUDA version
- falls back to CPU automatically if a matching GPU DGL wheel is not selected

Notes for Colab:

- For real training speed, use `Runtime -> Change runtime type -> T4 GPU`.
- If the notebook reports CPU fallback, switch to a Colab runtime whose CUDA version maps cleanly to `cu118`, `cu121`, or `cu124`, then rerun the setup cell.
- The notebook uses the environment variable `AMDGT_DEVICE` so you can force CPU if needed.

## Data

The datasets are already included under `data/`:

- `B-dataset`
- `C-dataset`
- `F-dataset`

Key files used by the pipeline:

- `Drug_mol2vec.csv`
- `DrugFingerprint.csv`
- `DrugGIP.csv`
- `DiseaseFeature.csv`
- `DiseasePS.csv`
- `DiseaseGIP.csv`
- `Protein_ESM.csv`
- `DrugDiseaseAssociationNumber.csv`
- `DrugProteinAssociationNumber.csv`
- `ProteinDiseaseAssociationNumber.csv`

## Run

Basic run:

```powershell
.\.venv\Scripts\python train_DDA.py
```

Quick smoke test:

```powershell
.\.venv\Scripts\python train_DDA.py --dataset C-dataset --epochs 1 --k_fold 2
```

Ablation examples:

```powershell
.\.venv\Scripts\python train_DDA.py --disable_metapath
.\.venv\Scripts\python train_DDA.py --disable_global_hgt
.\.venv\Scripts\python train_DDA.py --disable_topological
.\.venv\Scripts\python train_DDA.py --disable_relation_attention
```

## Important Notes

- The code now creates `data/<dataset>/fold/<i>/` automatically on first run.
- The current training loop still selects the best epoch using the test fold, so the reported metrics can be optimistic. This should be stated honestly in the thesis/report.
- The original AMDGT repository used hardcoded CUDA. This repository no longer does.

## Files

- `data_preprocess.py`: data loading, sampling, fold split, graph construction
- `metric.py`: AUC/AUPR and classification metrics
- `train_DDA.py`: training and evaluation entrypoint
- `model/AMNTDDA.py`: full model pipeline
- `model/rlg_hgt.py`: upgraded heterogeneous graph transformer

## Original Baseline

Original AMDGT repository:

- https://github.com/JK-Liu7/AMDGT

This improved repository was built on top of that baseline with changes concentrated in the HGT branch.
