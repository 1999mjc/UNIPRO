# UNIPRO enables protein inference and panel expansion across single-cell and spatial transcriptomics

UNIPRO is a graph-based framework for inferring protein expression from transcriptomic profiles while explicitly modeling spatial/feature neighborhood relationships. In this human lymph demo, UNIPRO learns from paired RNA-protein measurements in slice A and transfers the trained model to slice B to reconstruct protein profiles.

*Overview of the UNIPRO framework.*
![UNIPRO model figure](</model.png>)

## Overview

Given paired RNA and protein profiles from spatially resolved human lymph tissue, UNIPRO predicts protein expression in a target slice by learning spatially informed transcriptomic representations. The model is designed to preserve local tissue context while allowing edge weights to be adaptively refined during graph attention.

The demo consists of four main stages:

1. **Input preparation**  
   RNA and protein AnnData files are loaded for two human lymph slices. Shared genes are used as transcriptomic input features, and shared proteins are used as supervised prediction targets.

2. **Fused graph construction**  
   For each slice, UNIPRO constructs a directed graph by integrating spatial proximity and RNA expression similarity. Local density is used to balance spatial and expression-derived neighborhood information, and edges supported by both sources receive stronger initial weights.

3. **Edge-adaptive graph learning**  
   The initial fused graph is passed into an edge-aware graph attention network. Before each GAT layer, edge attributes are refined using source node features, target node features, and the previous edge attribute.

4. **Protein prediction and evaluation**  
   The model is trained on slice A and then applied to slice B. Predicted proteins are compared with measured proteins using Pearson correlation, Spearman correlation, and RMSE.

## Installation

UNIPRO is developed with Python and PyTorch. CUDA is used automatically when available;

Recommended Python version:

```text
Python 3.10
```

Install the core packages:

```bash
pip install anndata matplotlib numpy pandas scipy seaborn scikit-learn torch
pip install torch-geometric
```

Recommended package versions:

| Package | Version |
| --- | --- |
| `anndata` | `0.11.4` |
| `matplotlib` | `3.8.4` |
| `numpy` | `1.26.4` |
| `pandas` | `2.2.2` |
| `scipy` | `1.13.1` |
| `seaborn` | `0.13.2` |
| `scikit-learn` | `1.7.1` |
| `torch` | `2.6.0` |
| `torch-geometric` | compatible with the installed PyTorch/CUDA version |

For the remote server used in this project, the recommended environment is:

```bash
conda activate UNIPRO
```

## Human Lymph Demo

The recommended entry point is:

```text
human_lymph_demo.py
```

Run from the project root:

```bash
python human_lymph_demo.py
```

Remote-server example:

```bash
/home/user/miniconda3/envs/UNIPRO/bin/python -u /home/user/Code/UNIPRO/human_lymph_demo.py
```

The demo performs the complete experiment:

1. loads paired RNA and protein files for human lymph slices A and B;
2. selects shared genes and shared proteins;
3. constructs fused spatial-expression graphs for both slices;
4. trains the edge-adaptive GAT model on slice A;
5. predicts protein expression on slice B;
6. saves predictions, ground truth, metrics, plots, and the best model checkpoint.

## Inputs

The script reads paired RNA and protein `.h5ad` files defined in `PATHS` inside `human_lymph_demo.py`.

| Slice | Modality | Location |
| --- | --- | --- |
| A | RNA | `data/human_lymph/GSM8195494_A1_LN.h5ad` |
| A | Protein | `data/human_lymph/GSM8195498_A1_LN_Protein.h5ad` |
| B | RNA | `data/human_lymph/GSM8195496_D1_LN.h5ad` |
| B | Protein | `data/human_lymph/GSM8195500_D1_LN_Protein.h5ad` |

The AnnData objects should contain:

| Location | Description |
| --- | --- |
| `adata.X` | RNA or protein expression matrix |
| `adata.var_names` | gene or protein names |
| `adata.obs_names` | spot barcodes |
| `adata.obsm["spatial"]` | spatial coordinates for graph construction |
| `adata.var["highly_variable"]` | optional highly variable gene annotation |

## Outputs

The workflow writes reproducible artifacts to the following locations:

| Output | Location |
| --- | --- |
| Predicted protein matrix for slice B | `results_demo/predicted_protein_B.csv` |
| Measured protein matrix for slice B | `results_demo/true_protein_B.csv` |
| Protein-level evaluation metrics | `results_demo/evaluation_metrics_B.csv` |
| Metric boxplot | `results_demo/metrics_combined_A_to_Be.png` |
| Best model checkpoint | `weights_dir_demo/best_model_A_to_B.pt` |

## Evaluation

UNIPRO reports three complementary metrics on the shared proteins measured in slice B:

| Metric | Interpretation | Preferred direction |
| --- | --- | --- |
| Pearson correlation | Linear association between predicted and measured protein expression | Higher is better |
| Spearman correlation | Rank-based association between predicted and measured protein expression | Higher is better |
| RMSE | Magnitude of prediction error | Lower is better |

The metric table is saved to:

```text
results_demo/evaluation_metrics_B_fused_edge.csv
```

The metric plot is saved to:

```text
results_demo/metrics_combined_A_to_B_fused_edge.png
```

## datasets
The datasets used by UNIPRO and main results can be download from https://zenodo.org/uploads/21290508.

