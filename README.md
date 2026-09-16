# crisp

**crisp** is a Python package for quantifying interactions between biological signals in single-cell RNA-seq data using projection-based similarity scoring.

Given two signals, for example spatial zonation and circadian rhythm, crisp measures how much the structure of one signal is reflected in the other, at the level of the whole sample, individual cells, and individual genes.

## Method

For signals A and B and a gene expression matrix X, crisp computes:

```
proj_A = component of X explained by A
proj_B = component of X explained by B
```

For each cell, the interaction score is the Pearson correlation between that cell's `proj_A` and `proj_B` values across genes. A high score means the two signals share transcriptional structure in that cell — i.e., they interact.

Scores are computed at two additional levels, both derived from the per-cell score:
- **Global** : the mean per-cell score across all cells
- **Gene** : each gene's average contribution to the per-cell score, decomposing it across genes

## Installation

```bash
pip install crisp-sc
```

For GPU acceleration:

```bash
pip install crisp-sc[gpu]
```

## Quick start

```python
import scanpy as sc
import crisp

# Load data
adata = sc.read_h5ad("your_data.h5ad")
sc.pp.pca(adata)

# Encode your signals
crisp.encode_onehot(adata, "cell_type")          # categorical signal
crisp.encode_circular(adata, "cell_cycle_phase") # cyclic signal

# Compute interaction score
score = crisp.compute_projection_similarity(
    adata,
    signal_A="onehot_cell_type",
    signal_B="circular_cell_cycle_phase",
)
print(f"Global interaction score: {score:.4f}")

# Per-cell and per-gene scores are written to adata
# adata.obs["proj_pearson_cell"]         — cell-level scores
# adata.var["proj_pearson_gene_contrib"] — gene-level contribution
# adata.uns["proj_mean_cell_pearson"]    — global score (same as the return value)
```

## Signal encoders

| Function | Input | Use case |
|---|---|---|
| `encode_obs` | continuous 1D | pseudotime, position |
| `encode_onehot` | categorical | cell type, condition |
| `encode_circular` | cyclic discrete | cell cycle phase, ZT timepoint |
| `encode_spatial` | spatial coordinates | spatial transcriptomics |

## Reproducibility

`pyproject.toml` lists minimum versions matching the environment analyses were run in (Python 3.11, conda-forge). For an exact, full reproduction of that environment (including transitive dependencies), see [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
```

For the notebooks reproducing the analyses in the paper, see [nitzanlab/crisp_reproducibility](https://github.com/nitzanlab/crisp_reproducibility).

## Citation

If you use crisp, please cite:



> Nitzan et al. (2024). *Gene expression cartography*. Nature Communications.
> https://doi.org/10.1038/s41467-024-44758-2

## License

MIT
