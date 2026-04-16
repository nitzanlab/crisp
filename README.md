# projint

**projint** is a Python package for quantifying interactions between biological signals in single-cell RNA-seq data using projection-based similarity scoring.

Given two signals — for example, spatial zonation and circadian rhythm — projint measures how much the structure of one signal is reflected in the other, at the level of the whole sample, individual cells, and individual genes.

## Method

For signals A and B and a gene expression matrix X, projint computes:

```
proj_A = X − SIFT(X | A)   # component of X explained by A's graph
proj_B = X − SIFT(X | B)   # component of X explained by B's graph

score = cosine_similarity(proj_A, proj_B)
```

where SIFT ([Nitzan et al., Nature Communications 2024](https://www.nature.com/articles/s41467-024-44758-2)) filters gene expression through the graph defined by signal A (or B). A high score means the two signals share transcriptional structure — i.e., they interact.

Scores are computed at three levels:
- **Global** — a single scalar summarizing the interaction across all cells and genes
- **Cell** — per-cell cosine similarity (which cells drive the interaction)
- **Gene** — per-gene cosine similarity (which genes carry the interaction)

## Installation

```bash
pip install projint
```

The `sift` package is required. Install it from the [nitzanlab/sift](https://github.com/nitzanlab/sift) repository.

For GPU acceleration:

```bash
pip install projint[gpu]
```

## Quick start

```python
import scanpy as sc
import projint

# Load data
adata = sc.read_h5ad("your_data.h5ad")
sc.pp.pca(adata)

# Encode your signals
projint.encode_onehot(adata, "cell_type")          # categorical signal
projint.encode_circular(adata, "cell_cycle_phase") # cyclic signal

# Compute interaction score
score = projint.compute_projection_similarity(
    adata,
    signal_A="onehot_cell_type",
    signal_B="circular_cell_cycle_phase",
)
print(f"Global interaction score: {score:.4f}")

# Per-cell and per-gene scores are written to adata
# adata.obs["proj_sim_cell"]  — cell-level scores
# adata.var["proj_sim_gene"]  — gene-level scores
```

## Signal encoders

| Function | Input | Use case |
|---|---|---|
| `encode_obs` | continuous 1D | pseudotime, position |
| `encode_onehot` | categorical | cell type, condition |
| `encode_circular` | cyclic discrete | cell cycle phase, ZT timepoint |
| `encode_spatial` | spatial coordinates | spatial transcriptomics |

## Tutorials

See the [`tutorials/`](tutorials/) folder for example notebooks:

- [`simulation.ipynb`](tutorials/simulation.ipynb) — Validates the framework on synthetic data with controlled interaction levels

## Citation

If you use projint, please cite:

> Nitzan et al. (2024). *Gene expression cartography*. Nature Communications.
> https://doi.org/10.1038/s41467-024-44758-2

## License

MIT