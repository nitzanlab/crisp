"""
projint.core
============
Core computation: projection-based interaction scoring between two biological signals.

For signals A and B, the projection of gene expression X onto signal S is:
    proj_S = X - SIFT(X | S)
i.e. the component of X that is structurally explained by S's graph.

The interaction score is the cosine similarity between proj_A and proj_B,
measured at three levels:
  - Global : single scalar over all cells and genes
  - Cell   : per-cell cosine similarity over the gene axis
  - Gene   : per-gene cosine similarity over the cell axis
"""

import numpy as np
import scipy.sparse as sp
import scanpy as sc
try:
    from sift import sifter as _sifter
except ImportError:
    from sift._sift import sifter as _sifter


def compute_projection_similarity(
    adata,
    signal_A,
    signal_B,
    n_neighbors=None,
    embedding_key="X",
    metric="knn",
    device="cpu",
):
    """
    Compute projection-based interaction scores between two signals.

    Both signals must already be encoded in adata.obsm (use projint.encode.*).

    Parameters
    ----------
    adata : AnnData
        Single sample. Gene expression in adata.X, encoded signals in adata.obsm.
    signal_A : str
        Key in adata.obsm for the first signal (the one being "sifted out").
    signal_B : str
        Key in adata.obsm for the second signal.
    n_neighbors : int, optional
        Number of neighbors for the KNN graph used by SIFT. Defaults to sqrt(n_cells).
    embedding_key : str
        Embedding used to build the SIFT graph. 'X' uses raw gene expression,
        'X_pca' uses PCA (computed internally if not present).
    metric : str
        Distance metric for SIFT's KNN graph.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    global_score : float
        Cosine similarity between proj_A and proj_B over all cells and genes.

    Side effects
    ------------
    Writes to adata.obs:
        'proj_sim_cell'       — per-cell cosine similarity (cell interaction score)
        'proj_sim_cell_mag_A' — ||proj_A|| per cell (signal A magnitude)
        'proj_sim_cell_mag_B' — ||proj_B|| per cell (signal B magnitude)
    Writes to adata.var:
        'proj_sim_gene'       — per-gene cosine similarity (gene interaction score)
        'proj_sim_gene_mag_A' — ||proj_A|| per gene
        'proj_sim_gene_mag_B' — ||proj_B|| per gene
    Writes to adata.uns:
        'proj_sim_global'     — global score (same as return value)
    """
    bdata = adata.copy()

    if n_neighbors is None:
        n_neighbors = int(np.sqrt(bdata.n_obs))

    # PCA needed for SIFT's graph building regardless of embedding_key
    sc.pp.pca(bdata)
    sc.pp.neighbors(bdata, n_neighbors=n_neighbors, use_rep="X_pca")

    # Ensure signals are in obsm (should already be after encoding)
    _ensure_obsm(bdata, signal_A)
    _ensure_obsm(bdata, signal_B)

    # Densify X if needed (sift package does not handle sparse output correctly)
    if embedding_key == "X" and sp.issparse(bdata.X):
        bdata.X = bdata.X.toarray()

    X_original = np.array(bdata.X)

    # --- Projection for signal A: proj_A = X - SIFT(X | A) ---
    bdata_A = bdata.copy()
    _sifter(
        adata=bdata_A,
        kernel_key=signal_A,
        embedding_key=embedding_key,
        metric=metric,
        n_neighbors=n_neighbors,
        copy=False,
        device=device,
    )
    X_sifted_A = bdata_A.X.toarray() if sp.issparse(bdata_A.X) else np.array(bdata_A.X)
    proj_A = X_original - X_sifted_A

    # --- Projection for signal B: proj_B = X - SIFT(X | B) ---
    bdata_B = bdata.copy()
    _sifter(
        adata=bdata_B,
        kernel_key=signal_B,
        embedding_key=embedding_key,
        metric=metric,
        n_neighbors=n_neighbors,
        copy=False,
        device=device,
    )
    X_sifted_B = bdata_B.X.toarray() if sp.issparse(bdata_B.X) else np.array(bdata_B.X)
    proj_B = X_original - X_sifted_B

    # --- Global score ---
    denom_global = np.linalg.norm(proj_A) * np.linalg.norm(proj_B)
    global_score = float(np.dot(proj_A.ravel(), proj_B.ravel()) / (denom_global + 1e-12))

    # --- Cell scores: cosine similarity over genes, per cell ---
    num_cell = np.sum(proj_A * proj_B, axis=1)
    denom_cell = np.linalg.norm(proj_A, axis=1) * np.linalg.norm(proj_B, axis=1)
    cell_scores = num_cell / np.where(denom_cell == 0, 1, denom_cell)

    # --- Gene scores: cosine similarity over cells, per gene ---
    num_gene = np.sum(proj_A * proj_B, axis=0)
    denom_gene = np.linalg.norm(proj_A, axis=0) * np.linalg.norm(proj_B, axis=0)
    gene_scores = num_gene / np.where(denom_gene == 0, 1, denom_gene)

    # --- Write results ---
    adata.obs["proj_sim_cell"] = cell_scores
    adata.obs["proj_sim_cell_mag_A"] = np.linalg.norm(proj_A, axis=1)
    adata.obs["proj_sim_cell_mag_B"] = np.linalg.norm(proj_B, axis=1)

    adata.var["proj_sim_gene"] = gene_scores
    adata.var["proj_sim_gene_mag_A"] = np.linalg.norm(proj_A, axis=0)
    adata.var["proj_sim_gene_mag_B"] = np.linalg.norm(proj_B, axis=0)

    adata.uns["proj_sim_global"] = global_score

    print(f"Global projection similarity ({signal_A} vs {signal_B}): {global_score:.4f}")
    return global_score


def _ensure_obsm(adata, key):
    """Move a signal from obs to obsm if it isn't already in obsm."""
    if key not in adata.obsm and key in adata.obs:
        signal = adata.obs[key].values.astype(float)
        norm = np.linalg.norm(signal)
        adata.obsm[key] = (signal / norm if norm > 0 else signal).reshape(-1, 1)
