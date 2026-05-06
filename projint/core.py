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
import scipy.spatial.distance as sdist
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
    store_projections=False,
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
    store_projections : bool
        If True, store proj_A and proj_B as adata.layers['proj_A'] and adata.layers['proj_B'].

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
        'proj_l2_gene'        — per-gene L2 distance ||proj_A - proj_B|| (lower = more interaction)
        'proj_dot_scaled_gene' — Frobenius-normalized dot product r_A·r_B·cosine (jointly rewards both signals)
        'proj_spearman_gene'   — Spearman correlation between proj_A and proj_B per gene (rank-based, scale-invariant)

    Writes to adata.uns:
        'proj_sim_global'     — global score (same as return value)
    """
    bdata = adata.copy()
    import time
    t0 = time.time()

    if n_neighbors is None:
        n_neighbors = int(np.sqrt(bdata.n_obs))

    # PCA needed for SIFT's graph building regardless of embedding_key
    sc.pp.pca(bdata)
    sc.pp.neighbors(bdata, n_neighbors=n_neighbors, use_rep="X_pca")
    print(f"[timing] PCA + neighbors:   {time.time()-t0:.2f}s"); t0 = time.time()

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
    print(f"[timing] SIFT signal A:     {time.time()-t0:.2f}s"); t0 = time.time()

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
    print(f"[timing] SIFT signal B:     {time.time()-t0:.2f}s"); t0 = time.time()

    # --- Global score (cosine) ---
    denom_global = np.linalg.norm(proj_A) * np.linalg.norm(proj_B)
    global_score = float(np.dot(proj_A.ravel(), proj_B.ravel()) / (denom_global + 1e-12))

    # Global Spearman is computed as mean of per-cell scores after cell_spearman is available.

    # --- Cell scores: cosine similarity over genes, per cell ---
    num_cell = np.sum(proj_A * proj_B, axis=1)
    denom_cell = np.linalg.norm(proj_A, axis=1) * np.linalg.norm(proj_B, axis=1)
    cell_scores = num_cell / np.where(denom_cell == 0, 1, denom_cell)
    print(f"[timing] cell cosine:       {time.time()-t0:.2f}s"); t0 = time.time()

    # --- Cell scores: Spearman correlation over genes, per cell ---
    ranks_A_cell = np.argsort(np.argsort(proj_A, axis=1), axis=1).astype(float)
    ranks_B_cell = np.argsort(np.argsort(proj_B, axis=1), axis=1).astype(float)
    ranks_A_cell_c = ranks_A_cell - ranks_A_cell.mean(axis=1, keepdims=True)
    ranks_B_cell_c = ranks_B_cell - ranks_B_cell.mean(axis=1, keepdims=True)
    num_sp_cell   = np.sum(ranks_A_cell_c * ranks_B_cell_c, axis=1)
    denom_sp_cell = np.linalg.norm(ranks_A_cell_c, axis=1) * np.linalg.norm(ranks_B_cell_c, axis=1)
    cell_spearman = num_sp_cell / (denom_sp_cell + 1e-12)
    print(f"[timing] cell Spearman:     {time.time()-t0:.2f}s"); t0 = time.time()

    # --- Global Spearman: rank proj_A and proj_B flattened, then Pearson on ranks ---
    flat_A = proj_A.ravel()
    flat_B = proj_B.ravel()
    ranks_flat_A = np.argsort(np.argsort(flat_A)).astype(float)
    ranks_flat_B = np.argsort(np.argsort(flat_B)).astype(float)
    ranks_flat_A -= ranks_flat_A.mean()
    ranks_flat_B -= ranks_flat_B.mean()
    denom_sp_global = np.linalg.norm(ranks_flat_A) * np.linalg.norm(ranks_flat_B)
    global_spearman = float(np.dot(ranks_flat_A, ranks_flat_B) / (denom_sp_global + 1e-12))
    print(f"[timing] global Spearman:   {time.time()-t0:.2f}s"); t0 = time.time()

    # --- Gene scores: cosine similarity over cells, per gene ---
    num_gene = np.sum(proj_A * proj_B, axis=0)
    denom_gene = np.linalg.norm(proj_A, axis=0) * np.linalg.norm(proj_B, axis=0)
    gene_scores = num_gene / np.where(denom_gene == 0, 1, denom_gene)

    # --- Write results ---
    adata.obs["proj_sim_cell"]      = cell_scores
    adata.obs["proj_spearman_cell"] = cell_spearman
    adata.obs["proj_sim_cell_mag_A"] = np.linalg.norm(proj_A, axis=1)
    adata.obs["proj_sim_cell_mag_B"] = np.linalg.norm(proj_B, axis=1)

    adata.var["proj_sim_gene"] = gene_scores
    adata.var["proj_sim_gene_mag_A"] = np.linalg.norm(proj_A, axis=0)
    adata.var["proj_sim_gene_mag_B"] = np.linalg.norm(proj_B, axis=0)

    adata.var["proj_l2_gene"] = np.linalg.norm(proj_A - proj_B, axis=0)

    # Frobenius-normalized dot product: r_A * r_B * cosine, where r_A = mag_A[g] / ||proj_A||_F.
    # Rewards genes important in BOTH signals (large r_A and r_B) AND directionally aligned.
    # Fixes the scale imbalance of proj_sim_gene_weighted without inverting the ranking like L2.
    proj_A_scaled = proj_A / (np.linalg.norm(proj_A) + 1e-12)
    proj_B_scaled = proj_B / (np.linalg.norm(proj_B) + 1e-12)
    adata.var["proj_dot_scaled_gene"] = np.sum(proj_A_scaled * proj_B_scaled, axis=0)
    print(f"[timing] gene cosine+dot:   {time.time()-t0:.2f}s"); t0 = time.time()

    # Spearman correlation between proj_A[:,g] and proj_B[:,g] per gene.
    ranks_A = np.argsort(np.argsort(proj_A, axis=0), axis=0).astype(float)
    ranks_B = np.argsort(np.argsort(proj_B, axis=0), axis=0).astype(float)
    ranks_A_c = ranks_A - ranks_A.mean(axis=0)
    ranks_B_c = ranks_B - ranks_B.mean(axis=0)
    num_sp   = np.sum(ranks_A_c * ranks_B_c, axis=0)
    denom_sp = np.linalg.norm(ranks_A_c, axis=0) * np.linalg.norm(ranks_B_c, axis=0)
    adata.var["proj_spearman_gene"] = num_sp / (denom_sp + 1e-12)
    print(f"[timing] gene Spearman:     {time.time()-t0:.2f}s"); t0 = time.time()

    # Penalized gene score: suppress genes not structured by either signal.
    # compute_gene_score(adata, signal_A, signal_B, n_neighbors=n_neighbors)
    # print(f"[timing] compute_gene_score:{time.time()-t0:.2f}s"); t0 = time.time()
    # Uses min_smoothness if available (written by add_gene_smoothness).
    # if "gene_min_smoothness" in adata.var:
    #     penalty = np.exp(-adata.var["gene_min_smoothness"].values)
    #     adata.var["proj_sim_gene_penalized"] = gene_scores * penalty

    if store_projections:
        adata.layers["proj_A"] = proj_A
        adata.layers["proj_B"] = proj_B

    adata.uns["proj_sim_global"]      = global_score
    adata.uns["proj_spearman_global"] = global_spearman
    print(f"Global projection similarity ({signal_A} vs {signal_B}): cosine={global_score:.4f}  spearman={global_spearman:.4f}")
    return global_score


def compute_gene_score(adata, signal_a_key, signal_b_key, n_neighbors=10):
    """
    Compute per-gene smoothness w.r.t. two signals encoded in adata.obsm.

    For each signal, builds a KNN graph in signal space, then measures smoothness
    as the normalized Laplacian quadratic form x_g^T L x_g / total_weight per gene.
    Works for any encoding shape: (n_cells, 1), (n_cells, d), or (n_cells, n_cells)
    affinity matrices (as produced by encode_circular).

    Parameters
    ----------
    adata : AnnData
    signal_a_key : str — key in adata.obsm for signal A
    signal_b_key : str — key in adata.obsm for signal B
    n_neighbors : int — number of neighbors for the KNN graph in signal space

    Side effects
    ------------
    Writes adata.var['smooth_A'], adata.var['smooth_B'], adata.var['gene_min_smoothness'].
    Lower = smoother = gene is more structured by that signal.
    If gene_min_smoothness is present, also updates adata.var['proj_sim_gene_penalized'].
    """
    print("hey")
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    n = adata.n_obs

    def _build_knn_weights(signal):
        if signal.shape == (n, n):
            # Already an affinity matrix — pick top-k neighbours per row
            aff = signal.copy()
            np.fill_diagonal(aff, -np.inf)
            knn_idx = np.argsort(-aff, axis=1)[:, :n_neighbors]
            rows = np.repeat(np.arange(n), n_neighbors)
            cols = knn_idx.ravel()
            vals = np.maximum(aff[rows, cols], 0)
        else:
            # Embedding — Gaussian kernel on Euclidean distances
            D = sdist.cdist(signal, signal)
            np.fill_diagonal(D, np.inf)
            knn_idx = np.argsort(D, axis=1)[:, :n_neighbors]
            rows = np.repeat(np.arange(n), n_neighbors)
            cols = knn_idx.ravel()
            d = D[rows, cols]
            sigma = np.median(d)
            vals = np.exp(-d ** 2 / (2 * sigma ** 2 + 1e-12))

        W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        return (W + W.T) / 2

    def _laplacian_smoothness(W):
        deg = np.array(W.sum(axis=1)).ravel()
        total_weight = W.sum()
        # x_g^T L x_g = sum_i D_i x_gi^2 - sum_{ij} W_ij x_gi x_gj
        D_term = (deg[:, None] * X ** 2).sum(axis=0)
        W_term = np.array((W @ X) * X).sum(axis=0)
        return (D_term - W_term) / (total_weight + 1e-12)

    smooth_A = _laplacian_smoothness(_build_knn_weights(adata.obsm[signal_a_key]))
    smooth_B = _laplacian_smoothness(_build_knn_weights(adata.obsm[signal_b_key]))

    adata.var["smooth_A"] = smooth_A
    adata.var["smooth_B"] = smooth_B
    adata.var["gene_min_smoothness"] = np.minimum(smooth_A, smooth_B)
    adata.var["gene_max_smoothness"] = np.maximum(smooth_A, smooth_B)
    k=1
    if "proj_sim_gene" in adata.var:
        penalty = np.exp(-k*adata.var["gene_min_smoothness"].values)
        print(k)
        adata.var["proj_sim_gene_penalized"] = adata.var["proj_sim_gene"].values * penalty


def _ensure_obsm(adata, key):
    """Move a signal from obs to obsm if it isn't already in obsm."""
    if key not in adata.obsm and key in adata.obs:
        signal = adata.obs[key].values.astype(float)
        norm = np.linalg.norm(signal)
        adata.obsm[key] = (signal / norm if norm > 0 else signal).reshape(-1, 1)
