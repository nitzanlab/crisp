"""
crisp.core
==========
Core computation: projection-based interaction scoring between two biological signals.

For signals A and B, the projection of gene expression X onto signal S is:
    proj_S = X - SIFT(X | S)
i.e. the component of X that is structurally explained by S's graph.

The interaction score is the Pearson correlation between proj_A and proj_B
over genes, computed per cell (proj_pearson_cell). The global score is the
mean of the per-cell scores (proj_mean_cell_pearson); proj_pearson_gene_contrib
decomposes that same score into each gene's average contribution.
"""

import numpy as np
import scipy.sparse as sp
import scipy.spatial.distance as sdist
import scanpy as sc
try:
    from sift import sifter as _sifter
except ImportError:
    from sift._sift import sifter as _sifter


def _rbf_kernel_params(adata, signal_key, user_kernel_params):
    """Return kernel_params dict with length_scale set via median heuristic for RBF.

    Recognises one extra key that is consumed here and not forwarded to SiFT:
      median_factor (float, default 0.5) — multiplier applied to the median pairwise
        distance when auto-computing length_scale.
    """
    kp = dict(user_kernel_params or {})
    median_factor = kp.pop("median_factor", 0.5)
    if "length_scale" not in kp:
        signal = adata.obsm[signal_key]
        if hasattr(signal, "toarray"):
            signal = signal.toarray()
        dists = sdist.pdist(np.array(signal, dtype=float))
        dists = dists[dists > 0]
        kp["length_scale"] = median_factor * float(np.median(dists)) if len(dists) > 0 else 1.0
        print(f"[crisp] RBF length_scale for '{signal_key}' (median * {median_factor}): {kp['length_scale']:.6f}")
    print(kp["length_scale"])
    return kp


def compute_projection_similarity(
        adata,
        signal_A,
        signal_B,
        n_neighbors=15,
        embedding_key="X",
        metric_A="rbf",
        metric_B="rbf",
        device="cpu",
        store_projections=False,
        kernel_params_A=None,
        kernel_params_B=None,
):
    """
    Compute the projection-based interaction score between two signals.

    Both signals must already be encoded in adata.obsm (use crisp.encode.*).

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
    metric_A : str
    Distance metric for SIFT's KNN graph when sifting out signal_A.
    metric_B : str
    Distance metric for SIFT's KNN graph when sifting out signal_B.
    device : str
    'cpu' or 'cuda'.
    kernel_params_A : dict, optional
    Kernel parameters for signal A, e.g. {"length_scale": 0.01} for RBF/Laplacian.
    kernel_params_B : dict, optional
    Kernel parameters for signal B, e.g. {"length_scale": 0.007} for RBF/Laplacian.

    Returns
    -------
    global_score : float
    Mean per-cell Pearson correlation between proj_A and proj_B across all cells
    (same value as adata.uns['proj_mean_cell_pearson']).

    Side effects
    ------------
    Writes to adata.obs:
    'proj_pearson_cell' — per-cell Pearson correlation between proj_A and proj_B
        over genes (cell interaction score)
    Writes to adata.var:
    'proj_pearson_gene_contrib' — per-gene mean contribution to proj_pearson_cell,
        decomposing the cell-level Pearson score across genes
    Writes to adata.uns:
    'proj_mean_cell_pearson' — mean of proj_pearson_cell across cells (same as
        the return value)
    If store_projections is True, also writes adata.layers['proj_A'] and
    adata.layers['proj_B'].
    """
    bdata = adata.copy()
    _kparams_A_user = kernel_params_A or {}
    _kparams_B_user = kernel_params_B or {}

    if n_neighbors is None:
        n_neighbors = int(np.sqrt(bdata.n_obs))
    import time
    t0 = time.time()

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
    _kparams_A = _rbf_kernel_params(bdata, signal_A, _kparams_A_user) if metric_A.lower() in ("rbf", "laplacian") else dict(_kparams_A_user)
    bdata_A = bdata.copy()
    _sifter(
        adata=bdata_A,
        kernel_key=signal_A,
        embedding_key=embedding_key,
        metric=metric_A,
        n_neighbors=n_neighbors,
        copy=False,
        device=device,
        kernel_params=_kparams_A,
    )
    X_sifted_A = bdata_A.X.toarray() if sp.issparse(bdata_A.X) else np.array(bdata_A.X)
    proj_A = X_original - X_sifted_A
    print(f"[timing] SIFT signal A:     {time.time()-t0:.2f}s"); t0 = time.time()

    # --- Projection for signal B: proj_B = X - SIFT(X | B) ---
    _kparams_B = _rbf_kernel_params(bdata, signal_B, _kparams_B_user) if metric_B.lower() in ("rbf", "laplacian") else dict(_kparams_B_user)
    bdata_B = bdata.copy()
    _sifter(
        adata=bdata_B,
        kernel_key=signal_B,
        embedding_key=embedding_key,
        metric=metric_B,
        n_neighbors=n_neighbors,
        copy=False,
        device=device,
        kernel_params=_kparams_B,
    )
    X_sifted_B = bdata_B.X.toarray() if sp.issparse(bdata_B.X) else np.array(bdata_B.X)
    proj_B = X_original - X_sifted_B
    print(f"[timing] SIFT signal B:     {time.time()-t0:.2f}s"); t0 = time.time()

    # --- Cell scores: Pearson correlation over genes, per cell ---
    proj_A_c = proj_A - proj_A.mean(axis=1, keepdims=True)
    proj_B_c = proj_B - proj_B.mean(axis=1, keepdims=True)
    num_pearson_cell   = np.sum(proj_A_c * proj_B_c, axis=1)
    denom_pearson_cell = np.linalg.norm(proj_A_c, axis=1) * np.linalg.norm(proj_B_c, axis=1)
    cell_pearson = num_pearson_cell / (denom_pearson_cell + 1e-12)

    # --- Per-gene Pearson contribution ---
    # contrib[i, g] sums exactly to cell_pearson[i] across genes.
    # mean over cells gives a per-gene average contribution to the interaction score.
    n_genes = proj_A.shape[1]
    std_A_cell = proj_A.std(axis=1, keepdims=True)
    std_B_cell = proj_B.std(axis=1, keepdims=True)
    gene_contrib = (proj_A_c * proj_B_c) / (n_genes * std_A_cell * std_B_cell + 1e-12)
    mean_gene_contrib = gene_contrib.mean(axis=0)   # (n_genes,)

    # --- Write results ---

    adata.obs["proj_pearson_cell"]  = cell_pearson
    adata.var["proj_pearson_gene_contrib"] = mean_gene_contrib
    adata.uns["proj_mean_cell_pearson"] = float(np.mean(cell_pearson))


    if store_projections:
        adata.layers["proj_A"] = proj_A
        adata.layers["proj_B"] = proj_B

    return float(np.mean(cell_pearson))



def _ensure_obsm(adata, key):
    """Move a signal from obs to obsm if it isn't already in obsm."""
    if key not in adata.obsm and key in adata.obs:
        signal = adata.obs[key].values.astype(float)
        norm = np.linalg.norm(signal)
        adata.obsm[key] = (signal / norm if norm > 0 else signal).reshape(-1, 1)
