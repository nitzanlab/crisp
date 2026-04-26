"""
projint.simulate
================
Synthetic data generation for validating the projection similarity framework.
"""

import numpy as np
import anndata as ad


def simulate_cyclic(
    n_cells,
    n_cyclic_genes,
    w_cyclic=0.1,
    noise_level=0.1,
    perm_fraction=1.0,
    ratio=1,
    std_cyclic=None,
    label_perm_fraction=0.01,
    n_non_interacting=0,
):
    """
    Simulate gene expression with two cyclic signals and optional non-interacting genes.

    Signal A (label_a) and signal B (label_b) are cyclic (angle in [0, 2π]).
    The second block of genes is controlled by signal B. Setting perm_fraction < 1
    means only a fraction of cells are permuted, creating a partial interaction.

    Parameters
    ----------
    n_cells : int
    n_cyclic_genes : int
        Number of genes for each cyclic signal block.
    w_cyclic : float
        Window length as a fraction of n_cells; sets std if std_cyclic is None.
    noise_level : float
        Gaussian noise added to gene expression.
    perm_fraction : float in [0, 1]
        Fraction of cells to permute for signal B genes. 1.0 = full interaction,
        0.0 = no interaction.
    ratio : float
        Ratio of genes for the second cyclic block relative to n_cyclic_genes.
    std_cyclic : float, optional
        Gaussian std for cyclic gene profiles. Defaults to w_cyclic * n_cells.
    label_perm_fraction : float in [0, 1]
        Fraction of cells to permute in the label vectors.
    n_non_interacting : int
        Number of additional non-interacting (randomly permuted cyclic) genes.

    Returns
    -------
    AnnData with:
        obs['label_a'], obs['label_b'] — cyclic signal angles
        obs['permuted']               — bool mask of permuted cells
    """
    if std_cyclic is None:
        std_cyclic = w_cyclic * n_cells

    genes_A = _make_cyclic_genes(n_cells, n_cyclic_genes, w_cyclic, noise_level, std_cyclic)

    n_genes_B = int(np.floor(n_cyclic_genes * ratio))
    genes_B = _make_cyclic_genes(n_cells, n_genes_B, w_cyclic, noise_level, std_cyclic)

    if n_non_interacting > 0:
        genes_C = _make_cyclic_genes(n_cells, n_non_interacting, w_cyclic, noise_level, std_cyclic)
        genes_C = genes_C[np.random.permutation(n_cells)]
    else:
        genes_C = np.zeros((n_cells, 0))

    label_a = np.linspace(0, 2 * np.pi, n_cells)
    label_b = np.roll(label_a, int(np.floor(np.random.rand() * n_cells)))

    permuted = np.zeros(n_cells, dtype=bool)
    perm = None
    if perm_fraction > 0.0:
        m = int(round(n_cells * np.clip(perm_fraction, 0, 1)))
        if m >= n_cells:
            perm = np.random.permutation(n_cells)
            genes_B = genes_B[perm]
            label_b = label_b[perm]
        elif m > 0:
            idx = np.random.choice(n_cells, size=m, replace=False)
            perm_sub = np.random.permutation(idx)
            perm = np.arange(n_cells)
            perm[idx] = perm_sub
            genes_B = genes_B[perm]
            label_b = label_b[perm]
            permuted[idx] = True

    if label_perm_fraction > 0.0:
        m = int(round(n_cells * np.clip(label_perm_fraction, 0, 1)))
        if m > 0:
            idx = np.random.choice(n_cells, size=m, replace=False)
            perm_sub = np.random.permutation(idx)
            lp = np.arange(n_cells)
            lp[idx] = perm_sub
            label_a = label_a[lp]
            label_b = label_b[lp]

    X = np.hstack([genes_A, genes_B, genes_C])
    adata = ad.AnnData(X=X)
    adata.obs["label_a"] = label_a
    adata.obs["label_b"] = label_b
    adata.obs["permuted"] = permuted if perm is None else permuted[perm]
    return adata


def add_gene_smoothness(adata, signal_a_key="label_a", signal_b_key="label_b"):
    """
    Compute per-gene smoothness w.r.t. two signals and store the minimum in adata.var.

    For each signal, cells are sorted by signal value and the mean squared consecutive
    difference of gene expression is computed. Lower = smoother = gene is structured
    by that signal. The minimum across both signals is stored.

    Parameters
    ----------
    adata : AnnData
        Must have signal_a_key and signal_b_key in adata.obs, and adata.X as a dense array.
    signal_a_key : str
    signal_b_key : str

    Side effects
    ------------
    Writes adata.var['smooth_A'], adata.var['smooth_B'], adata.var['gene_min_smoothness'].
    Lower = smoother = gene is more structured by that signal.
    """
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()

    def _smoothness(signal):
        order = np.argsort(signal)
        X_sorted = X[order]
        return np.mean(np.diff(X_sorted, axis=0) ** 2, axis=0)

    smooth_A = _smoothness(adata.obs[signal_a_key].values)
    smooth_B = _smoothness(adata.obs[signal_b_key].values)
    adata.var["smooth_A"] = smooth_A
    adata.var["smooth_B"] = smooth_B
    adata.var["gene_min_smoothness"] = np.minimum(smooth_A, smooth_B)
    adata.var["gene_max_smoothness"] = np.maximum(smooth_A, smooth_B)


def _make_cyclic_genes(n_cells, n_genes, w_cyclic, noise_level, std_cyclic):
    genes = np.zeros((n_cells, n_genes))
    for i in range(n_genes):
        start = int(np.floor(np.random.rand() * n_cells))
        for j in range(n_cells):
            dist = min((j - start) % n_cells, (start - j) % n_cells)
            genes[j, i] = np.exp(-dist ** 2 / (2 * std_cyclic ** 2))
    genes += np.random.normal(0, noise_level, genes.shape)
    return genes
