"""
crisp.simulate
===============
Synthetic data generation for validating the projection similarity framework.

Simulates gene expression driven by two cyclic signals (label_a, label_b),
with four gene types:
    A  — cyclic Gaussians driven by label_a only
    B  — cyclic Gaussians driven by label_b only
    C  — cyclic Gaussians driven by an independent background label
    AB — centered-product genes jointly driven by label_a and label_b,
         representing genes structured by the interaction between the two signals
"""

import numpy as np
import anndata as ad


def _cyclic_genes(angles, n_genes, std, noise, rng):
    n = len(angles)
    G = np.zeros((n, n_genes))
    for g in range(n_genes):
        peak = rng.uniform(0, 2 * np.pi)
        d = np.abs(angles - peak)
        G[:, g] = np.exp(-np.minimum(d, 2 * np.pi - d) ** 2 / (2 * std ** 2))
    return G + rng.normal(0, noise, G.shape)


def _centered_product_genes(angles_a, angles_b, n_genes, std, noise, rng):
    """AB interaction genes: (gauss(la) - mean) * (gauss(lb) - mean).

    Each gene peaks jointly at one (la, lb) combination. Because each factor
    is zero-mean, proj_A and proj_B both retain the joint signal, yielding a
    Spearman score higher than pure A, B, or C genes. In practice the score
    is well above zero but not close to 1 due to noise and partial permutation.
    """
    n = len(angles_a)
    G = np.zeros((n, n_genes))
    for g in range(n_genes):
        pa = rng.uniform(0, 2 * np.pi)
        pb = rng.uniform(0, 2 * np.pi)
        da = np.minimum(np.abs(angles_a - pa), 2 * np.pi - np.abs(angles_a - pa))
        db = np.minimum(np.abs(angles_b - pb), 2 * np.pi - np.abs(angles_b - pb))
        f = np.exp(-da ** 2 / (2 * std ** 2))
        g_b = np.exp(-db ** 2 / (2 * std ** 2))
        G[:, g] = f * g_b
    return G + rng.normal(0, noise, G.shape)


def _background_genes(n_cells, n_genes, std, noise, rng):
    """Genes C: cyclic genes on a fixed independent label, uncorrelated
    with label_a and label_b."""
    label_bg = np.random.default_rng(42).permutation(
        np.linspace(0, 2 * np.pi, n_cells, endpoint=False)
    )
    return _cyclic_genes(label_bg, n_genes, std, noise, rng)


def simulate_cyclic_processes(
    n_cells, n_genes_A, n_genes_B, n_genes_AB,
    n_genes_background=0,
    noise=0.1, std=np.pi / 4, seed=None,
    perm_frac_b=0.0,
    label_noise=0.1,
):
    """
    Simulate gene expression from two cyclic interacting processes.

    Gene layout: n_genes_A (A) | n_genes_B (B) | n_genes_background (C) | n_genes_AB (AB)
    - A genes:  cyclic Gaussians driven by label_a
    - B genes:  cyclic Gaussians driven by label_b
    - C genes:  cyclic Gaussians driven by an independent background label
    - AB genes: centered-product (gauss(la)-mean)*(gauss(lb)-mean)

    Both labels are cyclic, with label_b = (label_a + random_phase) % 2pi.

    Parameters
    ----------
    n_cells : int
    n_genes_A, n_genes_B, n_genes_AB : int
    n_genes_background : int, optional
    noise : float
        Gaussian noise added to gene expression.
    std : float
        Width of each gene's cyclic Gaussian profile.
    seed : int, optional
    perm_frac_b : float in [0, 1]
        Fraction of cells whose B-genes and label_b are jointly permuted,
        breaking their alignment with label_a.
    label_noise : float
        Std of wrapped Gaussian noise added to both labels before gene
        generation (simulates imprecise phase measurement).

    Returns
    -------
    AnnData with:
        obs['label_a'], obs['label_b'] — cyclic signal angles
        obs['permuted']                — bool mask of permuted cells
        var['gene_type']               — 'A' / 'B' / 'C' / 'AB' per gene
    """
    rng_genes = np.random.default_rng(seed)

    label_a = np.linspace(0, 2 * np.pi, n_cells, endpoint=False)
    phase = rng_genes.uniform(0, 2 * np.pi)
    label_b = (label_a + phase) % (2 * np.pi)

    genes_A = _cyclic_genes(label_a, n_genes_A, std, noise, rng_genes)
    genes_B = _cyclic_genes(label_b, n_genes_B, std, noise, rng_genes)

    # Jointly permute B-genes and label_b for a fraction of cells
    permuted = np.zeros(n_cells, dtype=bool)
    if perm_frac_b > 0.0:
        k = int(round(perm_frac_b * n_cells))
        idx = rng_genes.choice(n_cells, size=k, replace=False)
        perm = rng_genes.permutation(k)
        genes_B[idx] = genes_B[idx][perm].copy()
        label_b = label_b.copy()
        label_b[idx] = label_b[idx][perm].copy()
        permuted[idx] = True

    parts = [genes_A, genes_B]
    if n_genes_background > 0:
        parts.append(_background_genes(n_cells, n_genes_background, std, noise, rng_genes))
    if n_genes_AB > 0:
        parts.append(_centered_product_genes(
            label_a, label_b, n_genes_AB, std,
            noise=noise, rng=rng_genes))
    if label_noise > 0.0:
        label_a = (label_a + rng_genes.normal(0, label_noise, n_cells)) % (2 * np.pi)
        label_b = (label_b + rng_genes.normal(0, label_noise, n_cells)) % (2 * np.pi)

    gene_types = (
        ["A"] * n_genes_A +
        ["B"] * n_genes_B +
        (["C"] * n_genes_background if n_genes_background > 0 else []) +
        (["AB"] * n_genes_AB if n_genes_AB > 0 else [])
    )
    adata = ad.AnnData(X=np.hstack(parts).astype(np.float32))
    adata.obs["label_a"] = label_a
    adata.obs["label_b"] = label_b
    adata.obs["permuted"] = permuted
    adata.var["gene_type"] = gene_types
    return adata
