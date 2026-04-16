"""Smoke tests for projint."""

import numpy as np
import pytest
import projint


def make_adata(n_cells=200, n_genes=50, seed=0):
    np.random.seed(seed)
    import anndata as ad
    X = np.random.rand(n_cells, n_genes).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs["label_a"] = np.linspace(0, 2 * np.pi, n_cells)
    adata.obs["label_b"] = np.roll(adata.obs["label_a"].values, n_cells // 4)
    adata.obs["category"] = (np.arange(n_cells) % 4).astype(str)
    return adata


def test_import():
    assert hasattr(projint, "compute_projection_similarity")
    assert hasattr(projint, "simulate_cyclic")


def test_simulate_cyclic():
    adata = projint.simulate_cyclic(n_cells=100, n_cyclic_genes=10)
    assert adata.n_obs == 100
    assert adata.n_vars == 20  # two blocks of 10
    assert "label_a" in adata.obs
    assert "label_b" in adata.obs
    assert "permuted" in adata.obs


def test_encode_obs():
    adata = make_adata()
    key = projint.encode_obs(adata, "label_a")
    assert key in adata.obsm
    assert adata.obsm[key].shape == (200, 1)
    assert abs(np.linalg.norm(adata.obsm[key]) - 1.0) < 1e-5


def test_encode_onehot():
    adata = make_adata()
    key = projint.encode_onehot(adata, "category")
    assert key in adata.obsm
    assert adata.obsm[key].shape[0] == 200
    assert adata.obsm[key].shape[1] == 4  # 4 categories


def test_encode_circular():
    adata = make_adata()
    key = projint.encode_circular(adata, "category")
    assert key in adata.obsm
    assert adata.obsm[key].shape == (200, 200)


def test_compute_projection_similarity():
    adata = projint.simulate_cyclic(n_cells=100, n_cyclic_genes=10, perm_fraction=1.0)
    projint.encode_circular(adata, "label_a", obsm_key="sig_a")
    projint.encode_circular(adata, "label_b", obsm_key="sig_b")

    score = projint.compute_projection_similarity(adata, "sig_a", "sig_b")

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
    assert "proj_sim_cell" in adata.obs
    assert "proj_sim_gene" in adata.var
    assert "proj_sim_global" in adata.uns
    assert adata.obs["proj_sim_cell"].shape[0] == 100
    assert adata.var["proj_sim_gene"].shape[0] == 20


def test_interaction_higher_than_no_interaction():
    """Interacting signals should score higher than non-interacting ones."""
    adata_int = projint.simulate_cyclic(n_cells=150, n_cyclic_genes=15, perm_fraction=1.0, noise_level=0.05)
    adata_no  = projint.simulate_cyclic(n_cells=150, n_cyclic_genes=15, perm_fraction=0.0, noise_level=0.05)

    for adata in [adata_int, adata_no]:
        projint.encode_circular(adata, "label_a", obsm_key="sig_a")
        projint.encode_circular(adata, "label_b", obsm_key="sig_b")

    score_int = projint.compute_projection_similarity(adata_int, "sig_a", "sig_b")
    score_no  = projint.compute_projection_similarity(adata_no,  "sig_a", "sig_b")

    assert score_int > score_no, f"Expected interaction score ({score_int:.3f}) > no-interaction ({score_no:.3f})"