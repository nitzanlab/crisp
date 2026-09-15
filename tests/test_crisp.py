"""Smoke tests for crisp."""

import numpy as np
import pytest
import crisp


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
    assert hasattr(crisp, "compute_projection_similarity")
    assert hasattr(crisp, "simulate_cyclic_processes")


def test_simulate_cyclic_processes():
    adata = crisp.simulate_cyclic_processes(
        n_cells=100, n_genes_A=10, n_genes_B=10, n_genes_AB=0, seed=0,
    )
    assert adata.n_obs == 100
    assert adata.n_vars == 20  # 10 A genes + 10 B genes
    assert "label_a" in adata.obs
    assert "label_b" in adata.obs
    assert "permuted" in adata.obs
    assert list(adata.var["gene_type"]) == ["A"] * 10 + ["B"] * 10


def test_encode_obs():
    adata = make_adata()
    key = crisp.encode_obs(adata, "label_a")
    assert key in adata.obsm
    assert adata.obsm[key].shape == (200, 1)
    assert abs(np.linalg.norm(adata.obsm[key]) - 1.0) < 1e-5


def test_encode_onehot():
    adata = make_adata()
    key = crisp.encode_onehot(adata, "category")
    assert key in adata.obsm
    assert adata.obsm[key].shape[0] == 200
    assert adata.obsm[key].shape[1] == 4  # 4 categories


def test_encode_circular():
    adata = make_adata()
    key = crisp.encode_circular(adata, "category")
    assert key in adata.obsm
    assert adata.obsm[key].shape == (200, 200)


def test_compute_projection_similarity():
    adata = crisp.simulate_cyclic_processes(
        n_cells=100, n_genes_A=10, n_genes_B=10, n_genes_AB=10, seed=0,
    )
    crisp.encode_circular(adata, "label_a", obsm_key="sig_a")
    crisp.encode_circular(adata, "label_b", obsm_key="sig_b")

    score = crisp.compute_projection_similarity(adata, "sig_a", "sig_b")

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
    assert "proj_pearson_cell" in adata.obs
    assert "proj_pearson_gene_contrib" in adata.var
    assert "proj_mean_cell_pearson" in adata.uns
    assert adata.obs["proj_pearson_cell"].shape[0] == 100
    assert adata.var["proj_pearson_gene_contrib"].shape[0] == 30


def test_gene_contrib_sums_to_global_score():
    """Per-gene mean contribution should sum to the global interaction score."""
    adata = crisp.simulate_cyclic_processes(
        n_cells=100, n_genes_A=10, n_genes_B=10, n_genes_AB=10, seed=0,
    )
    crisp.encode_circular(adata, "label_a", obsm_key="sig_a")
    crisp.encode_circular(adata, "label_b", obsm_key="sig_b")

    score = crisp.compute_projection_similarity(adata, "sig_a", "sig_b")

    gene_contrib_sum = adata.var["proj_pearson_gene_contrib"].sum()
    assert gene_contrib_sum == pytest.approx(score, abs=1e-4)
    assert gene_contrib_sum == pytest.approx(adata.uns["proj_mean_cell_pearson"], abs=1e-4)


def test_interaction_higher_with_ab_genes():
    """AB (jointly-driven) genes should raise the interaction score vs. A/B-only genes."""
    adata_ab = crisp.simulate_cyclic_processes(
        n_cells=150, n_genes_A=15, n_genes_B=15, n_genes_AB=15, noise=0.05, seed=0,
    )
    adata_no_ab = crisp.simulate_cyclic_processes(
        n_cells=150, n_genes_A=15, n_genes_B=15, n_genes_AB=0, noise=0.05, seed=0,
    )

    for adata in [adata_ab, adata_no_ab]:
        crisp.encode_circular(adata, "label_a", obsm_key="sig_a")
        crisp.encode_circular(adata, "label_b", obsm_key="sig_b")

    score_ab = crisp.compute_projection_similarity(adata_ab, "sig_a", "sig_b")
    score_no_ab = crisp.compute_projection_similarity(adata_no_ab, "sig_a", "sig_b")

    assert score_ab > score_no_ab, (
        f"Expected AB-interaction score ({score_ab:.3f}) > no-AB score ({score_no_ab:.3f})"
    )
