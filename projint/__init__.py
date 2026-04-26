from .core import compute_projection_similarity, compute_gene_score
from .encode import encode_obs, encode_onehot, encode_circular, encode_spatial
from .simulate import simulate_cyclic, add_gene_smoothness
from .plot import plot_pca, plot_umap, save_plot, plot_score_pca, plot_score_histogram

__version__ = "0.1.0"
__all__ = [
    "compute_projection_similarity",
    "compute_gene_score",
    "encode_obs",
    "encode_onehot",
    "encode_circular",
    "encode_spatial",
    "simulate_cyclic",
    "add_gene_smoothness",
    "plot_pca",
    "plot_umap",
    "save_plot",
    "plot_score_pca",
    "plot_score_histogram",
]
