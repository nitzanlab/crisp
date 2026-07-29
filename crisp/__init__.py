from .core import compute_projection_similarity
from .encode import encode_obs, encode_onehot, encode_circular,  encode_spatial
from .plot import plot_pca, plot_umap, save_plot, plot_score_pca, plot_score_histogram

__version__ = "0.1.0"
__all__ = [
    "compute_projection_similarity",
    "encode_obs",
    "encode_onehot",
    "encode_circular",
    "encode_spatial",
    "plot_pca",
    "plot_umap",
    "save_plot",
    "plot_score_pca",
    "plot_score_histogram",
]
