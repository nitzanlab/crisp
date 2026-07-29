"""
crisp.plot
==========
Plotting helpers for crisp results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 20,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2,
})

_PCA_KWARGS = dict(legend_loc="right margin", legend_fontsize=16, wspace=0.3)
_UMAP_KWARGS = dict(legend_loc="right margin", legend_fontsize=16, wspace=0.3)


def plot_pca(adata, save=None, **kwargs):
    sc.pl.pca(adata, save=save, **{**_PCA_KWARGS, **kwargs})


def plot_umap(adata, save=None, **kwargs):
    sc.pl.umap(adata, save=save, **{**_UMAP_KWARGS, **kwargs})


def save_plot(path):
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_score_pca(adata, score_key="proj_sim_cell", group_key="permuted", save=None):
    """
    PCA plot colored by cell score and an optional grouping variable (e.g. 'permuted').

    Parameters
    ----------
    adata : AnnData
        Must have adata.obsm['X_pca'] (run sc.pp.pca first if needed).
    score_key : str
        Key in adata.obs for the cell score.
    group_key : str or None
        Key in adata.obs for a categorical grouping (e.g. 'permuted').
    save : str or None
        File path to save the figure.
    """
    sc.pp.pca(adata)
    vals = adata.obs[score_key].values
    vmin, vmax = np.percentile(vals, [2.5, 97.5])
    color = [score_key] + ([group_key] if group_key and group_key in adata.obs else [])
    sc.pl.pca(
        adata,
        color=color,
        vmin=vmin,
        vmax=vmax,
        save=save,
        **_PCA_KWARGS,
    )


def plot_score_histogram(adata, score_key="proj_sim_cell", group_key="permuted", save=None):
    """
    Histogram of cell scores, optionally split by a grouping variable.

    Parameters
    ----------
    adata : AnnData
    score_key : str
        Key in adata.obs for the cell score.
    group_key : str or None
        Key in adata.obs for hue grouping (e.g. 'permuted').
    save : str or None
        File path to save the figure.
    """
    df = adata.obs[[score_key] + ([group_key] if group_key and group_key in adata.obs else [])].copy()
    vals = df[score_key].dropna()
    data_range = vals.max() - vals.min()
    binwidth = data_range / 50 if data_range > 0 else 1.0

    fig, ax = plt.subplots()
    sns.histplot(
        data=df,
        x=score_key,
        hue=group_key if group_key and group_key in adata.obs else None,
        binwidth=binwidth,
        stat="count",
        common_norm=False,
        element="step",
        alpha=0.6,
        ax=ax,
    )
    ax.set_xlabel("Cell Score")
    ax.set_ylabel("Count")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=300)
    plt.close(fig)
