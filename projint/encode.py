"""
projint.encode
==============
Signal encoding methods that prepare a signal for use in SIFT.

Each encoder reads a signal from adata.obs or adata.obsm and writes
a matrix representation into adata.obsm under a canonical key.
The key is returned so it can be passed directly to compute_projection_similarity.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD


def encode_obs(adata, key):
    """
    Encode a 1D continuous signal from adata.obs into adata.obsm.

    Normalizes the signal to unit norm and stores it as a (n_cells, 1) column
    in adata.obsm under the same key.

    Parameters
    ----------
    adata : AnnData
    key : str
        Column name in adata.obs containing the signal (e.g. 'zonation_position').

    Returns
    -------
    str : the obsm key where the encoded signal is stored (same as key).
    """
    signal = adata.obs[key].values.astype(float)
    norm = np.linalg.norm(signal)
    if norm > 0:
        signal = signal / norm
    adata.obsm[key] = signal.reshape(-1, 1)
    return key


def encode_onehot(adata, label_key, obsm_key=None):
    """
    Encode a categorical label from adata.obs as a size-normalized one-hot matrix.

    Each class column is normalized by 1/sqrt(n_k) where n_k is the class size,
    so that the Frobenius norm of each column is 1.

    Parameters
    ----------
    adata : AnnData
    label_key : str
        Column name in adata.obs containing the categorical labels.
    obsm_key : str, optional
        Key to store the result in adata.obsm. Defaults to f'onehot_{label_key}'.

    Returns
    -------
    str : the obsm key where the encoded signal is stored.
    """
    if obsm_key is None:
        obsm_key = f"onehot_{label_key}"

    labels = adata.obs[label_key].values
    categories = np.unique(labels)
    n = len(labels)
    k = len(categories)

    matrix = np.zeros((n, k), dtype=np.float32)
    for j, cat in enumerate(categories):
        mask = labels == cat
        n_k = mask.sum()
        if n_k > 0:
            matrix[mask, j] = 1.0 / np.sqrt(n_k)

    adata.obsm[obsm_key] = matrix
    return obsm_key


def encode_circular(adata, label_key, obsm_key=None):
    """
    Encode a circular/cyclic label from adata.obs as a pairwise affinity matrix.

    Labels are mapped to evenly spaced angles in [0, 2π]. Pairwise affinity is
    computed as exp(-circular_distance²).

    Parameters
    ----------
    adata : AnnData
    label_key : str
        Column name in adata.obs containing the cyclic labels (e.g. 'Phase_corrected').
    obsm_key : str, optional
        Key to store the result in adata.obsm. Defaults to f'circular_{label_key}'.

    Returns
    -------
    str : the obsm key where the encoded signal is stored.
    """
    if obsm_key is None:
        obsm_key = f"circular_{label_key}"

    raw = adata.obs[label_key]
    try:
        unique_labels = np.array(sorted(raw.unique(), key=float))
    except (ValueError, TypeError):
        unique_labels = np.array(sorted(raw.unique(), key=str))
    label_to_angle = {v: (i / len(unique_labels)) * 2 * np.pi for i, v in enumerate(unique_labels)}
    angles = np.array([label_to_angle[v] for v in raw.values])

    # Vectorized pairwise circular distance
    diff = np.abs(angles[:, None] - angles[None, :])
    circ_dist = np.minimum(diff, 2 * np.pi - diff)
    affinity = np.exp(-(circ_dist ** 2))

    adata.obsm[obsm_key] = affinity.astype(np.float32)
    return obsm_key


def encode_spatial(adata, spatial_key="spatial", obsm_key="spatial_encoded",
                   n_neighbors=None, n_components=50):
    """
    Encode spatial coordinates as a low-dimensional embedding via KNN + SVD.

    Builds a spatial KNN graph with Gaussian kernel weights, symmetrizes it,
    then reduces with TruncatedSVD to produce a dense embedding suitable for SIFT.

    Parameters
    ----------
    adata : AnnData
    spatial_key : str
        Key in adata.obsm containing (x, y) spatial coordinates.
    obsm_key : str
        Key to store the result in adata.obsm.
    n_neighbors : int, optional
        Number of spatial neighbors. Defaults to sqrt(n_cells).
    n_components : int
        Number of SVD components for dimensionality reduction.

    Returns
    -------
    str : the obsm key where the encoded signal is stored.
    """
    coords = adata.obsm[spatial_key]
    n = coords.shape[0]
    if n_neighbors is None:
        n_neighbors = int(np.sqrt(n))

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree")
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    rows = np.repeat(np.arange(n), n_neighbors)
    cols = indices.flatten()
    dists = distances.flatten()

    sigma = np.median(dists)
    weights = np.exp(-(dists ** 2) / (sigma ** 2))

    W = sp.coo_matrix((weights, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T).tocsr()

    svd = TruncatedSVD(n_components=n_components)
    embedding = svd.fit_transform(W)

    adata.obsm[obsm_key] = embedding.astype(np.float32)
    return obsm_key
