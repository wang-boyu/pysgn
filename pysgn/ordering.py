# ordering.py


import geopandas as gpd
import numpy as np
from sklearn.neighbors import KDTree, KernelDensity


def random_order(gdf: gpd.GeoDataFrame, random_state: int | None = None) -> np.ndarray:
    """
    Return a random ordering of node indices.

    Parameters:
        gdf: GeoDataFrame containing nodes.

        random_state: Optional random seed for reproducibility.

    Returns:
        np.ndarray: A permutation of node indices.
    """
    rng = np.random.default_rng(seed=random_state)
    indices = np.arange(len(gdf))
    rng.shuffle(indices)
    return indices


def attribute_order(gdf: gpd.GeoDataFrame, by: str | list[str], **kwargs) -> np.ndarray:
    """
    Return an ordering of node indices based on one or more attribute columns.

    The GeoDataFrame is sorted by the specified column(s) in ascending order by default.
    Additional keyword arguments are passed to `gdf.sort_values <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html>`_.

    Parameters:
        gdf: GeoDataFrame containing nodes.

        by: A column name or list of column names to sort by.

        **kwargs: Additional keyword arguments to pass to `gdf.sort_values <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html>`_.
                  These can include 'ascending', 'na_position', 'ignore_index', etc.

    Returns:
        np.ndarray: Array of indices representing the sorted order.
    """
    if isinstance(by, str):
        by = [by]
    gdf_with_range_index = gdf.copy().reset_index(drop=True)
    # Use pandas sorting (mergesort is stable) so that the relative order is preserved.
    sorted_df = gdf_with_range_index.sort_values(by=by, kind="mergesort", **kwargs)
    # Return the ordering as the locations of original gdf
    return sorted_df.index.to_numpy()


def density_order_knn(gdf: gpd.GeoDataFrame, k: int = 5, **kwargs) -> np.ndarray:
    """
    Return an ordering of node indices based on local density using k-nearest neighbors.

    The density is estimated as the average distance to the k nearest neighbors.
    Nodes with a lower average distance (i.e. higher density) will be ordered first.
    Additional keyword arguments are passed to the KDTree query method: `KDTree.query <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree.query>`_.

    Parameters:
        gdf: GeoDataFrame containing nodes.

        k: Number of nearest neighbors to consider (default 5).

        **kwargs: Additional keyword arguments passed to `KDTree.query <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree.query>`_.

    Returns:
        np.ndarray: Array of indices representing the density ordering.
    """
    if len(gdf) <= k:
        raise ValueError(f"Number of nodes ({len(gdf)}) must be greater than k ({k}).")
    if k < 1:
        raise ValueError("k must be a positive integer.")
    if len(gdf) == 1:
        return np.array([0])
    if len(gdf) == 0:
        return np.array([])

    # Get positions: if the geometry is polygonal, use centroids; otherwise assume points.
    if gdf.geometry.geom_type.iloc[0] == "Polygon":
        positions = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    else:
        positions = np.column_stack([gdf.geometry.x, gdf.geometry.y])

    # Build a KDTree for fast neighbor lookup.
    tree = KDTree(positions)
    # Query for k+1 neighbors because the first neighbor is the point itself (distance zero).
    distances, _ = tree.query(positions, k=k + 1, **kwargs)
    # Exclude the self-distance (first column) and compute the average distance.
    avg_dist = distances[:, 1:].mean(axis=1)
    # Nodes with lower average distance are denser.
    order = np.argsort(avg_dist)
    return order


def density_order_kde(
    gdf: gpd.GeoDataFrame, bandwidth: float = 1.0, kernel: str = "gaussian", **kwargs
) -> np.ndarray:
    """
    Return an ordering of node indices based on density estimated by Kernel Density Estimation (KDE).

    The density is estimated at each node position using KDE. Nodes with higher density estimates are ordered first.
    Additional keyword arguments are passed to the KernelDensity constructor: `KernelDensity <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html>`_.

    Parameters:
        gdf: GeoDataFrame containing nodes.

        bandwidth: Bandwidth parameter for the KDE (default 1.0).

        kernel: The kernel to use in KDE (default "gaussian").

        **kwargs: Additional keyword arguments passed to `KernelDensity <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html>`_.

    Returns:
        np.ndarray: Array of indices representing the density ordering.
    """
    if len(gdf) == 1:
        return np.array([0])
    if len(gdf) == 0:
        return np.array([])

    # Get positions: use centroids for polygons or coordinates for points.
    if gdf.geometry.geom_type.iloc[0] == "Polygon":
        positions = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    else:
        positions = np.column_stack([gdf.geometry.x, gdf.geometry.y])

    # Fit a Kernel Density Estimator to the positions.
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel, **kwargs)
    kde.fit(positions)
    # Get log-density estimates for each node.
    log_density = kde.score_samples(positions)
    # Higher density should come first. Since np.argsort sorts in ascending order,
    # we sort by the negative of the log-density.
    order = np.argsort(-log_density)
    return order


def density_order(gdf: gpd.GeoDataFrame, method: str = "kde", **kwargs) -> np.ndarray:
    """
    Return an ordering of node indices based on density using the specified method.

    Parameters:
        gdf: GeoDataFrame containing nodes.

        method: The density ordering method to use. Options are:
                - "knn": Uses the average distance to k-nearest neighbors.
                - "kde": Uses kernel density estimation.

        **kwargs: Additional keyword arguments passed to the selected density ordering function.

    Returns:
        np.ndarray: Array of indices representing the density ordering.
    """
    method = method.lower()
    if method == "knn":
        return density_order_knn(gdf, **kwargs)
    elif method == "kde":
        return density_order_kde(gdf, **kwargs)
    else:
        raise ValueError(
            f"Unknown density ordering method: {method}. Use 'knn' or 'kde'."
        )
