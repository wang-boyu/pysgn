from collections.abc import Callable
from typing import Literal

import numpy as np
from geopandas import GeoDataFrame
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors


def density_order(
    method: Literal["kde", "knn"] = "kde",
    bandwidth: float | None = None,
    n_neighbors: int = 10,
) -> Callable[[GeoDataFrame], np.ndarray]:
    """Create a density-based ordering function.

    Args:
        method: Method to use for density calculation. One of:
            - 'kde': Kernel Density Estimation
            - 'knn': K-Nearest Neighbors density
        bandwidth: Bandwidth for KDE. If None, Scott's rule is used.
        n_neighbors: Number of neighbors for KNN density calculation.

    Returns:
        Function that takes a GeoDataFrame and returns indices ordered by distance to density center
            (from highest density to lowest).

    Examples:
        >>> G = geo_barabasi_albert_network(gdf, m=3, node_order=density_order(method="kde"))
    """

    def order_func(gdf: GeoDataFrame) -> np.ndarray:
        # Extract coordinates
        if gdf.geometry.geom_type.iloc[0] == "Polygon":
            points = np.column_stack(
                [gdf.geometry.centroid.x.values, gdf.geometry.centroid.y.values]
            )
        else:
            points = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])

        if method == "kde":
            # Calculate KDE
            kde = gaussian_kde(points.T, bw_method=bandwidth)
            density = kde(points.T)

            # Find the point with highest density
            center_idx = np.argmax(density)
            center_point = points[center_idx]

            # Calculate distances to density center
            distances = np.sqrt(np.sum((points - center_point) ** 2, axis=1))

            # Return indices sorted by distance
            return np.argsort(distances)

        elif method == "knn":
            # Calculate local density using kNN
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
            distances, _ = nbrs.kneighbors(points)
            local_density = 1 / np.mean(distances, axis=1)

            # Find the point with highest density
            center_idx = np.argmax(local_density)
            center_point = points[center_idx]

            # Calculate distances to density center
            distances = np.sqrt(np.sum((points - center_point) ** 2, axis=1))

            # Return indices sorted by distance
            return np.argsort(distances)

        else:
            raise ValueError(f"Unknown method: {method}")

    return order_func


def random_order(seed: int | None = None) -> Callable[[GeoDataFrame], np.ndarray]:
    """Create a random ordering function.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Function that takes a GeoDataFrame and returns randomly ordered indices.

    Examples:
        >>> G = geo_barabasi_albert_network(gdf, m=3, node_order=random_order(seed=42))
    """

    def order_func(gdf: GeoDataFrame) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.permutation(len(gdf))

    return order_func


def attribute_order(
    by: str | list[str], ascending: bool = True
) -> Callable[[GeoDataFrame], np.ndarray]:
    """Create an attribute-based ordering function.

    Args:
        by: Column name(s) to sort by. Can be a single column name or a list of column names.
        ascending: Sort order. If True, sort in ascending order, if False, sort in descending order.

    Returns:
        Function that takes a GeoDataFrame and returns indices ordered by the specified column(s).

    Examples:
        >>> # Order by a single column
        >>> G1 = geo_barabasi_albert_network(gdf, m=3, node_order=attribute_order(by="population"))
        >>>
        >>> # Order by multiple columns
        >>> G2 = geo_barabasi_albert_network(gdf, m=3, node_order=attribute_order(by=["population", "density"]))
    """

    def order_func(gdf: GeoDataFrame) -> np.ndarray:
        if isinstance(by, str):
            if by not in gdf.columns:
                raise ValueError(f"Column '{by}' not found in GeoDataFrame")
            return (
                np.argsort(gdf[by].values)
                if ascending
                else np.argsort(gdf[by].values)[::-1]
            )
        else:
            # Validate all columns exist
            missing_cols = [col for col in by if col not in gdf.columns]
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} not found in GeoDataFrame")
            # Handle multiple columns
            return np.lexsort([gdf[col].values for col in reversed(by)])

    return order_func
