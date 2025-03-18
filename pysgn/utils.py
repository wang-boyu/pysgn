import networkx as nx
import numpy as np
from loguru import logger


def _compute_probabilities(
    distances: np.ndarray, a: int, scaling_factor: float
) -> np.ndarray:
    """
    Compute the probability of connecting two nodes based on their distance.

    The probability is calculated as:
    p(distance) = min(1, (distance * scaling_factor) ^ (-a))

    p = 1 if distance < 1 / scaling_factor
    p = (distance * scaling_factor) ^ (-a) if distance >= 1 / scaling_factor

    Args:
        distances: An array of distances between nodes.
        a: The distance decay exponent parameter.
        scaling_factor: The inverse of the minimum distance between nodes.
    """
    factors = np.ones_like(distances)
    mask = distances >= (1 / scaling_factor)
    factors[mask] = (distances[mask] * scaling_factor) ** (-a)
    return factors


def _create_k_col(k, n, random_state=None) -> np.ndarray:
    """
    Creates a column of integer degree centrality so that its mean is equal to k/2.

    - If k is even, all values are k//2.
    - If k is odd, half of the values are k//2 and the other half are k//2 + 1.
    - If k is float, a fraction of the values are floor(k/2) and the rest are ceil(k/2).
      The fraction is determined by the decimal part of k/2. For example, if k/2 = 3.2,
      80% of the values are 3 and 20% are 4.

    Args:
        k: The target mean degree centrality.
        n: The number of elements in the column.
        random_state: Seed for random number generator for reproducibility.

    Returns:
        np.ndarray: An array of integer degree centralities.
    """
    rng = np.random.default_rng(random_state)
    if isinstance(k, int):
        if k % 2 == 0:
            return np.full(n, k // 2, dtype=int)
        else:
            return rng.choice([k // 2, k // 2 + 1], n, p=[0.5, 0.5])
    elif isinstance(k, float):
        half_k = k / 2
        lower = int(np.floor(half_k))
        upper = int(np.ceil(half_k))
        # prevent `ZeroDivisionError: integer modulo by zero` if half_k < 1
        upper_prob = half_k if half_k < 1 else half_k % lower
        return rng.choice([lower, upper], n, p=[1 - upper_prob, upper_prob])
    else:
        raise ValueError("k must be an integer or a float.")


def _find_scaling_factor(gdf) -> float:
    logger.debug("Finding scaling factor")
    min_x, min_y, max_x, max_y = gdf.geometry.total_bounds
    # calculate the distance between the two corners of the bounding box
    dist = ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5
    # minimum distance is 1/20 of the bounding box diagonal
    # distances less than this will have a probability of 1 to be connected if chosen
    # distances greater than this will have a probability of (distance / min_dist) ^ (-a) to be connected if chosen
    min_dist = dist / 20
    logger.debug(
        f"Scaling factor: {1 / min_dist:.10f}, Minimum distance: {min_dist:.2f}, Bounding box diagonal: {dist:.2f}"
    )
    return 1 / min_dist


def _set_node_attributes(graph, gdf, id_col, node_attributes):
    if gdf.geometry.geom_type.iloc[0] == "Polygon":
        pos_x_array = gdf.geometry.centroid.x.values
        pos_y_array = gdf.geometry.centroid.y.values
    else:
        pos_x_array = gdf.geometry.x.values
        pos_y_array = gdf.geometry.y.values
    if node_attributes is True:
        node_attributes = gdf.columns.tolist()
    if isinstance(node_attributes, str):
        node_attributes = [node_attributes]
    if id_col is None:
        nx.set_node_attributes(
            graph,
            dict(enumerate(zip(pos_x_array, pos_y_array))),
            name="pos",
        )
        if node_attributes:
            for attribute in node_attributes:
                nx.set_node_attributes(
                    graph,
                    dict(enumerate(gdf[attribute].values)),
                    name=attribute,
                )
    else:
        id_col_array = gdf.index.values if id_col == "index" else gdf[id_col].values
        nx.set_node_attributes(
            graph,
            {
                row[0]: row[1]
                for row in zip(id_col_array, zip(pos_x_array, pos_y_array))
            },
            name="pos",
        )
        if node_attributes:
            for attribute in node_attributes:
                nx.set_node_attributes(
                    graph,
                    {
                        row[0]: row[1]
                        for row in zip(id_col_array, gdf[attribute].values)
                    },
                    name=attribute,
                )
