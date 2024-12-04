import sys
import warnings
from collections import defaultdict
from collections.abc import Callable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm

from .utils import _create_k_col, _find_scaling_factor, _set_node_attributes


def _get_nearest_nodes(
    gdf: gpd.GeoDataFrame,
    k: int | float | str,
    *,
    max_degree: int,
    query_factor: int = 2,
    constraint: Callable | None = None,
    random_state: int | None = None,
    verbose: bool = False,
) -> tuple[dict[int, list[int]], np.ndarray]:
    """
    Find the nearest neighbors for each node

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing nodes
        k (int | float | str): number of nearest neighbors to connect initially
                               If a number, it determines the number of nearest neighbors to connect initially, also the average degree of the network.
                               If a string, it determines the column name containing expected degree centrality for each node, when connecting to neighbors initially.
        max_degree (int): maximum degree centrality allowed
        query_factor (int): factor of k to query neighbors initially to handle constraint filtering
        constraint (Callable | None): constraint function to filter out invalid neighbors, default is None
                                      Example: constraint=lambda u, v: u.household != v.household
                                      This will ensure that nodes from the same household are not connected.
        random_state (int | None): random seed for reproducibility, default is None.
        verbose (bool): whether to show detailed progress messages, default is False

    Returns:
        dict[int, list[int]]: dictionary of nearest neighbors for each node
        np.ndarray: array of degree centrality for each node
    """
    logger.debug("Building KDTree for efficient nearest neighbor search")
    degree_centrality_array = np.zeros(len(gdf))
    if isinstance(k, str):
        k_col = gdf[k].values // 2
    elif isinstance(k, int | float):
        k_col = _create_k_col(k, len(gdf), random_state=random_state)
    else:
        raise ValueError("k must be an integer, a float, or a string")
    k_col = np.clip(k_col, 0, max_degree)

    # Get positions from either points or polygon centroids
    if gdf.geometry.geom_type.iloc[0] == "Polygon":
        positions = np.stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y], axis=1)
    else:
        positions = np.stack([gdf.geometry.x, gdf.geometry.y], axis=1)

    kdtree = KDTree(positions, metric="euclidean")

    nearest_neighbors = defaultdict(list)
    desc = (
        f"Finding {k} nearest neighbors"
        if isinstance(k, int)
        else f"Finding nearest neighbors based on column {k}"
    )

    # Step 2: Find nearest neighbors using KDTree queries with constraint handling
    for this_node_idx in tqdm(range(len(gdf)), desc=desc, disable=not verbose):
        expected_num_neighbors = k_col[this_node_idx]
        if expected_num_neighbors > max_degree:
            logger.error(
                f"Node {this_node_idx} has expected degree {expected_num_neighbors} > max degree {max_degree}. Skipping node."
            )
            continue
        if expected_num_neighbors > len(gdf):
            logger.error(
                f"Node {this_node_idx} has expected degree {expected_num_neighbors} > number of nodes {len(gdf)}. Skipping node."
            )
            continue
        neighbors_set = set()  # Track neighbors in a set to avoid duplicates
        # Initially query more neighbors to account for filtering
        query_k = min(expected_num_neighbors * query_factor, len(gdf) - 1)
        while len(neighbors_set) < expected_num_neighbors:
            # Query KDTree for the next batch of neighbors
            _, idxs = kdtree.query(
                [positions[this_node_idx]], k=query_k + 1
            )  # +1 to avoid self-loop

            for new_node_idx in idxs[0]:
                if new_node_idx == this_node_idx:  # Skip the node itself
                    continue
                if new_node_idx in neighbors_set:  # Skip if already added
                    continue
                # Avoid nodes with degree centrality >= max_degree
                if degree_centrality_array[this_node_idx] >= max_degree:
                    continue
                # Avoid nodes with degree centrality >= max_degree
                expected_num_neighbors_of_new_node = k_col[new_node_idx]
                if (
                    degree_centrality_array[new_node_idx]
                    + expected_num_neighbors_of_new_node
                    >= max_degree
                ):
                    continue
                # Avoid double counting neighbors
                if this_node_idx in nearest_neighbors[new_node_idx]:
                    continue
                # Apply constraint if provided
                if constraint is not None and not constraint(
                    gdf.iloc[this_node_idx], gdf.iloc[new_node_idx]
                ):
                    continue

                neighbors_set.add(new_node_idx)
                degree_centrality_array[this_node_idx] += 1
                degree_centrality_array[new_node_idx] += 1

                # If we've found enough valid neighbors, break the loop
                if len(neighbors_set) >= expected_num_neighbors:
                    break

            # If we've reached all nodes, break the loop
            if query_k == len(gdf) - 1:
                if len(neighbors_set) < expected_num_neighbors:
                    warnings.warn(
                        f"Node at index {this_node_idx} has only {len(neighbors_set)} neighbors out of expected {expected_num_neighbors}. "
                        f"Consider reducing the expected degree for this node:\n{gdf.iloc[this_node_idx].to_frame().T}",
                        UserWarning,
                        stacklevel=2,
                    )
                break

            # If we still don't have enough neighbors, increase query size and try again
            if len(neighbors_set) < expected_num_neighbors:
                # Increase query size to find more neighbors
                query_k = min(query_k + query_factor, len(gdf) - 1)

        # Step 3: Assign the valid neighbors and update degree centrality
        nearest_neighbors[this_node_idx] = list(neighbors_set)

    return nearest_neighbors, degree_centrality_array


def geo_watts_strogatz_network(
    gdf,
    k: int | str,
    p: float,
    *,
    a=3,
    scaling_factor: float | None = None,
    max_degree=150,
    id_col: str | None = None,
    query_factor: int = 2,
    node_attributes: bool | str | list[str] = True,
    constraint: Callable | None = None,
    random_state: int | None = None,
    verbose: bool = False,
) -> nx.Graph:
    r"""Construct a geo watts-strogatz network using the Geospatial Watts-Strogatz model

    The Geospatial Watts-Strogatz model is a variant of the Watts-Strogatz model that incorporates spatial considerations.

    First, the model connects each node to its k nearest neighbors.

    Then, it rewires each edge with probability p. When an edge is rewired, it is removed and a new edge is added to a random node.
    The probability of being rewired to a new node is determined by the distance between the nodes:

    .. math::
        p(\textrm{distance}|a, \textrm{min_dist}) = \textrm{min}\left(1, \left(\frac{\textrm{distance}}{\textrm{min_dist}}\right) ^ {-a}\right)

    where min_dist is the minimum distance between nodes, and a is the distance decay exponent parameter, default is 3.
    The minimum distance is a threshold, below which nodes are connected with probability 1, if an edge is chosen to be rewired.
    It is 1/20 of the bounding box diagonal by default. Users can set the scaling factor directly if needed, which is the inverse of the minimum distance.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing nodes

        k (int | str): number of nearest neighbors to connect initially
                       If a number, it determines the number of nearest neighbors to connect initially, also the average degree of the network.
                       If a string, it determines the column name containing expected degree centrality for each node, when connecting to neighbors initially.

        p (float): probability of rewiring an edge

    Keyword Args:
        a (int): distance decay exponent parameter, default is 3

        scaling_factor (float): scaling factor is the inverse of the minimum distance between nodes, default is None.
                                The minimum distance is a threshold, below which nodes are connected with probability 1,
                                if an edge is chosen to be rewired.
                                If None, the scaling factor will be calculated based on the bounding box of the GeoDataFrame.

        max_degree (int): maximum degree centrality allowed, default is 150

        id_col (str): column name containing unique IDs, default is None.
                      If "index", the index of the GeoDataFrame will be used as the unique ID.
                      If a column name, the values in the column will be used as the unique ID.
                      If None, the positional index of the node will be used as the unique ID.

        query_factor (int): factor of k to query neighbors initially to handle constraint filtering, default is 2

        node_attributes (bool | str | list[str]): node attributes to save in the graph, default is True.
                                                  If True, all attributes will be saved as node attributes.
                                                  If False, only the position of the nodes will be saved as a `pos` attribute.
                                                  If a string or a list of strings, the attributes will be saved as node attributes.

        constraint (Callable | None): constraint function to filter out invalid neighbors, default is None
                                      Example: constraint=lambda u, v: u.household != v.household
                                      This will ensure that nodes from the same household are not connected.

        random_state (int | None): random seed for reproducibility, default is None.

        verbose (bool): whether to show detailed progress messages, default is False

    Returns:
        nx.Graph: a geo watts-strogatz network graph with average degree k, maximum degree max_degree
    """
    # Set logger level based on verbose flag
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")
    logger.debug(
        f"Building geo watts-strogatz network with k={k}, p={p}, a={a}, scaling_factor={scaling_factor}, max_degree={max_degree}"
    )
    if gdf.crs and gdf.crs.is_geographic:
        warnings.warn(
            "Geometry is in a geographic CRS. "
            "Results from distance calculations are likely incorrect. "
            "Use 'GeoDataFrame.to_crs()' to re-project geometries to a "
            "projected CRS before this operation.\n",
            UserWarning,
            stacklevel=2,
        )
    if k == 0:
        raise ValueError("k must be greater than 0")
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    if id_col is not None:
        if id_col == "index" and isinstance(gdf.index, pd.MultiIndex):
            raise ValueError("Multi-index is not supported")
        id_col_array = gdf.index.values if id_col == "index" else gdf[id_col].values
        if len(np.unique(id_col_array)) != len(id_col_array):
            raise ValueError("ID column must contain unique values")
    if isinstance(k, int | float | str):
        nearest_neighbors, degree_centrality_array = _get_nearest_nodes(
            gdf,
            k,
            max_degree=max_degree,
            query_factor=query_factor,
            constraint=constraint,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        raise ValueError("k must be an integer, a float, or a string")
    rewire_count = 0
    graph = nx.Graph()
    # use centroid if geometry is a polygon
    if gdf.geometry.geom_type.iloc[0] == "Polygon":
        pos_x_array = gdf.geometry.centroid.x.values
        pos_y_array = gdf.geometry.centroid.y.values
    else:
        pos_x_array = gdf.geometry.x.values
        pos_y_array = gdf.geometry.y.values
    for this_node_idx in tqdm(
        nearest_neighbors,
        desc="Creating initial network from nearest neighbors",
        disable=not verbose,
    ):
        for neighboring_node_idx in nearest_neighbors[this_node_idx]:
            distance = (
                float(pos_x_array[this_node_idx] - pos_x_array[neighboring_node_idx])
                ** 2
                + (
                    float(
                        pos_y_array[this_node_idx] - pos_y_array[neighboring_node_idx]
                    )
                    ** 2
                )
            ) ** 0.5
            this_node_graph_id = (
                id_col_array[this_node_idx] if id_col else this_node_idx
            )
            if this_node_graph_id not in graph:
                graph.add_node(this_node_graph_id)
            neighboring_node_graph_id = (
                id_col_array[neighboring_node_idx] if id_col else neighboring_node_idx
            )
            if neighboring_node_graph_id not in graph:
                graph.add_node(neighboring_node_graph_id)
            graph.add_edge(
                this_node_graph_id, neighboring_node_graph_id, length=distance
            )
    if scaling_factor is None:
        scaling_factor = _find_scaling_factor(gdf)
    np_rng = np.random.default_rng(seed=random_state)
    # connect each node to k/2 neighbors
    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for this_node_idx in tqdm(
        nearest_neighbors,
        desc="Rewiring edges in geo watts-strogatz network",
        disable=not verbose,
    ):
        for neighboring_node_idx in nearest_neighbors[this_node_idx]:
            this_node_graph_id = (
                id_col_array[this_node_idx] if id_col else this_node_idx
            )
            neighboring_node_graph_id = (
                id_col_array[neighboring_node_idx] if id_col else neighboring_node_idx
            )
            if np_rng.random() < p:
                chosen = False
                while not chosen:
                    # get a random position index from gdf
                    random_node_idx = np_rng.integers(0, len(gdf))
                    random_node_graph_id = (
                        id_col_array[random_node_idx] if id_col else random_node_idx
                    )
                    checked_nodes = {random_node_idx}
                    # Enforce no self-loops, or multiple edges, or degree >= max_degree, or constraint
                    while (
                        random_node_idx == this_node_idx
                        or graph.has_edge(this_node_graph_id, random_node_graph_id)
                        or degree_centrality_array[random_node_idx] >= max_degree
                        or (
                            constraint is not None
                            and not constraint(
                                gdf.iloc[this_node_idx], gdf.iloc[random_node_idx]
                            )
                        )
                    ):
                        random_node_idx = np_rng.integers(0, len(gdf))
                        random_node_graph_id = (
                            id_col_array[random_node_idx] if id_col else random_node_idx
                        )
                        checked_nodes.add(random_node_idx)
                        if len(checked_nodes) == len(gdf):
                            break

                    if len(checked_nodes) == len(gdf):
                        warnings.warn(
                            f"Node {this_node_graph_id} has exhausted all possible rewiring options. Skipping."
                            f"Consider reducing the constraints for this node:\n{gdf.iloc[this_node_idx].to_frame().T}",
                            UserWarning,
                            stacklevel=2,
                        )
                        break
                    distance = (
                        float(pos_x_array[this_node_idx] - pos_x_array[random_node_idx])
                        ** 2
                        + (
                            float(
                                pos_y_array[this_node_idx]
                                - pos_y_array[random_node_idx]
                            )
                            ** 2
                        )
                    ) ** 0.5
                    # if distance is less than minimum distance, connect with probability 1
                    # minimum distance is determined by method _find_scaling_factor
                    if distance < 1 / scaling_factor:
                        q = 1
                    # else, connect with probability (distance / min_dist) ^ (-a)
                    # where min_dist is the minimum distance
                    # and a is the distance decay parameter
                    else:
                        q = (distance * scaling_factor) ** (-a)
                    if np_rng.random() < q:
                        graph.remove_edge(this_node_graph_id, neighboring_node_graph_id)
                        graph.add_edge(
                            this_node_graph_id, random_node_graph_id, length=distance
                        )
                        degree_centrality_array[neighboring_node_idx] -= 1
                        degree_centrality_array[random_node_idx] += 1
                        rewire_count += 1
                        chosen = True
    _set_node_attributes(graph, gdf, id_col, node_attributes)
    total_edges = graph.number_of_edges()
    logger.debug(
        f"Rewire Count: {rewire_count:,} edges out of {total_edges:,}. {rewire_count / total_edges * 100:.2f}% of edges rewired"
    )
    return graph
