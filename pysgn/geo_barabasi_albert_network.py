import sys
import warnings
from collections.abc import Callable

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from .utils import _compute_probabilities, _find_scaling_factor, _set_node_attributes


def geo_barabasi_albert_network(
    gdf,
    m: int,
    *,
    a: int = 3,
    scaling_factor: float | None = None,
    max_degree: int = 150,
    id_col: str | None = None,
    node_attributes: bool | str | list[str] = True,
    constraint: Callable | None = None,
    node_order: Callable[[pd.DataFrame], np.ndarray] | str | None = None,
    random_state: int | None = None,
    verbose: bool = False,
) -> nx.Graph:
    r"""Construct a geo Barabási-Albert network with geospatial preferential attachment.

    The Geospatial Barabási-Albert (BA) model is a geospatial modification of the classical BA model,
    incorporating spatial factors into the preferential attachment mechanism. When adding a new node,
    the probability of attaching to an existing node is proportional to the existing node's degree and
    a geospatial decay function based on the distance between the nodes:

    .. math::
        p_i(\textrm{distance}|a, \textrm{min_dist}) \propto k_i \cdot \textrm{min}\left(1, \left(\frac{\textrm{distance}}{\textrm{min_dist}}\right) ^ {-a}\right)

    where :math:`k_i` is the degree of existing node i, min_dist is the minimum distance between nodes,
    and a is the distance decay exponent parameter, default is 3. The minimum distance is a threshold,
    below which nodes are connected with probability 1, if an edge is chosen to be rewired. It is 1/20
    of the bounding box diagonal by default. Users can set the scaling factor directly if needed, which
    is the inverse of the minimum distance.

    The new node attaches to m different nodes chosen without replacement with these normalized
    probabilities.

    For the first m nodes, a seed network is created by fully connecting them. The seed network
    is then used to grow the network by adding one node at a time.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing nodes.

        m (int): Number of edges to attach from a new node to existing nodes (and size of the seed network).

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

        node_attributes (bool | str | list[str]): node attributes to save in the graph, default is True.
                                                  If True, all attributes will be saved as node attributes.
                                                  If False, only the position of the nodes will be saved as a `pos` attribute.
                                                  If a string or a list of strings, the attributes will be saved as node attributes.

        constraint (Callable | None): constraint function to filter out invalid neighbors, default is None
                                      Example: constraint=lambda u, v: u.household != v.household
                                      This will ensure that nodes from the same household are not connected.

        node_order (Callable[[gpd.GeoDataFrame], np.ndarray] | str | None): A function or column name to determine the order in which nodes are added.
                                                                            If None, nodes are added sequentially as they appear in the GeoDataFrame.
                                                                            If a callable, the function should take a GeoDataFrame and return an array of indices.
                                                                            If a string, the string is interpreted as a column name that contains order indices.

        random_state (int | None): random seed for reproducibility, default is None.

        verbose (bool): whether to show detailed progress messages, default is False


    Returns:
        nx.Graph: a geo barabasi-albert network graph
    """
    # Set logger level based on verbose flag
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")
    logger.debug(
        f"Building geo Barabási-Albert network with m={m}, a={a}, scaling_factor={scaling_factor}, max_degree={max_degree}"
    )

    if gdf.crs and gdf.crs.is_geographic:
        warnings.warn(
            "Geometry is in a geographic CRS. Results from distance calculations may be incorrect. "
            "Consider re-projecting using 'to_crs()'.",
            UserWarning,
            stacklevel=2,
        )

    # Determine node addition order.
    if node_order is None:
        order = np.arange(len(gdf))
    elif callable(node_order):
        order = node_order(gdf)
    elif isinstance(node_order, str):
        # Interpret the string as a column name that contains order indices.
        order = np.argsort(gdf[node_order].values)
    else:
        raise ValueError(
            "node_order must be None, a callable, or a string representing a column name"
        )

    if len(order) != len(gdf):
        raise ValueError(
            "The node_order must return an array of indices of the same length as gdf"
        )

    # Determine node IDs based on id_col if provided.
    if id_col is not None:
        if id_col == "index" and isinstance(gdf.index, pd.MultiIndex):
            raise ValueError("Multi-index is not supported")
        id_values = gdf.index.values if id_col == "index" else gdf[id_col].values
        # Reorder IDs according to node addition order.
        id_values = id_values[order]
    else:
        id_values = np.arange(len(gdf))[order]

    if len(np.unique(id_values)) != len(id_values):
        raise ValueError("ID column must contain unique values")

    # Compute scaling_factor if not provided.
    if scaling_factor is None:
        scaling_factor = _find_scaling_factor(gdf)

    # Extract positions from the GeoDataFrame and reorder them.
    if gdf.geometry.geom_type.iloc[0] == "Polygon":
        positions = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])[
            order
        ]
    else:
        positions = np.column_stack([gdf.geometry.x, gdf.geometry.y])[order]

    # Initialize the graph and create the seed network.
    if len(gdf) < m:
        raise ValueError(
            f"Number of nodes ({len(gdf)}) must be at least m ({m}) for the seed network"
        )
    degree_centrality_array = np.zeros(len(gdf))
    graph = nx.Graph()
    seed_count = m
    for i in range(seed_count):
        node_id = id_values[i]
        graph.add_node(node_id)
    # Fully connect the seed nodes (subject to max_degree and constraint).
    for i in range(seed_count):
        for j in range(i + 1, seed_count):
            if (
                graph.degree(id_values[i]) >= max_degree
                or graph.degree(id_values[j]) >= max_degree
                or (
                    constraint is not None
                    and not constraint(gdf.iloc[order[i]], gdf.iloc[order[j]])
                )
            ):
                continue
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distance = (dx**2 + dy**2) ** 0.5
            graph.add_edge(id_values[i], id_values[j], length=distance)
            degree_centrality_array[i] += 1
            degree_centrality_array[j] += 1

    np_rng = np.random.default_rng(seed=random_state)

    # Grow the network by adding one node at a time.
    for new_idx in tqdm(
        range(seed_count, len(gdf)),
        desc="Creating geo barabasi-albert network",
        disable=not verbose,
    ):
        new_node_id = id_values[new_idx]
        new_node_pos = positions[new_idx]
        new_node_data = gdf.iloc[order[new_idx]]
        graph.add_node(new_node_id)

        max_degree_mask = degree_centrality_array[:new_idx] < max_degree
        if constraint is not None:
            constraint_mask = np.array(
                [constraint(new_node_data, gdf.iloc[order[i]]) for i in range(new_idx)]
            )
            candidate_node_idx = np.arange(new_idx)[max_degree_mask & constraint_mask]
        else:
            candidate_node_idx = np.arange(new_idx)[max_degree_mask]
        if len(candidate_node_idx) == 0:
            logger.warning(
                f"Skipping attachment for new node {new_node_id}: no available nodes to attach to."
            )
            continue

        # Compute the weight for each candidate node.
        candidate_weights = (
            degree_centrality_array[candidate_node_idx] + 1
        ) * _compute_probabilities(
            np.linalg.norm(positions[candidate_node_idx] - new_node_pos, axis=1),
            a=a,
            scaling_factor=scaling_factor,
        )
        candidate_weights /= candidate_weights.sum()

        # The new node will try to attach to m distinct existing nodes.
        num_attachments = min(m, len(candidate_node_idx))
        selected_indices = np_rng.choice(
            candidate_node_idx, size=num_attachments, replace=False, p=candidate_weights
        )

        for idx in selected_indices:
            target_node_id = id_values[idx]
            distance = np.linalg.norm(new_node_pos - positions[idx])
            if not graph.has_edge(new_node_id, target_node_id):
                graph.add_edge(new_node_id, target_node_id, length=distance)
                degree_centrality_array[new_idx] += 1
                degree_centrality_array[idx] += 1

    _set_node_attributes(graph, gdf, id_col, node_attributes)
    total_edges = graph.number_of_edges()
    if total_edges == 0:
        warnings.warn(
            "No edges were created. Try adjusting the parameters.",
            UserWarning,
            stacklevel=2,
        )
    else:
        logger.debug(
            f"Finished building geo Barabási-Albert network with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges",
            f"average degree: {total_edges * 2 / graph.number_of_nodes():.2f}",
        )
    return graph
