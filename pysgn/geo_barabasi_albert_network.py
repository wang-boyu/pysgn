import sys
import warnings
from collections.abc import Callable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from .utils import _find_scaling_factor, _set_node_attributes


def geo_barabasi_albert_network(
    gdf,
    *,
    a: int = 3,
    scaling_factor: float | None = None,
    max_degree: int = 150,
    id_col: str | None = None,
    node_attributes: bool | str | list[str] = True,
    constraint: Callable | None = None,
    node_order: Callable[[gpd.GeoDataFrame], np.ndarray] | str | None = None,
    random_state: int | None = None,
    verbose: bool = False,
) -> nx.Graph:
    r"""Construct a geo barabasi-albert network using the Geospatial Barabási-Albert model

    The Geospatial Barabási-Albert model is a variant of the Barabási-Albert model that incorporates spatial considerations.
    Each new node connects to m existing nodes with probability proportional to both their degree and geographic distance:

    .. math::
        P(i) \propto k_i \cdot \textrm{min}\left(1, \left(\frac{\textrm{distance}}{\textrm{min_dist}}\right) ^ {-a}\right)

    where k_i is the degree of node i, min_dist is the minimum distance between nodes, and a is the distance decay
    exponent parameter, default is 3.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing nodes

    Keyword Args:
        a (int): distance decay exponent parameter, default is 3

        scaling_factor (float): scaling factor is the inverse of the minimum distance between nodes, default is None.
                               The minimum distance is a threshold, below which nodes are connected with probability 1.
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

        node_order (Callable[[gpd.GeoDataFrame], np.ndarray] | str | None): function or column name to determine the order of node addition, default is None.
                                                                        If None, nodes are added sequentially (traditional BA model).
                                                                        If a function, it should take a GeoDataFrame as input and return an array of indices.
                                                                        If a string, it should be a column name in the GeoDataFrame containing the order indices.
                                                                        Built-in ordering strategies are available in pysgn.ordering module.

        random_state (int | None): random seed for reproducibility, default is None

        verbose (bool): whether to show detailed progress messages, default is False

    Returns:
        nx.Graph: a geo barabasi-albert network graph with m edges per new node

    Examples:
        >>> import geopandas as gpd
        >>> from pysgn import geo_barabasi_albert_network
        >>> from pysgn.ordering import density_order, random_order, attribute_order
        >>>
        >>> # Load data
        >>> gdf = gpd.read_file("points.gpkg")
        >>>
        >>> # Create network with sequential ordering (traditional)
        >>> G1 = geo_barabasi_albert_network(gdf)
        >>>
        >>> # Create network with density-based ordering
        >>> node_order = density_order(gdf, method="kde")
        >>> G2 = geo_barabasi_albert_network(gdf, node_order=node_order)
        >>>
        >>> # Create network with population-based ordering
        >>> node_order = attribute_order(gdf, by="population")
        >>> G3 = geo_barabasi_albert_network(gdf, node_order=node_order)
        >>>
        >>> # Use a precomputed order stored in a column
        >>> gdf['order'] = attribute_order(gdf, by="population")
        >>> G4 = geo_barabasi_albert_network(gdf, node_order='order')
    """
    # Set logger level based on verbose flag
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")
    logger.debug(
        f"Building geo barabasi-albert network with a={a}, scaling_factor={scaling_factor}, max_degree={max_degree}"
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

    if id_col is not None:
        if id_col == "index" and isinstance(gdf.index, pd.MultiIndex):
            raise ValueError("Multi-index is not supported")
        id_col_array = gdf.index.values if id_col == "index" else gdf[id_col].values
        if len(np.unique(id_col_array)) != len(id_col_array):
            raise ValueError("ID column must contain unique values")

    # Get positions from either points or polygon centroids
    if gdf.geometry.geom_type.iloc[0] == "Polygon":
        pos_x_array = gdf.geometry.centroid.x.values
        pos_y_array = gdf.geometry.centroid.y.values
    else:
        pos_x_array = gdf.geometry.x.values
        pos_y_array = gdf.geometry.y.values

    if scaling_factor is None:
        scaling_factor = _find_scaling_factor(gdf)

    np_rng = np.random.default_rng(seed=random_state)
    graph = nx.Graph()
    degree_centrality_array = np.ones(len(gdf))

    # Get node order if provided
    if node_order is not None:
        if isinstance(node_order, str):
            if node_order not in gdf.columns:
                raise ValueError(f"Column '{node_order}' not found in GeoDataFrame")
            order = gdf[node_order].values
        elif callable(node_order):
            try:
                order = node_order(gdf)
                if not isinstance(order, np.ndarray):
                    raise TypeError("node_order function must return a numpy array")
                if len(order) != len(gdf):
                    raise ValueError(
                        "node_order function must return an array of length equal to the GeoDataFrame"
                    )
                if not np.array_equal(np.sort(order), np.arange(len(gdf))):
                    raise ValueError(
                        "node_order function must return a permutation of indices"
                    )
            except Exception as e:
                raise ValueError(f"Error in node_order function: {e!s}") from e
        else:
            raise TypeError("node_order must be a callable or a column name")
    else:
        order = np.arange(len(gdf))

    node_idx_ordered = np.argsort(order)

    # Starting from the second node with index 1
    for i, source_idx in tqdm(
        enumerate(node_idx_ordered[1:], 1),
        desc="Creating geo barabasi-albert network",
        disable=not verbose,
    ):
        source_id = id_col_array[source_idx] if id_col else source_idx
        if source_id not in graph:
            graph.add_node(source_id)

        target_idx_array = node_idx_ordered[:i]
        target_idx_mask = np.zeros(len(gdf), dtype=bool)
        # Only consider nodes that have been added
        target_idx_mask[target_idx_array] = True
        # Only consider nodes with degree less than max_degree
        target_idx_mask &= degree_centrality_array[target_idx_mask] < max_degree
        # Only consider nodes that satisfy constraint
        for target_idx in np.where(target_idx_mask)[0]:
            if constraint is not None and not constraint(
                gdf.iloc[source_idx], gdf.iloc[target_idx]
            ):
                target_idx_mask[target_idx] = False
        target_idx_array = np.where(target_idx_mask)[0]

        valid_target_positions = np.stack(
            [pos_x_array[target_idx_mask], pos_y_array[target_idx_mask]], axis=1
        )
        distances = np.linalg.norm(
            valid_target_positions - [pos_x_array[source_idx], pos_y_array[source_idx]],
            axis=1,
        )
        distance_factors = np.where(
            distances < 1 / scaling_factor, 1, (distances * scaling_factor) ** (-a)
        )
        k_array = degree_centrality_array[target_idx_mask]
        # Avoid zero probabilities when degree is zero
        k_array[k_array == 0] = 1
        probs = k_array * distance_factors
        if probs.sum() == 0:
            warnings.warn(
                f"Node {source_id} has no valid targets to connect to. "
                f"Consider reducing constraints for this node:\n{gdf.iloc[source_idx].to_frame().T}",
                UserWarning,
                stacklevel=2,
            )
        elif probs.sum() < 1:
            # Normalize probabilities
            probs /= probs.sum()

        # Decide whether to connect to each target
        selected_targets = target_idx_array[np_rng.random(size=len(probs)) < probs]
        # Make sure source node degree is less than max_degree
        selected_targets = selected_targets[
            : max_degree - degree_centrality_array[source_idx]
        ]

        for j, target_idx in enumerate(selected_targets):
            target_id = id_col_array[target_idx] if id_col else target_idx
            graph.add_edge(source_id, target_id, length=distances[j])
            degree_centrality_array[source_idx] += 1
            degree_centrality_array[target_idx] += 1

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
            f"Finished building geo barabasi-albert network with {total_edges:,} edges and {graph.number_of_nodes():,} nodes. "
            f"Average degree: {total_edges * 2 / graph.number_of_nodes():.2f}"
        )

    return graph
