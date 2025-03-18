import sys
import warnings
from collections.abc import Callable

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from .utils import _compute_probabilities, _find_scaling_factor, _set_node_attributes


def geo_erdos_renyi_network(
    gdf,
    *,
    a=3,
    scaling_factor: float | None = None,
    max_degree=150,
    id_col: str | None = None,
    node_attributes: bool | str | list[str] = True,
    constraint: Callable | None = None,
    random_state: int | None = None,
    verbose: bool = False,
) -> nx.Graph:
    r"""Construct a geo erdos-renyi network using the Geospatial Erdős-Rényi model

    The Geospatial Erdős-Rényi model is a variant of the Erdős-Rényi model that incorporates spatial considerations.

    Each possible edge in the network is connected with probability:

    .. math::
        p(\textrm{distance}|a, \textrm{min_dist}) = \textrm{min}\left(1, \left(\frac{\textrm{distance}}{\textrm{min_dist}}\right) ^ {-a}\right)

    where min_dist is the minimum distance between nodes, and a is the distance decay exponent parameter, default is 3.
    The minimum distance is a threshold, below which nodes are connected with probability 1, if an edge is chosen to be rewired.
    It is 1/20 of the bounding box diagonal by default. Users can set the scaling factor directly if needed, which is the inverse of the minimum distance.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing nodes
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

        random_state (int | None): random seed for reproducibility, default is None.

        verbose (bool): whether to show detailed progress messages, default is False

    Returns:
        nx.Graph: a geo erdos-renyi network graph
    """
    # Set logger level based on verbose flag
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")
    logger.debug(
        f"Building geo erdos-renyi network with a={a}, scaling_factor={scaling_factor}, max_degree={max_degree}"
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
    else:
        id_col_array = np.arange(len(gdf))

    # use centroid if geometry is a polygon
    if gdf.geometry.geom_type.iloc[0] == "Polygon":
        positions = np.column_stack(
            [gdf.geometry.centroid.x.values, gdf.geometry.centroid.y.values]
        )
    else:
        positions = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])

    if scaling_factor is None:
        scaling_factor = _find_scaling_factor(gdf)
    np_rng = np.random.default_rng(seed=random_state)
    degree_centrality_array = np.zeros(len(gdf))
    graph = nx.Graph()
    for this_node_idx in tqdm(
        range(len(gdf)), desc="Creating geo erdos-renyi network", disable=not verbose
    ):
        if degree_centrality_array[this_node_idx] >= max_degree:
            continue
        this_node_graph_id = id_col_array[this_node_idx]
        if this_node_graph_id not in graph:
            graph.add_node(this_node_graph_id)

        candidate_node_idx = np.arange(len(gdf))
        # avoid self-loops
        candidate_node_mask = candidate_node_idx != this_node_idx
        # check if the maximum degree centrality has been reached
        candidate_node_mask &= degree_centrality_array < max_degree
        if constraint is not None:
            constraint_mask = np.array(
                [
                    constraint(gdf.iloc[this_node_idx], gdf.iloc[i])
                    for i in range(len(gdf))
                ]
            )
            candidate_node_mask &= constraint_mask
        candidate_node_idx = candidate_node_idx[candidate_node_mask]

        if len(candidate_node_idx) == 0:
            logger.warning(
                f"No valid nodes to connect for node {this_node_graph_id}. Try adjusting the parameters."
            )
            continue
        # compute probabilities of connecting to other nodes
        distances = np.linalg.norm(
            positions[this_node_idx] - positions[candidate_node_idx], axis=1
        )
        q = _compute_probabilities(distances, a=a, scaling_factor=scaling_factor)
        # for each possible edge, decide whether to connect or not
        selected = np_rng.random(size=len(candidate_node_idx)) < q
        for i, that_node_idx in enumerate(candidate_node_idx[selected]):
            # avoid exceeding the maximum degree centrality
            if degree_centrality_array[this_node_idx] >= max_degree:
                break
            that_node_graph_id = id_col_array[that_node_idx]
            # avoid multiple edges
            if graph.has_edge(this_node_graph_id, that_node_graph_id):
                continue
            if that_node_graph_id not in graph:
                graph.add_node(that_node_graph_id)
            graph.add_edge(this_node_graph_id, that_node_graph_id, length=distances[i])
            degree_centrality_array[this_node_idx] += 1
            degree_centrality_array[that_node_idx] += 1

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
            f"Finished building geo erdos-renyi network with {total_edges:,} edges and {graph.number_of_nodes():,} nodes",
            f"average degree: {total_edges * 2 / graph.number_of_nodes():.2f}",
        )
    return graph
