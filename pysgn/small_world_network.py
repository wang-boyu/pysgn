import random as rd
import warnings
from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
from loguru import logger
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm


def _find_scaling_factor(gdf) -> float:
    logger.info("Finding scaling factor")
    min_x, min_y, max_x, max_y = gdf.geometry.total_bounds
    # calculate the distance between the two corners of the bounding box
    dist = ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5
    # minimum distance is 1/20 of the bounding box diagonal
    # distances less than this will have a probability of 1 to be connected if chosen
    # distances greater than this will have a probability of (distance / min_dist) ^ (-a) to be connected if chosen
    min_dist = dist / 20
    logger.info(
        f"Scaling factor: {1 / min_dist:.10f}, Minimum distance: {min_dist:.2f}, Bounding box diagonal: {dist:.2f}"
    )
    return 1 / min_dist


def _get_nearest_nodes(
    gdf: gpd.GeoDataFrame,
    k: int,
    max_degree: int,
    query_factor: int = 2,
) -> tuple[dict[int, list[int]], np.ndarray]:
    """
    Find the nearest neighbors for each agent

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing agents
        k (int): number of nearest neighbors to find
        max_degree (int): maximum degree centrality allowed
        query_factor (int): factor of k to query neighbors initially to handle constraint filtering

    Returns:
        dict[int, list[int]]: dictionary of nearest neighbors for each agent
        np.ndarray: array of degree centrality for each agent
    """
    logger.info("Building KDTree for efficient nearest neighbor search")
    # manipulate numpy arrays directly for faster performance
    # instead of using the .iloc method in pandas
    degree_centrality_array = np.zeros(len(gdf))

    # Step 1: Create a KDTree for fast neighbor lookups
    positions = np.stack([gdf.geometry.x, gdf.geometry.y], axis=1)
    kdtree = KDTree(positions, metric="euclidean")

    nearest_neighbors = defaultdict(list)
    desc = f"Finding {k} nearest neighbors"

    # Step 2: Find nearest neighbors using KDTree queries with constraint handling
    expected_num_neighbors = k
    for this_node_idx in tqdm(range(len(gdf)), desc=desc):
        if expected_num_neighbors > max_degree:
            logger.error(
                f"Agent {this_node_idx} has expected degree {expected_num_neighbors} > max degree {max_degree}. Skipping agent."
            )
            continue
        if expected_num_neighbors > len(gdf):
            logger.error(
                f"Agent {this_node_idx} has expected degree {expected_num_neighbors} > number of agents {len(gdf)}. Skipping agent."
            )
            continue
        neighbors_set = set()  # Track neighbors in a set to avoid duplicates
        # Initially query more neighbors to account for filtering
        query_k = expected_num_neighbors * query_factor
        while len(neighbors_set) < expected_num_neighbors:
            # Query KDTree for the next batch of neighbors
            _, idxs = kdtree.query(
                [positions[this_node_idx]], k=query_k + 1
            )  # +1 to avoid self-loop

            for new_node_idx in idxs[0]:
                if new_node_idx == this_node_idx:  # Skip the agent itself
                    continue
                if new_node_idx in neighbors_set:  # Skip if already added
                    continue
                # Avoid agents with degree centrality >= max_degree
                if degree_centrality_array[this_node_idx] >= max_degree:
                    continue
                # Avoid agents with degree centrality >= max_degree
                expected_num_neighbors_of_n = k
                if (
                    degree_centrality_array[new_node_idx] + expected_num_neighbors_of_n
                    >= max_degree
                ):
                    continue
                # Avoid double counting neighbors
                if this_node_idx in nearest_neighbors[new_node_idx]:
                    continue

                neighbors_set.add(new_node_idx)
                degree_centrality_array[this_node_idx] += 1
                degree_centrality_array[new_node_idx] += 1

                # If we've found enough valid neighbors, break the loop
                if len(neighbors_set) >= expected_num_neighbors:
                    break

            # If we still don't have enough neighbors, increase query size and try again
            if len(neighbors_set) < expected_num_neighbors:
                query_k += (
                    expected_num_neighbors  # Increase the number of neighbors to query
                )

        # Step 3: Assign the valid neighbors and update degree centrality
        nearest_neighbors[this_node_idx] = list(neighbors_set)

    return nearest_neighbors, degree_centrality_array


def _set_node_attributes(graph, gdf, id_col, save_attributes):
    if gdf.geometry.geom_type[0] == "Polygon":
        pos_x_array = gdf.geometry.centroid.x.values
        pos_y_array = gdf.geometry.centroid.y.values
    else:
        pos_x_array = gdf.geometry.x.values
        pos_y_array = gdf.geometry.y.values
    if save_attributes is True:
        save_attributes = gdf.columns.tolist()
    if isinstance(save_attributes, str):
        save_attributes = [save_attributes]
    if id_col is None:
        nx.set_node_attributes(
            graph,
            dict(enumerate(zip(pos_x_array, pos_y_array))),
            name="pos",
        )
        if save_attributes:
            for attribute in save_attributes:
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
        if save_attributes:
            for attribute in save_attributes:
                nx.set_node_attributes(
                    graph,
                    {
                        row[0]: row[1]
                        for row in zip(id_col_array, gdf[attribute].values)
                    },
                    name=attribute,
                )


def small_world_network(
    gdf,
    k,
    p,
    *,
    a=3,
    scaling_factor: float | None = None,
    max_degree=150,
    id_col: str | None = "index",
    query_factor: int = 2,
    save_attributes: bool | str | list[str] = True,
) -> nx.Graph:
    """Construct a small world network using the Geospatial Watts-Strogatz model

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing agents
        k (int): number of nearest neighbors to connect initially, also the average degree of the network
        p (float): probability of rewiring an edge
    Keyword Args:
        a (int): distance decay exponent parameter, default is 3
        scaling_factor (float): scaling factor is the inverse of the minimum distance between agents, default is None.
                                The minimum distance is a threshold, below which agents are connected with probability 1,
                                if an edge is chosen to be rewired.
                                If None, the scaling factor will be calculated based on the bounding box of the GeoDataFrame.
        max_degree (int): maximum degree centrality allowed, default is 150
        id_col (str): column name containing unique IDs, default is "index".
                      If "index", the index of the GeoDataFrame will be used as the unique ID.
                      If a column name, the values in the column will be used as the unique ID.
                      If None, the positional index of the agent will be used as the unique ID.
        query_factor (int): factor of k to query neighbors initially to handle constraint filtering, default is 2
        save_attributes (bool | str | list[str]): attributes to save in the graph, default is True.
                                                  If True, all attributes will be saved as node attributes.
                                                  If False, only the position of the nodes will be saved as a `pos` attribute.
                                                  If a string or a list of strings, the attributes will be saved as node attributes.

    Returns:
        nx.Graph: small world network graph with average degree k, maximum degree max_degree
    """
    logger.info(
        f"Building small world network with k={k}, p={p}, a={a}, max_degree={max_degree}"
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
    nearest_neighbors, degree_centrality_array = _get_nearest_nodes(
        gdf, k // 2, max_degree=max_degree, query_factor=query_factor
    )
    rewire_count = 0
    graph = nx.Graph()
    if id_col is not None:
        id_col_array = gdf.index.values if id_col == "index" else gdf[id_col].values
        if len(np.unique(id_col_array)) != len(id_col_array):
            raise RuntimeError("ID column must contain unique values")
    # use centroid if geometry is a polygon
    if gdf.geometry.geom_type[0] == "Polygon":
        pos_x_array = gdf.geometry.centroid.x.values
        pos_y_array = gdf.geometry.centroid.y.values
    else:
        pos_x_array = gdf.geometry.x.values
        pos_y_array = gdf.geometry.y.values
    for this_node_idx in tqdm(
        nearest_neighbors, desc="Creating initial network from nearest neighbors"
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
            neighboring_node_graph_id = (
                id_col_array[neighboring_node_idx] if id_col else neighboring_node_idx
            )
            graph.add_edge(
                this_node_graph_id, neighboring_node_graph_id, length=distance
            )
    if scaling_factor is None:
        scaling_factor = _find_scaling_factor(gdf)
    # connect each node to k/2 neighbors
    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for this_node_idx in tqdm(
        nearest_neighbors, desc="Rewiring edges in small world network"
    ):
        for neighboring_node_idx in nearest_neighbors[this_node_idx]:
            this_node_graph_id = (
                id_col_array[this_node_idx] if id_col else this_node_idx
            )
            neighboring_node_graph_id = (
                id_col_array[neighboring_node_idx] if id_col else neighboring_node_idx
            )
            if rd.random() < p:
                chosen = False
                while not chosen:
                    # get a random position index from gdf
                    random_node_idx = rd.randint(0, len(gdf) - 1)
                    random_node_graph_id = (
                        id_col_array[random_node_idx] if id_col else random_node_idx
                    )
                    while (
                        random_node_idx == this_node_idx
                        or graph.has_edge(this_node_graph_id, random_node_graph_id)
                        or degree_centrality_array[random_node_idx] >= max_degree
                    ):  # Enforce no self-loops, or multiple edges, or degree >= max_degree
                        random_node_idx = rd.randint(0, len(gdf) - 1)
                        random_node_graph_id = (
                            id_col_array[random_node_idx] if id_col else random_node_idx
                        )

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
                    if rd.random() < q:
                        graph.remove_edge(this_node_graph_id, neighboring_node_graph_id)
                        graph.add_edge(
                            this_node_graph_id, random_node_graph_id, length=distance
                        )
                        degree_centrality_array[neighboring_node_idx] -= 1
                        degree_centrality_array[random_node_idx] += 1
                        rewire_count += 1
                        chosen = True
    _set_node_attributes(graph, gdf, id_col, save_attributes)
    total_edges = graph.number_of_edges()
    logger.info(
        f"Rewire Count: {rewire_count:,} edges out of {total_edges:,}. {rewire_count / total_edges * 100:.2f}% of edges rewired"
    )
    return graph