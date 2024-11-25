import networkx as nx
from loguru import logger


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
    if gdf.geometry.geom_type[0] == "Polygon":
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
