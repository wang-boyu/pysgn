import random
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from pysgn.geo_barabasi_albert_network import geo_barabasi_albert_network
from pysgn.ordering import attribute_order, density_order_knn, random_order


@pytest.fixture
def point_gdf() -> gpd.GeoDataFrame:
    """Create a simple GeoDataFrame with points for testing.

    Returns:
        gpd.GeoDataFrame: Test GeoDataFrame with points.
    """
    points = [Point(x, y) for x in range(10) for y in range(10)]
    data = {
        "id": range(len(points)),
        "group": random.choices(["A", "B", "C"], k=len(points)),
        "expected_degree": random.choices(range(1, 10), k=len(points)),
        "geometry": points,
    }
    return gpd.GeoDataFrame(data, crs="EPSG:3857")


@pytest.fixture
def polygon_gdf() -> gpd.GeoDataFrame:
    """Create a simple GeoDataFrame with polygons for testing.

    Returns:
        gpd.GeoDataFrame: Test GeoDataFrame with polygons.
    """
    polygons = [
        Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
        for x, y in [(0, 0), (2, 2), (4, 4)]
    ]
    data = {"id": range(3), "group": ["A", "B", "C"], "geometry": polygons}
    return gpd.GeoDataFrame(data, crs="EPSG:3857")


def test_basic_network_creation(point_gdf: gpd.GeoDataFrame) -> None:
    """Test basic network creation with default parameters.

    Args:
        point_gdf: Test GeoDataFrame with points.
    """
    # Use a small scaling_factor (large minimum distance) to help ensure edges are created.
    g = geo_barabasi_albert_network(point_gdf, m=2, scaling_factor=1e-3)
    assert isinstance(g, nx.Graph)
    assert g.number_of_nodes() == len(point_gdf)
    assert g.number_of_edges() > 0


def test_max_degree_constraint(point_gdf: gpd.GeoDataFrame) -> None:
    """Test enforcement of maximum degree constraint.

    Args:
        point_gdf: Test GeoDataFrame with points.
    """
    max_degree = 3
    g = geo_barabasi_albert_network(point_gdf, m=2, max_degree=max_degree)
    degrees = dict(g.degree())
    assert all(d <= max_degree for d in degrees.values())


def test_polygon_geometry(polygon_gdf: gpd.GeoDataFrame) -> None:
    """Test network creation with polygon geometries.

    Args:
        polygon_gdf: Test GeoDataFrame with polygons.
    """
    # Use a small scaling_factor to ensure edges are created.
    g = geo_barabasi_albert_network(polygon_gdf, m=1, scaling_factor=1e-3)
    assert isinstance(g, nx.Graph)
    assert g.number_of_edges() > 0
    # Check that node positions are set to the centroids.
    for node in g.nodes:
        pos = g.nodes[node]["pos"]
        centroid = polygon_gdf.loc[node, "geometry"].centroid
        assert pos == (centroid.x, centroid.y)


def test_custom_constraint(point_gdf: gpd.GeoDataFrame) -> None:
    """Test network creation with a custom constraint function.

    Args:
        point_gdf: Test GeoDataFrame with points.
    """

    def group_constraint(u: Any, v: Any) -> bool:
        return u["group"] != v["group"]

    g = geo_barabasi_albert_network(point_gdf, m=2, constraint=group_constraint)
    # Check that no edge connects nodes from the same group.
    for u, v in g.edges:
        u_group = point_gdf.loc[u, "group"]
        v_group = point_gdf.loc[v, "group"]
        assert u_group != v_group


def test_node_attributes(point_gdf: gpd.GeoDataFrame) -> None:
    """Test saving of node attributes.

    Args:
        point_gdf: Test GeoDataFrame with points.
    """
    # Test saving a specific attribute ("group")
    g1 = geo_barabasi_albert_network(point_gdf, m=2, node_attributes="group")
    assert all("group" in g1.nodes[n] for n in g1.nodes)

    # Test saving all attributes
    g2 = geo_barabasi_albert_network(point_gdf, m=2, node_attributes=True)
    assert all("group" in g2.nodes[n] for n in g2.nodes)

    # Test saving no attributes (only the "pos" attribute should be present)
    g3 = geo_barabasi_albert_network(point_gdf, m=2, node_attributes=False)
    assert all(len(g3.nodes[n]) == 1 for n in g3.nodes)  # Only pos attribute exists


def test_geographic_warning(point_gdf: gpd.GeoDataFrame) -> None:
    """Test that a warning is raised when using a geographic CRS.

    Args:
        point_gdf: Test GeoDataFrame with points.
    """
    geographic_gdf = point_gdf.to_crs("EPSG:4326")
    with pytest.warns(UserWarning):
        geo_barabasi_albert_network(geographic_gdf, m=2)


def test_index_as_id(point_gdf: gpd.GeoDataFrame) -> None:
    """Test network creation using DataFrame index as node IDs.

    Args:
        point_gdf: Test GeoDataFrame with points.
    """
    # Test using the default index.
    g = geo_barabasi_albert_network(point_gdf, m=2, id_col="index")
    assert set(g.nodes()) == set(point_gdf.index)

    # Test with a custom index.
    custom_gdf = point_gdf.copy()
    custom_gdf.index = [f"node_{i}" for i in range(len(point_gdf))]
    g_custom = geo_barabasi_albert_network(custom_gdf, m=2, id_col="index")
    assert set(g_custom.nodes()) == set(custom_gdf.index)

    # Test with duplicated indices.
    dup_gdf = point_gdf.copy()
    dup_gdf.index = random.choices(
        range(5), k=len(point_gdf)
    )  # Introduce duplicate indices.
    with pytest.raises(ValueError, match="ID column must contain unique values"):
        geo_barabasi_albert_network(dup_gdf, m=2, id_col="index")

    # Test with a multi-index.
    multi_gdf = point_gdf.copy()
    multi_gdf.index = pd.MultiIndex.from_tuples(
        [(i, random.choice(["a", "b", "c"])) for i in range(len(point_gdf))]
    )
    with pytest.raises(ValueError, match="Multi-index is not supported"):
        geo_barabasi_albert_network(multi_gdf, m=2, id_col="index")


def test_ordering_function_usage(point_gdf: gpd.GeoDataFrame) -> None:
    """Test usage of ordering functions with the geo_barabasi_albert_network method.

    This test uses ordering functions from the ordering module to change the node addition order.
    """

    g_random = geo_barabasi_albert_network(
        point_gdf, m=2, node_order=random_order, random_state=42
    )
    assert isinstance(g_random, nx.Graph)

    g_attr = geo_barabasi_albert_network(
        point_gdf, m=2, node_order=lambda gdf: attribute_order(gdf, "expected_degree")
    )
    assert isinstance(g_attr, nx.Graph)

    g_density = geo_barabasi_albert_network(
        point_gdf, m=2, node_order=lambda gdf: density_order_knn(gdf, k=5)
    )
    assert isinstance(g_density, nx.Graph)


def test_node_order_column_name(point_gdf: gpd.GeoDataFrame) -> None:
    """Ensure string-based node ordering columns are respected."""

    ordered_gdf = point_gdf.head(10).copy()
    ordered_gdf["custom_order"] = list(reversed(range(len(ordered_gdf))))

    g = geo_barabasi_albert_network(
        ordered_gdf, m=2, node_order="custom_order", random_state=0
    )
    assert isinstance(g, nx.Graph)


def test_node_order_missing_column(point_gdf: gpd.GeoDataFrame) -> None:
    """Expect a ValueError when node_order references a missing column."""

    with pytest.raises(
        ValueError,
        match="Column 'does_not_exist' was not found in the provided GeoDataFrame",
    ):
        geo_barabasi_albert_network(point_gdf, m=2, node_order="does_not_exist")


def test_m_greater_than_number_of_nodes(point_gdf: gpd.GeoDataFrame) -> None:
    """Ensure specifying m larger than available nodes raises ValueError."""

    tiny = point_gdf.head(3).copy()
    with pytest.raises(
        ValueError,
        match=r"Number of nodes \(3\) must be at least m \(4\)",
    ):
        geo_barabasi_albert_network(tiny, m=4)


def test_no_edges_created_warning(point_gdf: gpd.GeoDataFrame) -> None:
    """Trigger the 'No edges were created' warning in geo_barabasi_albert_network."""

    tiny = point_gdf.head(4).copy()
    with pytest.warns(UserWarning, match="No edges were created"):
        g = geo_barabasi_albert_network(
            tiny, m=2, max_degree=0, scaling_factor=1e-6, random_state=0
        )
        assert g.number_of_edges() == 0


def test_node_order_incorrect_length(point_gdf: gpd.GeoDataFrame) -> None:
    """Expect ValueError when node_order returns indices of incorrect length."""

    def bad_order(gdf):
        return np.arange(len(gdf) - 1)

    with pytest.raises(
        ValueError,
        match="The node_order must return an array of indices of the same length as gdf",
    ):
        geo_barabasi_albert_network(point_gdf, m=2, node_order=bad_order)
