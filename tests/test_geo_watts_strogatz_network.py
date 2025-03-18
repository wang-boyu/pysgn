import random
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from pysgn.geo_watts_strogatz_network import geo_watts_strogatz_network


@pytest.fixture
def point_gdf() -> gpd.GeoDataFrame:
    """Create a simple GeoDataFrame with points for testing.

    Returns:
        gpd.GeoDataFrame: Test GeoDataFrame with points
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
        gpd.GeoDataFrame: Test GeoDataFrame with polygons
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
        point_gdf: Test GeoDataFrame with points
    """
    g = geo_watts_strogatz_network(point_gdf, k=2, p=0.1)

    assert isinstance(g, nx.Graph)
    assert g.number_of_nodes() == len(point_gdf)
    assert g.number_of_edges() > 0
    assert all("pos" in g.nodes[n] for n in g.nodes)


def test_mean_degree_equals_k(point_gdf: gpd.GeoDataFrame) -> None:
    """Test if the mean degree of the resulting graph equals the parameter k.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    k = 2
    g = geo_watts_strogatz_network(point_gdf, k=k, p=0.1)

    # Calculate the mean degree of the graph
    mean_degree = sum(dict(g.degree()).values()) / g.number_of_nodes()

    # Assert that the mean degree is equal to k
    assert mean_degree == k


def test_k_as_column(point_gdf: gpd.GeoDataFrame) -> None:
    """Test network creation using a column for k values.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    # Ensure expected_degree values don't exceed number of points - 1
    point_gdf["expected_degree"] = point_gdf["expected_degree"].clip(
        upper=len(point_gdf) - 1
    )
    g = geo_watts_strogatz_network(
        point_gdf, k="expected_degree", p=0.1, query_factor=1
    )

    assert isinstance(g, nx.Graph)


def test_polygon_geometry(polygon_gdf: gpd.GeoDataFrame) -> None:
    """Test network creation with polygon geometries.

    Args:
        polygon_gdf: Test GeoDataFrame with polygons
    """
    # Use k=2 instead of k=1 to ensure edges are created
    g = geo_watts_strogatz_network(polygon_gdf, k=2, p=0.1)

    assert isinstance(g, nx.Graph)
    assert g.number_of_edges() > 0  # Verify that edges were created
    # Check that node positions are centroids
    for node in g.nodes:
        pos = g.nodes[node]["pos"]
        centroid = polygon_gdf.loc[node, "geometry"].centroid
        assert pos == (centroid.x, centroid.y)


def test_custom_constraint(point_gdf: gpd.GeoDataFrame) -> None:
    """Test network creation with custom constraint function.

    Args:
        point_gdf: Test GeoDataFrame with points
    """

    def group_constraint(u: Any, v: Any) -> bool:
        return u["group"] != v["group"]

    g = geo_watts_strogatz_network(point_gdf, k=2, p=0.1, constraint=group_constraint)

    # Check that no edges exist between nodes of same group
    for u, v in g.edges:
        u_group = point_gdf.loc[u, "group"]
        v_group = point_gdf.loc[v, "group"]
        assert u_group != v_group


def test_node_attributes(point_gdf: gpd.GeoDataFrame) -> None:
    """Test saving of node attributes.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    # Test saving specific attributes
    g1 = geo_watts_strogatz_network(point_gdf, k=2, p=0.1, node_attributes="group")
    assert all("group" in g1.nodes[n] for n in g1.nodes)
    assert all("expected_degree" not in g1.nodes[n] for n in g1.nodes)

    g2 = geo_watts_strogatz_network(point_gdf, k=2, p=0.1, node_attributes=["group"])
    assert all("group" in g2.nodes[n] for n in g2.nodes)
    assert all("expected_degree" not in g2.nodes[n] for n in g2.nodes)

    # Test saving all attributes
    g3 = geo_watts_strogatz_network(point_gdf, k=2, p=0.1, node_attributes=True)
    assert all("group" in g3.nodes[n] for n in g3.nodes)
    assert all("expected_degree" in g3.nodes[n] for n in g3.nodes)

    # Test saving no attributes
    g4 = geo_watts_strogatz_network(point_gdf, k=2, p=0.1, node_attributes=False)
    assert all(len(g4.nodes[n]) == 1 for n in g4.nodes)  # Only pos attribute


def test_max_degree_constraint(point_gdf: gpd.GeoDataFrame) -> None:
    """Test enforcement of maximum degree constraint.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    max_degree = 3
    g = geo_watts_strogatz_network(point_gdf, k=2, p=0.1, max_degree=max_degree)

    degrees = dict(g.degree())
    assert all(d <= max_degree for d in degrees.values())


def test_invalid_inputs(point_gdf: gpd.GeoDataFrame) -> None:
    """Test handling of invalid inputs.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    # Test invalid k value
    with pytest.raises(ValueError):
        geo_watts_strogatz_network(point_gdf, k=0, p=0.1)

    # Test invalid probability
    with pytest.raises(ValueError):
        geo_watts_strogatz_network(point_gdf, k=2, p=1.5)

    # Test invalid id column
    with pytest.raises(KeyError):
        geo_watts_strogatz_network(point_gdf, k=2, p=0.1, id_col="nonexistent")


def test_geographic_warning(point_gdf: gpd.GeoDataFrame) -> None:
    """Test warning for geographic CRS.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    geographic_gdf = point_gdf.to_crs("EPSG:4326")

    with pytest.warns(UserWarning):
        geo_watts_strogatz_network(geographic_gdf, k=2, p=0.1)


def test_rewiring_probability(point_gdf: gpd.GeoDataFrame) -> None:
    """Test impact of rewiring probability.

    This test verifies that different rewiring probabilities produce networks with
    distinct structural properties. For p=0.0, all connections should be to geographic
    neighbors, while p=1.0 should produce some long-range connections.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    # Create networks with different rewiring probabilities
    g1 = geo_watts_strogatz_network(point_gdf, k=2, p=0.0)
    g2 = geo_watts_strogatz_network(point_gdf, k=2, p=1.0)

    # For p=0.0, all connections should be to geographic neighbors
    # Calculate geographic distances between all connected nodes in G1
    distances_g1 = []
    for u, v in g1.edges():
        pos_u = g1.nodes[u]["pos"]
        pos_v = g1.nodes[v]["pos"]
        dist = ((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2) ** 0.5
        distances_g1.append(dist)

    # For p=1.0, we should see some longer-range connections
    distances_g2 = []
    for u, v in g2.edges():
        pos_u = g2.nodes[u]["pos"]
        pos_v = g2.nodes[v]["pos"]
        dist = ((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2) ** 0.5
        distances_g2.append(dist)

    # The maximum distance in G2 should be greater than in G1
    # This indicates the presence of long-range connections
    assert max(distances_g2) > max(distances_g1)


def test_index_as_id(point_gdf: gpd.GeoDataFrame) -> None:
    """Test network creation using DataFrame index as node IDs.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    # Test with default index
    g = geo_watts_strogatz_network(point_gdf, k=2, p=0.1, id_col="index")
    assert set(g.nodes()) == set(point_gdf.index)

    # Test with custom index
    custom_gdf = point_gdf.copy()
    custom_gdf.index = [f"node_{i}" for i in range(len(point_gdf))]
    g_custom = geo_watts_strogatz_network(custom_gdf, k=2, p=0.1, id_col="index")
    assert set(g_custom.nodes()) == set(custom_gdf.index)

    # Test with duplicated index
    dup_gdf = point_gdf.copy()
    dup_gdf.index = random.choices(range(5), k=len(point_gdf))
    with pytest.raises(ValueError, match="ID column must contain unique values"):
        geo_watts_strogatz_network(dup_gdf, k=2, p=0.1, id_col="index")

    # Test with multi-index
    multi_gdf = point_gdf.copy()
    multi_gdf.index = pd.MultiIndex.from_tuples(
        [(i, random.choice(["a", "b", "c"])) for i in range(len(point_gdf))]
    )
    with pytest.raises(ValueError, match="Multi-index is not supported"):
        geo_watts_strogatz_network(multi_gdf, k=2, p=0.1, id_col="index")
