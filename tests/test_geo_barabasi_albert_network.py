import random
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from pysgn.geo_barabasi_albert_network import geo_barabasi_albert_network


@pytest.fixture
def point_gdf() -> gpd.GeoDataFrame:
    """Create a simple GeoDataFrame with points for testing.

    Returns:
        gpd.GeoDataFrame: Test GeoDataFrame with points
    """
    points = [Point(x, y) for x in range(10) for y in range(10)]
    data = {
        "geometry": points,
        "group": ["A" if i % 2 == 0 else "B" for i in range(len(points))],
        "expected_degree": [
            min(i % 5 + 1, len(points) - 1) for i in range(len(points))
        ],
    }
    return gpd.GeoDataFrame(data)


@pytest.fixture
def polygon_gdf() -> gpd.GeoDataFrame:
    """Create a simple GeoDataFrame with polygons for testing.

    Returns:
        gpd.GeoDataFrame: Test GeoDataFrame with polygons
    """
    polygons = [
        Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
        for x in range(10)
        for y in range(10)
    ]
    data = {
        "geometry": polygons,
        "group": ["A" if i % 2 == 0 else "B" for i in range(len(polygons))],
    }
    return gpd.GeoDataFrame(data)


def test_basic_network_creation(point_gdf: gpd.GeoDataFrame) -> None:
    """Test basic network creation functionality.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    g = geo_barabasi_albert_network(point_gdf, m=2)

    assert isinstance(g, nx.Graph)
    assert g.number_of_nodes() == len(point_gdf)
    assert g.number_of_edges() > 0


def test_polygon_geometry(polygon_gdf: gpd.GeoDataFrame) -> None:
    """Test network creation with polygon geometries.

    Args:
        polygon_gdf: Test GeoDataFrame with polygons
    """
    g = geo_barabasi_albert_network(polygon_gdf, m=2)

    assert isinstance(g, nx.Graph)
    assert g.number_of_edges() > 0
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

    g = geo_barabasi_albert_network(point_gdf, m=2, constraint=group_constraint)

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
    g1 = geo_barabasi_albert_network(point_gdf, m=2, node_attributes="group")
    assert all("group" in g1.nodes[n] for n in g1.nodes)
    assert all("expected_degree" not in g1.nodes[n] for n in g1.nodes)

    g2 = geo_barabasi_albert_network(point_gdf, m=2, node_attributes=["group"])
    assert all("group" in g2.nodes[n] for n in g2.nodes)
    assert all("expected_degree" not in g2.nodes[n] for n in g2.nodes)

    # Test saving all attributes
    g3 = geo_barabasi_albert_network(point_gdf, m=2, node_attributes=True)
    assert all("group" in g3.nodes[n] for n in g3.nodes)
    assert all("expected_degree" in g3.nodes[n] for n in g3.nodes)

    # Test saving no attributes
    g4 = geo_barabasi_albert_network(point_gdf, m=2, node_attributes=False)
    assert all(len(g4.nodes[n]) == 1 for n in g4.nodes)  # Only pos attribute


def test_geographic_warning(point_gdf: gpd.GeoDataFrame) -> None:
    """Test warning for geographic CRS.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    geographic_gdf = point_gdf.to_crs("EPSG:4326")

    with pytest.warns(UserWarning):
        geo_barabasi_albert_network(geographic_gdf, m=2)


def test_index_as_id(point_gdf: gpd.GeoDataFrame) -> None:
    """Test network creation using DataFrame index as node IDs.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    # Test with default index
    g = geo_barabasi_albert_network(point_gdf, m=2, id_col="index")
    assert set(g.nodes()) == set(point_gdf.index)

    # Test with custom index
    custom_gdf = point_gdf.copy()
    custom_gdf.index = [f"node_{i}" for i in range(len(point_gdf))]
    g_custom = geo_barabasi_albert_network(custom_gdf, m=2, id_col="index")
    assert set(g_custom.nodes()) == set(custom_gdf.index)

    # Test with duplicated index
    dup_gdf = point_gdf.copy()
    dup_gdf.index = random.choices(range(5), k=len(point_gdf))
    with pytest.raises(ValueError, match="ID column must contain unique values"):
        geo_barabasi_albert_network(dup_gdf, m=2, id_col="index")

    # Test with multi-index
    multi_gdf = point_gdf.copy()
    multi_gdf.index = pd.MultiIndex.from_tuples(
        [(i, random.choice(["a", "b", "c"])) for i in range(len(point_gdf))]
    )
    with pytest.raises(ValueError, match="Multi-index is not supported"):
        geo_barabasi_albert_network(multi_gdf, m=2, id_col="index")


def test_power_law_degree_distribution(point_gdf: gpd.GeoDataFrame) -> None:
    """Test if the network exhibits power-law degree distribution.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    g = geo_barabasi_albert_network(point_gdf, m=2)
    degrees = [d for _, d in g.degree()]

    # Calculate degree distribution
    unique_degrees = sorted(set(degrees))
    degree_counts = [degrees.count(d) for d in unique_degrees]

    # In a power-law distribution, log(P(k)) ~ -gamma * log(k)
    # where P(k) is the fraction of nodes with degree k
    # We check if the relationship is roughly linear in log-log space
    log_degrees = [pd.np.log(d) for d in unique_degrees[1:]]  # Skip degree 0
    log_counts = [pd.np.log(c / len(degrees)) for c in degree_counts[1:]]

    # Calculate correlation coefficient
    correlation = pd.np.corrcoef(log_degrees, log_counts)[0, 1]

    # Strong negative correlation indicates power-law-like behavior
    assert correlation < -0.8
