import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from pysgn.geo_erdos_renyi_network import geo_erdos_renyi_network


@pytest.fixture
def point_gdf() -> gpd.GeoDataFrame:
    """Create a simple GeoDataFrame with points for testing.

    Returns:
        gpd.GeoDataFrame: Test GeoDataFrame with points
    """
    points = [Point(x, y) for x, y in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]]
    data = {
        "id": range(5),
        "group": ["A", "A", "B", "B", "C"],
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
    # use a small scaling factor (large minimum distance) to ensure edges are created
    g = geo_erdos_renyi_network(point_gdf, scaling_factor=1e-3)

    assert isinstance(g, nx.Graph)
    assert g.number_of_nodes() == len(point_gdf)
    assert g.number_of_edges() > 0


def test_max_degree_constraint(point_gdf: gpd.GeoDataFrame) -> None:
    """Test enforcement of maximum degree constraint.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    max_degree = 3
    g = geo_erdos_renyi_network(point_gdf, max_degree=max_degree)

    degrees = dict(g.degree())
    assert all(d <= max_degree for d in degrees.values())


def test_polygon_geometry(polygon_gdf: gpd.GeoDataFrame) -> None:
    """Test network creation with polygon geometries.

    Args:
        polygon_gdf: Test GeoDataFrame with polygons
    """
    # use a small scaling factor (large minimum distance) to ensure edges are created
    g = geo_erdos_renyi_network(polygon_gdf, scaling_factor=1e-3)

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

    def group_constraint(u, v) -> bool:
        return u["group"] != v["group"]

    g = geo_erdos_renyi_network(point_gdf, constraint=group_constraint)

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
    g1 = geo_erdos_renyi_network(point_gdf, node_attributes="group")
    assert all("group" in g1.nodes[n] for n in g1.nodes)

    # Test saving all attributes
    g2 = geo_erdos_renyi_network(point_gdf, node_attributes=True)
    assert all("group" in g2.nodes[n] for n in g2.nodes)

    # Test saving no attributes
    g3 = geo_erdos_renyi_network(point_gdf, node_attributes=False)
    assert all(len(g3.nodes[n]) == 1 for n in g3.nodes)  # Only pos attribute


def test_geographic_warning(point_gdf: gpd.GeoDataFrame) -> None:
    """Test warning for geographic CRS.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    geographic_gdf = point_gdf.to_crs("EPSG:4326")

    with pytest.warns(UserWarning):
        geo_erdos_renyi_network(geographic_gdf)


def test_index_as_id(point_gdf: gpd.GeoDataFrame) -> None:
    """Test network creation using DataFrame index as node IDs.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    # Test with default index
    g = geo_erdos_renyi_network(point_gdf, id_col="index")
    assert set(g.nodes()) == set(point_gdf.index)

    # Test with custom index
    custom_gdf = point_gdf.copy()
    custom_gdf.index = [f"node_{i}" for i in range(len(point_gdf))]
    g_custom = geo_erdos_renyi_network(custom_gdf, id_col="index")
    assert set(g_custom.nodes()) == set(custom_gdf.index)

    # Test with duplicated index
    dup_gdf = point_gdf.copy()
    dup_gdf.index = [0, 0, 1, 1, 2]  # Duplicated indices
    with pytest.raises(ValueError, match="ID column must contain unique values"):
        geo_erdos_renyi_network(dup_gdf, id_col="index")

    # Test with multi-index
    multi_gdf = point_gdf.copy()
    multi_gdf.index = pd.MultiIndex.from_tuples(
        [(0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "e")]
    )
    with pytest.raises(ValueError, match="Multi-index is not supported"):
        geo_erdos_renyi_network(multi_gdf, id_col="index")
