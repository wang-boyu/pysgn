import re

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from pysgn.utils import (
    _create_k_col,
    _get_id_col_array,
    _set_node_attributes,
    graph_to_gdf,
)


def test_create_k_col_even_integer():
    """Test _create_k_col with an even integer k."""
    k = 4
    n = 10
    result = _create_k_col(k, n)
    expected = np.full(n, k // 2, dtype=int)
    assert np.array_equal(result, expected), "Failed on even integer k"


def test_create_k_col_odd_integer():
    """Test _create_k_col with an odd integer k."""
    k = 5
    n = 10
    result = _create_k_col(k, n, random_state=42)
    assert np.all(np.isin(result, [k // 2, k // 2 + 1])), "Failed on odd integer k"
    assert np.isclose(result.mean(), k / 2, atol=0.1), (
        "Mean not as expected for odd integer k"
    )


def test_create_k_col_float():
    """Test _create_k_col with a float k."""
    k = 6.4
    n = 10
    result = _create_k_col(k, n, random_state=42)
    lower = int(np.floor(k / 2))
    upper = int(np.ceil(k / 2))
    assert np.all(np.isin(result, [lower, upper])), "Failed on float k"
    assert np.isclose(result.mean(), k / 2, atol=0.1), (
        "Mean not as expected for float k"
    )


def test_create_k_col_invalid_k():
    """Test _create_k_col with an invalid k."""
    k = "invalid"
    n = 10
    with pytest.raises(ValueError, match=r"k must be an integer or a float."):
        _create_k_col(k, n)


def test_create_k_col_reproducibility():
    """Test _create_k_col reproducibility with a random state."""
    k = 5
    n = 10
    random_state = 42
    result1 = _create_k_col(k, n, random_state=random_state)
    result2 = _create_k_col(k, n, random_state=random_state)
    assert np.array_equal(result1, result2), (
        "Results are not reproducible with the same random state"
    )


def test_get_id_col_array_defaults_to_positional_index():
    """_get_id_col_array should return positional ids when id_col=None."""
    gdf = gpd.GeoDataFrame({"id": [10, 11], "geometry": [Point(0, 0), Point(1, 1)]})

    result = _get_id_col_array(gdf, None)

    assert np.array_equal(result, np.arange(len(gdf)))


def test_get_id_col_array_uses_index():
    """_get_id_col_array should use the GeoDataFrame index when requested."""
    gdf = gpd.GeoDataFrame({"id": [10, 11], "geometry": [Point(0, 0), Point(1, 1)]})
    gdf.index = ["a", "b"]

    result = _get_id_col_array(gdf, "index")

    assert np.array_equal(result, gdf.index.values)


def test_get_id_col_array_uses_column():
    """_get_id_col_array should use a named column."""
    gdf = gpd.GeoDataFrame({"id": [10, 11], "geometry": [Point(0, 0), Point(1, 1)]})

    result = _get_id_col_array(gdf, "id")

    assert np.array_equal(result, gdf["id"].values)


def test_get_id_col_array_duplicate_values_error():
    """Duplicate ids should raise a ValueError."""
    gdf = gpd.GeoDataFrame({"id": [10, 10], "geometry": [Point(0, 0), Point(1, 1)]})

    with pytest.raises(ValueError, match="ID column must contain unique values"):
        _get_id_col_array(gdf, "id")


def test_get_id_col_array_multiindex_error():
    """MultiIndex should raise a ValueError when using index ids."""
    gdf = gpd.GeoDataFrame({"id": [10, 11], "geometry": [Point(0, 0), Point(1, 1)]})
    gdf.index = pd.MultiIndex.from_tuples([(0, "a"), (1, "b")])

    with pytest.raises(ValueError, match="Multi-index is not supported"):
        _get_id_col_array(gdf, "index")


def test_id_col_not_duplicated_as_node_attribute_when_used_as_node_key():
    """Test that the id column is not duplicated as a node attribute when used as a node key."""
    gdf = gpd.GeoDataFrame(
        {
            "id": [101, 102, 103],
            "foo": ["a", "b", "c"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        }
    )

    graph = nx.Graph()
    graph.add_nodes_from(gdf["id"].tolist())

    _set_node_attributes(graph, gdf=gdf, id_col="id", node_attributes=True)

    for node_id in gdf["id"].tolist():
        assert node_id in graph.nodes

        # id_col should NOT be stored redundantly as a node attribute
        assert "id" not in graph.nodes[node_id]

        # still store other attributes and required fields
        assert "foo" in graph.nodes[node_id]
        assert "geometry" in graph.nodes[node_id]
        assert "pos" in graph.nodes[node_id]

    # check an example value matches
    assert graph.nodes[101]["foo"] == "a"


def test_graph_to_gdf_requires_layer_selection():
    """nodes=False and edges=False should raise."""
    g = nx.Graph()

    with pytest.raises(
        ValueError,
        match=re.escape(
            "At least one layer must be requested; set nodes=True or edges=True (received nodes=False, edges=False).",
        ),
    ):
        graph_to_gdf(g, nodes=False, edges=False)


def test_graph_to_gdf_nodes_include_attributes_and_geometry():
    """Nodes export should include node id, attrs, and geometry."""
    g = nx.Graph()
    g.graph["crs"] = "EPSG:3857"
    g.graph["id_col"] = "index"
    g.graph["index_name"] = "node_id"
    geom = Point(3, 4)
    g.add_node(10, geometry=geom, color="red")

    nodes_gdf, edges_gdf = graph_to_gdf(g, edges=False)

    assert edges_gdf is None
    assert nodes_gdf is not None and len(nodes_gdf) == 1
    assert nodes_gdf.index.tolist() == [10]
    assert nodes_gdf.index.name == "node_id"
    assert nodes_gdf.loc[10, "color"] == "red"
    assert nodes_gdf.loc[10, "geometry"].equals(geom)
    assert nodes_gdf.crs == g.graph["crs"]
    assert "node" not in nodes_gdf.columns


def test_graph_to_gdf_nodes_geometry_from_pos_when_missing_geometry():
    """Nodes export should fall back to Point from pos."""
    g = nx.Graph()
    g.graph["crs"] = "EPSG:3857"
    g.graph["id_col"] = "index"
    g.add_node("a", pos=(1.5, -2.0))

    nodes_gdf, _ = graph_to_gdf(g, edges=False)

    assert nodes_gdf is not None and len(nodes_gdf) == 1
    assert nodes_gdf.index.tolist() == ["a"]
    assert nodes_gdf.index.name is None
    assert nodes_gdf.loc["a", "geometry"].coords[:] == [(1.5, -2.0)]


def test_graph_to_gdf_nodes_missing_geometry_and_pos_errors():
    """Missing geometry and pos should raise a RuntimeError."""
    g = nx.Graph()
    g.graph["crs"] = "EPSG:3857"
    g.graph["id_col"] = "index"
    g.add_node(1, value="x")

    with pytest.raises(RuntimeError, match="Cannot reconstruct geometry for node"):
        graph_to_gdf(g, edges=False)


def test_graph_to_gdf_nodes_use_id_col_for_identifier():
    """Nodes export should include id_col field with node ids."""
    g = nx.Graph()
    g.graph["crs"] = "EPSG:3857"
    g.graph["id_col"] = "custom_id"
    g.add_node("n-1", pos=(0, 0))

    nodes_gdf, _ = graph_to_gdf(g, edges=False)

    assert nodes_gdf is not None
    assert "custom_id" in nodes_gdf.columns
    assert nodes_gdf.loc[0, "custom_id"] == "n-1"
    assert nodes_gdf.index.name is None
    assert "node" not in nodes_gdf.columns


def test_graph_to_gdf_nodes_positional_index_without_extra_fields():
    """Positional ids should stay as index with no extra columns."""
    g = nx.Graph()
    g.graph["crs"] = "EPSG:3857"
    g.graph["id_col"] = None
    g.add_node(0, pos=(0, 0))
    g.add_node(1, pos=(1, 1))

    nodes_gdf, _ = graph_to_gdf(g, edges=False)

    assert nodes_gdf is not None
    assert list(nodes_gdf.index) == [0, 1]
    assert nodes_gdf.index.name is None
    assert "node" not in nodes_gdf.columns
    assert "id" not in nodes_gdf.columns
    assert nodes_gdf.columns.tolist() == ["pos", "geometry"]


def test_set_node_attributes_with_positional_ids():
    """_set_node_attributes should map positional ids and attrs."""
    gdf = gpd.GeoDataFrame(
        {"group": ["a", "b"], "geometry": [Point(0, 0), Point(1, 1)]}
    )
    graph = nx.Graph()
    graph.add_nodes_from([0, 1])

    _set_node_attributes(graph, gdf=gdf, id_col=None, node_attributes=True)

    assert graph.nodes[0]["pos"] == (0.0, 0.0)
    assert graph.nodes[1]["pos"] == (1.0, 1.0)
    assert graph.nodes[0]["group"] == "a"
    assert graph.nodes[1]["group"] == "b"


def test_set_node_attributes_with_index_ids():
    """_set_node_attributes should map index ids when id_col='index'."""
    gdf = gpd.GeoDataFrame(
        {"group": ["a", "b"], "geometry": [Point(0, 0), Point(1, 1)]}
    )
    gdf.index = ["n0", "n1"]
    graph = nx.Graph()
    graph.add_nodes_from(["n0", "n1"])

    _set_node_attributes(graph, gdf=gdf, id_col="index", node_attributes=["group"])

    assert graph.nodes["n0"]["pos"] == (0.0, 0.0)
    assert graph.nodes["n1"]["pos"] == (1.0, 1.0)
    assert graph.nodes["n0"]["group"] == "a"
    assert graph.nodes["n1"]["group"] == "b"


def test_set_node_attributes_false_keeps_only_pos():
    """_set_node_attributes should only set pos when node_attributes=False."""
    gdf = gpd.GeoDataFrame(
        {"group": ["a", "b"], "geometry": [Point(0, 0), Point(1, 1)]}
    )
    graph = nx.Graph()
    graph.add_nodes_from([0, 1])

    _set_node_attributes(graph, gdf=gdf, id_col=None, node_attributes=False)

    assert set(graph.nodes[0].keys()) == {"pos"}
    assert set(graph.nodes[1].keys()) == {"pos"}


def test_graph_to_gdf_edges_use_existing_geometry():
    """Edges export should preserve existing shapely geometry."""
    g = nx.Graph()
    g.graph["crs"] = "EPSG:3857"
    g.graph["id_col"] = None
    g.add_node(0, pos=(0, 0))
    g.add_node(1, pos=(1, 1))
    edge_geom = LineString([(0, 0), (2, 2)])
    g.add_edge(0, 1, geometry=edge_geom, weight=2.5)

    _, edges_gdf = graph_to_gdf(g, nodes=False)

    assert edges_gdf is not None and len(edges_gdf) == 1
    assert set(edges_gdf.loc[0, ["source", "target"]]) == {0, 1}
    assert edges_gdf.loc[0, "weight"] == 2.5
    assert edges_gdf.loc[0, "geometry"].equals(edge_geom)


def test_graph_to_gdf_edges_geometry_from_node_geometries():
    """Edges export should use node geometry centroids when available."""
    g = nx.Graph()
    g.graph["crs"] = "EPSG:3857"
    g.graph["id_col"] = None
    g.add_node(0, geometry=Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]))
    g.add_node(1, geometry=Polygon([(3, 3), (3, 5), (5, 5), (5, 3)]))
    g.add_edge(0, 1)

    _, edges_gdf = graph_to_gdf(g, nodes=False)

    assert edges_gdf is not None and len(edges_gdf) == 1
    assert set(edges_gdf.loc[0, ["source", "target"]]) == {0, 1}
    line = edges_gdf.loc[0, "geometry"]
    assert line.coords[:] == [(1.0, 1.0), (4.0, 4.0)]


def test_graph_to_gdf_edges_geometry_from_node_pos():
    """Edges export should use node pos when geometry missing."""
    g = nx.Graph()
    g.graph["crs"] = "EPSG:3857"
    g.graph["id_col"] = None
    g.add_node(0, pos=(0, 0))
    g.add_node(1, pos=(2, 0))
    g.add_edge(0, 1)

    _, edges_gdf = graph_to_gdf(g, nodes=False)

    assert edges_gdf is not None and len(edges_gdf) == 1
    assert set(edges_gdf.loc[0, ["source", "target"]]) == {0, 1}
    line = edges_gdf.loc[0, "geometry"]
    assert line.coords[:] == [(0, 0), (2, 0)]


def test_graph_to_gdf_edges_missing_node_geometry_and_pos_errors():
    """Missing node geometry and pos should raise a RuntimeError."""
    g = nx.Graph()
    g.graph["crs"] = "EPSG:3857"
    g.graph["id_col"] = None
    g.add_node(0)
    g.add_node(1)
    g.add_edge(0, 1)

    with pytest.raises(
        RuntimeError,
        match="Cannot reconstruct geometry for edge endpoint",
    ):
        graph_to_gdf(g, nodes=False)
