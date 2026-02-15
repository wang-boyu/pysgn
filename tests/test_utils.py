import re
from functools import partial

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from pysgn.utils import (
    _bbox_to_polygon,
    _create_k_col,
    _get_id_col_array,
    _resolve_sampling_domain,
    _set_node_attributes,
    _validate_polygon,
    graph_to_gdf,
    sample_points,
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


def test_bbox_to_polygon_valid_bbox_returns_rectangle():
    """_bbox_to_polygon should return a rectangle with expected bounds."""
    poly = _bbox_to_polygon((0, 1, 3, 5))

    assert isinstance(poly, Polygon)
    assert poly.bounds == (0.0, 1.0, 3.0, 5.0)
    assert poly.area == 12.0


def test_bbox_to_polygon_accepts_numeric_sequence_types():
    """_bbox_to_polygon should accept mixed real numeric sequence inputs."""
    poly = _bbox_to_polygon(np.array([0, 0.5, 2, 1.5]))

    assert poly.bounds == (0.0, 0.5, 2.0, 1.5)


@pytest.mark.parametrize("bbox", [None, 5, object()])
def test_bbox_to_polygon_rejects_non_sequence_input(bbox):
    """_bbox_to_polygon should reject non-sequence bbox values."""
    with pytest.raises(
        ValueError,
        match="bbox must be a sequence of 4 values",
    ):
        _bbox_to_polygon(bbox)  # type: ignore[arg-type]


@pytest.mark.parametrize("bbox", [(0, 1, 2), (0, 1, 2, 3, 4)])
def test_bbox_to_polygon_rejects_wrong_length(bbox):
    """_bbox_to_polygon should reject bbox with incorrect length."""
    with pytest.raises(
        ValueError,
        match="bbox must be a sequence of 4 values",
    ):
        _bbox_to_polygon(bbox)


def test_bbox_to_polygon_rejects_non_numeric_values():
    """_bbox_to_polygon should reject non-numeric bbox entries."""
    with pytest.raises(ValueError, match="bbox values must be numeric"):
        _bbox_to_polygon((0, "x", 2, 3))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_bbox_to_polygon_rejects_non_finite_values(bad_value):
    """_bbox_to_polygon should reject NaN/Inf values."""
    with pytest.raises(ValueError, match="bbox values must be finite"):
        _bbox_to_polygon((0, 1, bad_value, 4))


@pytest.mark.parametrize(
    "bbox", [(0, 0, 0, 1), (0, 2, 1, 2), (2, 0, 1, 1), (0, 3, 1, 2)]
)
def test_bbox_to_polygon_rejects_invalid_ordering(bbox):
    """_bbox_to_polygon should enforce xmin<xmax and ymin<ymax."""
    with pytest.raises(
        ValueError,
        match="bbox ordering must satisfy xmin < xmax and ymin < ymax",
    ):
        _bbox_to_polygon(bbox)


def test_validate_polygon_accepts_polygon_and_multipolygon():
    """_validate_polygon should pass through valid polygonal inputs."""
    polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    multipolygon = MultiPolygon(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ]
    )

    assert _validate_polygon(polygon) is polygon
    assert _validate_polygon(multipolygon) is multipolygon


@pytest.mark.parametrize("bad_polygon", [Point(0, 0), "not-a-geometry", None, 123])
def test_validate_polygon_rejects_non_polygon_types(bad_polygon):
    """_validate_polygon should reject non-polygon geometry types."""
    with pytest.raises(
        ValueError, match="polygon must be a shapely Polygon or MultiPolygon"
    ):
        _validate_polygon(bad_polygon)  # type: ignore[arg-type]


def test_validate_polygon_rejects_empty_polygon():
    """_validate_polygon should reject empty polygon geometries."""
    with pytest.raises(ValueError, match="polygon must be non-empty"):
        _validate_polygon(Polygon())


def test_resolve_sampling_domain_bbox_only():
    """_resolve_sampling_domain should return bbox polygon when only bbox is provided."""
    domain = _resolve_sampling_domain(bbox=(0, 0, 2, 4), polygon=None)

    assert isinstance(domain, Polygon)
    assert domain.bounds == (0.0, 0.0, 2.0, 4.0)
    assert domain.area == 8.0


def test_resolve_sampling_domain_polygon_only():
    """_resolve_sampling_domain should return polygon when only polygon is provided."""
    polygon = Polygon([(0, 0), (3, 0), (3, 2), (0, 2)])
    domain = _resolve_sampling_domain(bbox=None, polygon=polygon)

    assert isinstance(domain, Polygon)
    assert domain.equals(polygon)
    assert domain.area == 6.0


def test_resolve_sampling_domain_intersection_when_both_provided():
    """_resolve_sampling_domain should use intersection of bbox and polygon."""
    polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    domain = _resolve_sampling_domain(bbox=(1, 1, 3, 5), polygon=polygon)

    assert domain.bounds == (1.0, 1.0, 3.0, 4.0)
    assert domain.area == 6.0
    assert domain.covers(Point(2, 2))
    assert not domain.covers(Point(0.5, 2))


def test_resolve_sampling_domain_requires_bbox_or_polygon():
    """_resolve_sampling_domain should fail when both inputs are missing."""
    with pytest.raises(
        ValueError, match="At least one of bbox or polygon must be provided"
    ):
        _resolve_sampling_domain(bbox=None, polygon=None)


def test_resolve_sampling_domain_rejects_empty_intersection():
    """_resolve_sampling_domain should fail when intersection has no positive area."""
    polygon = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
    with pytest.raises(
        ValueError, match="Final sampling domain must have positive area"
    ):
        _resolve_sampling_domain(bbox=(0, 0, 1, 1), polygon=polygon)


def test_resolve_sampling_domain_rejects_zero_area_polygon_domain():
    """_resolve_sampling_domain should fail for polygon domains with zero area."""
    zero_area_polygon = Polygon([(0, 0), (1, 1), (2, 2), (0, 0)])
    with pytest.raises(
        ValueError, match="Final sampling domain must have positive area"
    ):
        _resolve_sampling_domain(bbox=None, polygon=zero_area_polygon)


def test_resolve_sampling_domain_propagates_bbox_validation_errors():
    """_resolve_sampling_domain should surface bbox validation errors."""
    with pytest.raises(
        ValueError, match="bbox ordering must satisfy xmin < xmax and ymin < ymax"
    ):
        _resolve_sampling_domain(bbox=(2, 0, 1, 1), polygon=None)


def test_resolve_sampling_domain_propagates_polygon_validation_errors():
    """_resolve_sampling_domain should surface polygon validation errors."""
    with pytest.raises(
        ValueError, match="polygon must be a shapely Polygon or MultiPolygon"
    ):
        _resolve_sampling_domain(bbox=None, polygon=Point(0, 0))  # type: ignore[arg-type]


def test_sample_points_returns_geodataframe_with_points():
    """sample_points should return geometry-only Point GeoDataFrame with n rows."""
    gdf = sample_points(n=20, bbox=(0, 0, 2, 1), random_state=42)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 20
    assert gdf.columns.tolist() == ["geometry"]
    assert gdf.geometry.geom_type.eq("Point").all()


def test_sample_points_bbox_only_points_within_bbox():
    """sample_points should keep all bbox-only points inside bbox domain."""
    bbox = (10, -2, 12, 1)
    gdf = sample_points(n=30, bbox=bbox, random_state=7)
    bbox_polygon = _bbox_to_polygon(bbox)

    assert all(bbox_polygon.covers(point) for point in gdf.geometry)


def test_sample_points_polygon_only_excludes_hole_interiors():
    """sample_points should accept polygon covers and exclude hole interiors."""
    hole_ring = [(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]
    polygon = Polygon(
        [(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)],
        holes=[hole_ring],
    )
    hole_polygon = Polygon(hole_ring)
    gdf = sample_points(n=40, polygon=polygon, random_state=8, max_attempts=50_000)

    assert all(polygon.covers(point) for point in gdf.geometry)
    assert not any(hole_polygon.contains(point) for point in gdf.geometry)


def test_sample_points_bbox_and_polygon_uses_intersection_domain():
    """sample_points should restrict output to bbox/polygon intersection."""
    polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    bbox = (2, 1, 6, 3)
    intersection = polygon.intersection(_bbox_to_polygon(bbox))
    gdf = sample_points(n=30, bbox=bbox, polygon=polygon, random_state=21)

    assert all(intersection.covers(point) for point in gdf.geometry)


def test_sample_points_reproducible_with_same_random_state():
    """sample_points should be deterministic for fixed random_state."""
    gdf_a = sample_points(n=25, bbox=(0, 0, 1, 1), random_state=1234)
    gdf_b = sample_points(n=25, bbox=(0, 0, 1, 1), random_state=1234)

    coords_a = np.column_stack((gdf_a.geometry.x.values, gdf_a.geometry.y.values))
    coords_b = np.column_stack((gdf_b.geometry.x.values, gdf_b.geometry.y.values))
    assert np.array_equal(coords_a, coords_b)


def test_sample_points_uniform_mode_with_explicit_none_sampler():
    """sample_points should support explicit sampler=None path."""
    gdf = sample_points(n=15, bbox=(0, 0, 2, 2), sampler=None, random_state=9)

    assert len(gdf) == 15
    assert gdf.geometry.geom_type.eq("Point").all()


def _mvn_sampler(rng: np.random.Generator, k: int, mean, cov) -> np.ndarray:
    """Parameterized multivariate normal sampler used for tests."""
    return rng.multivariate_normal(mean=mean, cov=cov, size=k)


def test_sample_points_accepts_parameterized_sampler_and_filters_domain():
    """sample_points should accept custom sampler and enforce final domain."""
    sampler = partial(
        _mvn_sampler,
        mean=np.array([0.4, 0.4]),
        cov=np.array([[0.2, 0.0], [0.0, 0.2]]),
    )
    bbox = (0, 0, 1, 1)
    bbox_polygon = _bbox_to_polygon(bbox)
    gdf = sample_points(
        n=25,
        bbox=bbox,
        sampler=sampler,
        random_state=11,
        max_attempts=50_000,
    )

    assert len(gdf) == 25
    assert all(bbox_polygon.covers(point) for point in gdf.geometry)


@pytest.mark.parametrize("n", [0, -1, 1.5, "10"])
def test_sample_points_invalid_n(n):
    """sample_points should reject non-positive/non-integer n."""
    with pytest.raises(ValueError, match="n must be a positive integer"):
        sample_points(n=n, bbox=(0, 0, 1, 1), random_state=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("bbox", [(0, 0, 1), (2, 0, 1, 1), (0, 0, np.nan, 1)])
def test_sample_points_invalid_bbox_inputs(bbox):
    """sample_points should reject malformed bbox values."""
    with pytest.raises(ValueError):
        sample_points(n=5, bbox=bbox, random_state=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("polygon", [Point(0, 0), Polygon()])
def test_sample_points_invalid_or_empty_polygon(polygon):
    """sample_points should reject invalid polygon inputs."""
    with pytest.raises(ValueError):
        sample_points(n=5, polygon=polygon, random_state=1)  # type: ignore[arg-type]


def test_sample_points_requires_bbox_or_polygon():
    """sample_points should fail when no domain inputs are provided."""
    with pytest.raises(
        ValueError, match="At least one of bbox or polygon must be provided"
    ):
        sample_points(n=5, random_state=1)


def test_sample_points_rejects_empty_intersection():
    """sample_points should reject empty intersection final domains."""
    polygon = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
    with pytest.raises(
        ValueError, match="Final sampling domain must have positive area"
    ):
        sample_points(n=5, bbox=(0, 0, 1, 1), polygon=polygon, random_state=1)


def test_sample_points_rejects_non_callable_sampler():
    """sample_points should reject non-callable sampler values."""
    with pytest.raises(ValueError, match="sampler must be callable or None"):
        sample_points(n=5, bbox=(0, 0, 1, 1), sampler=123, random_state=1)  # type: ignore[arg-type]


def test_sample_points_rejects_sampler_output_wrong_shape():
    """sample_points should reject sampler outputs with invalid shape."""

    def bad_sampler(rng: np.random.Generator, k: int) -> np.ndarray:
        return np.zeros((k, 3))

    with pytest.raises(ValueError, match="sampler output must have exact shape"):
        sample_points(n=5, bbox=(0, 0, 1, 1), sampler=bad_sampler, random_state=1)


def test_sample_points_rejects_sampler_output_non_numeric():
    """sample_points should reject non-numeric sampler outputs."""

    def bad_sampler(rng: np.random.Generator, k: int):
        return [["x", "y"]] * k

    with pytest.raises(ValueError, match="sampler output must be numeric array-like"):
        sample_points(n=5, bbox=(0, 0, 1, 1), sampler=bad_sampler, random_state=1)


def test_sample_points_rejects_sampler_output_non_finite():
    """sample_points should reject sampler outputs containing NaN/Inf."""

    def bad_sampler(rng: np.random.Generator, k: int) -> np.ndarray:
        out = np.zeros((k, 2), dtype=float)
        out[0, 0] = np.nan
        return out

    with pytest.raises(
        ValueError, match="sampler output must contain only finite values"
    ):
        sample_points(n=5, bbox=(0, 0, 1, 1), sampler=bad_sampler, random_state=1)


@pytest.mark.parametrize("max_attempts", [0, -1])
def test_sample_points_invalid_max_attempts(max_attempts):
    """sample_points should reject non-positive max_attempts."""
    with pytest.raises(ValueError, match="max_attempts must be a positive integer"):
        sample_points(n=5, bbox=(0, 0, 1, 1), random_state=1, max_attempts=max_attempts)


def test_sample_points_runtime_exhaustion_path():
    """sample_points should raise RuntimeError when attempts are exhausted."""

    def always_outside_sampler(rng: np.random.Generator, k: int) -> np.ndarray:
        return np.full((k, 2), 10.0)

    with pytest.raises(
        RuntimeError, match="Unable to sample the requested number of points"
    ):
        sample_points(
            n=5,
            bbox=(0, 0, 1, 1),
            sampler=always_outside_sampler,
            random_state=1,
            max_attempts=5,
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
    assert "pos" not in nodes_gdf.columns
    assert nodes_gdf.columns.tolist() == ["geometry"]


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
