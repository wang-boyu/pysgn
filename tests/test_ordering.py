import numpy as np
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon

from pysgn.ordering import attribute_order, density_order, random_order


@pytest.fixture
def point_gdf() -> GeoDataFrame:
    """Create a simple GeoDataFrame with points for testing.

    Returns:
        GeoDataFrame: Test GeoDataFrame with points
    """
    points = [Point(x, y) for x in range(5) for y in range(5)]
    data = {
        "geometry": points,
        "value": range(len(points)),
        "category": ["A" if i % 2 == 0 else "B" for i in range(len(points))],
    }
    return GeoDataFrame(data)


@pytest.fixture
def polygon_gdf() -> GeoDataFrame:
    """Create a simple GeoDataFrame with polygons for testing.

    Returns:
        GeoDataFrame: Test GeoDataFrame with polygons
    """
    polygons = [
        Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
        for x in range(5)
        for y in range(5)
    ]
    data = {
        "geometry": polygons,
        "value": range(len(polygons)),
        "category": ["A" if i % 2 == 0 else "B" for i in range(len(polygons))],
    }
    return GeoDataFrame(data)


def test_density_order_kde(point_gdf: GeoDataFrame) -> None:
    """Test KDE-based density ordering.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    order = density_order(point_gdf, method="kde")

    # Check output properties
    assert isinstance(order, np.ndarray)
    assert len(order) == len(point_gdf)
    assert set(order) == set(range(len(point_gdf)))

    # Center point should be first (highest density)
    center_idx = order[0]
    center_x = point_gdf.geometry.x[center_idx]
    center_y = point_gdf.geometry.y[center_idx]

    # Points should be ordered by distance from center
    distances = []
    for idx in order:
        x = point_gdf.geometry.x[idx]
        y = point_gdf.geometry.y[idx]
        dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
        distances.append(dist)

    assert np.all(np.diff(distances) >= 0)  # Distances should be non-decreasing


def test_density_order_knn(point_gdf: GeoDataFrame) -> None:
    """Test KNN-based density ordering.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    order = density_order(point_gdf, method="knn", k=5)

    # Check output properties
    assert isinstance(order, np.ndarray)
    assert len(order) == len(point_gdf)
    assert set(order) == set(range(len(point_gdf)))

    # Test with different n_neighbors
    order2 = density_order(point_gdf, method="knn", k=3)
    assert len(order2) == len(point_gdf)


def test_density_order_polygon(polygon_gdf: GeoDataFrame) -> None:
    """Test density ordering with polygon geometries.

    Args:
        polygon_gdf: Test GeoDataFrame with polygons
    """
    # Test both methods with polygons
    kde_order = density_order(polygon_gdf, method="kde")
    knn_order = density_order(polygon_gdf, method="knn")

    assert isinstance(kde_order, np.ndarray)
    assert isinstance(knn_order, np.ndarray)
    assert len(kde_order) == len(polygon_gdf)
    assert len(knn_order) == len(polygon_gdf)


def test_density_order_invalid_method(point_gdf: GeoDataFrame) -> None:
    """Test density ordering with invalid method.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    with pytest.raises(
        ValueError,
        match="Unknown density ordering method: invalid. Use 'knn' or 'kde'.",
    ):
        density_order(point_gdf, method="invalid")


def test_random_order(point_gdf: GeoDataFrame) -> None:
    """Test random ordering.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    # Test reproducibility with seed
    order1 = random_order(point_gdf, random_state=42)
    order2 = random_order(point_gdf, random_state=42)
    assert np.array_equal(order1, order2)

    # Test different seeds give different orders
    order3 = random_order(point_gdf, random_state=43)
    assert not np.array_equal(order1, order3)

    # Check output properties
    assert isinstance(order1, np.ndarray)
    assert len(order1) == len(point_gdf)
    assert set(order1) == set(range(len(point_gdf)))


def test_attribute_order_single(point_gdf: GeoDataFrame) -> None:
    """Test attribute ordering with single column.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    # Test ascending order
    order = attribute_order(point_gdf, by="value", ascending=True)
    assert np.array_equal(order, np.arange(len(point_gdf)))

    # Test descending order
    order_desc = attribute_order(point_gdf, by="value", ascending=False)
    assert np.array_equal(order_desc, np.arange(len(point_gdf))[::-1])


def test_attribute_order_multiple(point_gdf: GeoDataFrame) -> None:
    """Test attribute ordering with multiple columns.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    order = attribute_order(point_gdf, by=["category", "value"])
    assert isinstance(order, np.ndarray)
    assert len(order) == len(point_gdf)

    # Check if ordering is correct
    sorted_df = point_gdf.iloc[order]
    assert sorted_df.equals(point_gdf.sort_values(["category", "value"]))


def test_attribute_order_invalid_column(point_gdf: GeoDataFrame) -> None:
    """Test attribute ordering with invalid column.

    Args:
        point_gdf: Test GeoDataFrame with points
    """
    with pytest.raises(KeyError, match="invalid"):
        attribute_order(point_gdf, by="invalid")

    with pytest.raises(KeyError, match="invalid"):
        attribute_order(point_gdf, by=["value", "invalid"])


def test_empty_geodataframe() -> None:
    """Test ordering functions with empty GeoDataFrame."""
    empty_gdf = GeoDataFrame({"geometry": [], "value": []})

    # Test all ordering functions with empty GeoDataFrame
    assert len(density_order(empty_gdf, method="kde")) == 0
    assert len(random_order(empty_gdf)) == 0
    assert len(attribute_order(empty_gdf, by="value")) == 0


def test_single_point() -> None:
    """Test ordering functions with single point."""
    single_gdf = GeoDataFrame(
        {"geometry": [Point(0, 0)], "value": [1]}, geometry="geometry"
    )

    # Test all ordering functions with single point
    assert len(density_order(single_gdf, method="kde")) == 1
    assert len(random_order(single_gdf)) == 1
    assert len(attribute_order(single_gdf, by="value")) == 1
