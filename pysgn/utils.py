from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from numbers import Real
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from pyproj import CRS
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry


def sample_points(
    n: int,
    *,
    bbox: Sequence[float] | None = None,
    polygon: Polygon | MultiPolygon | None = None,
    sampler: Callable[[np.random.Generator, int], np.ndarray] | None = None,
    random_state: int | None = None,
    crs: Any | None = None,
    max_attempts: int = 1_000_000,
) -> gpd.GeoDataFrame:
    """Sample point geometries within a spatial domain.

    Phase 1 API for node synthesis. This function returns a geometry-only
    ``GeoDataFrame`` with exactly ``n`` point geometries in the ``geometry``
    column.

    Domain definition:
    - ``bbox`` only: sample in ``(xmin, ymin, xmax, ymax)`` bounds
    - ``polygon`` only: sample in polygon area
    - both: sample in ``polygon.intersection(bbox_polygon)``
    - at least one of ``bbox`` or ``polygon`` must be provided

    Sampling modes:
    - ``sampler is None``: uniform candidate generation over the final domain
    - ``sampler is callable``: user-driven candidate generation using the
      contract ``sampler(rng, k) -> array-like`` with exact shape ``(k, 2)``,
      numeric finite values, then rejection against the final domain

    Determinism:
    - ``random_state`` may be an ``int`` or ``None``
    - RNG initialization uses ``np.random.default_rng(seed=random_state)``

    CRS:
    - CRS is set only from explicit ``crs``
    - explicit CRS values are normalized with
      ``pyproj.CRS.from_user_input(crs)``
      (https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input)
    - no CRS inference is performed from ``polygon``

    Error behavior:
    - Raises ``ValueError`` for invalid inputs (for example: ``n <= 0``,
      invalid ``bbox``, invalid/empty ``polygon``, missing domain, non-callable
      ``sampler``, invalid sampler output, ``max_attempts <= 0``, or final
      domain area ``<= 0``)
    - Raises ``RuntimeError`` if sampling cannot produce ``n`` accepted points
      within ``max_attempts`` evaluated candidate points

    Args:
        n: Number of points to generate; must be > 0.
        bbox: Bounding box sequence ``(xmin, ymin, xmax, ymax)``.
        polygon: Domain polygon geometry (``Polygon`` or ``MultiPolygon`` only).
        sampler: Optional callable producing candidate coordinates.
        random_state: Random seed for deterministic output.
        crs: Optional CRS user input passed through
            ``pyproj.CRS.from_user_input`` before assignment.
        max_attempts: Maximum number of evaluated candidate points.

    Returns:
        A geometry-only ``geopandas.GeoDataFrame`` with ``n`` sampled points.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer.")

    if not isinstance(max_attempts, (int, np.integer)) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer.")

    if sampler is not None and not callable(sampler):
        raise ValueError("sampler must be callable or None.")

    final_domain = _resolve_sampling_domain(bbox=bbox, polygon=polygon)
    rng = np.random.default_rng(seed=random_state)
    parsed_crs = CRS.from_user_input(crs) if crs is not None else None

    accepted_coords: list[tuple[float, float]] = []
    attempts = 0

    while len(accepted_coords) < n and attempts < max_attempts:
        remaining_points = n - len(accepted_coords)
        remaining_attempts = max_attempts - attempts
        batch_size = max(1, min(4096, remaining_points, remaining_attempts))

        candidates = _candidate_batch(
            rng=rng,
            final_domain=final_domain,
            sampler=sampler,
            k=batch_size,
        )
        attempts += batch_size

        for x, y in candidates:
            point = Point(float(x), float(y))
            if final_domain.covers(point):
                accepted_coords.append((float(x), float(y)))
                if len(accepted_coords) == n:
                    break

    if len(accepted_coords) < n:
        raise RuntimeError(
            "Unable to sample the requested number of points within max_attempts; "
            "try increasing max_attempts or adjusting the domain/sampler."
        )

    accepted_array = np.asarray(accepted_coords, dtype=float)
    geometry = gpd.points_from_xy(accepted_array[:, 0], accepted_array[:, 1])
    return gpd.GeoDataFrame(geometry=geometry, crs=parsed_crs)


def _bbox_to_polygon(bbox: Sequence[float]) -> Polygon:
    """Validate bbox and return a rectangle polygon."""
    try:
        bbox_values = list(bbox)
    except TypeError as exc:
        raise ValueError(
            "bbox must be a sequence of 4 values: (xmin, ymin, xmax, ymax)."
        ) from exc

    if len(bbox_values) != 4:
        raise ValueError(
            "bbox must be a sequence of 4 values: (xmin, ymin, xmax, ymax)."
        )

    validated: list[float] = []
    for idx, value in enumerate(bbox_values):
        if not isinstance(value, Real):
            raise ValueError(
                f"bbox values must be numeric; got {type(value).__name__} at index {idx}."
            )
        value_f = float(value)
        if not np.isfinite(value_f):
            raise ValueError(f"bbox values must be finite; got {value} at index {idx}.")
        validated.append(value_f)

    xmin, ymin, xmax, ymax = validated
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("bbox ordering must satisfy xmin < xmax and ymin < ymax.")

    return Polygon(
        [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
    )


def _validate_polygon(polygon: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    """Validate polygon input for node synthesis domain construction."""
    if not isinstance(polygon, (Polygon, MultiPolygon)):
        raise ValueError("polygon must be a shapely Polygon or MultiPolygon.")
    if polygon.is_empty:
        raise ValueError("polygon must be non-empty.")
    return polygon


def _resolve_sampling_domain(
    *,
    bbox: Sequence[float] | None,
    polygon: Polygon | MultiPolygon | None,
) -> BaseGeometry:
    """Resolve bbox/polygon inputs into a validated final sampling domain."""
    bbox_polygon: Polygon | None = _bbox_to_polygon(bbox) if bbox is not None else None
    validated_polygon: Polygon | MultiPolygon | None = (
        _validate_polygon(polygon) if polygon is not None else None
    )

    if bbox_polygon is None and validated_polygon is None:
        raise ValueError("At least one of bbox or polygon must be provided.")

    if bbox_polygon is not None and validated_polygon is not None:
        final_domain = validated_polygon.intersection(bbox_polygon)
    else:
        final_domain = bbox_polygon if bbox_polygon is not None else validated_polygon

    if final_domain is None or final_domain.area <= 0:
        raise ValueError(
            "Final sampling domain must have positive area; check bbox/polygon overlap."
        )
    return final_domain


def _validate_sampler_output(candidates: Any, k: int) -> np.ndarray:
    """Validate sampler output against the strict Phase 1 contract."""
    candidate_array = np.asarray(candidates)
    if not np.issubdtype(candidate_array.dtype, np.number):
        raise ValueError("sampler output must be numeric array-like with shape (k, 2).")

    if candidate_array.ndim != 2 or candidate_array.shape != (k, 2):
        raise ValueError(
            f"sampler output must have exact shape ({k}, 2); got {candidate_array.shape}."
        )

    candidate_array = candidate_array.astype(float, copy=False)
    if not np.isfinite(candidate_array).all():
        raise ValueError("sampler output must contain only finite values.")
    return candidate_array


def _candidate_batch(
    *,
    rng: np.random.Generator,
    final_domain: BaseGeometry,
    sampler: Callable[[np.random.Generator, int], np.ndarray] | None,
    k: int,
) -> np.ndarray:
    """Generate one candidate batch via uniform or custom sampler mode."""
    if sampler is None:
        xmin, ymin, xmax, ymax = final_domain.bounds
        x = rng.uniform(low=xmin, high=xmax, size=k)
        y = rng.uniform(low=ymin, high=ymax, size=k)
        return np.column_stack((x, y))
    if not callable(sampler):
        raise ValueError("sampler must be callable or None.")
    return _validate_sampler_output(sampler(rng, k), k=k)


def _compute_probabilities(
    distances: np.ndarray, a: int, scaling_factor: float
) -> np.ndarray:
    """
    Compute the probability of connecting two nodes based on their distance.

    The probability is calculated as:
    p(distance) = min(1, (distance * scaling_factor) ^ (-a))

    p = 1 if distance < 1 / scaling_factor
    p = (distance * scaling_factor) ^ (-a) if distance >= 1 / scaling_factor

    Args:
        distances: An array of distances between nodes.
        a: The distance decay exponent parameter.
        scaling_factor: The inverse of the minimum distance between nodes.
    """
    factors = np.ones_like(distances)
    mask = distances >= (1 / scaling_factor)
    factors[mask] = (distances[mask] * scaling_factor) ** (-a)
    return factors


def _create_k_col(k, n, random_state=None) -> np.ndarray:
    """
    Creates a column of integer degree centrality so that its mean is equal to k/2.

    - If k is even, all values are k//2.
    - If k is odd, half of the values are k//2 and the other half are k//2 + 1.
    - If k is float, a fraction of the values are floor(k/2) and the rest are ceil(k/2).
      The fraction is determined by the decimal part of k/2. For example, if k/2 = 3.2,
      80% of the values are 3 and 20% are 4.

    Args:
        k: The target mean degree centrality.
        n: The number of elements in the column.
        random_state: Seed for random number generator for reproducibility.

    Returns:
        np.ndarray: An array of integer degree centralities.
    """
    rng = np.random.default_rng(random_state)
    if isinstance(k, int):
        if k % 2 == 0:
            return np.full(n, k // 2, dtype=int)
        else:
            return rng.choice([k // 2, k // 2 + 1], n, p=[0.5, 0.5])
    elif isinstance(k, float):
        half_k = k / 2
        lower = int(np.floor(half_k))
        upper = int(np.ceil(half_k))
        # prevent `ZeroDivisionError: integer modulo by zero` if half_k < 1
        upper_prob = half_k if half_k < 1 else half_k % lower
        return rng.choice([lower, upper], n, p=[1 - upper_prob, upper_prob])
    else:
        raise ValueError("k must be an integer or a float.")


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


def _get_id_col_array(gdf: gpd.GeoDataFrame, id_col: str | None) -> np.ndarray:
    """Return node id values based on id_col settings."""
    if id_col is None:
        return np.arange(len(gdf))
    if id_col == "index":
        if isinstance(gdf.index, pd.MultiIndex):
            raise ValueError("Multi-index is not supported")
        id_values = gdf.index.values
    else:
        id_values = gdf[id_col].values
    if len(np.unique(id_values)) != len(id_values):
        raise ValueError("ID column must contain unique values")
    return id_values


def _set_node_attributes(graph, gdf, id_col, node_attributes):
    if gdf.geometry.geom_type.iloc[0] == "Polygon":
        pos_x_array = gdf.geometry.centroid.x.values
        pos_y_array = gdf.geometry.centroid.y.values
    else:
        pos_x_array = gdf.geometry.x.values
        pos_y_array = gdf.geometry.y.values
    if node_attributes is True:
        node_attributes = gdf.columns.tolist()
    if isinstance(node_attributes, str):
        node_attributes = [node_attributes]
    # Avoid storing the id column again as a node attribute when it is already
    # being used for the node keys.
    if id_col is not None and node_attributes:
        node_attributes = [attr for attr in node_attributes if attr != id_col]
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


def graph_to_gdf(
    graph: nx.Graph,
    *,
    nodes: bool = True,
    edges: bool = True,
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    """Convert a geospatial NetworkX graph into node and edge GeoDataFrames.

    This function reconstructs node and edge GeoDataFrames from a graph that
    stores geospatial attributes. Node geometry is taken from the stored
    geometry attribute when available, otherwise it falls back to a Point
    created from the ``pos`` coordinates. Edge geometry uses a stored shapely
    geometry when present; otherwise it is built as a straight LineString
    between endpoint positions or centroids. CRS metadata is read from
    ``graph.graph["crs"]`` and applied to outputs when present.

    Args:
        graph: Graph containing geospatial node/edge attributes.
        nodes: Whether to build the node GeoDataFrame.
        edges: Whether to build the edge GeoDataFrame.

    Returns:
        A tuple of ``(nodes_gdf, edges_gdf)`` with ``None`` for any layer not
        requested.

    Raises:
        ValueError: If both ``nodes`` and ``edges`` are ``False``.
        RuntimeError: If geometry cannot be reconstructed for a node or edge.
    """
    if not nodes and not edges:
        raise ValueError(
            "At least one layer must be requested; set nodes=True or edges=True (received nodes=False, edges=False)."
        )

    id_col = graph.graph.get("id_col")
    index_name = graph.graph.get("index_name")
    crs = graph.graph.get("crs")

    if crs is None:
        warnings.warn(
            "Graph has no CRS; output GeoDataFrames will have an undefined coordinate reference system.",
            UserWarning,
            stacklevel=2,
        )

    nodes_gdf: gpd.GeoDataFrame | None = None
    edges_gdf: gpd.GeoDataFrame | None = None

    if nodes:
        node_records: list[dict[str, Any]] = []
        node_ids: list[Any] = []
        for node_id, attrs in graph.nodes(data=True):
            record = dict(attrs)
            geometry = _node_geometry_from_attrs(node_id, attrs)
            record["geometry"] = geometry
            node_records.append(record)
            node_ids.append(node_id)

        node_df = pd.DataFrame.from_records(node_records)
        nodes_gdf = gpd.GeoDataFrame(node_df, geometry="geometry", crs=crs)

        if id_col == "index":
            nodes_gdf.index = pd.Index(node_ids, name=index_name)
        elif isinstance(id_col, str):
            nodes_gdf[id_col] = node_ids
        elif id_col is None:
            nodes_gdf.index = pd.Index(node_ids)

    if edges:
        edge_records: list[dict[str, Any]] = []
        for u, v, data in graph.edges(data=True):
            record = {"source": u, "target": v, **data}
            geometry = data.get("geometry")
            if geometry is None or not isinstance(geometry, BaseGeometry):
                geometry = _edge_geometry_from_nodes(graph, u, v)
            record["geometry"] = geometry
            edge_records.append(record)

        edge_df = pd.DataFrame.from_records(edge_records)
        edges_gdf = gpd.GeoDataFrame(edge_df, geometry="geometry", crs=crs)

    return nodes_gdf, edges_gdf


def _node_geometry_from_attrs(node_id: Any, attrs: dict[str, Any]):
    if attrs.get("geometry") is not None:
        return attrs["geometry"]
    if attrs.get("pos") is not None:
        x, y = attrs["pos"]
        return Point(x, y)
    raise RuntimeError(
        f"Cannot reconstruct geometry for node {node_id}; expected 'geometry' or 'pos' attributes."
    )


def _edge_geometry_from_nodes(graph: nx.Graph, u: Any, v: Any) -> LineString:
    source_point = _node_point(graph, u)
    target_point = _node_point(graph, v)
    return LineString(
        [(source_point.x, source_point.y), (target_point.x, target_point.y)]
    )


def _node_point(graph: nx.Graph, node_id: Any) -> Point:
    attrs = graph.nodes[node_id]
    if attrs.get("geometry") is not None:
        geom = attrs["geometry"]
        if hasattr(geom, "centroid"):
            centroid = geom.centroid
            return Point(centroid.x, centroid.y)
    if attrs.get("pos") is not None:
        x, y = attrs["pos"]
        return Point(x, y)
    raise RuntimeError(
        f"Cannot reconstruct geometry for edge endpoint {node_id}; missing 'geometry' or 'pos' on node."
    )
