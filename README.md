# PySGN: A Python package for constructing synthetic geospatial networks

[![GitHub CI](https://github.com/wang-boyu/pysgn/actions/workflows/build.yml/badge.svg)](https://github.com/wang-boyu/pysgn/actions) [![Read the Docs](https://readthedocs.org/projects/pysgn/badge/?version=stable)](https://pysgn.readthedocs.io/stable) [![Codecov](https://codecov.io/gh/wang-boyu/pysgn/branch/main/graph/badge.svg)](https://codecov.io/gh/wang-boyu/pysgn) [![PyPI](https://img.shields.io/pypi/v/pysgn.svg)](https://pypi.org/project/pysgn) [![PyPI - License](https://img.shields.io/pypi/l/pysgn)](https://pypi.org/project/pysgn/) [![DOI](https://zenodo.org/badge/DOI/10.3847/xxxxx.svg)](https://doi.org/10.3847/xxxxx)


## Introduction

PySGN (**Py**thon for **S**ynthetic **G**eospatial **N**etworks) is a Python package for constructing synthetic geospatial networks. It is built on top of the [NetworkX](https://networkx.github.io/) package, which provides a flexible and efficient data structure for representing complex networks and [GeoPandas](https://geopandas.org/), which extends the datatypes used by pandas to allow spatial operations on geometric types. PySGN is designed to be easy to use and flexible, allowing users to generate networks with a wide range of characteristics.

## Installation

PySGN can be installed using pip:

```bash
pip install pysgn
```

## Usage Example

### Geospatial Erdős-Rényi Network

Here's a simple example of how to use the `geo_erdos_renyi_network` function to create a geospatial Erdős-Rényi network. It generates a network where each pair of nodes is connected with probability `p`, which depends on the spatial distance between the nodes. The parameter `a` controls the rate of decay of the connection probability with distance.

```python
import geopandas as gpd
from pysgn import geo_erdos_renyi_network

# Load your geospatial data into a GeoDataFrame
gdf = gpd.read_file('path/to/your/geospatial_data.shp')

# Create a geospatial Erdős-Rényi network
graph = geo_erdos_renyi_network(gdf, a=3)

# Output the number of nodes and edges
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")
```

### Geospatial Watts-Strogatz Network

Similarly you can use the `geo_watts_strogatz_network` function to create a geospatial Watts-Strogatz network. It first creates a network where each node is connected to its `k` nearest neighbors. Then, it rewires each edge with probability `p`. If an edge is chosen to be rewired, it is replaced with a new edge to a random node, where the probability of connecting to this new node is inversely proportional to the spatial distance.

```python
import geopandas as gpd
from pysgn import geo_watts_strogatz_network

# Load your geospatial data into a GeoDataFrame
gdf = gpd.read_file('path/to/your/geospatial_data.shp')

# Create a geospatial Watts-Strogatz network
graph = geo_watts_strogatz_network(
    gdf,
    k=4,    # Each node is connected to k nearest neighbors
    p=0.1,  # Probability of rewiring each edge
    a=2,    # Distance decay exponent
)

# Output the number of nodes and edges
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")
```

### Geospatial Barabási-Albert Network

You can also use the `geo_barabasi_albert_network` function to create a geospatial Barabási-Albert network. It creates a network using geospatial preferential attachment, where the probability of connecting to existing nodes depends on both their degrees and the spatial distances.

```python
import geopandas as gpd
from pysgn import geo_barabasi_albert_network
from pysgn.ordering import density_order

# Load your geospatial data into a GeoDataFrame
gdf = gpd.read_file('path/to/your/geospatial_data.shp')

# Create a geospatial Barabási-Albert network
graph = geo_barabasi_albert_network(
    gdf,
    m=3,                # Each new node connects to 3 existing nodes
    a=2,                # Distance decay exponent
    max_degree=150,     # Maximum degree constraint
    # Use density-based node ordering (nodes in dense areas join first)
    node_order=lambda gdf: density_order(gdf, method='knn'),
)

# Output the number of nodes and edges
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")
```

## Documentation

For more information on how to use PySGN, please refer to the [documentation](https://pysgn.readthedocs.io/).

## Contributing

If you run into an issue, please file a [ticket](https://github.com/wang-boyu/pysgn/issues) for us to discuss. If possible, follow up with a pull request.

If you would like to add a feature, please reach out via [ticket](https://github.com/wang-boyu/pysgn/issues) or start a [discussion](https://github.com/wang-boyu/pysgn/discussions).
A feature is most likely to be added if you build it!

Don't forget to check out the [Contributors guide](https://github.com/wang-boyu/pysgn/blob/main/CONTRIBUTING.md).

## License

PySGN is released under the MIT License.
