# PySGN: A Python package for constructing synthetic geospatial networks

[![GitHub CI](https://github.com/wang-boyu/pysgn/actions/workflows/build.yml/badge.svg)](https://github.com/wang-boyu/pysgn/actions) [![Read the Docs](https://readthedocs.org/projects/pysgn/badge/?version=stable)](https://pysgn.readthedocs.io/stable) [![Codecov](https://codecov.io/gh/wang-boyu/pysgn/branch/main/graph/badge.svg)](https://codecov.io/gh/wang-boyu/pysgn) [![PyPI](https://img.shields.io/pypi/v/pysgn.svg)](https://pypi.org/project/pysgn) [![PyPI - License](https://img.shields.io/pypi/l/pysgn)](https://pypi.org/project/pysgn/) [![PyPI - Downloads](https://img.shields.io/pypi/dw/pysgn)](https://pypistats.org/packages/pysgn) [![DOI](https://zenodo.org/badge/DOI/10.3847/xxxxx.svg)](https://doi.org/10.3847/xxxxx)


## Introduction

PySGN (**Py**thon for **S**ynthetic **G**eospatial **N**etworks) is a Python package for constructing synthetic geospatial networks. It is built on top of the [NetworkX](https://networkx.github.io/) package, which provides a flexible and efficient data structure for representing complex networks and [GeoPandas](https://geopandas.org/), which extends the datatypes used by pandas to allow spatial operations on geometric types. PySGN is designed to be easy to use and flexible, allowing users to generate networks with a wide range of characteristics.

## Installation

PySGN can be installed using pip:

```bash
pip install pysgn
```

## Usage Example

### Geospatial Erdős-Rényi Network

Here's a simple example of how to use the `geo_erdos_renyi_network` function to create a geospatial Erdős-Rényi network:

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

Similarly you can use the `geo_watts_strogatz_network` function to create a geospatial Watts-Strogatz network:

```python
import geopandas as gpd
from pysgn import geo_watts_strogatz_network

# Load your geospatial data into a GeoDataFrame
gdf = gpd.read_file('path/to/your/geospatial_data.shp')

# Create a geospatial Watts-Strogatz network
graph = geo_watts_strogatz_network(gdf, k=4, p=0.1)

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
