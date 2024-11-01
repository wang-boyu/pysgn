"""Python package for constructing synthetic geospatial networks."""

import datetime

from pysgn.small_world_network import get_nearest_nodes, small_world_network

__all__ = [
    "get_nearest_nodes",
    "small_world_network",
]

__title__ = "pysgn"
__version__ = "0.1.0"
__license__ = "MIT"
_this_year = datetime.datetime.now(tz=datetime.timezone.utc).date().year
__copyright__ = f"Copyright {_this_year} Wang Boyu"
