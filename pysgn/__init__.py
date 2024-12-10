"""Python package for constructing synthetic geospatial networks."""

import datetime

from pysgn.geo_barabasi_albert_network import geo_barabasi_albert_network
from pysgn.geo_erdos_renyi_network import geo_erdos_renyi_network
from pysgn.geo_watts_strogatz_network import geo_watts_strogatz_network

__all__ = [
    "geo_erdos_renyi_network",
    "geo_watts_strogatz_network",
    "geo_barabasi_albert_network",
]

__title__ = "pysgn"
__version__ = "0.1.0"
__license__ = "MIT"
_this_year = datetime.datetime.now(tz=datetime.timezone.utc).date().year
__copyright__ = f"Copyright {_this_year} Wang Boyu"
