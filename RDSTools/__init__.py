"""
RDS Tools - A Python package for Respondent-Driven Sampling analysis
"""
#print("__init__.py is being executed")


from .data_processing import RDSdata
from .bootstrap import RDSboot
from .mean import RDSmean
from .table import RDStable
from .regression import RDSlm
from .parallel_bootstrap import RDSBootOptimizedParallel
from .rds_map import (
    RDSmap,
    get_available_seeds,
    get_available_waves,
    print_map_info
)
from .network_graph import RDSnetgraph

__version__ = "0.1.0"
__all__ = [
    "RDSdata",
    "RDSboot",
    "RDStable",
    "RDSmean",
    "RDSlm",
    "RDSBootOptimizedParallel",
    "RDSmap",
    "get_available_seeds",
    "get_available_waves",
    "print_map_info",
    "RDSnetgraph"
]