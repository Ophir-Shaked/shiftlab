#   from src import run_experiment
from .runner import run_experiment


# __all__ defines the public API of this package.
# Only names listed here will be exported when someone writes:
#   from src import *
#
# In our case, we explicitly expose only run_experiment
# and hide all internal modules (utils, metrics, generators, etc.).
__all__ = ["run_experiment"]
