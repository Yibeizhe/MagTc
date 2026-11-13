"""MagTc 2: classical Monte Carlo Curie/Neel temperatures of 2D and 3D magnets, with VASP energy mapping."""
__version__ = "2.0.0"

from .model import ModelError, apply_overrides, load_model  # noqa: E402,F401
from .simulate import run_seed  # noqa: E402,F401
from .analysis import analyse, load_runs  # noqa: E402,F401
from .ground import ground  # noqa: E402,F401
