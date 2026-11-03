"""MagTc: classical Monte Carlo Curie/Neel temperatures of 2D magnets (JCP 2026 protocol)."""
__version__ = "1.0.0"

from .model import load_model, ModelError  # noqa: E402,F401
from .runner import run_seed, apply_overrides  # noqa: E402,F401
from .analysis import analyse, load_runs  # noqa: E402,F401
from .ground import ground  # noqa: E402,F401
