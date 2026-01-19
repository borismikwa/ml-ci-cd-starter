"""
src package for ml-ci-cd-starter.

This file makes the src directory a package so imports like 'from src.data import ...' work
in CI and when running python -m src.train.
"""

__all__ = ["data", "train", "evaluate"]
__version__ = "0.1.0"

# End of file