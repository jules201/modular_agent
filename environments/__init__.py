"""
Environments Package

This package contains the grid-world environment and helper functions
for the Modular DQN Agent.

Main Components:
    - World: The main grid-world environment
    - Helper functions: multivariate_gaussian, get_n_patches, get_n_resources, etc.
"""

from .world import World
from .helper_functions import (
    multivariate_gaussian,
    norm,
    get_n_patches,
    get_n_resources,
    get_spaced_means,
)

__all__ = [
    'World',
    'multivariate_gaussian',
    'norm',
    'get_n_patches',
    'get_n_resources',
    'get_spaced_means',
]