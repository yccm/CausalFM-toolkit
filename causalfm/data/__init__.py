"""
Data generation and loading utilities for CausalFM.

This module provides:
- Data generators for creating synthetic causal datasets
- Data loaders for training and evaluation
"""

from causalfm.data.generators import (
    StandardCATEGenerator,
    IVDataGenerator,
    FrontdoorDataGenerator,
)

from causalfm.data.loaders import (
    create_standard_dataloader,
    create_iv_dataloader,
    create_frontdoor_dataloader,
)

__all__ = [
    # Generators
    "StandardCATEGenerator",
    "IVDataGenerator",
    "FrontdoorDataGenerator",
    # Loaders
    "create_standard_dataloader",
    "create_iv_dataloader",
    "create_frontdoor_dataloader",
]

