"""
Model wrappers for CausalFM.

This module provides easy-to-use interfaces for loading and using
CausalFM models for different causal inference settings.
"""

from causalfm.models.standard import StandardCATEModel
from causalfm.models.iv import IVModel
from causalfm.models.frontdoor import FrontdoorModel

__all__ = [
    "StandardCATEModel",
    "IVModel",
    "FrontdoorModel",
]


