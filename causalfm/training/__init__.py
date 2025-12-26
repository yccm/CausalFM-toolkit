"""
Training utilities for CausalFM.

This module provides Trainer classes for training CausalFM models
across different causal inference settings.
"""

from causalfm.training.standard import StandardCATETrainer
from causalfm.training.iv import IVTrainer
from causalfm.training.frontdoor import FrontdoorTrainer

__all__ = [
    "StandardCATETrainer",
    "IVTrainer",
    "FrontdoorTrainer",
]

