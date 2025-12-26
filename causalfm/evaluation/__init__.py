"""
Evaluation utilities for CausalFM.

This module provides functions and classes for evaluating
CausalFM models on causal inference tasks.
"""

from causalfm.evaluation.metrics import (
    compute_pehe,
    compute_ate_error,
    compute_mse,
    compute_rmse,
    evaluate_model,
    EvaluationResult,
)

__all__ = [
    "compute_pehe",
    "compute_ate_error",
    "compute_mse",
    "compute_rmse",
    "evaluate_model",
    "EvaluationResult",
]

