"""
Evaluation metrics for causal inference.

This module provides standard metrics for evaluating CATE estimation:
- PEHE (Precision in Estimation of Heterogeneous Effects)
- ATE Error (Average Treatment Effect Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch


@dataclass
class EvaluationResult:
    """
    Container for evaluation results.
    
    Attributes:
        dataset_name: Name of the evaluated dataset
        num_samples: Number of samples evaluated
        pehe: Precision in Estimation of Heterogeneous Effects
        ate_error: Average Treatment Effect Error
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
    """
    dataset_name: str
    num_samples: int
    pehe: float
    ate_error: float
    mse: float
    rmse: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'dataset': self.dataset_name,
            'samples': self.num_samples,
            'pehe': self.pehe,
            'ate_error': self.ate_error,
            'mse': self.mse,
            'rmse': self.rmse,
        }
    
    def __repr__(self) -> str:
        return (
            f"EvaluationResult("
            f"dataset={self.dataset_name}, "
            f"PEHE={self.pehe:.4f}, "
            f"ATE_Error={self.ate_error:.4f})"
        )


def compute_pehe(
    cate_pred: Union[np.ndarray, torch.Tensor],
    ite_true: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute PEHE (Precision in Estimation of Heterogeneous Effects).
    
    PEHE = sqrt(E[(tau_pred - tau_true)^2])
    
    where tau is the individual treatment effect.
    
    Args:
        cate_pred: Predicted CATE values
        ite_true: True ITE values
        
    Returns:
        PEHE value
        
    Example:
        >>> pehe = compute_pehe(predictions, ground_truth)
        >>> print(f"PEHE: {pehe:.4f}")
    """
    if isinstance(cate_pred, torch.Tensor):
        cate_pred = cate_pred.detach().cpu().numpy()
    if isinstance(ite_true, torch.Tensor):
        ite_true = ite_true.detach().cpu().numpy()
    
    cate_pred = np.asarray(cate_pred).flatten()
    ite_true = np.asarray(ite_true).flatten()
    
    mse = np.mean((cate_pred - ite_true) ** 2)
    return float(np.sqrt(mse))


def compute_ate_error(
    cate_pred: Union[np.ndarray, torch.Tensor],
    ite_true: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute ATE (Average Treatment Effect) Error.
    
    ATE_Error = |E[tau_pred] - E[tau_true]|
    
    Args:
        cate_pred: Predicted CATE values
        ite_true: True ITE values
        
    Returns:
        Absolute ATE error
        
    Example:
        >>> ate_err = compute_ate_error(predictions, ground_truth)
        >>> print(f"ATE Error: {ate_err:.4f}")
    """
    if isinstance(cate_pred, torch.Tensor):
        cate_pred = cate_pred.detach().cpu().numpy()
    if isinstance(ite_true, torch.Tensor):
        ite_true = ite_true.detach().cpu().numpy()
    
    cate_pred = np.asarray(cate_pred).flatten()
    ite_true = np.asarray(ite_true).flatten()
    
    ate_pred = np.mean(cate_pred)
    ate_true = np.mean(ite_true)
    
    return float(np.abs(ate_pred - ate_true))


def compute_mse(
    cate_pred: Union[np.ndarray, torch.Tensor],
    ite_true: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute Mean Squared Error between predicted CATE and true ITE.
    
    Args:
        cate_pred: Predicted CATE values
        ite_true: True ITE values
        
    Returns:
        MSE value
    """
    if isinstance(cate_pred, torch.Tensor):
        cate_pred = cate_pred.detach().cpu().numpy()
    if isinstance(ite_true, torch.Tensor):
        ite_true = ite_true.detach().cpu().numpy()
    
    cate_pred = np.asarray(cate_pred).flatten()
    ite_true = np.asarray(ite_true).flatten()
    
    return float(np.mean((cate_pred - ite_true) ** 2))


def compute_rmse(
    cate_pred: Union[np.ndarray, torch.Tensor],
    ite_true: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        cate_pred: Predicted CATE values
        ite_true: True ITE values
        
    Returns:
        RMSE value
    """
    return float(np.sqrt(compute_mse(cate_pred, ite_true)))


def evaluate_model(
    model,
    data_path: str,
    train_ratio: float = 0.8,
    device: Optional[str] = None
) -> EvaluationResult:
    """
    Evaluate a CausalFM model on a dataset.
    
    Uses a portion of the data for training context and evaluates
    on the remaining portion.
    
    Args:
        model: CausalFM model (StandardCATEModel, IVModel, or FrontdoorModel)
        data_path: Path to the evaluation CSV file
        train_ratio: Fraction of data to use as training context
        device: Device to use for evaluation
        
    Returns:
        EvaluationResult with metrics
        
    Example:
        >>> from causalfm.models import StandardCATEModel
        >>> model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
        >>> result = evaluate_model(model, "data/test.csv")
        >>> print(f"PEHE: {result.pehe:.4f}")
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Load data
    df = pd.read_csv(data_path)
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    
    # Extract columns
    x_cols = [col for col in df.columns if col.startswith('x')]
    X = torch.FloatTensor(df[x_cols].values).to(device)
    A = torch.FloatTensor(df['treatment'].values).to(device)
    Y = torch.FloatTensor(df['outcome'].values).to(device)
    ITE = torch.FloatTensor(df['ite'].values).to(device)
    
    # Split into train/test
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    
    x_train = X[:train_size]
    a_train = A[:train_size]
    y_train = Y[:train_size]
    x_test = X[train_size:]
    ite_test = ITE[train_size:]
    
    # Get predictions
    model.eval_mode()
    with torch.no_grad():
        result = model.estimate_cate(x_train, a_train, y_train, x_test)
        cate_pred = result['cate']
    
    # Convert to numpy
    cate_pred = cate_pred.detach().cpu().numpy().flatten()
    ite_test = ite_test.detach().cpu().numpy().flatten()
    
    # Compute metrics
    pehe = compute_pehe(cate_pred, ite_test)
    ate_error = compute_ate_error(cate_pred, ite_test)
    mse = compute_mse(cate_pred, ite_test)
    rmse = compute_rmse(cate_pred, ite_test)
    
    return EvaluationResult(
        dataset_name=dataset_name,
        num_samples=len(cate_pred),
        pehe=pehe,
        ate_error=ate_error,
        mse=mse,
        rmse=rmse
    )


def evaluate_multiple_datasets(
    model,
    data_dir: str,
    file_pattern: str = "*.csv",
    train_ratio: float = 0.8,
    device: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluate a model on multiple datasets and return summary.
    
    Args:
        model: CausalFM model
        data_dir: Directory containing test datasets
        file_pattern: Pattern for matching files
        train_ratio: Fraction of data for training context
        device: Device to use
        
    Returns:
        DataFrame with evaluation results for all datasets
        
    Example:
        >>> results_df = evaluate_multiple_datasets(
        ...     model,
        ...     "data/test/",
        ...     file_pattern="test_*.csv"
        ... )
        >>> print(results_df)
    """
    import glob
    
    files = glob.glob(os.path.join(data_dir, file_pattern))
    
    if not files:
        raise ValueError(f"No files found matching {file_pattern} in {data_dir}")
    
    results = []
    for file_path in sorted(files):
        try:
            result = evaluate_model(model, file_path, train_ratio, device)
            results.append(result.to_dict())
            print(f"{result.dataset_name}: PEHE={result.pehe:.4f}")
        except Exception as e:
            print(f"Error evaluating {file_path}: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    # Add summary statistics
    if len(df) > 0:
        print("\n=== Summary ===")
        print(f"Avg PEHE: {df['pehe'].mean():.4f} ± {df['pehe'].std():.4f}")
        print(f"Avg ATE Error: {df['ate_error'].mean():.4f} ± {df['ate_error'].std():.4f}")
    
    return df

