"""
Data normalization utilities for CausalFM.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


def normalize_data(
    X: np.ndarray,
    Y: np.ndarray,
    Y0: Optional[np.ndarray] = None,
    Y1: Optional[np.ndarray] = None,
    x_scaler: Optional[StandardScaler] = None,
    y_scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """
    Normalize features and outcomes for causal inference.
    
    This function standardizes both features (X) and outcomes (Y) to have zero mean 
    and unit variance, which is important for model training and evaluation consistency.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        Y: Outcome vector of shape (n_samples,) or (n_samples, 1)
        Y0: Potential outcome under control (optional)
        Y1: Potential outcome under treatment (optional)
        x_scaler: Pre-fitted scaler for X (if None, will fit new one)
        y_scaler: Pre-fitted scaler for Y (if None, will fit new one)
        
    Returns:
        Tuple of (X_normalized, Y_normalized, x_scaler, y_scaler)
        
    Example:
        >>> # Training: fit and transform
        >>> X_train_norm, Y_train_norm, x_scaler, y_scaler = normalize_data(
        ...     X_train, Y_train, Y0_train, Y1_train
        ... )
        >>> 
        >>> # Testing: transform only using fitted scalers
        >>> X_test_norm, Y_test_norm, _, _ = normalize_data(
        ...     X_test, Y_test, x_scaler=x_scaler, y_scaler=y_scaler
        ... )
    """
    # Handle Y shape
    Y_reshaped = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    
    # Normalize X
    if x_scaler is None:
        x_scaler = StandardScaler()
        X_normalized = x_scaler.fit_transform(X)
    else:
        X_normalized = x_scaler.transform(X)
    
    # Normalize Y
    if y_scaler is None:
        y_scaler = StandardScaler()
        # If Y0 and Y1 provided, fit on both for better scaling
        if Y0 is not None and Y1 is not None:
            Y0_reshaped = Y0.reshape(-1, 1) if Y0.ndim == 1 else Y0
            Y1_reshaped = Y1.reshape(-1, 1) if Y1.ndim == 1 else Y1
            y_combined = np.concatenate([Y0_reshaped, Y1_reshaped])
            y_scaler.fit(y_combined)
        Y_normalized = y_scaler.transform(Y_reshaped)
    else:
        Y_normalized = y_scaler.transform(Y_reshaped)
    
    # Flatten if original Y was 1D
    if Y.ndim == 1:
        Y_normalized = Y_normalized.flatten()
    
    return X_normalized, Y_normalized, x_scaler, y_scaler


def normalize_ite(
    Y0: np.ndarray,
    Y1: np.ndarray,
    y_scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize potential outcomes and compute normalized ITE.
    
    Args:
        Y0: Potential outcome under control
        Y1: Potential outcome under treatment  
        y_scaler: Pre-fitted scaler (if None, will fit new one)
        
    Returns:
        Tuple of (ITE_normalized, y_scaler)
        
    Example:
        >>> ITE_norm, y_scaler = normalize_ite(Y0_test, Y1_test, y_scaler)
    """
    Y0_reshaped = Y0.reshape(-1, 1) if Y0.ndim == 1 else Y0
    Y1_reshaped = Y1.reshape(-1, 1) if Y1.ndim == 1 else Y1
    
    if y_scaler is None:
        y_scaler = StandardScaler()
        y_combined = np.concatenate([Y0_reshaped, Y1_reshaped])
        y_scaler.fit(y_combined)
    
    Y0_normalized = y_scaler.transform(Y0_reshaped).flatten()
    Y1_normalized = y_scaler.transform(Y1_reshaped).flatten()
    
    ITE_normalized = Y1_normalized - Y0_normalized
    
    return ITE_normalized, y_scaler
