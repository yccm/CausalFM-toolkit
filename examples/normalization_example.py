"""
Simple example demonstrating data normalization for evaluation.

This example shows how to properly normalize test data before evaluation
to ensure consistency with training data preprocessing.
"""

import pandas as pd
import torch
from causalfm.data import normalize_data, normalize_ite
from causalfm.models import StandardCATEModel
from causalfm.evaluation import compute_pehe, compute_ate_error


def evaluate_with_normalization(model_path, test_data_path):
    """
    Evaluate a model on test data with proper normalization.
    
    Args:
        model_path: Path to pretrained model checkpoint
        test_data_path: Path to test CSV file
    """
    # Load model
    model = StandardCATEModel.from_pretrained(model_path)
    model.eval_mode()
    
    # Load test data
    df = pd.read_csv(test_data_path)
    print(f"Loaded {len(df)} samples from {test_data_path}")
    
    # Extract columns
    x_cols = [c for c in df.columns if c.startswith('x')]
    
    # Normalize data (IMPORTANT!)
    X_norm, Y_norm, x_scaler, y_scaler = normalize_data(
        df[x_cols].values,
        df['outcome'].values,
        df['y0'].values,
        df['y1'].values
    )
    print(f"Normalized {len(x_cols)} features")
    
    # Prepare tensors
    X = torch.FloatTensor(X_norm)
    A = torch.FloatTensor(df['treatment'].values).unsqueeze(1)
    Y = torch.FloatTensor(Y_norm).unsqueeze(1)
    
    # Normalize ITE for evaluation
    true_ite_norm, _ = normalize_ite(
        df['y0'].values, 
        df['y1'].values, 
        y_scaler
    )
    
    # Split into train/test for in-context learning
    n_train = int(0.8 * len(X))
    x_train, x_test = X[:n_train], X[n_train:]
    a_train, y_train = A[:n_train], Y[:n_train]
    ite_test = true_ite_norm[n_train:]
    
    print(f"Train size: {n_train}, Test size: {len(x_test)}")
    
    # Predict
    result = model.estimate_cate(x_train, a_train, y_train, x_test)
    pred_cate = result['cate'].cpu().numpy()
    
    # Evaluate
    pehe = compute_pehe(pred_cate, ite_test)
    ate_error = compute_ate_error(pred_cate, ite_test)
    
    print(f"\nResults:")
    print(f"  PEHE: {pehe:.4f}")
    print(f"  ATE Error: {ate_error:.4f}")
    
    return {
        'pehe': pehe,
        'ate_error': ate_error,
        'predictions': pred_cate,
        'true_ite': ite_test
    }


if __name__ == "__main__":
    # Example usage
    model_path = "checkpoints/checkpoints_standard/best_model.pth"
    test_data_path = "DATA_standard/standard_cate_TEST/cate_test_dataset_1.csv"
    
    results = evaluate_with_normalization(model_path, test_data_path)
    print(f"\nEvaluation complete!")
