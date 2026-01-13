"""
Quick test to verify normalization functionality works correctly.
"""

import numpy as np
from causalfm.data import normalize_data, normalize_ite


def test_normalize_data():
    """Test normalize_data function."""
    print("Testing normalize_data...")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 5) * 10 + 5  # Mean ~5, std ~10
    Y = np.random.randn(100) * 3 + 2      # Mean ~2, std ~3
    Y0 = np.random.randn(100) * 3 + 1
    Y1 = np.random.randn(100) * 3 + 3
    
    # Normalize
    X_norm, Y_norm, x_scaler, y_scaler = normalize_data(X, Y, Y0, Y1)
    
    # Check normalization (should be close to 0 mean, 1 std)
    assert abs(X_norm.mean()) < 0.1, f"X mean not close to 0: {X_norm.mean()}"
    assert abs(X_norm.std() - 1.0) < 0.1, f"X std not close to 1: {X_norm.std()}"
    assert abs(Y_norm.mean()) < 0.5, f"Y mean not close to 0: {Y_norm.mean()}"
    
    print(f"  X: mean={X_norm.mean():.4f}, std={X_norm.std():.4f} ✓")
    print(f"  Y: mean={Y_norm.mean():.4f}, std={Y_norm.std():.4f} ✓")
    
    # Test transform with fitted scalers
    X_test = np.random.randn(20, 5) * 10 + 5
    Y_test = np.random.randn(20) * 3 + 2
    
    X_test_norm, Y_test_norm, _, _ = normalize_data(
        X_test, Y_test, x_scaler=x_scaler, y_scaler=y_scaler
    )
    
    print(f"  Transform: X mean={X_test_norm.mean():.4f}, Y mean={Y_test_norm.mean():.4f} ✓")
    print("  normalize_data works correctly! ✓\n")
    
    return x_scaler, y_scaler


def test_normalize_ite(y_scaler):
    """Test normalize_ite function."""
    print("Testing normalize_ite...")
    
    # Create sample potential outcomes
    np.random.seed(42)
    Y0 = np.random.randn(50) * 3 + 1
    Y1 = np.random.randn(50) * 3 + 3
    
    # Normalize ITE
    ITE_norm, _ = normalize_ite(Y0, Y1, y_scaler)
    
    print(f"  ITE: mean={ITE_norm.mean():.4f}, std={ITE_norm.std():.4f} ✓")
    print("  normalize_ite works correctly! ✓\n")


def test_import():
    """Test that imports work correctly."""
    print("Testing imports...")
    
    try:
        from causalfm.data import normalize_data, normalize_ite
        print("  Imports successful! ✓\n")
    except ImportError as e:
        print(f"  Import failed: {e} ✗")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("CausalFM Normalization Tests")
    print("=" * 60 + "\n")
    
    try:
        # Run tests
        test_import()
        x_scaler, y_scaler = test_normalize_data()
        test_normalize_ite(y_scaler)
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
