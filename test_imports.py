"""
Test script to verify all CausalFM imports work correctly.
"""

print("=" * 70)
print("Testing CausalFM Package Imports")
print("=" * 70)

# Test 1: Main package
print("\n[1] Testing main package...")
try:
    import causalfm
    print(f"    ✓ causalfm version: {causalfm.__version__}")
except Exception as e:
    print(f"    ✗ Error: {e}")
    exit(1)

# Test 2: Data generators
print("\n[2] Testing data generators...")
try:
    from causalfm.data import StandardCATEGenerator, IVDataGenerator, FrontdoorDataGenerator
    print("    ✓ StandardCATEGenerator")
    print("    ✓ IVDataGenerator")
    print("    ✓ FrontdoorDataGenerator")
except Exception as e:
    print(f"    ✗ Error: {e}")
    exit(1)

# Test 3: Generate a small dataset
print("\n[3] Testing data generation...")
try:
    import numpy as np
    np.random.seed(42)
    
    # Test Standard CATE generator
    gen = StandardCATEGenerator(num_samples=50, num_features=3, seed=42)
    df = gen.generate()
    print(f"    ✓ Standard CATE: shape={df.shape}, columns={list(df.columns)[:6]}...")
    
    # Test IV generator
    iv_gen = IVDataGenerator(num_samples=50, num_features=3, seed=42)
    iv_df = iv_gen.generate()
    print(f"    ✓ IV Data: shape={iv_df.shape}")
    
    # Test Frontdoor generator
    fd_gen = FrontdoorDataGenerator(num_samples=50, num_features=3, seed=42)
    fd_df = fd_gen.generate()
    print(f"    ✓ Frontdoor Data: shape={fd_df.shape}")
    
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Evaluation metrics
print("\n[4] Testing evaluation metrics...")
try:
    from causalfm.evaluation import (
        compute_pehe, 
        compute_ate_error, 
        compute_mse, 
        compute_rmse,
        EvaluationResult
    )
    
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    true = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    pehe = compute_pehe(pred, true)
    ate_err = compute_ate_error(pred, true)
    mse = compute_mse(pred, true)
    rmse = compute_rmse(pred, true)
    
    print(f"    ✓ PEHE: {pehe:.4f}")
    print(f"    ✓ ATE Error: {ate_err:.4f}")
    print(f"    ✓ MSE: {mse:.4f}")
    print(f"    ✓ RMSE: {rmse:.4f}")
except Exception as e:
    print(f"    ✗ Error: {e}")
    exit(1)

# Test 5: Models
print("\n[5] Testing model imports...")
try:
    from causalfm.models import StandardCATEModel, IVModel, FrontdoorModel
    print("    ✓ StandardCATEModel")
    print("    ✓ IVModel")
    print("    ✓ FrontdoorModel")
except Exception as e:
    print(f"    ✗ Error: {e}")
    exit(1)

# Test 6: Training modules
print("\n[6] Testing training imports...")
try:
    from causalfm.training import (
        StandardCATETrainer, 
        IVTrainer, 
        FrontdoorTrainer,
        TrainingConfig
    )
    print("    ✓ StandardCATETrainer")
    print("    ✓ IVTrainer")
    print("    ✓ FrontdoorTrainer")
    print("    ✓ TrainingConfig")
except Exception as e:
    print(f"    ✗ Error: {e}")
    exit(1)

# Test 7: Data loaders
print("\n[7] Testing data loader imports...")
try:
    from causalfm.data.loaders import (
        create_standard_dataloader,
        create_iv_dataloader,
        create_frontdoor_dataloader,
        StandardCausalDataset,
        IVCausalDataset,
        FrontdoorCausalDataset
    )
    print("    ✓ create_standard_dataloader")
    print("    ✓ create_iv_dataloader")
    print("    ✓ create_frontdoor_dataloader")
    print("    ✓ Dataset classes")
except Exception as e:
    print(f"    ✗ Error: {e}")
    exit(1)

# Test 8: Create a simple model instance
print("\n[8] Testing model instantiation...")
try:
    model = StandardCATEModel(device='cpu')
    print(f"    ✓ StandardCATEModel created: {model}")
except Exception as e:
    print(f"    ✗ Error: {e}")
    exit(1)

# Test 9: Create a TrainingConfig
print("\n[9] Testing TrainingConfig...")
try:
    config = TrainingConfig(
        data_path="data/*.csv",
        epochs=10,
        batch_size=4,
        num_workers=0
    )
    print(f"    ✓ TrainingConfig created")
    print(f"      - epochs: {config.epochs}")
    print(f"      - batch_size: {config.batch_size}")
    print(f"      - learning_rate: {config.learning_rate}")
except Exception as e:
    print(f"    ✗ Error: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✅ All Import Tests Passed Successfully!")
print("=" * 70)
print("\nThe CausalFM package is properly installed and ready to use.")
print("\nNext steps:")
print("  1. Generate data: from causalfm.data import StandardCATEGenerator")
print("  2. Train model: from causalfm.training import StandardCATETrainer")
print("  3. Evaluate: from causalfm.evaluation import compute_pehe")
print("=" * 70)


