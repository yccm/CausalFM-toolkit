Standard CATE Estimation Example
==================================

This example demonstrates a complete workflow for standard CATE (Conditional Average 
Treatment Effect) estimation using CausalFM.

Overview
--------

In this example, we will:

1. Generate synthetic training data
2. Train a Standard CATE model
3. Evaluate the model on test data
4. Visualize the results

Complete Example
----------------

.. code-block:: python

   """
   Complete example for Standard CATE estimation with CausalFM
   """
   
   import torch
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from pathlib import Path
   
   from causalfm.data import StandardCATEGenerator
   from causalfm.models import StandardCATEModel
   from causalfm.training import StandardCATETrainer, TrainingConfig
   from causalfm.evaluation import compute_pehe, compute_ate_error, compute_rmse
   
   
   def generate_data():
       """Step 1: Generate synthetic datasets"""
       print("=" * 60)
       print("STEP 1: Data Generation")
       print("=" * 60)
       
       # Training data
       train_gen = StandardCATEGenerator(
           num_samples=1024,
           num_features=10,
           seed=42
       )
       
       print("Generating 500 training datasets...")
       train_gen.generate_multiple(
           num_datasets=500,
           output_dir="data/standard_cate/train/",
           filename_prefix="train"
       )
       
       # Test data
       test_gen = StandardCATEGenerator(
           num_samples=1024,
           num_features=10,
           seed=999  # Different seed for test
       )
       
       print("Generating 50 test datasets...")
       test_gen.generate_multiple(
           num_datasets=50,
           output_dir="data/standard_cate/test/",
           filename_prefix="test"
       )
       
       print("✓ Data generation complete!\n")
   
   
   def train_model():
       """Step 2: Train the model"""
       print("=" * 60)
       print("STEP 2: Model Training")
       print("=" * 60)
       
       # Configure training
       config = TrainingConfig(
           # Data
           data_path="data/standard_cate/train/*.csv",
           val_split=0.2,
           
           # Training
           epochs=100,
           batch_size=16,
           learning_rate=0.001,
           weight_decay=1e-5,
           
           # Early stopping
           early_stop_patience=30,
           
           # Model
           use_gmm_head=True,
           gmm_n_components=5,
           
           # Checkpointing
           save_dir="checkpoints/standard_cate/",
           save_freq=10,
           
           # Logging
           log_dir="logs/standard_cate/",
           
           # Hardware
           device='auto',
           num_workers=0,  # Set to 0 to avoid multiprocessing issues
           
           # Reproducibility
           seed=42
       )
       
       # Train
       print("Starting training...")
       trainer = StandardCATETrainer(config)
       trainer.train()
       
       print("\n✓ Training complete!\n")
   
   
   def evaluate_model():
       """Step 3: Evaluate the model"""
       print("=" * 60)
       print("STEP 3: Model Evaluation")
       print("=" * 60)
       
       # Load trained model
       model = StandardCATEModel.from_pretrained(
           "checkpoints/standard_cate/best_model.pth",
           device='cpu'
       )
       model.eval_mode()
       
       # Evaluate on all test datasets
       test_dir = Path("data/standard_cate/test/")
       test_files = sorted(test_dir.glob("test_*.csv"))
       
       results = []
       for file in test_files:
           # Load dataset
           df = pd.read_csv(file)
           
           # Extract features
           x_cols = [c for c in df.columns if c.startswith('x')]
           X = torch.FloatTensor(df[x_cols].values)
           A = torch.FloatTensor(df['treatment'].values).unsqueeze(1)
           Y = torch.FloatTensor(df['outcome'].values).unsqueeze(1)
           true_ite = df['ite'].values
           
           # Split into train/test for in-context learning
           n_train = int(0.8 * len(X))
           x_train = X[:n_train]
           x_test = X[n_train:]
           a_train = A[:n_train]
           y_train = Y[:n_train]
           ite_test = true_ite[n_train:]
           
           # Predict
           with torch.no_grad():
               result = model.estimate_cate(x_train, a_train, y_train, x_test)
           
           pred_cate = result['cate'].cpu().numpy()
           
           # Compute metrics
           pehe = compute_pehe(pred_cate, ite_test)
           ate_error = compute_ate_error(pred_cate, ite_test)
           rmse = compute_rmse(pred_cate, ite_test)
           
           results.append({
               'dataset': file.name,
               'pehe': pehe,
               'ate_error': ate_error,
               'rmse': rmse,
               'n_test': len(ite_test)
           })
           
           print(f"  {file.name}: PEHE={pehe:.4f}, ATE Error={ate_error:.4f}")
       
       # Aggregate results
       results_df = pd.DataFrame(results)
       results_df.to_csv("results/standard_cate_results.csv", index=False)
       
       print("\n" + "=" * 60)
       print("SUMMARY STATISTICS")
       print("=" * 60)
       print(f"Number of test datasets: {len(results_df)}")
       print(f"\nPEHE: {results_df['pehe'].mean():.4f} ± {results_df['pehe'].std():.4f}")
       print(f"ATE Error: {results_df['ate_error'].mean():.4f} ± {results_df['ate_error'].std():.4f}")
       print(f"RMSE: {results_df['rmse'].mean():.4f} ± {results_df['rmse'].std():.4f}")
       
       print("\n✓ Evaluation complete!\n")
       
       return results_df, model
   
   
   def visualize_results(results_df, model):
       """Step 4: Visualize results"""
       print("=" * 60)
       print("STEP 4: Visualization")
       print("=" * 60)
       
       # Load one test dataset for visualization
       df = pd.read_csv("data/standard_cate/test/test_dataset_1.csv")
       
       x_cols = [c for c in df.columns if c.startswith('x')]
       X = torch.FloatTensor(df[x_cols].values)
       A = torch.FloatTensor(df['treatment'].values).unsqueeze(1)
       Y = torch.FloatTensor(df['outcome'].values).unsqueeze(1)
       true_ite = df['ite'].values
       
       n_train = int(0.8 * len(X))
       x_train = X[:n_train]
       x_test = X[n_train:]
       a_train = A[:n_train]
       y_train = Y[:n_train]
       ite_test = true_ite[n_train:]
       
       # Get predictions with uncertainty
       with torch.no_grad():
           result = model.estimate_cate(x_train, a_train, y_train, x_test)
       
       pred_cate = result['cate'].cpu().numpy()
       pi = result['gmm_pi'].cpu().numpy()
       mu = result['gmm_mu'].cpu().numpy()
       sigma = result['gmm_sigma'].cpu().numpy()
       
       # Compute confidence intervals
       n_samples = 10000
       samples = np.zeros((len(pred_cate), n_samples))
       
       for i in range(len(pred_cate)):
           components = np.random.choice(len(pi[i]), size=n_samples, p=pi[i])
           for k in range(len(pi[i])):
               mask = (components == k)
               n_k = mask.sum()
               if n_k > 0:
                   samples[i, mask] = np.random.normal(mu[i, k], sigma[i, k], n_k)
       
       ci_lower = np.percentile(samples, 2.5, axis=1)
       ci_upper = np.percentile(samples, 97.5, axis=1)
       
       # Create visualizations
       fig, axes = plt.subplots(2, 2, figsize=(14, 10))
       
       # Plot 1: Predicted vs True
       ax1 = axes[0, 0]
       ax1.scatter(ite_test, pred_cate, alpha=0.6)
       min_val = min(ite_test.min(), pred_cate.min())
       max_val = max(ite_test.max(), pred_cate.max())
       ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
       ax1.set_xlabel('True ITE')
       ax1.set_ylabel('Predicted CATE')
       ax1.set_title('Predicted vs True Treatment Effects')
       ax1.legend()
       ax1.grid(True, alpha=0.3)
       
       # Plot 2: Error Distribution
       ax2 = axes[0, 1]
       errors = pred_cate - ite_test
       ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7)
       ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
       ax2.set_xlabel('Prediction Error')
       ax2.set_ylabel('Frequency')
       ax2.set_title('Error Distribution')
       ax2.legend()
       
       # Plot 3: Uncertainty Calibration
       ax3 = axes[1, 0]
       variance = (pi * (sigma**2 + mu**2)).sum(axis=-1) - pred_cate**2
       std_dev = np.sqrt(variance)
       ax3.scatter(std_dev, np.abs(errors), alpha=0.6)
       ax3.set_xlabel('Predicted Std Dev')
       ax3.set_ylabel('Absolute Error')
       ax3.set_title('Uncertainty Calibration')
       ax3.grid(True, alpha=0.3)
       
       # Plot 4: Predictions with Uncertainty
       ax4 = axes[1, 1]
       sorted_idx = np.argsort(pred_cate)
       x = np.arange(len(sorted_idx))
       ax4.plot(x, pred_cate[sorted_idx], label='Predicted CATE', color='blue', linewidth=2)
       ax4.fill_between(x, ci_lower[sorted_idx], ci_upper[sorted_idx], 
                        alpha=0.3, label='95% CI')
       ax4.scatter(x, ite_test[sorted_idx], s=20, alpha=0.6, 
                  color='red', label='True ITE')
       ax4.set_xlabel('Sample (sorted by prediction)')
       ax4.set_ylabel('Treatment Effect')
       ax4.set_title('Predictions with Uncertainty Bands')
       ax4.legend()
       ax4.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.savefig('results/standard_cate_visualization.png', dpi=300, bbox_inches='tight')
       print("✓ Visualization saved to results/standard_cate_visualization.png")
       
       # Plot 5: PEHE across datasets
       fig2, ax = plt.subplots(figsize=(10, 6))
       ax.bar(range(len(results_df)), results_df['pehe'])
       ax.axhline(results_df['pehe'].mean(), color='r', linestyle='--', 
                  linewidth=2, label=f"Mean: {results_df['pehe'].mean():.4f}")
       ax.set_xlabel('Test Dataset')
       ax.set_ylabel('PEHE')
       ax.set_title('PEHE Across Test Datasets')
       ax.legend()
       ax.grid(True, alpha=0.3, axis='y')
       
       plt.tight_layout()
       plt.savefig('results/pehe_across_datasets.png', dpi=300, bbox_inches='tight')
       print("✓ PEHE plot saved to results/pehe_across_datasets.png")
       
       print("\n✓ Visualization complete!\n")
   
   
   if __name__ == '__main__':
       # Create directories
       Path("data/standard_cate/train/").mkdir(parents=True, exist_ok=True)
       Path("data/standard_cate/test/").mkdir(parents=True, exist_ok=True)
       Path("checkpoints/standard_cate/").mkdir(parents=True, exist_ok=True)
       Path("logs/standard_cate/").mkdir(parents=True, exist_ok=True)
       Path("results/").mkdir(parents=True, exist_ok=True)
       
       # Run complete pipeline
       generate_data()
       train_model()
       results_df, model = evaluate_model()
       visualize_results(results_df, model)
       
       print("=" * 60)
       print("PIPELINE COMPLETE!")
       print("=" * 60)
       print("\nOutputs:")
       print("  - Model: checkpoints/standard_cate/best_model.pth")
       print("  - Results: results/standard_cate_results.csv")
       print("  - Plots: results/*.png")
       print("  - Logs: logs/standard_cate/")

Expected Output
---------------

When you run this example, you should see output similar to:

.. code-block:: text

   ============================================================
   STEP 1: Data Generation
   ============================================================
   Generating 500 training datasets...
   Generating 50 test datasets...
   ✓ Data generation complete!
   
   ============================================================
   STEP 2: Model Training
   ============================================================
   Starting training...
   Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━━━━ 100% | Train Loss: 1.23 | Val Loss: 1.34
   Epoch 2/100 ━━━━━━━━━━━━━━━━━━━━━━━ 100% | Train Loss: 1.15 | Val Loss: 1.25
   ✓ New best model saved!
   ...
   ✓ Training complete!
   
   ============================================================
   STEP 3: Model Evaluation
   ============================================================
     test_dataset_1.csv: PEHE=0.4523, ATE Error=0.0234
     test_dataset_2.csv: PEHE=0.4312, ATE Error=0.0189
   ...
   
   ============================================================
   SUMMARY STATISTICS
   ============================================================
   Number of test datasets: 50
   
   PEHE: 0.4456 ± 0.0234
   ATE Error: 0.0201 ± 0.0089
   RMSE: 0.4782 ± 0.0245
   
   ✓ Evaluation complete!

Key Takeaways
-------------

1. **Data Generation**: Generate many diverse synthetic datasets for training
2. **Training**: Use the TrainingConfig to control all aspects of training
3. **Evaluation**: Evaluate on multiple test datasets for robust estimates
4. **Uncertainty**: GMM head provides calibrated uncertainty quantification
5. **Visualization**: Visualize predictions and errors to understand model behavior

Next Steps
----------

* Try adjusting ``num_features`` or ``num_samples`` in data generation
* Experiment with different ``gmm_n_components`` values
* Test on real-world datasets like the Jobs dataset
* Compare with other causal inference methods
* Explore the IV and Front-door examples

