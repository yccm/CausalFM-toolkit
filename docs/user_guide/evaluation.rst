Evaluation
==========

CausalFM provides comprehensive evaluation tools for assessing model performance 
on causal inference tasks.

Evaluation Metrics
------------------

Standard Metrics
~~~~~~~~~~~~~~~~

CausalFM implements the following metrics for evaluating causal effect estimates:

**PEHE (Precision in Estimation of Heterogeneous Effects)**
   Measures the accuracy of individual treatment effect (ITE) predictions:
   
   .. math::
   
      \text{PEHE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (\tau_i - \hat{\tau}_i)^2}
   
   where :math:`\tau_i` is the true ITE and :math:`\hat{\tau}_i` is the predicted CATE.

**ATE Error (Average Treatment Effect Error)**
   Measures the accuracy of average treatment effect estimation:
   
   .. math::
   
      \text{ATE Error} = \left|\frac{1}{n}\sum_{i=1}^n \tau_i - \frac{1}{n}\sum_{i=1}^n \hat{\tau}_i\right|

**MSE (Mean Squared Error)**
   Standard squared error metric:
   
   .. math::
   
      \text{MSE} = \frac{1}{n}\sum_{i=1}^n (\tau_i - \hat{\tau}_i)^2

**RMSE (Root Mean Squared Error)**
   Square root of MSE for interpretability:
   
   .. math::
   
      \text{RMSE} = \sqrt{\text{MSE}}

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from causalfm.evaluation import (
       compute_pehe,
       compute_ate_error,
       compute_mse,
       compute_rmse
   )
   import numpy as np
   
   # Ground truth ITEs
   true_ite = np.array([1.5, 2.3, 0.8, -0.5, 1.2])
   
   # Model predictions
   pred_cate = np.array([1.4, 2.1, 0.9, -0.3, 1.1])
   
   # Compute metrics
   pehe = compute_pehe(pred_cate, true_ite)
   ate_error = compute_ate_error(pred_cate, true_ite)
   mse = compute_mse(pred_cate, true_ite)
   rmse = compute_rmse(pred_cate, true_ite)
   
   print(f"PEHE: {pehe:.4f}")
   print(f"ATE Error: {ate_error:.4f}")
   print(f"MSE: {mse:.4f}")
   print(f"RMSE: {rmse:.4f}")

With PyTorch Tensors
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from causalfm.evaluation import compute_pehe
   
   # Metrics work with both numpy arrays and torch tensors
   true_ite = torch.randn(100)
   pred_cate = torch.randn(100)
   
   pehe = compute_pehe(pred_cate, true_ite)
   print(f"PEHE: {pehe:.4f}")

Model Evaluation
----------------

Evaluating a Single Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.models import StandardCATEModel
   from causalfm.evaluation import compute_pehe, compute_ate_error
   import pandas as pd
   import torch
   
   # Load model
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
   
   # Load test data
   df = pd.read_csv("data/test/test_dataset_1.csv")
   
   # Extract features
   x_cols = [c for c in df.columns if c.startswith('x')]
   X = torch.FloatTensor(df[x_cols].values)
   A = torch.FloatTensor(df['treatment'].values).unsqueeze(1)
   Y = torch.FloatTensor(df['outcome'].values).unsqueeze(1)
   true_ite = df['ite'].values
   
   # Split into train/test for in-context learning
   n_train = int(0.8 * len(X))
   x_train, x_test = X[:n_train], X[n_train:]
   a_train, y_train = A[:n_train], Y[:n_train]
   ite_test = true_ite[n_train:]
   
   # Predict
   result = model.estimate_cate(x_train, a_train, y_train, x_test)
   pred_cate = result['cate'].cpu().numpy()
   
   # Evaluate
   pehe = compute_pehe(pred_cate, ite_test)
   ate_error = compute_ate_error(pred_cate, ite_test)
   
   print(f"Dataset: test_dataset_1")
   print(f"  PEHE: {pehe:.4f}")
   print(f"  ATE Error: {ate_error:.4f}")

Evaluating Multiple Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.models import StandardCATEModel
   from causalfm.evaluation import compute_pehe, compute_ate_error
   import pandas as pd
   import torch
   from pathlib import Path
   
   # Load model
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
   
   # Evaluate on multiple test datasets
   test_dir = Path("data/test/")
   test_files = sorted(test_dir.glob("test_*.csv"))
   
   results = []
   for file in test_files:
       df = pd.read_csv(file)
       
       # Extract features
       x_cols = [c for c in df.columns if c.startswith('x')]
       X = torch.FloatTensor(df[x_cols].values)
       A = torch.FloatTensor(df['treatment'].values).unsqueeze(1)
       Y = torch.FloatTensor(df['outcome'].values).unsqueeze(1)
       true_ite = df['ite'].values
       
       # Split
       n_train = int(0.8 * len(X))
       x_train, x_test = X[:n_train], X[n_train:]
       a_train, y_train = A[:n_train], Y[:n_train]
       ite_test = true_ite[n_train:]
       
       # Predict and evaluate
       result = model.estimate_cate(x_train, a_train, y_train, x_test)
       pred_cate = result['cate'].cpu().numpy()
       
       pehe = compute_pehe(pred_cate, ite_test)
       ate_error = compute_ate_error(pred_cate, ite_test)
       
       results.append({
           'dataset': file.name,
           'pehe': pehe,
           'ate_error': ate_error
       })
   
   # Create results DataFrame
   results_df = pd.DataFrame(results)
   
   print(results_df)
   print(f"\nAverage PEHE: {results_df['pehe'].mean():.4f} ± {results_df['pehe'].std():.4f}")
   print(f"Average ATE Error: {results_df['ate_error'].mean():.4f} ± {results_df['ate_error'].std():.4f}")

Automated Evaluation
~~~~~~~~~~~~~~~~~~~~

For convenience, use the built-in evaluation utilities:

.. code-block:: python

   from causalfm.evaluation.metrics import evaluate_model_on_dataset
   
   # Evaluate single dataset
   result = evaluate_model_on_dataset(
       model,
       data_path="data/test/test_dataset_1.csv",
       train_ratio=0.8
   )
   
   print(f"PEHE: {result['pehe']:.4f}")
   print(f"ATE Error: {result['ate_error']:.4f}")

.. code-block:: python

   from causalfm.evaluation.metrics import evaluate_model_on_directory
   
   # Evaluate all datasets in a directory
   results_df = evaluate_model_on_directory(
       model,
       data_dir="data/test/",
       file_pattern="test_*.csv",
       train_ratio=0.8
   )
   
   print(results_df)
   print(f"\nSummary:")
   print(results_df[['pehe', 'ate_error']].describe())

Comparing Models
----------------

Comparing Multiple Models
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.models import StandardCATEModel, IVModel, FrontdoorModel
   from causalfm.evaluation import compute_pehe
   import pandas as pd
   
   # Load models
   models = {
       'Standard': StandardCATEModel.from_pretrained("checkpoints/standard.pth"),
       'IV': IVModel.from_pretrained("checkpoints/iv.pth"),
       'Frontdoor': FrontdoorModel.from_pretrained("checkpoints/frontdoor.pth")
   }
   
   # Evaluate each model
   comparison_results = []
   
   for name, model in models.items():
       # ... load and prepare data ...
       result = model.estimate_cate(x_train, a_train, y_train, x_test)
       pred_cate = result['cate'].cpu().numpy()
       
       pehe = compute_pehe(pred_cate, true_ite)
       
       comparison_results.append({
           'model': name,
           'pehe': pehe
       })
   
   comparison_df = pd.DataFrame(comparison_results)
   print(comparison_df)

Baseline Comparisons
~~~~~~~~~~~~~~~~~~~~

Compare with simple baselines:

.. code-block:: python

   import numpy as np
   from causalfm.evaluation import compute_pehe
   
   # CausalFM prediction
   causalfm_pehe = compute_pehe(pred_cate, true_ite)
   
   # Baseline 1: Predict ATE for everyone
   ate_baseline = np.full_like(true_ite, true_ite.mean())
   baseline_ate_pehe = compute_pehe(ate_baseline, true_ite)
   
   # Baseline 2: Random predictions
   random_pred = np.random.randn(len(true_ite))
   random_pehe = compute_pehe(random_pred, true_ite)
   
   # Baseline 3: Zero effect
   zero_pred = np.zeros_like(true_ite)
   zero_pehe = compute_pehe(zero_pred, true_ite)
   
   print(f"CausalFM PEHE: {causalfm_pehe:.4f}")
   print(f"ATE Baseline PEHE: {baseline_ate_pehe:.4f}")
   print(f"Random PEHE: {random_pehe:.4f}")
   print(f"Zero Effect PEHE: {zero_pehe:.4f}")

Uncertainty Evaluation
----------------------

Calibration Analysis
~~~~~~~~~~~~~~~~~~~~

Evaluate the calibration of uncertainty estimates:

.. code-block:: python

   import numpy as np
   from causalfm.models import StandardCATEModel
   
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
   
   # Get predictions with uncertainty
   result = model.estimate_cate(x_train, a_train, y_train, x_test)
   pred_cate = result['cate'].cpu().numpy()
   
   # GMM parameters
   pi = result['gmm_pi'].cpu().numpy()
   mu = result['gmm_mu'].cpu().numpy()
   sigma = result['gmm_sigma'].cpu().numpy()
   
   # Compute predictive variance
   variance = (pi * (sigma**2 + mu**2)).sum(axis=-1) - pred_cate**2
   std_dev = np.sqrt(variance)
   
   # Compute standardized errors
   errors = pred_cate - true_ite
   standardized_errors = errors / std_dev
   
   # Check if standardized errors follow N(0,1)
   print(f"Mean of standardized errors: {standardized_errors.mean():.4f}")
   print(f"Std of standardized errors: {standardized_errors.std():.4f}")
   
   # Calibration plot
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(10, 5))
   
   # Plot 1: Predicted std vs absolute error
   plt.subplot(1, 2, 1)
   plt.scatter(std_dev, np.abs(errors), alpha=0.5)
   plt.xlabel('Predicted Std Dev')
   plt.ylabel('Absolute Error')
   plt.title('Uncertainty Calibration')
   
   # Plot 2: QQ plot of standardized errors
   plt.subplot(1, 2, 2)
   from scipy import stats
   stats.probplot(standardized_errors, dist="norm", plot=plt)
   plt.title('Q-Q Plot')
   
   plt.tight_layout()
   plt.savefig('calibration.png')

Coverage Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # Compute confidence intervals
   n_samples = 10000
   n_test = len(pred_cate)
   
   samples = np.zeros((n_test, n_samples))
   for i in range(n_test):
       # Sample component indices
       components = np.random.choice(
           len(pi[i]), 
           size=n_samples, 
           p=pi[i]
       )
       
       # Sample from selected components
       for k in range(len(pi[i])):
           mask = (components == k)
           n_k = mask.sum()
           if n_k > 0:
               samples[i, mask] = np.random.normal(
                   mu[i, k],
                   sigma[i, k],
                   n_k
               )
   
   # Compute 95% confidence intervals
   ci_lower = np.percentile(samples, 2.5, axis=1)
   ci_upper = np.percentile(samples, 97.5, axis=1)
   
   # Check coverage
   coverage = np.mean((true_ite >= ci_lower) & (true_ite <= ci_upper))
   print(f"95% CI Coverage: {coverage:.2%}")
   
   # Expected: ~95% for well-calibrated model

Visualization
-------------

Plotting Predictions
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   
   # Scatter plot: predicted vs true
   plt.figure(figsize=(8, 6))
   plt.scatter(true_ite, pred_cate, alpha=0.6)
   
   # Perfect prediction line
   min_val = min(true_ite.min(), pred_cate.min())
   max_val = max(true_ite.max(), pred_cate.max())
   plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
   
   plt.xlabel('True ITE')
   plt.ylabel('Predicted CATE')
   plt.title(f'CATE Predictions (PEHE: {pehe:.4f})')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.savefig('predictions.png')

Error Distribution
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   errors = pred_cate - true_ite
   
   plt.figure(figsize=(10, 4))
   
   # Histogram
   plt.subplot(1, 2, 1)
   plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
   plt.xlabel('Prediction Error')
   plt.ylabel('Frequency')
   plt.title('Error Distribution')
   plt.axvline(0, color='r', linestyle='--', label='Zero Error')
   plt.legend()
   
   # Box plot
   plt.subplot(1, 2, 2)
   plt.boxplot(errors)
   plt.ylabel('Prediction Error')
   plt.title('Error Box Plot')
   plt.axhline(0, color='r', linestyle='--')
   
   plt.tight_layout()
   plt.savefig('error_distribution.png')

Uncertainty Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sort by predicted CATE
   sorted_idx = np.argsort(pred_cate)
   
   plt.figure(figsize=(12, 6))
   x = np.arange(len(sorted_idx))
   
   # Plot predictions with uncertainty bands
   plt.plot(x, pred_cate[sorted_idx], label='Predicted CATE', color='blue')
   plt.fill_between(x, 
                     ci_lower[sorted_idx], 
                     ci_upper[sorted_idx], 
                     alpha=0.3, 
                     label='95% CI')
   plt.scatter(x, true_ite[sorted_idx], s=10, alpha=0.5, 
               color='red', label='True ITE')
   
   plt.xlabel('Sample (sorted by prediction)')
   plt.ylabel('Treatment Effect')
   plt.title('CATE Predictions with Uncertainty')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.savefig('uncertainty.png')

Real-World Evaluation
---------------------

Jobs Dataset Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.models import StandardCATEModel
   from causalfm.evaluation import compute_pehe
   import pandas as pd
   import torch
   
   # Load model
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
   
   # Load Jobs dataset (real-world data)
   df = pd.read_csv("DATA_standard/jobs_data/jobs_data.csv")
   
   # Prepare data
   feature_cols = ['age', 'education', 'black', 'hispanic', 'married', 
                   'nodegree', 're74', 're75']
   X = torch.FloatTensor(df[feature_cols].values)
   A = torch.FloatTensor(df['treat'].values).unsqueeze(1)
   Y = torch.FloatTensor(df['re78'].values).unsqueeze(1)
   
   # Split
   n_train = int(0.8 * len(X))
   x_train, x_test = X[:n_train], X[n_train:]
   a_train, y_train = A[:n_train], Y[:n_train]
   
   # Estimate treatment effects
   result = model.estimate_cate(x_train, a_train, y_train, x_test)
   cate = result['cate'].cpu().numpy()
   
   # Analyze results
   print(f"Estimated ATE: {cate.mean():.2f}")
   print(f"CATE range: [{cate.min():.2f}, {cate.max():.2f}]")
   print(f"Percentage with positive effect: {(cate > 0).mean():.1%}")

Best Practices
--------------

Evaluation Guidelines
~~~~~~~~~~~~~~~~~~~~~

1. **Use multiple test datasets** (10+ recommended) to get robust estimates
2. **Report standard deviations** along with mean metrics
3. **Check uncertainty calibration** if using GMM predictions
4. **Compare with baselines** to demonstrate improvement
5. **Visualize predictions** to identify systematic biases

Example Complete Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.models import StandardCATEModel
   from causalfm.evaluation import compute_pehe, compute_ate_error, compute_rmse
   import pandas as pd
   import numpy as np
   from pathlib import Path
   
   # Load model
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
   
   # Evaluate on all test datasets
   test_dir = Path("data/test/")
   results = []
   
   for file in sorted(test_dir.glob("test_*.csv")):
       # Load and prepare data
       df = pd.read_csv(file)
       # ... (data preparation) ...
       
       # Predict
       result = model.estimate_cate(x_train, a_train, y_train, x_test)
       pred_cate = result['cate'].cpu().numpy()
       
       # Compute metrics
       results.append({
           'dataset': file.name,
           'pehe': compute_pehe(pred_cate, true_ite),
           'ate_error': compute_ate_error(pred_cate, true_ite),
           'rmse': compute_rmse(pred_cate, true_ite),
           'n_test': len(true_ite)
       })
   
   # Aggregate results
   results_df = pd.DataFrame(results)
   
   print("=" * 60)
   print("EVALUATION RESULTS")
   print("=" * 60)
   print(f"\nNumber of test datasets: {len(results_df)}")
   print(f"\nMetric Summary:")
   print(results_df[['pehe', 'ate_error', 'rmse']].describe())
   
   print(f"\nFinal Results:")
   print(f"  PEHE: {results_df['pehe'].mean():.4f} ± {results_df['pehe'].std():.4f}")
   print(f"  ATE Error: {results_df['ate_error'].mean():.4f} ± {results_df['ate_error'].std():.4f}")
   print(f"  RMSE: {results_df['rmse'].mean():.4f} ± {results_df['rmse'].std():.4f}")
   
   # Save results
   results_df.to_csv("evaluation_results.csv", index=False)

API Reference
-------------

For complete API documentation, see:

* :func:`causalfm.evaluation.compute_pehe`
* :func:`causalfm.evaluation.compute_ate_error`
* :func:`causalfm.evaluation.compute_mse`
* :func:`causalfm.evaluation.compute_rmse`

