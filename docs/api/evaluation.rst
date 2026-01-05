Evaluation API
==============

This page documents the evaluation APIs in CausalFM.

.. module:: causalfm.evaluation

Metrics
-------

compute_pehe
~~~~~~~~~~~~

.. autofunction:: causalfm.evaluation.metrics.compute_pehe

   Compute Precision in Estimation of Heterogeneous Effects (PEHE).
   
   .. math::
   
      \text{PEHE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (\tau_i - \hat{\tau}_i)^2}
   
   :param predictions: Predicted CATE values
   :type predictions: np.ndarray or torch.Tensor
   :param ground_truth: True ITE values
   :type ground_truth: np.ndarray or torch.Tensor
   :return: PEHE score
   :rtype: float
   
   Example:
   
   .. code-block:: python
   
      from causalfm.evaluation import compute_pehe
      import numpy as np
      
      true_ite = np.array([1.5, 2.3, 0.8, -0.5, 1.2])
      pred_cate = np.array([1.4, 2.1, 0.9, -0.3, 1.1])
      
      pehe = compute_pehe(pred_cate, true_ite)
      print(f"PEHE: {pehe:.4f}")

compute_ate_error
~~~~~~~~~~~~~~~~~

.. autofunction:: causalfm.evaluation.metrics.compute_ate_error

   Compute Average Treatment Effect error.
   
   .. math::
   
      \text{ATE Error} = \left|\frac{1}{n}\sum_{i=1}^n \tau_i - \frac{1}{n}\sum_{i=1}^n \hat{\tau}_i\right|
   
   :param predictions: Predicted CATE values
   :type predictions: np.ndarray or torch.Tensor
   :param ground_truth: True ITE values
   :type ground_truth: np.ndarray or torch.Tensor
   :return: ATE error
   :rtype: float
   
   Example:
   
   .. code-block:: python
   
      from causalfm.evaluation import compute_ate_error
      
      ate_error = compute_ate_error(pred_cate, true_ite)
      print(f"ATE Error: {ate_error:.4f}")

compute_mse
~~~~~~~~~~~

.. autofunction:: causalfm.evaluation.metrics.compute_mse

   Compute Mean Squared Error.
   
   .. math::
   
      \text{MSE} = \frac{1}{n}\sum_{i=1}^n (\tau_i - \hat{\tau}_i)^2
   
   :param predictions: Predicted values
   :type predictions: np.ndarray or torch.Tensor
   :param ground_truth: True values
   :type ground_truth: np.ndarray or torch.Tensor
   :return: MSE
   :rtype: float
   
   Example:
   
   .. code-block:: python
   
      from causalfm.evaluation import compute_mse
      
      mse = compute_mse(pred_cate, true_ite)
      print(f"MSE: {mse:.4f}")

compute_rmse
~~~~~~~~~~~~

.. autofunction:: causalfm.evaluation.metrics.compute_rmse

   Compute Root Mean Squared Error.
   
   .. math::
   
      \text{RMSE} = \sqrt{\text{MSE}}
   
   :param predictions: Predicted values
   :type predictions: np.ndarray or torch.Tensor
   :param ground_truth: True values
   :type ground_truth: np.ndarray or torch.Tensor
   :return: RMSE
   :rtype: float
   
   Example:
   
   .. code-block:: python
   
      from causalfm.evaluation import compute_rmse
      
      rmse = compute_rmse(pred_cate, true_ite)
      print(f"RMSE: {rmse:.4f}")

Basic Usage
-----------

Computing Multiple Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.evaluation import (
       compute_pehe,
       compute_ate_error,
       compute_mse,
       compute_rmse
   )
   import numpy as np
   
   # Your predictions and ground truth
   predictions = np.random.randn(100)
   ground_truth = np.random.randn(100)
   
   # Compute all metrics
   pehe = compute_pehe(predictions, ground_truth)
   ate_error = compute_ate_error(predictions, ground_truth)
   mse = compute_mse(predictions, ground_truth)
   rmse = compute_rmse(predictions, ground_truth)
   
   print(f"PEHE: {pehe:.4f}")
   print(f"ATE Error: {ate_error:.4f}")
   print(f"MSE: {mse:.4f}")
   print(f"RMSE: {rmse:.4f}")

With PyTorch Tensors
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from causalfm.evaluation import compute_pehe
   
   # Works with torch tensors
   pred = torch.randn(100)
   true = torch.randn(100)
   
   pehe = compute_pehe(pred, true)
   print(f"PEHE: {pehe:.4f}")

Model Evaluation
----------------

Evaluating a Model on a Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
   
   # Split
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
   
   print(f"PEHE: {pehe:.4f}")
   print(f"ATE Error: {ate_error:.4f}")

Evaluating Multiple Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   import pandas as pd
   
   # Evaluate on all test datasets
   test_dir = Path("data/test/")
   results = []
   
   for file in test_dir.glob("test_*.csv"):
       df = pd.read_csv(file)
       
       # ... prepare data and predict ...
       
       pehe = compute_pehe(pred_cate, ite_test)
       ate_error = compute_ate_error(pred_cate, ite_test)
       
       results.append({
           'dataset': file.name,
           'pehe': pehe,
           'ate_error': ate_error
       })
   
   # Aggregate results
   results_df = pd.DataFrame(results)
   
   print(results_df)
   print(f"\nAverage PEHE: {results_df['pehe'].mean():.4f} ± {results_df['pehe'].std():.4f}")

Advanced Evaluation
-------------------

Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate calibration of uncertainty estimates:

.. code-block:: python

   import numpy as np
   from causalfm.models import StandardCATEModel
   
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
   
   # Get predictions with uncertainty
   result = model.estimate_cate(x_train, a_train, y_train, x_test)
   
   pred_cate = result['cate'].cpu().numpy()
   pi = result['gmm_pi'].cpu().numpy()
   mu = result['gmm_mu'].cpu().numpy()
   sigma = result['gmm_sigma'].cpu().numpy()
   
   # Compute predictive variance
   variance = (pi * (sigma**2 + mu**2)).sum(axis=-1) - pred_cate**2
   std_dev = np.sqrt(variance)
   
   # Check calibration
   errors = pred_cate - true_ite
   standardized_errors = errors / std_dev
   
   print(f"Mean standardized error: {standardized_errors.mean():.4f}")
   print(f"Std standardized error: {standardized_errors.std():.4f}")

Coverage Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sample from GMM for confidence intervals
   n_samples = 10000
   samples = np.zeros((len(pred_cate), n_samples))
   
   for i in range(len(pred_cate)):
       components = np.random.choice(len(pi[i]), size=n_samples, p=pi[i])
       for k in range(len(pi[i])):
           mask = (components == k)
           n_k = mask.sum()
           if n_k > 0:
               samples[i, mask] = np.random.normal(mu[i, k], sigma[i, k], n_k)
   
   # Compute 95% CI
   ci_lower = np.percentile(samples, 2.5, axis=1)
   ci_upper = np.percentile(samples, 97.5, axis=1)
   
   # Check coverage
   coverage = np.mean((true_ite >= ci_lower) & (true_ite <= ci_upper))
   print(f"95% CI Coverage: {coverage:.2%}")

Visualization
-------------

Plotting Results
~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Predicted vs True
   plt.figure(figsize=(8, 6))
   plt.scatter(true_ite, pred_cate, alpha=0.6)
   
   min_val = min(true_ite.min(), pred_cate.min())
   max_val = max(true_ite.max(), pred_cate.max())
   plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
   
   plt.xlabel('True ITE')
   plt.ylabel('Predicted CATE')
   plt.title(f'PEHE: {pehe:.4f}')
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
   plt.axvline(0, color='r', linestyle='--')
   
   # Box plot
   plt.subplot(1, 2, 2)
   plt.boxplot(errors)
   plt.ylabel('Prediction Error')
   plt.title('Error Box Plot')
   plt.axhline(0, color='r', linestyle='--')
   
   plt.tight_layout()
   plt.savefig('errors.png')

Comparison
----------

Comparing Multiple Models
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.models import StandardCATEModel, IVModel
   from causalfm.evaluation import compute_pehe
   
   # Load models
   models = {
       'Standard': StandardCATEModel.from_pretrained("checkpoints/standard.pth"),
       'IV': IVModel.from_pretrained("checkpoints/iv.pth")
   }
   
   # Evaluate each
   results = {}
   for name, model in models.items():
       result = model.estimate_cate(x_train, a_train, y_train, x_test)
       pred = result['cate'].cpu().numpy()
       pehe = compute_pehe(pred, true_ite)
       results[name] = pehe
   
   print("Model Comparison:")
   for name, pehe in results.items():
       print(f"  {name}: PEHE={pehe:.4f}")

Baseline Comparison
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # CausalFM
   causalfm_pehe = compute_pehe(pred_cate, true_ite)
   
   # Baseline 1: ATE for all
   ate_pred = np.full_like(true_ite, true_ite.mean())
   ate_pehe = compute_pehe(ate_pred, true_ite)
   
   # Baseline 2: Zero effect
   zero_pred = np.zeros_like(true_ite)
   zero_pehe = compute_pehe(zero_pred, true_ite)
   
   print("Comparison:")
   print(f"  CausalFM: {causalfm_pehe:.4f}")
   print(f"  ATE Baseline: {ate_pehe:.4f}")
   print(f"  Zero Baseline: {zero_pehe:.4f}")

Best Practices
--------------

Evaluation Guidelines
~~~~~~~~~~~~~~~~~~~~~

1. **Multiple test datasets**: Use 10+ test datasets for robust estimates
2. **Report statistics**: Include mean ± std deviation
3. **Check calibration**: Verify uncertainty estimates are well-calibrated
4. **Compare baselines**: Always compare with simple baselines
5. **Visualize**: Create plots to identify systematic biases

Complete Evaluation Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.models import StandardCATEModel
   from causalfm.evaluation import compute_pehe, compute_ate_error, compute_rmse
   from pathlib import Path
   import pandas as pd
   
   # Load model
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
   
   # Evaluate all test datasets
   results = []
   for file in Path("data/test/").glob("test_*.csv"):
       # ... load and prepare data ...
       
       # Predict
       result = model.estimate_cate(x_train, a_train, y_train, x_test)
       pred = result['cate'].cpu().numpy()
       
       # Evaluate
       results.append({
           'dataset': file.name,
           'pehe': compute_pehe(pred, true_ite),
           'ate_error': compute_ate_error(pred, true_ite),
           'rmse': compute_rmse(pred, true_ite)
       })
   
   # Summary
   df = pd.DataFrame(results)
   print("\nFinal Results:")
   print(f"PEHE: {df['pehe'].mean():.4f} ± {df['pehe'].std():.4f}")
   print(f"ATE Error: {df['ate_error'].mean():.4f} ± {df['ate_error'].std():.4f}")
   
   # Save
   df.to_csv("evaluation_results.csv", index=False)

See Also
--------

* :doc:`../user_guide/evaluation` - Detailed evaluation guide
* :doc:`models` - Model API reference
* :doc:`../examples/standard_cate` - Complete evaluation example

