Tutorial 4: Model Evaluation
============================

Learn how to evaluate CausalFM models effectively.

Coming Soon
-----------

This tutorial is under development. For now, see:

* :doc:`../user_guide/evaluation` - Complete evaluation guide
* :doc:`tutorial_01_basics` - Basic concepts
* :doc:`../examples/standard_cate` - Complete example

Quick Reference
---------------

Basic Metrics
~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.evaluation import (
       compute_pehe,
       compute_ate_error,
       compute_rmse
   )
   
   # Compute metrics
   pehe = compute_pehe(predictions, ground_truth)
   ate_error = compute_ate_error(predictions, ground_truth)
   rmse = compute_rmse(predictions, ground_truth)
   
   print(f"PEHE: {pehe:.4f}")
   print(f"ATE Error: {ate_error:.4f}")
   print(f"RMSE: {rmse:.4f}")

Evaluating a Model
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.models import StandardCATEModel
   import pandas as pd
   import torch
   
   # Load model
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
   
   # Load test data
   df = pd.read_csv("data/test/test_dataset_1.csv")
   
   # Prepare data
   x_cols = [c for c in df.columns if c.startswith('x')]
   X = torch.FloatTensor(df[x_cols].values)
   A = torch.FloatTensor(df['treatment'].values).unsqueeze(1)
   Y = torch.FloatTensor(df['outcome'].values).unsqueeze(1)
   true_ite = df['ite'].values
   
   # Split and predict
   n_train = int(0.8 * len(X))
   result = model.estimate_cate(
       X[:n_train], A[:n_train], Y[:n_train], X[n_train:]
   )
   
   # Evaluate
   pehe = compute_pehe(result['cate'].cpu().numpy(), true_ite[n_train:])
   print(f"PEHE: {pehe:.4f}")

Congratulations!
----------------

You've completed the CausalFM tutorials! 

Next Steps
----------

* Explore :doc:`../examples/standard_cate` for complete working code
* Read the :doc:`../user_guide/models` for advanced model usage
* Check out the API reference: :doc:`../api/index`
* Try CausalFM on your own causal inference problems!

