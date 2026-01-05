Quick Start
===========

This guide will help you get started with CausalFM Toolkit in minutes.

5-Minute Introduction
---------------------

Here's a complete workflow from data generation to model evaluation:

.. code-block:: python

   from causalfm.data import StandardCATEGenerator
   from causalfm.models import StandardCATEModel
   from causalfm.training import StandardCATETrainer, TrainingConfig
   from causalfm.evaluation import compute_pehe
   import torch
   
   # 1. Generate Training Data
   generator = StandardCATEGenerator(num_samples=1024, num_features=10, seed=42)
   generator.generate_multiple(num_datasets=100, output_dir="data/train/")
   
   # 2. Train Model
   config = TrainingConfig(
       data_path="data/train/*.csv",
       epochs=50,
       batch_size=16,
       save_dir="checkpoints/"
   )
   trainer = StandardCATETrainer(config)
   trainer.train()
   
   # 3. Load and Evaluate
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
   
   # Prepare test data
   x_train = torch.randn(800, 10)
   a_train = torch.randint(0, 2, (800, 1)).float()
   y_train = torch.randn(800, 1)
   x_test = torch.randn(200, 10)
   
   # Estimate CATE
   result = model.estimate_cate(x_train, a_train, y_train, x_test)
   print(f"CATE estimates: {result['cate'][:5]}")

Core Workflows
--------------

1. Data Generation
~~~~~~~~~~~~~~~~~~

**Standard CATE Data**

.. code-block:: python

   from causalfm.data import StandardCATEGenerator
   
   # Single dataset
   generator = StandardCATEGenerator(
       num_samples=1024,
       num_features=10,
       seed=42
   )
   df = generator.generate()
   
   # Save to file
   df.to_csv("my_dataset.csv", index=False)
   
   # Multiple datasets for training
   generator.generate_multiple(
       num_datasets=100,
       output_dir="data/standard/",
       filename_prefix="train_data"
   )

**Instrumental Variables Data**

.. code-block:: python

   from causalfm.data import IVDataGenerator
   
   generator = IVDataGenerator(
       num_samples=1024,
       num_features=10,
       instrument_type='binary',  # or 'continuous'
       seed=42
   )
   df = generator.generate()
   
   # Dataset includes: X (covariates), U (confounders), 
   # Z (instrument), A (treatment), Y (outcome)
   print(df.columns)

**Front-door Data**

.. code-block:: python

   from causalfm.data import FrontdoorDataGenerator
   
   generator = FrontdoorDataGenerator(
       num_samples=1024,
       num_features=10,
       num_confounders=5,  # or None for random
       seed=42
   )
   df = generator.generate()
   
   # Dataset includes: X, U, A, M (mediator), Y
   print(f"Mediator column: {df['mediator'][:5]}")

2. Model Training
~~~~~~~~~~~~~~~~~

**Basic Training**

.. code-block:: python

   from causalfm.training import StandardCATETrainer, TrainingConfig
   
   # Method 1: Using TrainingConfig
   config = TrainingConfig(
       data_path="data/standard/*.csv",
       epochs=100,
       batch_size=16,
       learning_rate=0.001,
       save_dir="checkpoints/standard",
       device='cuda'  # or 'cpu'
   )
   trainer = StandardCATETrainer(config)
   trainer.train()
   
   # Method 2: Simplified interface
   trainer = StandardCATETrainer.from_args(
       data_path="data/standard/*.csv",
       epochs=100,
       save_dir="checkpoints/standard"
   )
   trainer.train()

**Training Configuration Options**

.. code-block:: python

   config = TrainingConfig(
       # Data settings
       data_path="data/*.csv",
       val_split=0.2,              # Validation split ratio
       batch_size=16,
       num_workers=4,              # Data loading workers (0 for no multiprocessing)
       
       # Optimizer settings
       learning_rate=0.001,
       weight_decay=1e-5,
       
       # Training settings
       epochs=150,
       early_stop_patience=50,     # Stop if no improvement
       clip_grad=1.0,              # Gradient clipping
       
       # Model settings
       use_gmm_head=True,          # Use GMM for uncertainty
       gmm_n_components=5,
       
       # Logging and checkpointing
       save_dir="checkpoints/",
       log_dir="logs/",
       save_freq=10,               # Save every N epochs
       
       # Device
       device='auto',              # 'auto', 'cuda', or 'cpu'
       gpu_id=0,
       
       # Reproducibility
       seed=42
   )

**Training Different Settings**

.. code-block:: python

   # Instrumental Variables
   from causalfm.training import IVTrainer
   
   trainer = IVTrainer.from_args(
       data_path="data/iv/*.csv",
       epochs=100,
       save_dir="checkpoints/iv"
   )
   trainer.train()
   
   # Front-door Adjustment
   from causalfm.training import FrontdoorTrainer
   
   trainer = FrontdoorTrainer.from_args(
       data_path="data/frontdoor/*.csv",
       epochs=100,
       save_dir="checkpoints/frontdoor"
   )
   trainer.train()

3. Model Inference
~~~~~~~~~~~~~~~~~~

**Loading Pretrained Models**

.. code-block:: python

   from causalfm.models import StandardCATEModel
   
   # Load from checkpoint
   model = StandardCATEModel.from_pretrained(
       "checkpoints/best_model.pth",
       device='cuda'
   )
   
   # Set to evaluation mode
   model.eval_mode()

**Estimating CATE**

.. code-block:: python

   import torch
   
   # Prepare data (ensure correct shapes!)
   x_train = torch.randn(800, 10)       # (n_samples, n_features)
   a_train = torch.randint(0, 2, (800, 1)).float()  # (n_samples, 1)
   y_train = torch.randn(800, 1)        # (n_samples, 1)
   x_test = torch.randn(200, 10)        # (n_test, n_features)
   
   # Estimate CATE
   result = model.estimate_cate(x_train, a_train, y_train, x_test)
   
   # Extract results
   cate = result['cate']                # Point estimates
   gmm_pi = result['gmm_pi']            # Mixture weights
   gmm_mu = result['gmm_mu']            # Mixture means
   gmm_sigma = result['gmm_sigma']      # Mixture std devs
   
   print(f"CATE shape: {cate.shape}")   # (200,)
   print(f"Mean CATE: {cate.mean():.4f}")

**IV and Front-door Models**

.. code-block:: python

   from causalfm.models import IVModel, FrontdoorModel
   
   # IV Model (requires instrument)
   iv_model = IVModel.from_pretrained("checkpoints/iv_model.pth")
   result = iv_model.estimate_cate(
       x_train, z_train, a_train, y_train, x_test
   )
   
   # Front-door Model (requires mediator)
   fd_model = FrontdoorModel.from_pretrained("checkpoints/fd_model.pth")
   result = fd_model.estimate_cate(
       x_train, m_train, a_train, y_train, x_test
   )

4. Evaluation
~~~~~~~~~~~~~

**Computing Metrics**

.. code-block:: python

   from causalfm.evaluation import (
       compute_pehe,
       compute_ate_error,
       compute_mse,
       compute_rmse
   )
   
   # Assume we have predictions and ground truth
   cate_pred = model.estimate_cate(x_train, a_train, y_train, x_test)['cate']
   true_ite = ite_test  # Ground truth ITE
   
   # Compute metrics
   pehe = compute_pehe(cate_pred, true_ite)
   ate_error = compute_ate_error(cate_pred, true_ite)
   mse = compute_mse(cate_pred, true_ite)
   rmse = compute_rmse(cate_pred, true_ite)
   
   print(f"PEHE: {pehe:.4f}")
   print(f"ATE Error: {ate_error:.4f}")
   print(f"RMSE: {rmse:.4f}")

**Evaluating on a Dataset**

.. code-block:: python

   from causalfm.evaluation import evaluate_model
   
   result = evaluate_model(
       model,
       data_path="data/test/test_dataset_1.csv",
       train_ratio=0.8  # Use 80% for context, 20% for evaluation
   )
   
   print(result)
   # EvaluationResult(dataset=test_dataset_1, 
   #                  PEHE=0.4523, ATE_Error=0.0234)

**Evaluating Multiple Datasets**

.. code-block:: python

   from causalfm.evaluation.metrics import evaluate_multiple_datasets
   import pandas as pd
   
   results_df = evaluate_multiple_datasets(
       model,
       data_dir="data/test/",
       file_pattern="test_*.csv",
       train_ratio=0.8
   )
   
   # Display results
   print(results_df)
   
   # Summary statistics
   print(f"Average PEHE: {results_df['pehe'].mean():.4f}")
   print(f"Std PEHE: {results_df['pehe'].std():.4f}")

Common Patterns
---------------

Pattern 1: End-to-End Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   if __name__ == '__main__':
       # 1. Generate data
       from causalfm.data import StandardCATEGenerator
       
       gen = StandardCATEGenerator(num_samples=1024, num_features=10)
       gen.generate_multiple(100, "data/train/")
       gen.generate_multiple(10, "data/test/")
       
       # 2. Train
       from causalfm.training import StandardCATETrainer
       
       trainer = StandardCATETrainer.from_args(
           data_path="data/train/*.csv",
           epochs=100,
           batch_size=16,
           num_workers=0,  # Important for multiprocessing
           save_dir="checkpoints/"
       )
       trainer.train()
       
       # 3. Evaluate
       from causalfm.models import StandardCATEModel
       from causalfm.evaluation.metrics import evaluate_multiple_datasets
       
       model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
       results = evaluate_multiple_datasets(model, "data/test/")
       
       print(f"Final PEHE: {results['pehe'].mean():.4f}")

Pattern 2: Load and Predict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import pandas as pd
   from causalfm.models import StandardCATEModel
   
   # Load model
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
   
   # Load your data
   df = pd.read_csv("my_data.csv")
   
   # Extract features
   x_cols = [col for col in df.columns if col.startswith('x')]
   X = torch.FloatTensor(df[x_cols].values)
   A = torch.FloatTensor(df['treatment'].values).unsqueeze(1)
   Y = torch.FloatTensor(df['outcome'].values).unsqueeze(1)
   
   # Split train/test
   n_train = int(0.8 * len(X))
   x_train, x_test = X[:n_train], X[n_train:]
   a_train = A[:n_train]
   y_train = Y[:n_train]
   
   # Estimate
   result = model.estimate_cate(x_train, a_train, y_train, x_test)
   
   # Save predictions
   df_test = df[n_train:].copy()
   df_test['cate_pred'] = result['cate'].cpu().numpy()
   df_test.to_csv("predictions.csv", index=False)

Next Steps
----------

* Check out :doc:`user_guide/data_generation` for advanced data generation
* See :doc:`user_guide/training` for detailed training options
* Explore :doc:`examples/standard_cate` for complete examples
* Read the :doc:`api/models` for model API reference

Troubleshooting
---------------

**Multiprocessing Errors**: Use ``if __name__ == '__main__':`` wrapper or set ``num_workers=0``

**CUDA Out of Memory**: Reduce ``batch_size`` or use ``device='cpu'``

**Shape Mismatch**: Ensure treatments and outcomes have shape ``(n, 1)`` not ``(n,)``

For more help, see the :doc:`installation` guide's troubleshooting section.
