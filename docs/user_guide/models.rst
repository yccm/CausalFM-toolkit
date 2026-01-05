Models
======

CausalFM provides three main model classes for different causal inference settings, 
all built on the TabPFN foundation model architecture.

Model Architecture
------------------

All CausalFM models share a common architecture:

* **Transformer-based** encoder with per-feature attention
* **GMM prediction head** for uncertainty quantification
* **Context-based learning** using training samples as context
* **In-context adaptation** without gradient updates

Key Features
~~~~~~~~~~~~

✨ **Foundation Model Approach**
   Models are pre-trained on diverse synthetic datasets and can adapt to new 
   datasets in-context without fine-tuning.

🎯 **Gaussian Mixture Model Head**
   Instead of point estimates, models output a mixture of Gaussians, providing:
   
   * Point estimates (mixture mean)
   * Uncertainty quantification (mixture variance)
   * Full predictive distribution

📊 **Transformer Architecture**
   * Per-feature encoding
   * Multi-head attention mechanisms
   * Layer normalization and residual connections

Standard CATE Model
-------------------

The ``StandardCATEModel`` is designed for standard CATE estimation without 
unobserved confounding.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from causalfm.models import StandardCATEModel
   import torch
   
   # Create new model
   model = StandardCATEModel(
       use_gmm_head=True,
       gmm_n_components=5,
       device='cuda'
   )
   
   # Or load pretrained
   model = StandardCATEModel.from_pretrained(
       "checkpoints/best_model.pth",
       device='cuda'
   )

Model Parameters
~~~~~~~~~~~~~~~~

.. code-block:: python

   model = StandardCATEModel(
       use_gmm_head=True,        # Use GMM for uncertainty
       gmm_n_components=5,       # Number of mixture components
       gmm_min_sigma=1e-3,       # Minimum std dev
       gmm_pi_temp=1.0,          # Temperature for mixing weights
       device='cuda'             # 'cuda', 'cpu', or None (auto)
   )

Estimating CATE
~~~~~~~~~~~~~~~

.. code-block:: python

   # Prepare data
   x_train = torch.randn(800, 10)       # Covariates
   a_train = torch.randint(0, 2, (800, 1)).float()  # Treatments
   y_train = torch.randn(800, 1)        # Outcomes
   x_test = torch.randn(200, 10)        # Test covariates
   
   # Estimate CATE
   result = model.estimate_cate(x_train, a_train, y_train, x_test)
   
   # Extract results
   cate = result['cate']                # Point estimates (200,)
   
   # Uncertainty quantification
   pi = result['gmm_pi']                # Mixture weights (200, 5)
   mu = result['gmm_mu']                # Component means (200, 5)
   sigma = result['gmm_sigma']          # Component std devs (200, 5)
   
   # Compute confidence intervals
   import numpy as np
   lower = np.percentile(mu.cpu().numpy(), 2.5, axis=1)
   upper = np.percentile(mu.cpu().numpy(), 97.5, axis=1)

Input Format
~~~~~~~~~~~~

**Important**: Ensure correct tensor shapes:

.. code-block:: python

   # ✅ Correct shapes
   x_train: (n_train, n_features)      # e.g., (800, 10)
   a_train: (n_train, 1)               # e.g., (800, 1) - NOT (800,)
   y_train: (n_train, 1)               # e.g., (800, 1) - NOT (800,)
   x_test:  (n_test, n_features)       # e.g., (200, 10)
   
   # ❌ Wrong - will cause errors
   a_train_wrong: (800,)    # Missing dimension
   y_train_wrong: (800,)    # Missing dimension

Model Methods
~~~~~~~~~~~~~

.. code-block:: python

   # Set to evaluation mode
   model.eval_mode()
   
   # Set to training mode
   model.train_mode()
   
   # Save model
   model.save("my_model.pth")
   
   # Get model parameters
   params = model.parameters
   
   # Direct forward pass (for training)
   output = model.forward(x, a, y, single_eval_pos)

Instrumental Variables Model
-----------------------------

The ``IVModel`` handles settings with unobserved confounding using 
instrumental variables.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from causalfm.models import IVModel
   
   # Load pretrained IV model
   model = IVModel.from_pretrained(
       "checkpoints/iv_binary_model.pth",
       device='cuda'
   )

Estimating CATE with Instruments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prepare data (including instruments)
   x_train = torch.randn(800, 10)       # Covariates
   z_train = torch.randint(0, 2, (800, 1)).float()  # Binary instrument
   a_train = torch.randint(0, 2, (800, 1)).float()  # Treatment
   y_train = torch.randn(800, 1)        # Outcome
   x_test = torch.randn(200, 10)        # Test covariates
   
   # Estimate CATE using IV
   result = model.estimate_cate(
       x_train, z_train, a_train, y_train, x_test
   )
   
   cate = result['cate']

Input Requirements
~~~~~~~~~~~~~~~~~~

The IV model requires an additional instrument input:

.. code-block:: python

   # All inputs required:
   x_train: (n_train, n_features)      # Observed covariates
   z_train: (n_train, 1)               # Instrument (binary or continuous)
   a_train: (n_train, 1)               # Treatment
   y_train: (n_train, 1)               # Outcome
   x_test:  (n_test, n_features)       # Test covariates

Binary vs Continuous Instruments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Binary instrument (e.g., randomized encouragement)
   z_binary = torch.randint(0, 2, (800, 1)).float()
   
   # Continuous instrument (e.g., distance, price)
   z_continuous = torch.randn(800, 1)
   
   # Model handles both types
   result = model.estimate_cate(x_train, z_binary, a_train, y_train, x_test)
   result = model.estimate_cate(x_train, z_continuous, a_train, y_train, x_test)

Front-door Model
----------------

The ``FrontdoorModel`` uses mediators to identify causal effects when 
there are unobserved confounders.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from causalfm.models import FrontdoorModel
   
   # Load pretrained front-door model
   model = FrontdoorModel.from_pretrained(
       "checkpoints/frontdoor_model.pth",
       device='cuda'
   )

Estimating CATE with Mediators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prepare data (including mediators)
   x_train = torch.randn(800, 10)       # Covariates
   m_train = torch.randn(800, 1)        # Mediator values
   a_train = torch.randint(0, 2, (800, 1)).float()  # Treatment
   y_train = torch.randn(800, 1)        # Outcome
   x_test = torch.randn(200, 10)        # Test covariates
   
   # Estimate CATE using front-door adjustment
   result = model.estimate_cate(
       x_train, m_train, a_train, y_train, x_test
   )
   
   cate = result['cate']

Input Requirements
~~~~~~~~~~~~~~~~~~

The front-door model requires mediator observations:

.. code-block:: python

   # All inputs required:
   x_train: (n_train, n_features)      # Observed covariates
   m_train: (n_train, 1)               # Mediator values
   a_train: (n_train, 1)               # Treatment
   y_train: (n_train, 1)               # Outcome
   x_test:  (n_test, n_features)       # Test covariates

Model Loading and Saving
------------------------

Loading Pretrained Models
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.models import StandardCATEModel, IVModel, FrontdoorModel
   
   # Standard CATE
   standard_model = StandardCATEModel.from_pretrained(
       "checkpoints/standard/best_model.pth",
       device='cuda'
   )
   
   # IV
   iv_model = IVModel.from_pretrained(
       "checkpoints/iv/best_model.pth",
       device='cpu'  # Use CPU
   )
   
   # Front-door
   fd_model = FrontdoorModel.from_pretrained(
       "checkpoints/frontdoor/best_model.pth"
       # device='auto' by default
   )

Saving Models
~~~~~~~~~~~~~

.. code-block:: python

   # Save model state
   model.save("my_custom_model.pth")
   
   # The checkpoint includes:
   # - model_state_dict: Model parameters
   # - Other metadata

Creating New Models
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create a new untrained model
   new_model = StandardCATEModel(
       use_gmm_head=True,
       gmm_n_components=5,
       device='cuda'
   )
   
   # Train it (see Training guide)
   from causalfm.training import StandardCATETrainer, TrainingConfig
   
   config = TrainingConfig(data_path="data/*.csv")
   trainer = StandardCATETrainer(config)
   trainer.train()

Uncertainty Quantification
--------------------------

GMM Output Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~

The GMM head outputs a mixture of Gaussians:

.. code-block:: python

   result = model.estimate_cate(x_train, a_train, y_train, x_test)
   
   pi = result['gmm_pi']      # Shape: (n_test, n_components)
   mu = result['gmm_mu']      # Shape: (n_test, n_components)
   sigma = result['gmm_sigma']  # Shape: (n_test, n_components)
   
   # Point estimate (mixture mean)
   cate = result['cate']  # = sum(pi * mu, axis=-1)
   
   # Variance (mixture variance)
   variance = (pi * (sigma**2 + mu**2)).sum(dim=-1) - cate**2
   std_dev = torch.sqrt(variance)

Computing Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # Sample from the GMM
   n_samples = 1000
   n_test = len(pi)
   
   samples = np.zeros((n_test, n_samples))
   for i in range(n_test):
       # Sample component indices
       components = np.random.choice(
           len(pi[i]), 
           size=n_samples, 
           p=pi[i].cpu().numpy()
       )
       
       # Sample from selected components
       for k in range(len(pi[i])):
           mask = (components == k)
           n_k = mask.sum()
           if n_k > 0:
               samples[i, mask] = np.random.normal(
                   mu[i, k].cpu().numpy(),
                   sigma[i, k].cpu().numpy(),
                   n_k
               )
   
   # Compute percentiles
   ci_lower = np.percentile(samples, 2.5, axis=1)
   ci_upper = np.percentile(samples, 97.5, axis=1)

Model Comparison
----------------

Choosing the Right Model
~~~~~~~~~~~~~~~~~~~~~~~~

==================  ===================================  =========================
Setting             Model                                When to Use
==================  ===================================  =========================
Standard            ``StandardCATEModel``                No unobserved confounding
IV                  ``IVModel``                          Unobserved confounders + valid instrument
Front-door          ``FrontdoorModel``                   Unobserved confounders + mediator
==================  ===================================  =========================

Example Comparison
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # All models have the same interface
   models = {
       'Standard': StandardCATEModel.from_pretrained("checkpoints/standard.pth"),
       'IV': IVModel.from_pretrained("checkpoints/iv.pth"),
       'Frontdoor': FrontdoorModel.from_pretrained("checkpoints/fd.pth")
   }
   
   # Different inputs required
   results = {}
   
   # Standard
   results['Standard'] = models['Standard'].estimate_cate(
       x_train, a_train, y_train, x_test
   )
   
   # IV (needs instrument)
   results['IV'] = models['IV'].estimate_cate(
       x_train, z_train, a_train, y_train, x_test
   )
   
   # Front-door (needs mediator)
   results['Frontdoor'] = models['Frontdoor'].estimate_cate(
       x_train, m_train, a_train, y_train, x_test
   )

Advanced Usage
--------------

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from causalfm.models import StandardCATEModel
   
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
   
   # Process multiple test datasets
   test_files = ["data/test1.csv", "data/test2.csv", "data/test3.csv"]
   
   all_results = []
   for file in test_files:
       df = pd.read_csv(file)
       
       # Extract features
       x_cols = [c for c in df.columns if c.startswith('x')]
       X = torch.FloatTensor(df[x_cols].values)
       A = torch.FloatTensor(df['treatment'].values).unsqueeze(1)
       Y = torch.FloatTensor(df['outcome'].values).unsqueeze(1)
       
       # Split
       n_train = int(0.8 * len(X))
       result = model.estimate_cate(
           X[:n_train], A[:n_train], Y[:n_train], X[n_train:]
       )
       
       all_results.append(result['cate'])

GPU/CPU Management
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Auto-detect device
   model = StandardCATEModel.from_pretrained("model.pth", device='auto')
   
   # Force CPU
   model_cpu = StandardCATEModel.from_pretrained("model.pth", device='cpu')
   
   # Specific GPU
   model_gpu = StandardCATEModel.from_pretrained("model.pth", device='cuda:0')
   
   # Move between devices
   model = model.to('cuda:1')  # Move to GPU 1

API Reference
-------------

For complete API documentation, see:

* :class:`causalfm.models.StandardCATEModel`
* :class:`causalfm.models.IVModel`
* :class:`causalfm.models.FrontdoorModel`

