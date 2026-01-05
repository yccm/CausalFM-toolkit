Models API
==========

This page documents the model APIs in CausalFM.

Model Classes
-------------

StandardCATEModel
~~~~~~~~~~~~~~~~~

**Class:** ``causalfm.models.standard.StandardCATEModel``

Foundation model for standard CATE estimation.
   
   This model uses a transformer architecture with GMM prediction head
   for estimating conditional average treatment effects.
   
   Example:
   
   .. code-block:: python
   
      from causalfm.models import StandardCATEModel
      import torch
      
      # Create new model
      model = StandardCATEModel(use_gmm_head=True, gmm_n_components=5)
      
      # Or load pretrained
      model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
      
      # Estimate CATE
      x_train = torch.randn(800, 10)
      a_train = torch.randint(0, 2, (800, 1)).float()
      y_train = torch.randn(800, 1)
      x_test = torch.randn(200, 10)
      
      result = model.estimate_cate(x_train, a_train, y_train, x_test)
      cate = result['cate']  # Point estimates

IVModel
~~~~~~~

**Class:** ``causalfm.models.iv.IVModel``

Foundation model for instrumental variables setting.
   
   This model uses instruments to identify causal effects in the
   presence of unobserved confounding.
   
   Example:
   
   .. code-block:: python
   
      from causalfm.models import IVModel
      import torch
      
      model = IVModel.from_pretrained("checkpoints/iv_model.pth")
      
      # Requires instrument variable z
      x_train = torch.randn(800, 10)
      z_train = torch.randint(0, 2, (800, 1)).float()  # Binary instrument
      a_train = torch.randint(0, 2, (800, 1)).float()
      y_train = torch.randn(800, 1)
      x_test = torch.randn(200, 10)
      
      result = model.estimate_cate(x_train, z_train, a_train, y_train, x_test)
      cate = result['cate']

FrontdoorModel
~~~~~~~~~~~~~~

**Class:** ``causalfm.models.frontdoor.FrontdoorModel``

Foundation model for front-door adjustment setting.
   
   This model uses mediators to identify causal effects via
   front-door adjustment when backdoor paths are blocked.
   
   Example:
   
   .. code-block:: python
   
      from causalfm.models import FrontdoorModel
      import torch
      
      model = FrontdoorModel.from_pretrained("checkpoints/fd_model.pth")
      
      # Requires mediator variable m
      x_train = torch.randn(800, 10)
      m_train = torch.randn(800, 1)  # Mediator
      a_train = torch.randint(0, 2, (800, 1)).float()
      y_train = torch.randn(800, 1)
      x_test = torch.randn(200, 10)
      
      result = model.estimate_cate(x_train, m_train, a_train, y_train, x_test)
      cate = result['cate']

Common Methods
--------------

All model classes share the following interface:

``__init__(use_gmm_head, gmm_n_components, device)``
   Initialize a new model.
   
   :param bool use_gmm_head: Whether to use GMM prediction head
   :param int gmm_n_components: Number of mixture components
   :param str device: Device to use ('cuda', 'cpu', or 'auto')

``from_pretrained(checkpoint_path, device)``
   Load a pretrained model from checkpoint.
   
   :param str checkpoint_path: Path to checkpoint file
   :param str device: Device to use
   :return: Loaded model instance
   :rtype: Model

``estimate_cate(...)``
   Estimate conditional average treatment effects.
   
   :return: Dictionary with keys:
       - 'cate': Point estimates (n_test,)
       - 'gmm_pi': Mixture weights (n_test, n_components)
       - 'gmm_mu': Component means (n_test, n_components)
       - 'gmm_sigma': Component std devs (n_test, n_components)
   :rtype: dict

``save(path)``
   Save model checkpoint.
   
   :param str path: Path to save checkpoint

``eval_mode()``
   Set model to evaluation mode.

``train_mode()``
   Set model to training mode.

Input Shapes
------------

All models expect specific tensor shapes:

**Standard CATE:**

.. code-block:: python

   x_train: (n_train, n_features)
   a_train: (n_train, 1)          # NOT (n_train,)
   y_train: (n_train, 1)          # NOT (n_train,)
   x_test:  (n_test, n_features)

**IV Model:**

.. code-block:: python

   x_train: (n_train, n_features)
   z_train: (n_train, 1)          # Instrument
   a_train: (n_train, 1)
   y_train: (n_train, 1)
   x_test:  (n_test, n_features)

**Front-door Model:**

.. code-block:: python

   x_train: (n_train, n_features)
   m_train: (n_train, 1)          # Mediator
   a_train: (n_train, 1)
   y_train: (n_train, 1)
   x_test:  (n_test, n_features)

Output Format
-------------

GMM Output
~~~~~~~~~~

All models return a dictionary with GMM parameters:

.. code-block:: python

   result = model.estimate_cate(x_train, a_train, y_train, x_test)
   
   # Point estimate (mixture mean)
   cate = result['cate']  # Shape: (n_test,)
   
   # GMM parameters
   pi = result['gmm_pi']        # Mixing weights: (n_test, n_components)
   mu = result['gmm_mu']        # Component means: (n_test, n_components)
   sigma = result['gmm_sigma']  # Component stds: (n_test, n_components)
   
   # Compute predictive variance
   variance = (pi * (sigma**2 + mu**2)).sum(dim=-1) - cate**2

Confidence Intervals
~~~~~~~~~~~~~~~~~~~~

Sample from the GMM to get confidence intervals:

.. code-block:: python

   import numpy as np
   
   # Sample from GMM
   n_samples = 10000
   samples = np.zeros((len(cate), n_samples))
   
   for i in range(len(cate)):
       # Sample components
       components = np.random.choice(
           len(pi[i]),
           size=n_samples,
           p=pi[i].cpu().numpy()
       )
       
       # Sample from each component
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

Advanced Usage
--------------

Custom Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = StandardCATEModel(
       use_gmm_head=True,
       gmm_n_components=10,        # More components
       gmm_min_sigma=1e-4,         # Minimum variance
       gmm_pi_temp=0.8,            # Temperature for mixing
       device='cuda:0'
   )

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   import torch
   import pandas as pd
   
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
   
   # Process multiple files
   results = []
   for file in Path("data/test/").glob("*.csv"):
       df = pd.read_csv(file)
       
       # Prepare data
       x_cols = [c for c in df.columns if c.startswith('x')]
       X = torch.FloatTensor(df[x_cols].values)
       A = torch.FloatTensor(df['treatment'].values).unsqueeze(1)
       Y = torch.FloatTensor(df['outcome'].values).unsqueeze(1)
       
       # Split
       n_train = int(0.8 * len(X))
       
       # Predict
       result = model.estimate_cate(
           X[:n_train], A[:n_train], Y[:n_train], X[n_train:]
       )
       
       results.append({
           'file': file.name,
           'mean_cate': result['cate'].mean().item()
       })
   
   print(pd.DataFrame(results))

GPU Management
~~~~~~~~~~~~~~

.. code-block:: python

   # Auto-detect device
   model = StandardCATEModel.from_pretrained("model.pth", device='auto')
   
   # Specific GPU
   model = StandardCATEModel.from_pretrained("model.pth", device='cuda:1')
   
   # Move between devices
   model = model.to('cpu')
   
   # Check current device
   print(model.device)

Model Properties
----------------

Access model attributes:

.. code-block:: python

   # Get underlying model
   underlying_model = model.model
   
   # Get model parameters
   params = model.parameters
   
   # Count parameters
   n_params = sum(p.numel() for p in model.parameters)
   print(f"Model has {n_params:,} parameters")

See Also
--------

* :doc:`../user_guide/models` - Detailed model usage guide
* :doc:`../examples/standard_cate` - Complete example
* :doc:`training` - Training API reference

