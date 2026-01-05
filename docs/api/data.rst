Data API
========

This page documents the data generation and loading APIs in CausalFM.

Data Generators
---------------

StandardCATEGenerator
~~~~~~~~~~~~~~~~~~~~~

Generate synthetic datasets for standard CATE estimation.
   
   Example:
   
   .. code-block:: python
   
      from causalfm.data import StandardCATEGenerator
      
      generator = StandardCATEGenerator(
          num_samples=1024,
          num_features=10,
          seed=42
      )
      
      # Single dataset
      df = generator.generate()
      
      # Multiple datasets
      generator.generate_multiple(100, "data/train/")

IVDataGenerator
~~~~~~~~~~~~~~~

Generate synthetic datasets for instrumental variables setting.
   
   Example:
   
   .. code-block:: python
   
      from causalfm.data import IVDataGenerator
      
      generator = IVDataGenerator(
          num_samples=1024,
          num_features=10,
          instrument_type='binary',
          seed=42
      )
      
      df = generator.generate()

FrontdoorDataGenerator
~~~~~~~~~~~~~~~~~~~~~~

Generate synthetic datasets for front-door adjustment setting.
   
   Example:
   
   .. code-block:: python
   
      from causalfm.data import FrontdoorDataGenerator
      
      generator = FrontdoorDataGenerator(
          num_samples=1024,
          num_features=10,
          num_confounders=5,
          seed=42
      )
      
      df = generator.generate()

Base Classes
~~~~~~~~~~~~

DAGStructuredSCM
^^^^^^^^^^^^^^^^

Structural Causal Model with DAG structure for generating covariates.

**Key Methods:**

* ``__init__(num_features, num_layers, hidden_size, edge_drop_prob)`` - Initialize DAG-SCM
* ``generate(num_samples)`` - Generate samples following the DAG structure

BaseMLPGenerator
^^^^^^^^^^^^^^^^

Base class for MLP-based data generation components.

**Key Methods:**

* ``__init__(input_dim, hidden_dim, output_dim)`` - Initialize MLP generator
* ``forward(x)`` - Generate outputs given inputs

Data Loaders
------------

StandardDataLoader
~~~~~~~~~~~~~~~~~~

PyTorch Dataset for loading standard CATE training data.

**Class:** ``causalfm.data.loaders.standard.CausalDataset``
   
   Example:
   
   .. code-block:: python
   
      from causalfm.data.loaders.standard import CausalDataset
      from torch.utils.data import DataLoader
      
      dataset = CausalDataset("data/train/", file_pattern="*.csv")
      loader = DataLoader(dataset, batch_size=16, shuffle=True)

**Function:** ``causalfm.data.loaders.standard.collate_fn`` - Custom collate function for batching causal datasets.

StandardTestDataLoader
~~~~~~~~~~~~~~~~~~~~~~

PyTorch Dataset for loading standard CATE test data.

**Class:** ``causalfm.data.loaders.standard.CausalTestDataset``

**Function:** ``causalfm.data.loaders.standard.test_collate_fn`` - Custom collate function for batching test datasets.

IV Data Loaders
~~~~~~~~~~~~~~~

PyTorch Dataset for loading IV training data.

**Class:** ``causalfm.data.loaders.iv.IVDataset``

**Function:** ``causalfm.data.loaders.iv.iv_collate_fn`` - Custom collate function.

Front-door Data Loaders
~~~~~~~~~~~~~~~~~~~~~~~

PyTorch Dataset for loading front-door training data.

**Class:** ``causalfm.data.loaders.frontdoor.FrontdoorDataset``

**Function:** ``causalfm.data.loaders.frontdoor.frontdoor_collate_fn`` - Custom collate function.

Utility Functions
-----------------

Loading from CSV
~~~~~~~~~~~~~~~~

All data loaders support loading from CSV files with the following column conventions:

**Standard CATE Data:**

* ``x0, x1, ..., xN``: Covariates
* ``treatment``: Binary treatment (0 or 1)
* ``outcome``: Observed outcome
* ``y0, y1``: Potential outcomes (if available)
* ``ite``: Individual treatment effect (if available)

**IV Data:**

* ``x0, x1, ..., xN``: Observed covariates
* ``u0, u1, ..., uM``: Unobserved confounders (for synthetic data)
* ``z``: Instrument variable
* ``treatment``: Binary treatment
* ``outcome``: Observed outcome
* ``y0, y1, ite``: Ground truth (if available)

**Front-door Data:**

* ``x0, x1, ..., xN``: Observed covariates
* ``u0, u1, ..., uM``: Unobserved confounders (for synthetic data)
* ``treatment``: Binary treatment
* ``mediator``: Mediator variable
* ``outcome``: Observed outcome
* ``y0, y1, m0, m1, ite, ate``: Ground truth (if available)

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from causalfm.data import StandardCATEGenerator
   
   # Generate and save data
   generator = StandardCATEGenerator(num_samples=1024, num_features=10)
   df = generator.generate()
   df.to_csv("my_data.csv", index=False)
   
   # Load data
   df_loaded = pd.read_csv("my_data.csv")
   
   print(df_loaded.columns)
   # ['x0', 'x1', ..., 'x9', 'treatment', 'outcome', 'y0', 'y1', 'ite']

