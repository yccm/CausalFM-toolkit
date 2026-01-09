Data Generation
===============

CausalFM provides powerful data generators for creating synthetic causal datasets 
across different causal inference settings.

Overview
--------

All data generators inherit from ``BaseDataGenerator`` and provide:

* Configurable data generation parameters
* Ground truth causal effects (ITE, ATE)
* Support for single or multiple dataset generation
* Reproducibility through random seeds

Standard CATE Generator
-----------------------

The ``StandardCATEGenerator`` creates datasets for conditional average treatment 
effect estimation without confounding.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from causalfm.data import StandardCATEGenerator
   
   generator = StandardCATEGenerator(
       num_samples=1024,    # Number of observations
       num_features=10,     # Number of covariates
       seed=42              # Random seed for reproducibility
   )
   
   # Generate single dataset
   df = generator.generate()
   
   print(df.head())
   print(df.columns)
   # ['x0', 'x1', ..., 'x9', 'treatment', 'outcome', 'y0', 'y1', 'ite']

Data Generation Process
~~~~~~~~~~~~~~~~~~~~~~~

The generator follows a sophisticated DAG-structured process:

1. **Generate Covariates (X)**: Using DAG-structured SCM with MLP architecture
2. **Assign Treatments (A)**: Binary treatment based on covariates via neural propensity model
3. **Generate Outcomes (Y)**: Potential outcomes Y(0), Y(1) using neural outcome model
4. **Compute ITE**: Individual treatment effect = Y(1) - Y(0)

.. code-block:: python

   # The data generation uses:
   # - DAG-structured Structural Causal Models
   # - MLP-based treatment assignment
   # - MLP-based outcome generation
   # - Random noise from various distributions (normal, uniform, laplace, logistic)

Generated Columns
~~~~~~~~~~~~~~~~~

==================  ====================================================
Column              Description
==================  ====================================================
``x0, x1, ...``     Covariates (features)
``treatment``       Binary treatment (0 or 1)
``outcome``         Observed outcome Y
``y0``              Potential outcome under control (A=0)
``y1``              Potential outcome under treatment (A=1)
``ite``             Individual treatment effect (Y1 - Y0)
==================  ====================================================

Generate Multiple Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For training, you typically need many datasets:

.. code-block:: python

   generator = StandardCATEGenerator(num_samples=1024, num_features=10, seed=42)
   
   # Generate 100 training datasets
   generator.generate_multiple(
       num_datasets=100,
       output_dir="data/train/",
       filename_prefix="train_data"
   )
   
   # Generate 10 test datasets
   generator.generate_multiple(
       num_datasets=10,
       output_dir="data/test/",
       filename_prefix="test_data"
   )

Instrumental Variables Generator
---------------------------------

The ``IVDataGenerator`` creates datasets for IV settings where there are 
unobserved confounders but a valid instrument is available.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from causalfm.data import IVDataGenerator
   
   generator = IVDataGenerator(
       num_samples=1024,
       num_features=10,
       num_confounders_range=(2, 5),  # Random number of confounders
       instrument_type='binary',       # 'binary' or 'continuous'
       seed=42
   )
   
   df = generator.generate()
   
   print(df.columns)
   # ['x0', ..., 'u0', ..., 'z', 'treatment', 'outcome', 'y0', 'y1', 'ite']

IV Data Structure
~~~~~~~~~~~~~~~~~

The IV setting includes:

* **X**: Observed covariates
* **U**: Unobserved confounders (affect both treatment and outcome)
* **Z**: Instrument (affects treatment but not outcome directly)
* **A**: Treatment (affected by X, U, and Z)
* **Y**: Outcome (affected by X, U, and A)

.. code-block:: python

   # Binary instrument (common in RCTs)
   iv_binary = IVDataGenerator(
       num_samples=1024,
       num_features=10,
       instrument_type='binary'
   )
   df_binary = iv_binary.generate()
   
   # Continuous instrument
   iv_conti = IVDataGenerator(
       num_samples=1024,
       num_features=10,
       instrument_type='continuous'
   )
   df_conti = iv_conti.generate()

Generated Columns
~~~~~~~~~~~~~~~~~

==================  ====================================================
Column              Description
==================  ====================================================
``x0, x1, ...``     Observed covariates
``u0, u1, ...``     Unobserved confounders
``z``               Instrument variable
``treatment``       Binary treatment
``outcome``         Observed outcome
``y0, y1, ite``     Potential outcomes and ITE
==================  ====================================================

Front-door Generator
--------------------

The ``FrontdoorDataGenerator`` creates datasets for front-door adjustment 
where a mediator blocks the backdoor path through unobserved confounders.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from causalfm.data import FrontdoorDataGenerator
   
   generator = FrontdoorDataGenerator(
       num_samples=1024,
       num_features=10,
       num_confounders=5,    # Or None for random
       seed=42
   )
   
   df = generator.generate()
   
   print(df.columns)
   # ['x0', ..., 'u0', ..., 'treatment', 'mediator', 'outcome', 
   #  'y0', 'y1', 'm0', 'm1', 'ite', 'ate']

Front-door Data Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

The front-door setting includes:

* **X**: Observed covariates
* **U**: Unobserved confounders
* **A**: Treatment
* **M**: Mediator (on causal path from A to Y)
* **Y**: Outcome

.. code-block:: python

   # Fixed number of confounders
   fd_gen = FrontdoorDataGenerator(
       num_samples=1024,
       num_features=10,
       num_confounders=5
   )
   
   # Random number of confounders
   fd_gen_random = FrontdoorDataGenerator(
       num_samples=1024,
       num_features=10,
       num_confounders=None  # Will randomly sample between 1-15
   )

Generated Columns
~~~~~~~~~~~~~~~~~

==================  ====================================================
Column              Description
==================  ====================================================
``x0, x1, ...``     Observed covariates
``u0, u1, ...``     Unobserved confounders
``treatment``       Binary treatment
``mediator``        Mediator variable
``outcome``         Observed outcome
``y0, y1``          Potential outcomes
``m0, m1``          Potential mediators under A=0 and A=1
``ite``             Individual treatment effect
``ate``             Average treatment effect
==================  ====================================================

Advanced Usage
--------------

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

You can customize the data generation process:

.. code-block:: python

   from causalfm.data.generators.standard import (
       StandardCATEGenerator,
       DAGStructuredSCM
   )
   import numpy as np
   
   # Custom generator with specific parameters
   generator = StandardCATEGenerator(
       num_samples=2048,     # More samples
       num_features=20,      # More features
       seed=123              # Different seed
   )
   
   # The internal DAG-SCM uses:
   # - Random number of layers (3-7)
   # - Random hidden size (15-40)
   # - Edge dropout (0.5)
   # - Tanh activation

Batch Generation
~~~~~~~~~~~~~~~~

Generate data in batches for large-scale experiments:

.. code-block:: python

   import os
   from causalfm.data import StandardCATEGenerator
   
   # Training data
   train_gen = StandardCATEGenerator(num_samples=1024, num_features=10)
   train_gen.generate_multiple(
       num_datasets=1000,
       output_dir="data/large_train/",
       filename_prefix="train"
   )
   
   # Validation data
   val_gen = StandardCATEGenerator(num_samples=1024, num_features=10, seed=999)
   val_gen.generate_multiple(
       num_datasets=100,
       output_dir="data/large_val/",
       filename_prefix="val"
   )

Reproducibility
~~~~~~~~~~~~~~~

Ensure reproducible data generation:

.. code-block:: python

   # Same seed produces same data
   gen1 = StandardCATEGenerator(num_samples=100, num_features=5, seed=42)
   df1 = gen1.generate()
   
   gen2 = StandardCATEGenerator(num_samples=100, num_features=5, seed=42)
   df2 = gen2.generate()
   
   # DataFrames are identical
   assert df1.equals(df2)
   
   # Different seeds produce different data
   gen3 = StandardCATEGenerator(num_samples=100, num_features=5, seed=43)
   df3 = gen3.generate()
   
   assert not df1.equals(df3)

Data Properties
---------------

Understanding Generated Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generated datasets have realistic properties:

.. code-block:: python

   import matplotlib.pyplot as plt
   from causalfm.data import StandardCATEGenerator
   
   gen = StandardCATEGenerator(num_samples=10000, num_features=10)
   df = gen.generate()
   
   # Check treatment balance
   treatment_rate = df['treatment'].mean()
   print(f"Treatment rate: {treatment_rate:.2%}")
   
   # ITE distribution
   plt.hist(df['ite'], bins=50)
   plt.xlabel('Individual Treatment Effect')
   plt.ylabel('Frequency')
   plt.title('Distribution of ITEs')
   plt.show()
   
   # Average Treatment Effect
   ate = df['ite'].mean()
   print(f"ATE: {ate:.4f}")
   
   # Check overlap
   treated_outcomes = df[df['treatment'] == 1]['outcome']
   control_outcomes = df[df['treatment'] == 0]['outcome']
   print(f"Treated mean: {treated_outcomes.mean():.4f}")
   print(f"Control mean: {control_outcomes.mean():.4f}")

API Reference
-------------

For complete API documentation, see:

* :class:`causalfm.data.generators.StandardCATEGenerator`
* :class:`causalfm.data.generators.IVDataGenerator`
* :class:`causalfm.data.generators.FrontdoorDataGenerator`
* :class:`causalfm.data.generators.base.DAGStructuredSCM`

