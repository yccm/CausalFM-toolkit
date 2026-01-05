API Reference
=============

Complete API documentation for CausalFM Toolkit.

.. toctree::
   :maxdepth: 2

   data
   models
   training
   evaluation

Overview
--------

The CausalFM Toolkit API is organized into four main modules:

Data Module
~~~~~~~~~~~

:doc:`data`

The data module provides tools for generating and loading causal datasets:

* **Generators:** Create synthetic datasets for training and testing
* **Loaders:** PyTorch datasets for efficient data loading
* **Base Classes:** Reusable components for custom data generation

Key classes:

* :class:`causalfm.data.StandardCATEGenerator`
* :class:`causalfm.data.IVDataGenerator`
* :class:`causalfm.data.FrontdoorDataGenerator`

Models Module
~~~~~~~~~~~~~

:doc:`models`

The models module contains foundation model implementations:

* **StandardCATEModel:** Standard CATE estimation
* **IVModel:** Instrumental variables setting
* **FrontdoorModel:** Front-door adjustment

Key classes:

* :class:`causalfm.models.StandardCATEModel`
* :class:`causalfm.models.IVModel`
* :class:`causalfm.models.FrontdoorModel`

Training Module
~~~~~~~~~~~~~~~

:doc:`training`

The training module provides trainers and configuration for model training:

* **Trainers:** Handle the training loop for each setting
* **TrainingConfig:** Comprehensive training configuration
* **Utilities:** Helper functions for training

Key classes:

* :class:`causalfm.training.StandardCATETrainer`
* :class:`causalfm.training.IVTrainer`
* :class:`causalfm.training.FrontdoorTrainer`
* :class:`causalfm.training.TrainingConfig`

Evaluation Module
~~~~~~~~~~~~~~~~~

:doc:`evaluation`

The evaluation module provides metrics for assessing model performance:

* **PEHE:** Precision in Estimation of Heterogeneous Effects
* **ATE Error:** Average Treatment Effect error
* **MSE/RMSE:** Mean squared error metrics
* **Utilities:** Helper functions for evaluation

Key functions:

* :func:`causalfm.evaluation.compute_pehe`
* :func:`causalfm.evaluation.compute_ate_error`
* :func:`causalfm.evaluation.compute_mse`
* :func:`causalfm.evaluation.compute_rmse`

Quick Links
-----------

**Getting Started:**

* :doc:`../installation` - Installation instructions
* :doc:`../quickstart` - Quick start guide
* :doc:`../tutorials/index` - Step-by-step tutorials

**User Guides:**

* :doc:`../user_guide/data_generation` - Data generation guide
* :doc:`../user_guide/models` - Model usage guide
* :doc:`../user_guide/training` - Training guide
* :doc:`../user_guide/evaluation` - Evaluation guide

**Examples:**

* :doc:`../examples/standard_cate` - Complete example
* :doc:`../examples/instrumental_variables` - IV example
* :doc:`../examples/frontdoor_adjustment` - Front-door example

Module Index
------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   causalfm.data
   causalfm.models
   causalfm.training
   causalfm.evaluation

Search
------

* :ref:`genindex` - General index
* :ref:`modindex` - Module index
* :ref:`search` - Search documentation
