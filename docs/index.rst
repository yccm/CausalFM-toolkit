CausalFM Toolkit Documentation
===============================

Welcome to **CausalFM Toolkit**'s documentation!

CausalFM Toolkit is a comprehensive PyTorch-based framework for training foundation models 
for causal inference using Prior-Data Fitted Networks (PFNs). It provides a unified interface 
for multiple causal inference settings with state-of-the-art deep learning architectures.

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/pytorch-2.0+-orange.svg
   :target: https://pytorch.org/
   :alt: PyTorch Version

Key Features
------------

✨ **Multiple Causal Settings**
   - Standard CATE (Conditional Average Treatment Effect) estimation
   - Instrumental Variables (IV) with binary/continuous instruments
   - Front-door adjustment with mediators

🎯 **Foundation Model Architecture**
   - Built on TabPFN (Prior-Data Fitted Networks)
   - Transformer-based architecture with GMM prediction heads
   - Uncertainty quantification through mixture distributions

📦 **Clean Library Interface**
   - Easy-to-use API for data generation, training, and evaluation
   - Pre-built data generators for synthetic datasets
   - Comprehensive evaluation metrics (PEHE, ATE error, etc.)

🚀 **Production Ready**
   - Model checkpointing and loading
   - TensorBoard integration
   - Efficient PyTorch DataLoaders

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/data_generation
   user_guide/training
   user_guide/evaluation
   user_guide/models

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/standard_cate
   examples/instrumental_variables
   examples/frontdoor_adjustment

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/data
   api/models
   api/training
   api/evaluation

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   citation
   license

Quick Example
-------------

Here's a minimal example to get you started:

.. code-block:: python

   from causalfm.data import StandardCATEGenerator
   from causalfm.models import StandardCATEModel
   from causalfm.evaluation import compute_pehe

   # Generate synthetic data
   generator = StandardCATEGenerator(num_samples=1000, num_features=10)
   df = generator.generate()

   # Load pretrained model
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")

   # Estimate CATE
   result = model.estimate_cate(x_train, a_train, y_train, x_test)
   cate = result['cate']

   # Evaluate
   pehe = compute_pehe(cate, true_ite)
   print(f"PEHE: {pehe:.4f}")

Paper Reference
---------------

This toolkit implements the methods described in:

   Ma, Yuchen, Dennis Frauen, Emil Javurek, and Stefan Feuerriegel. 
   "Foundation Models for Causal Inference via Prior-Data Fitted Networks." 
   *arXiv preprint arXiv:2506.10914* (2025).

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
