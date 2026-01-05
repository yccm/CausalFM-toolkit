Tutorials
=========

Welcome to the CausalFM Toolkit tutorials! These step-by-step guides will help you 
get started with causal inference using foundation models.

.. toctree::
   :maxdepth: 2
   :caption: Tutorial Series

   tutorial_01_basics
   tutorial_02_data_generation
   tutorial_03_training
   tutorial_04_evaluation

Tutorial Overview
-----------------

Tutorial 1: Basics
~~~~~~~~~~~~~~~~~~

Learn the fundamentals of CausalFM:

* Understanding the foundation model approach
* Key concepts: PFNs, in-context learning, CATE estimation
* Basic workflow from data to predictions

:doc:`tutorial_01_basics`

Tutorial 2: Data Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Master data generation for different settings:

* Standard CATE data generation
* Instrumental variables data
* Front-door adjustment data
* Understanding DAG-structured SCMs

:doc:`tutorial_02_data_generation`

Tutorial 3: Training Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learn how to train your own models:

* Configuring training runs
* Monitoring training progress
* Saving and loading checkpoints
* Training for different causal settings

:doc:`tutorial_03_training`

Tutorial 4: Model Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate your models effectively:

* Computing causal inference metrics
* Uncertainty quantification
* Visualization techniques
* Comparing multiple models

:doc:`tutorial_04_evaluation`

Prerequisites
-------------

Before starting these tutorials, make sure you have:

* ✅ Installed CausalFM Toolkit (see :doc:`../installation`)
* ✅ Basic Python knowledge
* ✅ Familiarity with PyTorch (helpful but not required)
* ✅ Understanding of causal inference concepts (helpful but not required)

What You'll Learn
-----------------

By completing these tutorials, you will be able to:

1. Generate synthetic causal datasets
2. Train foundation models for causal inference
3. Make predictions with pretrained models
4. Evaluate model performance
5. Apply CausalFM to your own causal inference problems

Example Notebooks
-----------------

For hands-on examples, check out the Jupyter notebooks in the ``evaluation/notebook/`` directory:

* ``test_standard_cate.ipynb`` - Standard CATE estimation
* ``test_iv_binary.ipynb`` - Binary instrumental variables
* ``test_iv_conti.ipynb`` - Continuous instrumental variables
* ``test_fd.ipynb`` - Front-door adjustment
* ``test_jobs.ipynb`` - Real-world dataset example

These notebooks provide complete working examples you can run and modify.

Getting Help
------------

If you get stuck:

* Check the :doc:`../user_guide/index` for detailed explanations
* Look at the :doc:`../examples/standard_cate` for complete code
* Read the :doc:`../api/index` for API documentation
* Open an issue on GitHub if you find a bug

Let's Get Started!
------------------

Ready to begin? Start with :doc:`tutorial_01_basics` to learn the fundamentals!

