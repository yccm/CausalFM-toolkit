Tutorial 1: Basics
==================

Welcome to CausalFM! This tutorial introduces the fundamental concepts and workflow.

Learning Objectives
-------------------

By the end of this tutorial, you will understand:

* What are Prior-Data Fitted Networks (PFNs)
* The key concepts in causal inference with CausalFM
* The basic workflow from data to predictions
* When to use different causal settings

What are Foundation Models for Causal Inference?
-------------------------------------------------

Traditional Approach
~~~~~~~~~~~~~~~~~~~~

Traditional machine learning for causal inference:

1. Collect a single dataset
2. Train a model on that dataset
3. Make predictions on new samples from the same distribution

**Problem:** Requires large datasets and doesn't transfer well.

Foundation Model Approach (CausalFM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CausalFM uses a different paradigm:

1. Train once on many diverse synthetic datasets
2. Model learns the structure of causal inference problems
3. Apply to new datasets without retraining (zero-shot transfer)
4. Adapt in-context using just a few samples

**Advantage:** Transfer learning + in-context adaptation!

Key Concepts
------------

Prior-Data Fitted Networks (PFNs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PFNs are trained to solve a distribution of tasks, not just one task.

.. code-block:: python

   # Traditional: Train on one dataset
   model.fit(X, y)  # Requires large dataset
   
   # PFN: Trained on many datasets, adapts in-context
   model.estimate(X_context, y_context, X_query)  # Few samples needed!

In CausalFM, the "task" is estimating treatment effects for a particular dataset.

In-Context Learning
~~~~~~~~~~~~~~~~~~~

The model takes training examples as input and adapts its predictions:

.. code-block:: python

   # Provide context (training samples)
   result = model.estimate_cate(
       x_train,  # Context: observed covariates
       a_train,  # Context: treatments
       y_train,  # Context: outcomes
       x_test    # Query: new covariates to predict for
   )

The model learns the relationship between (X, A) → Y from the context samples!

CATE: Conditional Average Treatment Effect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CATE is the expected treatment effect for an individual with covariates X:

.. math::

   \tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]

Where:

* Y(1) = potential outcome under treatment
* Y(0) = potential outcome under control
* X = covariates (features)

.. code-block:: python

   # Individual treatment effects
   cate = model.estimate_cate(x_train, a_train, y_train, x_test)['cate']
   
   # Average treatment effect
   ate = cate.mean()

Causal Settings in CausalFM
----------------------------

CausalFM supports three causal inference settings:

1. Standard CATE Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** No unobserved confounding (all confounders measured).

.. code-block:: python

   from causalfm.models import StandardCATEModel
   
   model = StandardCATEModel.from_pretrained("checkpoints/standard.pth")
   result = model.estimate_cate(x_train, a_train, y_train, x_test)

2. Instrumental Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Unobserved confounding, but valid instrument available.

.. code-block:: python

   from causalfm.models import IVModel
   
   model = IVModel.from_pretrained("checkpoints/iv.pth")
   result = model.estimate_cate(x_train, z_train, a_train, y_train, x_test)

3. Front-door Adjustment
~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Unobserved confounding, mediator blocks backdoor path.

.. code-block:: python

   from causalfm.models import FrontdoorModel
   
   model = FrontdoorModel.from_pretrained("checkpoints/fd.pth")
   result = model.estimate_cate(x_train, m_train, a_train, y_train, x_test)

Basic Workflow
--------------

The CausalFM workflow has four main steps:

Step 1: Generate Data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.data import StandardCATEGenerator
   
   generator = StandardCATEGenerator(num_samples=1024, num_features=10)
   
   # Generate training data
   generator.generate_multiple(500, "data/train/")
   
   # Generate test data
   generator.generate_multiple(50, "data/test/")

Step 2: Train Model
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.training import StandardCATETrainer, TrainingConfig
   
   if __name__ == '__main__':
       config = TrainingConfig(
           data_path="data/train/*.csv",
           epochs=100,
           save_dir="checkpoints/"
       )
       trainer = StandardCATETrainer(config)
       trainer.train()

Step 3: Load and Predict
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.models import StandardCATEModel
   
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
   
   # Prepare data
   result = model.estimate_cate(x_train, a_train, y_train, x_test)
   cate = result['cate']  # Treatment effect estimates

Step 4: Evaluate
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.evaluation import compute_pehe, compute_ate_error
   
   pehe = compute_pehe(cate, true_ite)
   ate_error = compute_ate_error(cate, true_ite)
   
   print(f"PEHE: {pehe:.4f}")
   print(f"ATE Error: {ate_error:.4f}")

Your First CausalFM Script
---------------------------

Let's put it all together:

.. code-block:: python

   """
   My first CausalFM script
   """
   import torch
   from causalfm.data import StandardCATEGenerator
   from causalfm.models import StandardCATEModel
   from causalfm.training import StandardCATETrainer, TrainingConfig
   from causalfm.evaluation import compute_pehe
   
   # 1. Generate data
   print("Generating data...")
   gen = StandardCATEGenerator(num_samples=1024, num_features=10)
   gen.generate_multiple(100, "data/train/")
   gen.generate_multiple(10, "data/test/")
   
   # 2. Train model
   if __name__ == '__main__':
       print("Training model...")
       config = TrainingConfig(
           data_path="data/train/*.csv",
           epochs=50,
           batch_size=16,
           num_workers=0,
           save_dir="checkpoints/"
       )
       trainer = StandardCATETrainer(config)
       trainer.train()
       
       # 3. Load and predict
       print("Making predictions...")
       model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
       
       # Prepare some test data
       x_train = torch.randn(800, 10)
       a_train = torch.randint(0, 2, (800, 1)).float()
       y_train = torch.randn(800, 1)
       x_test = torch.randn(200, 10)
       
       result = model.estimate_cate(x_train, a_train, y_train, x_test)
       cate = result['cate']
       
       print(f"Estimated {len(cate)} treatment effects!")
       print(f"Mean CATE: {cate.mean():.4f}")

Key Takeaways
-------------

✅ **CausalFM uses foundation models** that learn from many datasets

✅ **In-context learning** allows adaptation with few samples

✅ **Three causal settings** for different identification strategies

✅ **Simple workflow:** Generate → Train → Predict → Evaluate

✅ **Zero-shot transfer** to new datasets without retraining

Next Steps
----------

Now that you understand the basics, continue to:

* :doc:`tutorial_02_data_generation` - Learn about data generation
* :doc:`tutorial_03_training` - Dive deep into training
* :doc:`../examples/standard_cate` - See a complete working example

Questions?
----------

* Check the :doc:`../quickstart` for quick reference
* Read the :doc:`../user_guide/models` for model details
* Look at example notebooks in ``evaluation/notebook/``

