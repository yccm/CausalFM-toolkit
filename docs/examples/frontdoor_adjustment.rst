Front-door Adjustment Example
=============================

This example demonstrates using CausalFM for front-door adjustment.

Overview
--------

In this example, we will:

1. Generate front-door training data with mediators
2. Train a front-door model
3. Evaluate on test data
4. Understand when front-door identification works

Coming Soon
-----------

This tutorial is under development. In the meantime, check out:

* :doc:`standard_cate` - Complete Standard CATE example
* :doc:`../user_guide/data_generation` - Front-door data generation guide
* :doc:`../user_guide/models` - Front-door model usage

Quick Example
-------------

.. code-block:: python

   from causalfm.data import FrontdoorDataGenerator
   from causalfm.models import FrontdoorModel
   from causalfm.training import FrontdoorTrainer, TrainingConfig
   
   # Generate front-door data
   generator = FrontdoorDataGenerator(
       num_samples=1024,
       num_features=10,
       num_confounders=5
   )
   generator.generate_multiple(500, "data/frontdoor_train/")
   
   # Train
   if __name__ == '__main__':
       config = TrainingConfig(
           data_path="data/frontdoor_train/*.csv",
           epochs=100,
           save_dir="checkpoints/frontdoor/"
       )
       trainer = FrontdoorTrainer(config)
       trainer.train()
   
   # Evaluate
   model = FrontdoorModel.from_pretrained("checkpoints/frontdoor/best_model.pth")
   
   # Use mediator m along with x, a, y
   result = model.estimate_cate(
       x_train, m_train, a_train, y_train, x_test
   )
   cate = result['cate']

For a complete working example, see the notebook at:
``evaluation/notebook/test_fd.ipynb``

