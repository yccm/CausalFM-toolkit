Instrumental Variables Example
==============================

This example demonstrates using CausalFM for causal inference with instrumental variables.

Overview
--------

In this example, we will:

1. Generate IV training data (binary instrument)
2. Train an IV model
3. Evaluate on test data with unobserved confounding
4. Compare with standard CATE (which fails with confounding)

Coming Soon
-----------

This tutorial is under development. In the meantime, check out:

* :doc:`standard_cate` - Complete Standard CATE example
* :doc:`../user_guide/data_generation` - IV data generation guide
* :doc:`../user_guide/models` - IV model usage

Quick Example
-------------

.. code-block:: python

   from causalfm.data import IVDataGenerator
   from causalfm.models import IVModel
   from causalfm.training import IVTrainer, TrainingConfig
   
   # Generate IV data
   generator = IVDataGenerator(
       num_samples=1024,
       num_features=10,
       instrument_type='binary'
   )
   generator.generate_multiple(500, "data/iv_train/")
   
   # Train
   if __name__ == '__main__':
       config = TrainingConfig(
           data_path="data/iv_train/*.csv",
           epochs=100,
           save_dir="checkpoints/iv/"
       )
       trainer = IVTrainer(config)
       trainer.train()
   
   # Evaluate
   model = IVModel.from_pretrained("checkpoints/iv/best_model.pth")
   
   # Use instrument z along with x, a, y
   result = model.estimate_cate(
       x_train, z_train, a_train, y_train, x_test
   )
   cate = result['cate']

For a complete working example, see the notebook at:
``evaluation/notebook/test_iv_binary.ipynb``

