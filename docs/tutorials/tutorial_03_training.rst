Tutorial 3: Training Models
===========================

Learn how to train CausalFM foundation models.

Coming Soon
-----------

This tutorial is under development. For now, see:

* :doc:`../user_guide/training` - Complete training guide
* :doc:`tutorial_01_basics` - Basic concepts
* :doc:`../examples/standard_cate` - Complete example

Quick Reference
---------------

Basic Training
~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.training import StandardCATETrainer, TrainingConfig
   
   config = TrainingConfig(
       data_path="data/train/*.csv",
       epochs=100,
       batch_size=16,
       save_dir="checkpoints/"
   )
   
   if __name__ == '__main__':
       trainer = StandardCATETrainer(config)
       trainer.train()

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TrainingConfig(
       # Data
       data_path="data/train/*.csv",
       val_split=0.2,
       batch_size=16,
       num_workers=4,
       
       # Training
       epochs=150,
       learning_rate=0.001,
       weight_decay=1e-5,
       early_stop_patience=50,
       
       # Model
       use_gmm_head=True,
       gmm_n_components=5,
       
       # Checkpointing
       save_dir="checkpoints/",
       
       # Hardware
       device='auto'
   )

Next Tutorial
-------------

Continue to :doc:`tutorial_04_evaluation` to learn about model evaluation.

