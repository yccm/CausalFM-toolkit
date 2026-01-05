Training API
============

This page documents the training APIs in CausalFM.

.. module:: causalfm.training

Trainer Classes
---------------

StandardCATETrainer
~~~~~~~~~~~~~~~~~~~

.. autoclass:: causalfm.training.standard.StandardCATETrainer
   :members:
   :undoc-members:
   :show-inheritance:

   Trainer for standard CATE models.
   
   Example:
   
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

IVTrainer
~~~~~~~~~

.. autoclass:: causalfm.training.iv.IVTrainer
   :members:
   :undoc-members:
   :show-inheritance:

   Trainer for instrumental variables models.
   
   Example:
   
   .. code-block:: python
   
      from causalfm.training import IVTrainer, TrainingConfig
      
      config = TrainingConfig(
          data_path="data/iv_train/*.csv",
          epochs=100,
          save_dir="checkpoints/iv/"
      )
      
      if __name__ == '__main__':
          trainer = IVTrainer(config)
          trainer.train()

FrontdoorTrainer
~~~~~~~~~~~~~~~~

.. autoclass:: causalfm.training.frontdoor.FrontdoorTrainer
   :members:
   :undoc-members:
   :show-inheritance:

   Trainer for front-door adjustment models.
   
   Example:
   
   .. code-block:: python
   
      from causalfm.training import FrontdoorTrainer, TrainingConfig
      
      config = TrainingConfig(
          data_path="data/fd_train/*.csv",
          epochs=100,
          save_dir="checkpoints/frontdoor/"
      )
      
      if __name__ == '__main__':
          trainer = FrontdoorTrainer(config)
          trainer.train()

Configuration
-------------

TrainingConfig
~~~~~~~~~~~~~~

.. autoclass:: causalfm.training.base.TrainingConfig
   :members:
   :undoc-members:

   Configuration class for training.
   
   Parameters:
   
   **Data Settings:**
   
   * ``data_path`` (str): Glob pattern for training data files
   * ``val_split`` (float): Validation split ratio (default: 0.2)
   * ``batch_size`` (int): Training batch size (default: 16)
   * ``num_workers`` (int): Number of data loading workers (default: 4)
   
   **Training Settings:**
   
   * ``epochs`` (int): Number of training epochs (default: 100)
   * ``learning_rate`` (float): Learning rate (default: 0.001)
   * ``weight_decay`` (float): L2 regularization (default: 1e-5)
   * ``clip_grad`` (float): Gradient clipping value (default: 1.0)
   * ``early_stop_patience`` (int): Early stopping patience (default: 50)
   
   **Model Settings:**
   
   * ``use_gmm_head`` (bool): Use GMM prediction head (default: True)
   * ``gmm_n_components`` (int): Number of mixture components (default: 5)
   * ``gmm_min_sigma`` (float): Minimum std dev (default: 1e-3)
   * ``gmm_pi_temp`` (float): Mixture weight temperature (default: 1.0)
   
   **Checkpointing:**
   
   * ``save_dir`` (str): Directory for checkpoints (default: "checkpoints/")
   * ``save_freq`` (int): Save every N epochs (default: 10)
   * ``save_best_only`` (bool): Only save best model (default: False)
   * ``resume_from`` (str): Path to resume training from (default: None)
   
   **Logging:**
   
   * ``log_dir`` (str): TensorBoard log directory (default: "logs/")
   * ``log_freq`` (int): Log every N steps (default: 100)
   
   **Hardware:**
   
   * ``device`` (str): Device to use ('auto', 'cuda', 'cpu') (default: 'auto')
   * ``gpu_id`` (int): GPU ID to use (default: 0)
   * ``pin_memory`` (bool): Pin memory for faster GPU transfer (default: True)
   
   **Reproducibility:**
   
   * ``seed`` (int): Random seed (default: 42)
   * ``deterministic`` (bool): Use deterministic mode (default: False)
   
   Example:
   
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
          clip_grad=1.0,
          early_stop_patience=50,
          
          # Model
          use_gmm_head=True,
          gmm_n_components=5,
          
          # Checkpointing
          save_dir="checkpoints/",
          save_freq=10,
          
          # Logging
          log_dir="logs/",
          
          # Hardware
          device='auto',
          num_workers=0,  # Set to 0 to avoid multiprocessing issues
          
          # Reproducibility
          seed=42
      )

Common Trainer Methods
----------------------

All trainer classes share the following interface:

``__init__(config)``
   Initialize trainer with configuration.
   
   :param TrainingConfig config: Training configuration

``train()``
   Start training loop.
   
   :return: None
   
   This method will:
   
   1. Load training and validation data
   2. Initialize model and optimizer
   3. Run training loop with progress bars
   4. Save checkpoints periodically
   5. Apply early stopping if validation loss doesn't improve
   6. Log metrics to TensorBoard

``from_args(**kwargs)``
   Class method to create trainer from keyword arguments.
   
   :param kwargs: Arguments to pass to TrainingConfig
   :return: Trainer instance
   
   Example:
   
   .. code-block:: python
   
      trainer = StandardCATETrainer.from_args(
          data_path="data/*.csv",
          epochs=100,
          batch_size=16,
          save_dir="checkpoints/"
      )

Training Workflow
-----------------

Basic Training
~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.training import StandardCATETrainer, TrainingConfig
   
   # Configure
   config = TrainingConfig(
       data_path="data/train/*.csv",
       epochs=100,
       save_dir="checkpoints/"
   )
   
   # Train (wrap in if __name__ == '__main__' for multiprocessing)
   if __name__ == '__main__':
       trainer = StandardCATETrainer(config)
       trainer.train()

Training with Custom Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TrainingConfig(
       # Data
       data_path="data/train/*.csv",
       val_split=0.15,
       batch_size=32,
       num_workers=8,
       
       # Optimization
       learning_rate=0.0005,
       weight_decay=1e-4,
       clip_grad=0.5,
       
       # Early stopping
       early_stop_patience=30,
       
       # Model
       use_gmm_head=True,
       gmm_n_components=10,
       
       # Hardware
       device='cuda:0',
       
       # Checkpointing
       save_dir="checkpoints/custom/",
       save_freq=5,
       
       # Logging
       log_dir="logs/custom/"
   )
   
   if __name__ == '__main__':
       trainer = StandardCATETrainer(config)
       trainer.train()

Resume Training
~~~~~~~~~~~~~~~

.. code-block:: python

   config = TrainingConfig(
       data_path="data/train/*.csv",
       save_dir="checkpoints/",
       resume_from="checkpoints/checkpoint_epoch_50.pth"
   )
   
   if __name__ == '__main__':
       trainer = StandardCATETrainer(config)
       trainer.train()  # Continues from epoch 51

Monitoring Training
-------------------

TensorBoard
~~~~~~~~~~~

Training automatically logs to TensorBoard:

.. code-block:: bash

   tensorboard --logdir logs/

Tracked metrics:

* Training loss (per epoch)
* Validation loss (per epoch)
* Learning rate (if using scheduler)
* Gradient norms
* Model statistics

Progress Output
~~~~~~~~~~~~~~~

During training, you'll see:

.. code-block:: text

   Epoch 1/100
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | Loss: 1.234
   Train Loss: 1.2345 | Val Loss: 1.3456
   
   Epoch 2/100
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | Loss: 1.123
   Train Loss: 1.1234 | Val Loss: 1.2345
   ✓ New best model saved!

Training Output
~~~~~~~~~~~~~~~

Access training history:

.. code-block:: python

   if __name__ == '__main__':
       trainer = StandardCATETrainer(config)
       trainer.train()
       
       # After training
       print(f"Final train loss: {trainer.train_losses[-1]:.4f}")
       print(f"Final val loss: {trainer.val_losses[-1]:.4f}")
       print(f"Best val loss: {min(trainer.val_losses):.4f}")
       print(f"Best epoch: {trainer.best_epoch}")

Checkpointing
-------------

Automatic Checkpointing
~~~~~~~~~~~~~~~~~~~~~~~

Checkpoints are saved automatically:

.. code-block:: python

   config = TrainingConfig(
       save_dir="checkpoints/",
       save_freq=10,           # Save every 10 epochs
       save_best_only=False    # Save all checkpoints
   )

This creates:

.. code-block:: text

   checkpoints/
   ├── best_model.pth              # Best model by validation loss
   ├── checkpoint_epoch_10.pth
   ├── checkpoint_epoch_20.pth
   └── ...

Checkpoint Contents
~~~~~~~~~~~~~~~~~~~

Each checkpoint includes:

.. code-block:: python

   checkpoint = {
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'epoch': epoch,
       'train_loss': train_loss,
       'val_loss': val_loss,
       'config': config
   }

Load Checkpoint
~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from causalfm.models import StandardCATEModel
   
   # Load checkpoint
   checkpoint = torch.load("checkpoints/best_model.pth")
   
   # Load model
   model = StandardCATEModel()
   model.load_state_dict(checkpoint['model_state_dict'])
   
   # Or use from_pretrained
   model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")

Best Practices
--------------

Recommended Settings
~~~~~~~~~~~~~~~~~~~~

For most use cases:

.. code-block:: python

   config = TrainingConfig(
       # Data
       data_path="data/train/*.csv",
       val_split=0.2,
       batch_size=16,
       num_workers=4,  # Or 0 if multiprocessing issues
       
       # Training
       epochs=150,
       learning_rate=0.001,
       weight_decay=1e-5,
       clip_grad=1.0,
       early_stop_patience=50,
       
       # Model
       use_gmm_head=True,
       gmm_n_components=5,
       
       # Checkpointing
       save_dir="checkpoints/",
       save_freq=10,
       
       # Logging
       log_dir="logs/",
       
       # Hardware
       device='auto',
       
       # Reproducibility
       seed=42
   )

Multiprocessing
~~~~~~~~~~~~~~~

Always wrap training code in ``if __name__ == '__main__':`` guard:

.. code-block:: python

   if __name__ == '__main__':
       trainer = StandardCATETrainer(config)
       trainer.train()

Or disable multiprocessing:

.. code-block:: python

   config = TrainingConfig(
       data_path="data/*.csv",
       num_workers=0  # No multiprocessing
   )

GPU Usage
~~~~~~~~~

For single GPU:

.. code-block:: python

   config = TrainingConfig(
       data_path="data/*.csv",
       device='cuda',     # Or 'cuda:0'
       batch_size=16
   )

For multi-GPU (DataParallel):

.. code-block:: python

   config = TrainingConfig(
       data_path="data/*.csv",
       device='cuda',     # Uses all available GPUs
       batch_size=64      # Increase for multi-GPU
   )

For CPU:

.. code-block:: python

   config = TrainingConfig(
       data_path="data/*.csv",
       device='cpu'
   )

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**CUDA Out of Memory:**

.. code-block:: python

   # Reduce batch size
   config = TrainingConfig(
       batch_size=8,  # Smaller batches
       num_workers=0
   )

**Multiprocessing Errors:**

.. code-block:: python

   # Use if __name__ == '__main__':
   if __name__ == '__main__':
       trainer.train()
   
   # Or disable multiprocessing
   config = TrainingConfig(num_workers=0)

**Not Converging:**

.. code-block:: python

   # Lower learning rate
   config = TrainingConfig(
       learning_rate=0.0001,
       epochs=200,
       clip_grad=0.5
   )

**Overfitting:**

.. code-block:: python

   # Increase regularization
   config = TrainingConfig(
       weight_decay=1e-3,
       early_stop_patience=20,
       val_split=0.3
   )

See Also
--------

* :doc:`../user_guide/training` - Detailed training guide
* :doc:`models` - Model API reference
* :doc:`../examples/standard_cate` - Complete training example

