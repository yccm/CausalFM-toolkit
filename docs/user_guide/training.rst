Training
========

CausalFM provides powerful training modules for all causal inference settings. 
Training is based on the PFN (Prior-Data Fitted Networks) paradigm where models 
learn from many synthetic datasets.

Training Overview
-----------------

Key Concepts
~~~~~~~~~~~~

**Prior-Data Fitted Networks (PFNs)**
   Instead of training on a single dataset, PFNs are trained on a distribution 
   of synthetic datasets. This enables:
   
   * Transfer to new datasets without fine-tuning
   * In-context learning capabilities
   * Robust performance across diverse settings

**Training Process**
   1. Generate many synthetic datasets (100-1000s)
   2. Train model to predict outcomes in-context
   3. Model learns to adapt to dataset characteristics
   4. Deploy on real data without retraining

**Loss Function**
   Models are trained using Gaussian Mixture Model (GMM) negative log-likelihood, 
   which encourages both accurate predictions and calibrated uncertainty.

Training Configuration
----------------------

All training is configured via ``TrainingConfig``:

.. code-block:: python

   from causalfm.training import TrainingConfig
   
   config = TrainingConfig(
       # Data
       data_path="data/train/*.csv",      # Path to training datasets
       val_split=0.2,                      # Validation split ratio
       
       # Training
       epochs=100,                         # Number of training epochs
       batch_size=16,                      # Batch size
       learning_rate=0.001,                # Learning rate
       weight_decay=1e-5,                  # L2 regularization
       
       # Optimization
       clip_grad=1.0,                      # Gradient clipping
       early_stop_patience=50,             # Early stopping patience
       
       # Model
       use_gmm_head=True,                  # Use GMM prediction head
       gmm_n_components=5,                 # Number of mixture components
       
       # Checkpointing
       save_dir="checkpoints/",            # Save directory
       save_freq=10,                       # Save every N epochs
       
       # Logging
       log_dir="logs/",                    # TensorBoard logs
       
       # Hardware
       device='auto',                      # 'auto', 'cuda', 'cpu'
       num_workers=4,                      # DataLoader workers
       
       # Reproducibility
       seed=42
   )

Standard CATE Training
----------------------

Training a Standard CATE Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.training import StandardCATETrainer, TrainingConfig
   from causalfm.data import StandardCATEGenerator
   
   # Step 1: Generate training data
   generator = StandardCATEGenerator(num_samples=1024, num_features=10)
   generator.generate_multiple(
       num_datasets=500,
       output_dir="data/train/"
   )
   
   # Step 2: Configure training
   config = TrainingConfig(
       data_path="data/train/*.csv",
       epochs=100,
       batch_size=16,
       save_dir="checkpoints/standard/"
   )
   
   # Step 3: Train (must be wrapped for multiprocessing)
   if __name__ == '__main__':
       trainer = StandardCATETrainer(config)
       trainer.train()

Training Output
~~~~~~~~~~~~~~~

During training, you'll see:

.. code-block:: text

   Epoch 1/100
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:15
   Train Loss: 1.2345 | Val Loss: 1.3456
   
   Epoch 2/100
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:14
   Train Loss: 1.1234 | Val Loss: 1.2345
   ✓ New best model saved!
   
   ...
   
   Early stopping triggered after epoch 75
   Best validation loss: 0.8765 at epoch 65
   Training completed!

Simplified Interface
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.training import StandardCATETrainer
   
   if __name__ == '__main__':
       # One-liner training
       trainer = StandardCATETrainer.from_args(
           data_path="data/train/*.csv",
           epochs=100,
           save_dir="checkpoints/"
       )
       trainer.train()

Instrumental Variables Training
--------------------------------

Training an IV Model
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.training import IVTrainer, TrainingConfig
   from causalfm.data import IVDataGenerator
   
   # Generate IV data
   generator = IVDataGenerator(
       num_samples=1024,
       num_features=10,
       instrument_type='binary'
   )
   generator.generate_multiple(
       num_datasets=500,
       output_dir="data/iv_train/"
   )
   
   # Train
   if __name__ == '__main__':
       config = TrainingConfig(
           data_path="data/iv_train/*.csv",
           epochs=100,
           save_dir="checkpoints/iv/"
       )
       trainer = IVTrainer(config)
       trainer.train()

Binary vs Continuous IV
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Binary IV (e.g., randomized encouragement)
   binary_gen = IVDataGenerator(
       num_samples=1024,
       num_features=10,
       instrument_type='binary'
   )
   binary_gen.generate_multiple(500, "data/iv_binary/")
   
   # Train binary IV model
   if __name__ == '__main__':
       trainer = IVTrainer.from_args(
           data_path="data/iv_binary/*.csv",
           save_dir="checkpoints/iv_binary/"
       )
       trainer.train()
   
   # Continuous IV
   conti_gen = IVDataGenerator(
       num_samples=1024,
       num_features=10,
       instrument_type='continuous'
   )
   conti_gen.generate_multiple(500, "data/iv_conti/")
   
   # Train continuous IV model
   if __name__ == '__main__':
       trainer = IVTrainer.from_args(
           data_path="data/iv_conti/*.csv",
           save_dir="checkpoints/iv_conti/"
       )
       trainer.train()

Front-door Training
-------------------

Training a Front-door Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.training import FrontdoorTrainer, TrainingConfig
   from causalfm.data import FrontdoorDataGenerator
   
   # Generate front-door data
   generator = FrontdoorDataGenerator(
       num_samples=1024,
       num_features=10
   )
   generator.generate_multiple(
       num_datasets=500,
       output_dir="data/frontdoor_train/"
   )
   
   # Train
   if __name__ == '__main__':
       config = TrainingConfig(
           data_path="data/frontdoor_train/*.csv",
           epochs=100,
           save_dir="checkpoints/frontdoor/"
       )
       trainer = FrontdoorTrainer(config)
       trainer.train()

Advanced Configuration
----------------------

Custom Training Settings
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = TrainingConfig(
       # Data
       data_path="data/train/*.csv",
       val_split=0.15,                     # 15% validation
       
       # Optimization
       learning_rate=0.0005,               # Lower learning rate
       weight_decay=1e-4,                  # Stronger regularization
       clip_grad=0.5,                      # Stricter gradient clipping
       
       # Scheduler (optional)
       use_scheduler=True,
       scheduler_patience=10,
       scheduler_factor=0.5,
       
       # Early stopping
       early_stop_patience=30,             # Stop after 30 epochs without improvement
       min_delta=1e-4,                     # Minimum improvement threshold
       
       # Batch size
       batch_size=32,                      # Larger batches
       
       # Model architecture
       use_gmm_head=True,
       gmm_n_components=10,                # More components for better uncertainty
       gmm_min_sigma=1e-4,
       gmm_pi_temp=0.8,
       
       # Checkpointing
       save_dir="checkpoints/custom/",
       save_freq=5,                        # Save every 5 epochs
       save_best_only=True,                # Only save best model
       
       # Logging
       log_dir="logs/custom/",
       log_freq=100,                       # Log every 100 steps
       
       # Hardware
       device='cuda:0',                    # Specific GPU
       num_workers=8,                      # More workers for data loading
       pin_memory=True,                    # Pin memory for faster GPU transfer
       
       # Reproducibility
       seed=42,
       deterministic=True                  # Deterministic mode (slower but reproducible)
   )

Multi-GPU Training
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use DataParallel for multi-GPU
   config = TrainingConfig(
       data_path="data/train/*.csv",
       device='cuda',                      # Will use all available GPUs
       batch_size=64,                      # Increase batch size for multi-GPU
       num_workers=16                      # More workers
   )
   
   if __name__ == '__main__':
       trainer = StandardCATETrainer(config)
       trainer.train()

Monitoring Training
-------------------

TensorBoard Integration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Training automatically logs to TensorBoard
   config = TrainingConfig(
       data_path="data/train/*.csv",
       log_dir="logs/experiment_1/"
   )
   
   if __name__ == '__main__':
       trainer = StandardCATETrainer(config)
       trainer.train()

View logs with:

.. code-block:: bash

   tensorboard --logdir logs/experiment_1/

Tracked metrics include:

* Training loss (per epoch)
* Validation loss (per epoch)
* Learning rate (if using scheduler)
* Gradient norms
* Model parameter statistics

Manual Logging
~~~~~~~~~~~~~~

.. code-block:: python

   # Access trainer's history
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

.. code-block:: python

   config = TrainingConfig(
       data_path="data/train/*.csv",
       save_dir="checkpoints/",
       save_freq=10,                       # Save every 10 epochs
       save_best_only=False                # Save all checkpoints
   )
   
   # This will create:
   # checkpoints/
   #   ├── best_model.pth         # Best model by validation loss
   #   ├── checkpoint_epoch_10.pth
   #   ├── checkpoint_epoch_20.pth
   #   └── ...

Resume Training
~~~~~~~~~~~~~~~

.. code-block:: python

   # Resume from checkpoint
   config = TrainingConfig(
       data_path="data/train/*.csv",
       save_dir="checkpoints/",
       resume_from="checkpoints/checkpoint_epoch_50.pth"
   )
   
   if __name__ == '__main__':
       trainer = StandardCATETrainer(config)
       trainer.train()  # Continues from epoch 51

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Multiprocessing Errors**

.. code-block:: python

   # ❌ Wrong - causes errors
   trainer = StandardCATETrainer(config)
   trainer.train()
   
   # ✅ Correct - wrap in if __name__ == '__main__':
   if __name__ == '__main__':
       trainer = StandardCATETrainer(config)
       trainer.train()
   
   # Or disable multiprocessing
   config = TrainingConfig(
       data_path="data/*.csv",
       num_workers=0  # No multiprocessing
   )

**CUDA Out of Memory**

.. code-block:: python

   # Reduce batch size
   config = TrainingConfig(
       data_path="data/*.csv",
       batch_size=8,    # Smaller batches
       num_workers=0
   )
   
   # Or use CPU
   config = TrainingConfig(
       data_path="data/*.csv",
       device='cpu'
   )

**Training Too Slow**

.. code-block:: python

   # Increase batch size
   config = TrainingConfig(
       data_path="data/*.csv",
       batch_size=32,           # Larger batches
       num_workers=8,           # More workers
       pin_memory=True          # Faster GPU transfer
   )

**Overfitting**

.. code-block:: python

   # Increase regularization
   config = TrainingConfig(
       data_path="data/*.csv",
       weight_decay=1e-3,       # Stronger L2
       early_stop_patience=20,  # Earlier stopping
       val_split=0.3            # More validation data
   )

**Not Converging**

.. code-block:: python

   # Adjust learning rate
   config = TrainingConfig(
       data_path="data/*.csv",
       learning_rate=0.0001,    # Lower LR
       clip_grad=0.5,           # Stricter clipping
       epochs=200               # More epochs
   )

Best Practices
--------------

Data Generation
~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate diverse training data
   generator = StandardCATEGenerator(num_samples=1024, num_features=10)
   
   # Use many datasets (500-1000 recommended)
   generator.generate_multiple(
       num_datasets=1000,
       output_dir="data/train/"
   )
   
   # Separate test data with different seed
   test_gen = StandardCATEGenerator(
       num_samples=1024,
       num_features=10,
       seed=999
   )
   test_gen.generate_multiple(100, "data/test/")

Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Recommended settings for most cases
   config = TrainingConfig(
       data_path="data/train/*.csv",
       
       # Training
       epochs=150,
       batch_size=16,
       learning_rate=0.001,
       weight_decay=1e-5,
       
       # Early stopping
       early_stop_patience=50,
       val_split=0.2,
       
       # Model
       use_gmm_head=True,
       gmm_n_components=5,
       
       # Checkpointing
       save_dir="checkpoints/",
       save_freq=10,
       
       # Hardware
       device='auto',
       num_workers=4,
       
       # Reproducibility
       seed=42
   )

Validation Strategy
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use held-out datasets for validation
   config = TrainingConfig(
       data_path="data/train/*.csv",
       val_split=0.2  # 20% for validation
   )
   
   # Or use separate validation files
   config = TrainingConfig(
       data_path="data/train/*.csv",
       val_path="data/val/*.csv"  # Explicit validation set
   )

API Reference
-------------

For complete API documentation, see:

* :class:`causalfm.training.StandardCATETrainer`
* :class:`causalfm.training.IVTrainer`
* :class:`causalfm.training.FrontdoorTrainer`
* :class:`causalfm.training.TrainingConfig`

