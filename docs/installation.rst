Installation
============

Requirements
------------

CausalFM Toolkit requires:

* Python 3.9 or higher
* PyTorch 2.0 or higher
* CUDA-capable GPU (optional, but recommended for training)

Python Dependencies
~~~~~~~~~~~~~~~~~~~

The main dependencies include:

* ``torch>=2.0.0`` - Deep learning framework
* ``numpy>=1.24.0`` - Numerical computing
* ``pandas>=2.0.0`` - Data manipulation
* ``networkx>=3.0`` - Graph operations for DAG construction
* ``scipy>=1.10.0`` - Scientific computing
* ``scikit-learn>=1.3.0`` - Machine learning utilities
* ``tqdm`` - Progress bars
* ``tensorboard`` - Training visualization

Install from Source
-------------------

Step 1: Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/yccm/CausalFM.git
   cd CausalFM-toolkit

Step 2: Create Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using Conda (Recommended):

.. code-block:: bash

   conda create -n causalfm python=3.10
   conda activate causalfm

Using venv:

.. code-block:: bash

   python -m venv causalfm_env
   source causalfm_env/bin/activate  # On Windows: causalfm_env\Scripts\activate

Step 3: Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt

GPU Support
-----------

For GPU acceleration, ensure you have:

1. CUDA Toolkit installed (version 11.8 or higher recommended)
2. cuDNN library
3. PyTorch with CUDA support

To install PyTorch with CUDA 11.8:

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

For other CUDA versions, visit: https://pytorch.org/get-started/locally/

Verify Installation
-------------------

Run the following to verify your installation:

.. code-block:: python

   import causalfm
   import torch
   
   print(f"CausalFM version: {causalfm.__version__}")
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   
   # Test basic import
   from causalfm.data import StandardCATEGenerator
   from causalfm.models import StandardCATEModel
   from causalfm.training import StandardCATETrainer
   from causalfm.evaluation import compute_pehe
   
   print("✅ All imports successful!")

Test Data Generation
~~~~~~~~~~~~~~~~~~~~

Quick test to ensure data generation works:

.. code-block:: python

   from causalfm.data import StandardCATEGenerator
   import numpy as np
   
   # Generate a small test dataset
   generator = StandardCATEGenerator(num_samples=100, num_features=5, seed=42)
   df = generator.generate()
   
   print(f"Generated dataset shape: {df.shape}")
   print(f"Columns: {list(df.columns)}")
   print("✅ Data generation working!")

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

If you encounter ``ModuleNotFoundError`` for ``tabpfn``:

The ``src/tabpfn`` directory is part of this repository and should be automatically 
accessible. Ensure you're running Python from the repository root directory.

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

If you encounter CUDA out of memory errors during training:

* Reduce batch size in ``TrainingConfig``
* Use ``num_workers=0`` to disable multiprocessing
* Consider using CPU for small experiments

.. code-block:: python

   config = TrainingConfig(
       batch_size=8,  # Reduce from default 16
       num_workers=0,
       device='cpu'   # Or 'cuda:0' for GPU
   )

Multiprocessing Issues (macOS/Windows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you see ``RuntimeError`` about multiprocessing, wrap your training code in:

.. code-block:: python

   if __name__ == '__main__':
       # Your training code here
       trainer.train()

Or set ``num_workers=0`` in ``TrainingConfig``.

Development Installation
------------------------

For development and contributing:

.. code-block:: bash

   git clone https://github.com/yccm/CausalFM.git
   cd CausalFM-toolkit
   pip install -e .  # Editable install
   pip install -r requirements-dev.txt  # Development dependencies

This allows you to modify the source code and see changes immediately.

Next Steps
----------

After installation, check out the :doc:`quickstart` guide to learn how to use CausalFM Toolkit.
