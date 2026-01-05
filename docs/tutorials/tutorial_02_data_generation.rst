Tutorial 2: Data Generation
===========================

Learn how to generate synthetic causal datasets for training and evaluation.

Coming Soon
-----------

This tutorial is under development. For now, see:

* :doc:`../user_guide/data_generation` - Complete data generation guide
* :doc:`tutorial_01_basics` - Basic concepts
* :doc:`../examples/standard_cate` - Complete example

Quick Reference
---------------

Standard CATE Data
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.data import StandardCATEGenerator
   
   generator = StandardCATEGenerator(
       num_samples=1024,
       num_features=10,
       seed=42
   )
   
   # Single dataset
   df = generator.generate()
   
   # Multiple datasets
   generator.generate_multiple(100, "data/train/")

IV Data
~~~~~~~

.. code-block:: python

   from causalfm.data import IVDataGenerator
   
   generator = IVDataGenerator(
       num_samples=1024,
       num_features=10,
       instrument_type='binary',
       seed=42
   )
   
   df = generator.generate()

Front-door Data
~~~~~~~~~~~~~~~

.. code-block:: python

   from causalfm.data import FrontdoorDataGenerator
   
   generator = FrontdoorDataGenerator(
       num_samples=1024,
       num_features=10,
       num_confounders=5,
       seed=42
   )
   
   df = generator.generate()

Next Tutorial
-------------

Continue to :doc:`tutorial_03_training` to learn about model training.

