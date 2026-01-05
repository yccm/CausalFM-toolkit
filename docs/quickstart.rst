Quick Start
===========

Basic Usage
-----------

Standard Causal Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from causalfm.models.standard import StandardCausalModel
   from causalfm.data.loaders.standard import StandardDataLoader
   
   # Load your data
   data_loader = StandardDataLoader(data_path="path/to/data.csv")
   
   # Initialize model
   model = StandardCausalModel()
   
   # Train model
   model.train(data_loader)
   
   # Estimate causal effects
   effects = model.predict(test_data)

Instrumental Variable Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from causalfm.models.iv import IVModel
   from causalfm.data.loaders.iv import IVDataLoader
   
   # Load IV data
   data_loader = IVDataLoader(data_path="path/to/iv_data.csv")
   
   # Initialize IV model
   model = IVModel()
   
   # Estimate effects
   effects = model.estimate(data_loader)

Front-door Adjustment
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from causalfm.models.frontdoor import FrontdoorModel
   from causalfm.data.loaders.frontdoor import FrontdoorDataLoader
   
   # Load front-door data
   data_loader = FrontdoorDataLoader(data_path="path/to/frontdoor_data.csv")
   
   # Initialize model
   model = FrontdoorModel()
   
   # Estimate effects
   effects = model.estimate(data_loader)

Examples
--------

Check out the `evaluation/notebook/` directory for detailed examples:

* ``test_standard_cate.ipynb`` - Standard CATE estimation
* ``test_iv_binary.ipynb`` - Binary IV methods
* ``test_iv_conti.ipynb`` - Continuous IV methods
* ``test_fd.ipynb`` - Front-door adjustment
* ``test_jobs.ipynb`` - Real-world dataset example

