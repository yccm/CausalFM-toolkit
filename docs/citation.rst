Citation
========

If you use CausalFM Toolkit in your research, please cite our paper:

BibTeX
------

.. code-block:: bibtex

   @article{ma2025causalfm,
     title={Foundation Models for Causal Inference via Prior-Data Fitted Networks},
     author={Ma, Yuchen and Frauen, Dennis and Javurek, Emil and Feuerriegel, Stefan},
     journal={arXiv preprint arXiv:2506.10914},
     year={2025}
   }

Paper Information
-----------------

**Title:** Foundation Models for Causal Inference via Prior-Data Fitted Networks

**Authors:** Yuchen Ma, Dennis Frauen, Emil Javurek, Stefan Feuerriegel

**Year:** 2025

**arXiv:** https://arxiv.org/abs/2506.10914

**Abstract:**

We introduce CausalFM, a comprehensive framework for training foundation models 
for causal inference using Prior-Data Fitted Networks (PFNs). Unlike traditional 
approaches that require training on individual datasets, CausalFM learns from 
distributions of synthetic datasets, enabling zero-shot transfer to new datasets 
without fine-tuning. We develop foundation models for multiple causal inference 
settings, including standard CATE estimation, instrumental variables, and 
front-door adjustment. Our experimental results demonstrate that CausalFM achieves 
state-of-the-art performance across diverse causal inference tasks while providing 
calibrated uncertainty quantification through Gaussian Mixture Model prediction heads.

Acknowledgments
---------------

This work was supported by [funding information if applicable].

We thank the contributors and users of the CausalFM Toolkit for their valuable 
feedback and contributions.

Related Work
------------

CausalFM builds upon several key ideas from the literature:

**Prior-Data Fitted Networks (PFNs):**

* Müller, S., Hollmann, N., Arango, S. P., Grabocka, J., & Hutter, F. (2022). 
  Transformers can do bayesian inference. *ICLR*.

**Causal Inference:**

* Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and 
  nonrandomized studies. *Journal of Educational Psychology*.

* Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). 
  Cambridge University Press.

**Heterogeneous Treatment Effects:**

* Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous 
  treatment effects using random forests. *Journal of the American Statistical Association*.

* Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for 
  estimating heterogeneous treatment effects using machine learning. 
  *Proceedings of the National Academy of Sciences*.

License
-------

CausalFM Toolkit is released under the Apache License 2.0. See the LICENSE file 
for more details.

Contact
-------

For questions, issues, or contributions, please:

* Open an issue on GitHub: https://github.com/yccm/CausalFM
* Contact the authors via email: [contact information]

Contributing
------------

We welcome contributions to CausalFM Toolkit! Please see our contributing guidelines 
on GitHub for more information on how to contribute code, documentation, or bug reports.

**Ways to Contribute:**

* Report bugs and issues
* Suggest new features
* Improve documentation
* Submit pull requests with bug fixes or enhancements
* Share your use cases and applications

Community
---------

Join our community to stay updated:

* **GitHub:** https://github.com/yccm/CausalFM
* **Documentation:** https://causalfm.readthedocs.io
* **Paper:** https://arxiv.org/abs/2506.10914

We appreciate any feedback and contributions from the community!

