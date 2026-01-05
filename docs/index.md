# CausalFM Toolkit

Welcome to **CausalFM Toolkit** - A comprehensive PyTorch framework for training foundation models for causal inference.

## What is CausalFM?

CausalFM implements **Prior-Data Fitted Networks (PFNs)** for causal inference, enabling zero-shot transfer to new datasets without fine-tuning. Unlike traditional methods that require training on individual datasets, CausalFM learns from distributions of synthetic datasets and can immediately adapt to new data in-context.

## Key Features

✨ **Multiple Causal Settings**
- Standard CATE (Conditional Average Treatment Effect) estimation  
- Instrumental Variables (IV) with binary/continuous instruments  
- Front-door adjustment with mediators  

🎯 **Foundation Model Architecture**
- Built on TabPFN (Prior-Data Fitted Networks)  
- Transformer-based with GMM prediction heads  
- Calibrated uncertainty quantification  

📦 **Clean Library Interface**
- Simple API for data generation, training, and evaluation  
- Pre-built synthetic data generators  
- Comprehensive evaluation metrics (PEHE, ATE error, etc.)  

🚀 **Production Ready**
- Model checkpointing and loading  
- TensorBoard integration  
- Efficient PyTorch DataLoaders  

## Quick Example

```python
from causalfm.data import StandardCATEGenerator
from causalfm.models import StandardCATEModel
from causalfm.evaluation import compute_pehe

# Generate synthetic data
generator = StandardCATEGenerator(num_samples=1000, num_features=10)
df = generator.generate()

# Load pretrained model
model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")

# Estimate CATE
result = model.estimate_cate(x_train, a_train, y_train, x_test)
cate = result['cate']

# Evaluate
pehe = compute_pehe(cate, true_ite)
print(f"PEHE: {pehe:.4f}")
```

## Documentation Structure

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/data_generation
user_guide/models
user_guide/training
user_guide/evaluation
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples/standard_cate
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
api/data
api/models
api/training
api/evaluation
```

```{toctree}
:maxdepth: 1
:caption: Additional Information

citation
license
```

## Installation

```bash
git clone https://github.com/yccm/CausalFM.git
cd CausalFM-toolkit
pip install -r requirements.txt
```

See the {doc}`installation` guide for detailed instructions.

## Citation

If you use CausalFM in your research, please cite:

```bibtex
@article{ma2025causalfm,
  title={Foundation Models for Causal Inference via Prior-Data Fitted Networks},
  author={Ma, Yuchen and Frauen, Dennis and Javurek, Emil and Feuerriegel, Stefan},
  journal={arXiv preprint arXiv:2506.10914},
  year={2025}
}
```

## License

CausalFM Toolkit is released under the Apache License 2.0. See {doc}`license` for details.

## Community

- **GitHub:** https://github.com/yccm/CausalFM  
- **Paper:** https://arxiv.org/abs/2506.10914  
- **Documentation:** https://causalfm.readthedocs.io  

## Indices and Tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
