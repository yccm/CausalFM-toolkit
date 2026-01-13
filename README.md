# CausalFM

PyTorch Implementation on Paper [Foundation Models for Causal Inference via Prior-Data Fitted Networks](https://arxiv.org/abs/2506.10914)

📖 **[Full Documentation](https://causalfm-toolkit.readthedocs.io/en/latest/index.html)** | 📄 **[Paper](https://arxiv.org/abs/2506.10914)** | 💻 **[GitHub](https://github.com/yccm/CausalFM-toolkit)**

## 📌 Introduction

In this paper, we introduce **CausalFM**, a comprehensive framework for training PFN-based foundation models in various causal inference settings.

CausalFM provides a **unified framework** for training foundation models across multiple causal inference tasks, including:  

- **Standard CATE estimation setting**  
- **Instrumental Variables (IV) setting**  
- **Front-door adjustment setting**  

This repository contains dataset generation pipelines, model implementations, and training/evaluation scripts.

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yccm/CausalFM-toolkit.git
cd CausalFM-toolkit
conda create -n causalfm python=3.10
conda activate causalfm
pip install -r requirements.txt
```

---

## 📚 Library Usage

CausalFM can be used as a library with a clean, intuitive API:

### Quick Start

```python
import causalfm

# Load a pretrained model
model = causalfm.StandardCATEModel.from_pretrained("checkpoints/best_model.pth")

# Estimate CATE for new samples
result = model.estimate_cate(x_train, a_train, y_train, x_test)
cate_estimates = result['cate']
```

### Data Generation

Generate synthetic datasets for training and evaluation:

```python
from causalfm.data import StandardCATEGenerator, IVDataGenerator, FrontdoorDataGenerator

# Standard CATE data
generator = StandardCATEGenerator(num_samples=1024, num_features=10, seed=42)
df = generator.generate()

# Generate multiple datasets
generator.generate_multiple(num_datasets=10, output_dir="data/standard/")

# Instrumental Variables data
iv_generator = IVDataGenerator(
    num_samples=1024,
    num_features=10,
    instrument_type='binary',  # or 'continuous'
    seed=42
)
iv_df = iv_generator.generate()

# Front-door adjustment data
fd_generator = FrontdoorDataGenerator(
    num_samples=1024,
    num_features=10,
    num_confounders=5,
    seed=42
)
fd_df = fd_generator.generate()
```

### Training

Train models using the Trainer classes:

```python
from causalfm.training import StandardCATETrainer, TrainingConfig

# Using configuration object
config = TrainingConfig(
    data_path="data/standard/*.csv",
    epochs=100,
    batch_size=16,
    learning_rate=0.001,
    save_dir="checkpoints/standard"
)
trainer = StandardCATETrainer(config)
trainer.train()

# Or use simplified interface
trainer = StandardCATETrainer.from_args(
    data_path="data/standard/*.csv",
    epochs=100,
    batch_size=16,
    save_dir="checkpoints/standard"
)
trainer.train()
```

### Model Loading and Inference

Load pretrained models and run inference:

```python
from causalfm.models import StandardCATEModel, IVModel, FrontdoorModel
import torch

# Standard CATE Model
model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")

# Prepare data
x_train = torch.randn(800, 10)  # Training covariates
a_train = torch.randint(0, 2, (800,)).float()  # Training treatments
y_train = torch.randn(800)  # Training outcomes
x_test = torch.randn(200, 10)  # Test covariates

# Estimate CATE
result = model.estimate_cate(x_train, a_train, y_train, x_test)
cate = result['cate']  # Shape: (200,)

# Access GMM distribution parameters (for uncertainty)
pi = result['gmm_pi']      # Mixture weights
mu = result['gmm_mu']      # Means
sigma = result['gmm_sigma'] # Standard deviations

# IV Model
iv_model = IVModel.from_pretrained("checkpoints/iv_model.pth")
result = iv_model.estimate_cate(x_train, z_train, a_train, y_train, x_test)

# Front-door Model
fd_model = FrontdoorModel.from_pretrained("checkpoints/fd_model.pth")
result = fd_model.estimate_cate(x_train, m_train, a_train, y_train, x_test)
```

### Evaluation

Evaluate models using standard metrics:

```python
from causalfm.evaluation import compute_pehe, compute_ate_error
from causalfm.data import normalize_data
import pandas as pd
import torch

# Load test data
df = pd.read_csv("data/test/test_dataset_1.csv")
x_cols = [c for c in df.columns if c.startswith('x')]

# Normalize data (important for consistency with training!)
X_norm, Y_norm, x_scaler, y_scaler = normalize_data(
    df[x_cols].values, 
    df['outcome'].values,
    df['y0'].values,
    df['y1'].values
)

# Prepare tensors
X = torch.FloatTensor(X_norm)
A = torch.FloatTensor(df['treatment'].values).unsqueeze(1)
Y = torch.FloatTensor(Y_norm).unsqueeze(1)

# Get normalized ITE for evaluation
from causalfm.data import normalize_ite
true_ite_norm, _ = normalize_ite(df['y0'].values, df['y1'].values, y_scaler)

# Split and evaluate
n_train = int(0.8 * len(X))
model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
result = model.estimate_cate(X[:n_train], A[:n_train], Y[:n_train], X[n_train:])

# Compute metrics
pehe = compute_pehe(result['cate'].cpu().numpy(), true_ite_norm[n_train:])
print(f"PEHE: {pehe:.4f}")
```

---

## 📊 Script-Based Usage

For backward compatibility, you can also use the original script-based approach:

### Data Generation

Standard CATE:
```bash
cd DATA_standard
python gen_standard_syn.py 
```

Instrumental Variables (IV):
```bash
cd DATA_IV
python gen_iv_data_binary.py  # Binary Instrument
python gen_iv_data_conti.py   # Continuous Instrument
```

Front-door adjustment:
```bash
cd DATA_FD
python gen_frontdoor.py
```

### Training

Standard CATE:
```bash
python src/tabpfn/train_standard/training_standard.py 
```

Instrumental Variables (IV):
```bash
python src/tabpfn/train_iv/training_iv_binary.py
python src/tabpfn/train_iv/training_iv_conti.py
```

Front-door adjustment:
```bash
python src/tabpfn/train_fd/training_fd.py
```

### Evaluation (Notebooks)

```
├── evaluation/notebook/
│   ├── test_fd.ipynb            # Front-door evaluation
│   ├── test_iv_binary.ipynb     # Binary IV evaluation
│   ├── test_iv_conti.ipynb      # Continuous IV evaluation
│   ├── test_jobs.ipynb          # Jobs dataset evaluation
│   └── test_standard_cate.ipynb # Standard CATE evaluation
```

---

## 📁 Project Structure

```
CausalFM-toolkit/
├── causalfm/                    # Main package (new library interface)
│   ├── __init__.py
│   ├── data/                    # Data generation and loading
│   │   ├── generators/          # Dataset generators
│   │   └── loaders/             # PyTorch data loaders
│   ├── models/                  # Model wrappers
│   │   ├── standard.py          # StandardCATEModel
│   │   ├── iv.py                # IVModel
│   │   └── frontdoor.py         # FrontdoorModel
│   ├── training/                # Training utilities
│   │   ├── base.py              # BaseTrainer
│   │   ├── standard.py          # StandardCATETrainer
│   │   ├── iv.py                # IVTrainer
│   │   └── frontdoor.py         # FrontdoorTrainer
│   └── evaluation/              # Evaluation metrics
│       └── metrics.py           # PEHE, ATE error, etc.
├── src/tabpfn/                  # Core TabPFN-based models
│   └── model/
│       ├── causalFM.py          # Standard CATE model
│       ├── causalFM4IV.py       # IV model
│       └── causalFM4FD.py       # Front-door model
├── DATA_standard/               # Standard CATE data
├── DATA_IV/                     # IV data
├── DATA_FD/                     # Front-door data
└── evaluation/notebook/         # Evaluation notebooks
```

---

## 📖 Documentation

For comprehensive guides, tutorials, and API reference, visit our documentation:

**👉 [https://causalfm-toolkit.readthedocs.io](https://causalfm-toolkit.readthedocs.io/en/latest/index.html)**

The documentation includes:
- **Installation Guide** - Detailed setup instructions
- **Quick Start** - Get started in 5 minutes
- **Tutorials** - Step-by-step learning path
- **User Guides** - In-depth coverage of all features
- **API Reference** - Complete API documentation
- **Examples** - Complete working examples

---

## 📖 Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{ma2025foundation,
  title={Foundation Models for Causal Inference via Prior-Data Fitted Networks},
  author={Ma, Yuchen and Frauen, Dennis and Javurek, Emil and Feuerriegel, Stefan},
  journal={arXiv preprint arXiv:2506.10914},
  year={2025}
}
```

---

## 🙏 Acknowledgement

This repo is based on the implementation of [TabPFN](https://github.com/PriorLabs/TabPFN/)
