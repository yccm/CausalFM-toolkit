"""
CausalFM: Foundation Models for Causal Inference via Prior-Data Fitted Networks.

This package provides a unified framework for training foundation models
across multiple causal inference tasks:
- Standard CATE (Conditional Average Treatment Effect) estimation
- Instrumental Variables (IV) setting
- Front-door adjustment setting

Example usage:
    >>> import causalfm
    >>> 
    >>> # Data generation
    >>> from causalfm.data import StandardCATEGenerator
    >>> generator = StandardCATEGenerator(num_samples=1024, num_features=10)
    >>> data = generator.generate()
    >>> 
    >>> # Model loading and inference
    >>> from causalfm.models import StandardCATEModel
    >>> model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
    >>> cate = model.estimate_cate(x_train, a_train, y_train, x_test)
    >>> 
    >>> # Training
    >>> from causalfm.training import StandardCATETrainer
    >>> trainer = StandardCATETrainer(data_path="data/*.csv")
    >>> trainer.train(epochs=100)
"""

__version__ = "1.0.0"

# Import main components for convenient access
from causalfm.models import (
    StandardCATEModel,
    IVModel,
    FrontdoorModel,
)

from causalfm.data import (
    StandardCATEGenerator,
    IVDataGenerator,
    FrontdoorDataGenerator,
)

from causalfm.training import (
    StandardCATETrainer,
    IVTrainer,
    FrontdoorTrainer,
)

from causalfm.evaluation import (
    compute_pehe,
    compute_ate_error,
    evaluate_model,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "StandardCATEModel",
    "IVModel", 
    "FrontdoorModel",
    # Data Generators
    "StandardCATEGenerator",
    "IVDataGenerator",
    "FrontdoorDataGenerator",
    # Trainers
    "StandardCATETrainer",
    "IVTrainer",
    "FrontdoorTrainer",
    # Evaluation
    "compute_pehe",
    "compute_ate_error",
    "evaluate_model",
]

