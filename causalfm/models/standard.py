"""
Standard CATE Model wrapper.

Provides a clean interface for loading and using the CausalFM model
for Standard CATE estimation.
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

# Import the underlying model from tabpfn
from src.tabpfn.model.causalFM import PerFeatureTransformerCATE


class StandardCATEModel:
    """
    Standard CATE Estimation Model.
    
    A wrapper around the PerFeatureTransformerCATE model that provides
    a clean, easy-to-use interface for CATE estimation.
    
    Example:
        >>> # Load pretrained model
        >>> model = StandardCATEModel.from_pretrained("checkpoints/best_model.pth")
        >>> 
        >>> # Estimate CATE
        >>> result = model.estimate_cate(x_train, a_train, y_train, x_test)
        >>> cate = result['cate']
        >>> 
        >>> # Get full distribution (if using GMM head)
        >>> pi, mu, sigma = result['gmm_pi'], result['gmm_mu'], result['gmm_sigma']
    """
    
    def __init__(
        self,
        use_gmm_head: bool = True,
        gmm_n_components: int = 5,
        gmm_min_sigma: float = 1e-3,
        gmm_pi_temp: float = 1.0,
        device: Optional[str] = None
    ):
        """
        Initialize the Standard CATE model.
        
        Args:
            use_gmm_head: Whether to use GMM head for uncertainty estimation
            gmm_n_components: Number of GMM components
            gmm_min_sigma: Minimum sigma for GMM
            gmm_pi_temp: Temperature for GMM mixing weights
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        self.model = PerFeatureTransformerCATE(
            use_gmm_head=use_gmm_head,
            gmm_n_components=gmm_n_components,
            gmm_min_sigma=gmm_min_sigma,
            gmm_pi_temp=gmm_pi_temp
        )
        self.model.to(self.device)
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
        **kwargs
    ) -> 'StandardCATEModel':
        """
        Load a pretrained model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            device: Device to use
            **kwargs: Additional arguments passed to __init__
            
        Returns:
            Loaded StandardCATEModel instance
            
        Example:
            >>> model = StandardCATEModel.from_pretrained(
            ...     "checkpoints/best_model.pth",
            ...     device="cuda:0"
            ... )
        """
        instance = cls(device=device, **kwargs)
        
        checkpoint = torch.load(checkpoint_path, map_location=instance.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        instance.model.load_state_dict(state_dict)
        instance.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        return instance
    
    def estimate_cate(
        self,
        x_train: torch.Tensor,
        a_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate CATE for test samples given training data.
        
        This method uses the trained model to estimate the Conditional
        Average Treatment Effect (CATE) for new samples.
        
        Args:
            x_train: Training covariates, shape (n_train, n_features)
            a_train: Training treatments, shape (n_train,) or (n_train, 1)
            y_train: Training outcomes, shape (n_train,) or (n_train, 1)
            x_test: Test covariates, shape (n_test, n_features)
            
        Returns:
            Dictionary containing:
                - 'cate': Estimated CATE values, shape (n_test,)
                - 'gmm_pi': GMM mixing weights (if using GMM head)
                - 'gmm_mu': GMM means (if using GMM head)
                - 'gmm_sigma': GMM standard deviations (if using GMM head)
                
        Example:
            >>> result = model.estimate_cate(X_train, A_train, Y_train, X_test)
            >>> cate_estimates = result['cate']
            >>> print(f"Mean CATE: {cate_estimates.mean():.4f}")
        """
        self.model.eval()
        
        # Ensure tensors are on the correct device
        x_train = self._to_device(x_train)
        a_train = self._to_device(a_train)
        y_train = self._to_device(y_train)
        x_test = self._to_device(x_test)
        
        with torch.no_grad():
            return self.model.estimate_cate(x_train, a_train, y_train, x_test)
    
    def forward(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        y: torch.Tensor,
        single_eval_pos: int
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        This is useful for training or when you need raw model outputs.
        
        Args:
            x: Covariates, shape (seq_len, batch_size, n_features)
            a: Treatments, shape (seq_len, batch_size, 1)
            y: Outcomes, shape (seq_len, batch_size, 1)
            single_eval_pos: Position to split train/test in the sequence
            
        Returns:
            Dictionary with model outputs
        """
        return self.model(x, a, y, single_eval_pos)
    
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to model device."""
        return tensor.to(self.device)
    
    def train_mode(self) -> None:
        """Set model to training mode."""
        self.model.train()
    
    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()
    
    def save(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    @property
    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()
    
    def __repr__(self) -> str:
        return f"StandardCATEModel(device={self.device})"

