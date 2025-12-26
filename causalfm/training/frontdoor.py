"""
Trainer for Front-door adjustment setting.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from causalfm.training.base import BaseTrainer, TrainingConfig
from causalfm.data.loaders import create_frontdoor_dataloader
from src.tabpfn.model.causalFM4FD import PerFeatureTransformerCATE as PerFeatureTransformerFD


class FrontdoorTrainer(BaseTrainer):
    """
    Trainer for Front-door adjustment setting.
    
    Trains the PerFeatureTransformerFD model on front-door datasets.
    
    Example:
        >>> config = TrainingConfig(
        ...     data_path="data/frontdoor/*.csv",
        ...     epochs=100,
        ...     batch_size=16
        ... )
        >>> trainer = FrontdoorTrainer(config)
        >>> trainer.train()
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the Front-door trainer.
        
        Args:
            config: Training configuration
        """
        super().__init__(config)
    
    @classmethod
    def from_args(
        cls,
        data_path: str,
        epochs: int = 150,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        save_dir: str = "checkpoints",
        device: str = "auto",
        **kwargs
    ) -> 'FrontdoorTrainer':
        """
        Create trainer from simple arguments.
        
        Args:
            data_path: Path to training data (glob pattern)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            save_dir: Directory to save checkpoints
            device: Device to use
            **kwargs: Additional config options
            
        Returns:
            Configured trainer instance
        """
        config = TrainingConfig(
            data_path=data_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_dir=save_dir,
            device=device,
            **kwargs
        )
        return cls(config)
    
    def _create_model(self) -> nn.Module:
        """Create the Front-door model."""
        return PerFeatureTransformerFD(
            use_gmm_head=self.config.use_gmm_head,
            gmm_n_components=self.config.gmm_n_components,
            gmm_min_sigma=self.config.gmm_min_sigma,
            gmm_pi_temp=self.config.gmm_pi_temp
        )
    
    def _create_dataloaders(self) -> Tuple:
        """Create data loaders."""
        return create_frontdoor_dataloader(
            self.config.data_path,
            batch_size=self.config.batch_size,
            val_split=self.config.val_split,
            shuffle=True,
            num_workers=self.config.num_workers
        )
    
    def _train_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Single training step for Front-door.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        x = batch['X'].to(self.device)
        m = batch['m'].to(self.device)
        a = batch['a'].to(self.device)
        y = batch['y'].to(self.device)
        ite = batch['ite'].to(self.device)
        
        single_eval_pos = int(len(x) * 0.8)
        
        # Forward pass
        out = self.model(x, m, a, y, single_eval_pos)
        
        # Compute loss
        gmm_pi = out['gmm_pi']
        gmm_mu = out['gmm_mu']
        gmm_sigma = out['gmm_sigma']
        
        ite_bs1 = ite.permute(1, 0, 2)
        loss = self.gmm_nll_loss(gmm_pi, gmm_mu, gmm_sigma, ite_bs1)
        
        # Compute metrics
        cate = (gmm_pi * gmm_mu).sum(dim=-1).unsqueeze(-1)
        rmse = torch.sqrt(((cate - ite_bs1) ** 2).mean()).item()
        
        return loss, {'rmse': rmse}
    
    def _val_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Single validation step for Front-door.
        
        Args:
            batch: Batch of validation data
            
        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        x = batch['X'].to(self.device)
        m = batch['m'].to(self.device)
        a = batch['a'].to(self.device)
        y = batch['y'].to(self.device)
        ite = batch['ite'].to(self.device)
        
        single_eval_pos = int(len(x) * 0.8)
        
        # Forward pass
        out = self.model(x, m, a, y, single_eval_pos)
        
        # Compute loss
        gmm_pi = out['gmm_pi']
        gmm_mu = out['gmm_mu']
        gmm_sigma = out['gmm_sigma']
        
        ite_bs1 = ite.permute(1, 0, 2)
        loss = self.gmm_nll_loss(gmm_pi, gmm_mu, gmm_sigma, ite_bs1)
        
        # Compute metrics
        cate = (gmm_pi * gmm_mu).sum(dim=-1).unsqueeze(-1)
        rmse = torch.sqrt(((cate - ite_bs1) ** 2).mean()).item()
        
        return loss, {'rmse': rmse}

