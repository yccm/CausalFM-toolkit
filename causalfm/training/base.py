"""
Base Trainer class for CausalFM.

Provides common functionality for all trainer implementations.
"""

import os
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainingConfig:
    """
    Configuration for training.
    
    Example:
        >>> config = TrainingConfig(
        ...     epochs=100,
        ...     batch_size=16,
        ...     learning_rate=0.001
        ... )
    """
    # Data settings
    data_path: str = "data/*.csv"
    val_split: float = 0.2
    batch_size: int = 16
    num_workers: int = 4
    
    # Optimizer settings
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Training settings
    epochs: int = 150
    early_stop_patience: int = 50
    warmup_steps: int = 100
    clip_grad: float = 1.0
    
    # Model settings
    use_gmm_head: bool = True
    gmm_n_components: int = 5
    gmm_min_sigma: float = 1e-3
    gmm_pi_temp: float = 1.0
    
    # Logging and saving
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_freq: int = 10
    log_interval: int = 10
    
    # Device settings
    device: str = "auto"  # 'cuda', 'cpu', or 'auto'
    gpu_id: int = 0
    
    # Random seed
    seed: int = 42


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


class BaseTrainer(ABC):
    """
    Abstract base class for CausalFM trainers.
    
    Provides common functionality like training loops, logging,
    checkpointing, and early stopping.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Set random seed
        set_seed(config.seed)
        
        # Set device
        if config.device == "auto":
            self.device = torch.device(
                f"cuda:{config.gpu_id}" 
                if torch.cuda.is_available() 
                else "cpu"
            )
        else:
            self.device = torch.device(config.device)
        
        # Initialize model and optimizers
        self.model = self._create_model()
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Create data loaders
        self.train_loader, self.val_loader, _ = self._create_dataloaders()
        
        # Training tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_metrics: List[float] = []
        self.best_val_loss = float('inf')
        self.epoch_times: List[float] = []
        self.total_training_time = 0.0
        
        # Numerical stability
        self.eps = 1e-12
        
        # Create directories
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _create_dataloaders(self) -> Tuple:
        """Create data loaders. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _train_step(self, batch: Dict) -> Tuple[float, Dict]:
        """Single training step. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _val_step(self, batch: Dict) -> Tuple[float, Dict]:
        """Single validation step. Must be implemented by subclasses."""
        pass
    
    def gmm_nll_loss(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood for GMM.
        
        Args:
            pi: Mixture weights
            mu: Means
            sigma: Standard deviations
            target: Target values
            
        Returns:
            NLL loss value
        """
        if target.dim() == 2:
            target = target.squeeze(-1)

        pi = torch.clamp(pi, min=self.eps)
        pi = pi / pi.sum(dim=-1, keepdim=True)
        sigma = torch.clamp(sigma, min=self.eps)

        log_norm_const = 0.5 * np.log(2.0 * np.pi)
        z = (target - mu) / sigma
        log_prob = -0.5 * z * z - torch.log(sigma) - log_norm_const

        log_mix = torch.logsumexp(torch.log(pi) + log_prob, dim=-1)

        return -log_mix.mean()
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        epoch_start = time.time()
        
        for batch in self.train_loader:
            loss, _ = self._train_step(batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.config.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.clip_grad
                )
            
            self.optimizer.step()
            total_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        self.epoch_times.append(epoch_time)
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Run validation.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, primary metric value)
        """
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                loss, metrics = self._val_step(batch)
                total_loss += loss.item()
                total_metric += metrics.get('rmse', 0.0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_metric = total_metric / len(self.val_loader)
        
        self.val_losses.append(avg_loss)
        self.val_metrics.append(avg_metric)
        self.scheduler.step(avg_loss)
        
        return avg_loss, avg_metric
    
    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Current validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'training_time': self.total_training_time,
            'epoch_times': self.epoch_times,
        }
        
        # Save periodic checkpoint
        path = os.path.join(self.config.save_dir, f'model_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.config.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved (val_loss: {val_loss:.4f})")
    
    def train(self) -> None:
        """
        Run the full training loop.
        
        Example:
            >>> trainer = StandardCATETrainer(config)
            >>> trainer.train()
        """
        training_start = time.time()
        
        print(f"Starting training on device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        
        best_val_loss = float('inf')
        early_stop_count = 0
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metric = self.validate(epoch)
            
            # Log
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"RMSE: {val_metric:.4f}"
            )
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(epoch + 1, val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_count = 0
            else:
                early_stop_count += 1
            
            if early_stop_count >= self.config.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        self.total_training_time = time.time() - training_start
        
        # Save final checkpoint
        self.save_checkpoint(epoch + 1, val_loss)
        
        print(f"\nTraining complete!")
        print(f"Total time: {format_time(self.total_training_time)}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

