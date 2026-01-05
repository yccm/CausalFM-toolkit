"""
Data loaders for Instrumental Variables setting.
"""

import glob
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class IVCausalDataset(Dataset):
    """
    PyTorch Dataset for IV setting data.
    
    Each sample is a complete CSV file (sequence of observations).
    Includes observed covariates, unobserved confounders, instrument, and outcomes.
    """
    
    def __init__(
        self, 
        csv_files: Union[str, List[str]], 
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_files: List of CSV file paths or a glob pattern
            transform: Optional transform to apply to samples
        """
        self.transform = transform
        
        if isinstance(csv_files, str):
            self.csv_files = glob.glob(csv_files)
        else:
            self.csv_files = csv_files
            
        print(f"Found {len(self.csv_files)} IV CSV files to load")
            
        self.sequences = []
        for file in self.csv_files:
            df = pd.read_csv(file)
            
            x_cols = [col for col in df.columns if col.startswith('x')]
            u_cols = [col for col in df.columns if col.startswith('u')]
            
            sequence = {
                'X': torch.FloatTensor(df[x_cols].values),
                'U': torch.FloatTensor(df[u_cols].values) if u_cols else torch.empty(0),
                'z': torch.FloatTensor(df['z'].values).unsqueeze(1),
                'a': torch.FloatTensor(df['treatment'].values).unsqueeze(1),
                'y': torch.FloatTensor(df['outcome'].values).unsqueeze(1),
                'y0': torch.FloatTensor(df['y0'].values).unsqueeze(1),
                'y1': torch.FloatTensor(df['y1'].values).unsqueeze(1),
                'ite': torch.FloatTensor(df['ite'].values).unsqueeze(1),
            }
            
            self.sequences.append(sequence)
        
        print(f"Total IV sequences: {len(self.sequences)}")
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sequences[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


def iv_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for IV data.
    
    Stacks sequences into format: (seq_len, batch_size, features)
    """
    result = {
        'X': torch.stack([item['X'] for item in batch], dim=1),
        'z': torch.stack([item['z'] for item in batch], dim=1),
        'a': torch.stack([item['a'] for item in batch], dim=1),
        'y': torch.stack([item['y'] for item in batch], dim=1),
        'y0': torch.stack([item['y0'] for item in batch], dim=1),
        'y1': torch.stack([item['y1'] for item in batch], dim=1),
        'ite': torch.stack([item['ite'] for item in batch], dim=1),
    }
    
    # Handle U if present
    if batch[0]['U'].numel() > 0:
        result['U'] = torch.stack([item['U'] for item in batch], dim=1)
    
    return result


def create_iv_dataloader(
    data_path: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    shuffle: bool = True,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train/val data loaders for IV setting.
    
    Args:
        data_path: Glob pattern for CSV files
        batch_size: Number of sequences per batch
        val_split: Fraction of data for validation
        shuffle: Whether to shuffle training data
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Example:
        >>> train_loader, val_loader, _ = create_iv_dataloader(
        ...     "data/iv/*.csv",
        ...     batch_size=16
        ... )
    """
    dataset = IVCausalDataset(data_path)
    
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=iv_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=iv_collate_fn
    )
    
    return train_loader, val_loader, None


