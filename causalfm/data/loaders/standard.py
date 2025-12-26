"""
Data loaders for Standard CATE estimation.
"""

import glob
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class StandardCausalDataset(Dataset):
    """
    PyTorch Dataset for Standard CATE estimation data.
    
    Each sample is a complete CSV file (sequence of observations).
    
    Example:
        >>> dataset = StandardCausalDataset("data/*.csv")
        >>> sample = dataset[0]
        >>> print(sample['X'].shape)  # (seq_len, num_features)
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
            
        print(f"Found {len(self.csv_files)} CSV files to load")
            
        self.sequences = []
        for file in self.csv_files:
            df = pd.read_csv(file)
            
            x_cols = [col for col in df.columns if col.startswith('x')]
            
            sequence = {
                'X': torch.FloatTensor(df[x_cols].values),
                'a': torch.FloatTensor(df['treatment'].values).unsqueeze(1),
                'y': torch.FloatTensor(df['outcome'].values).unsqueeze(1),
                'y0': torch.FloatTensor(df['y0'].values).unsqueeze(1),
                'y1': torch.FloatTensor(df['y1'].values).unsqueeze(1),
                'ite': torch.FloatTensor(df['ite'].values).unsqueeze(1),
            }
            
            self.sequences.append(sequence)
        
        print(f"Total sequences: {len(self.sequences)}")
        if len(self.sequences) > 0:
            print(f"Sequence length: {len(self.sequences[0]['X'])}")
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sequences[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


def standard_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for Standard CATE data.
    
    Stacks sequences into format: (seq_len, batch_size, features)
    """
    return {
        'X': torch.stack([item['X'] for item in batch], dim=1),
        'a': torch.stack([item['a'] for item in batch], dim=1),
        'y': torch.stack([item['y'] for item in batch], dim=1),
        'y0': torch.stack([item['y0'] for item in batch], dim=1),
        'y1': torch.stack([item['y1'] for item in batch], dim=1),
        'ite': torch.stack([item['ite'] for item in batch], dim=1),
    }


def create_standard_dataloader(
    data_path: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    shuffle: bool = True,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train/val data loaders for Standard CATE estimation.
    
    Args:
        data_path: Glob pattern for CSV files
        batch_size: Number of sequences per batch
        val_split: Fraction of data for validation
        shuffle: Whether to shuffle training data
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Example:
        >>> train_loader, val_loader, _ = create_standard_dataloader(
        ...     "data/standard/*.csv",
        ...     batch_size=16,
        ...     val_split=0.2
        ... )
    """
    dataset = StandardCausalDataset(data_path)
    
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
        collate_fn=standard_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=standard_collate_fn
    )
    
    return train_loader, val_loader, None

