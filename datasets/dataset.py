# datasets/dataset.py

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

class DatasetFactory:
    """Factory class for creating dataset loaders with proper augmentation and normalization"""
    
    DATASET_STATS = {
        'cifar10': {
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2470, 0.2435, 0.2616),
            'classes': 10
        },
        'cifar100': {
            'mean': (0.5071, 0.4867, 0.4408),
            'std': (0.2675, 0.2565, 0.2761),
            'classes': 100
        }
    }
    
    @staticmethod
    def get_transforms(dataset_name, train=True):
        """Get appropriate transforms for the dataset"""
        stats = DatasetFactory.DATASET_STATS[dataset_name]
        
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(stats['mean'], stats['std']),
                transforms.RandomErasing(p=0.2)
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(stats['mean'], stats['std'])
            ])

def get_data_loaders(
    dataset_name='cifar10',
    data_dir='./data',
    batch_size=64,
    num_workers=2,
    val_split=0.1,
    seed=42
):
    """
    Creates and returns PyTorch DataLoaders for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset ('cifar10' or 'cifar100')
        data_dir (str): Directory where dataset will be downloaded
        batch_size (int): Batch size for training and validation loaders
        num_workers (int): Number of subprocesses for data loading
        val_split (float): Fraction of training data to use for validation
        seed (int): Random seed for reproducibility

    Returns:
        train_loader, val_loader, test_loader (DataLoader, DataLoader, DataLoader)
    """
    if dataset_name not in DatasetFactory.DATASET_STATS:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(DatasetFactory.DATASET_STATS.keys())}")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Create transforms
    train_transform = DatasetFactory.get_transforms(dataset_name, train=True)
    test_transform = DatasetFactory.get_transforms(dataset_name, train=False)
    
    # Load dataset
    dataset_class = getattr(datasets, dataset_name.upper())
    
    # Load training data and split into train/val
    full_train_dataset = dataset_class(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Override transform for validation set
    val_dataset.dataset.transform = test_transform
    
    # Load test dataset
    test_dataset = dataset_class(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
