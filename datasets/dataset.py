# datasets/dataset.py

import torch
from torchvision import datasets, transforms

def get_data_loaders(data_dir='./data', batch_size=64, num_workers=2):
    """
    Creates and returns PyTorch DataLoaders for CIFAR-10 dataset.

    Args:
        data_dir (str): Directory where CIFAR-10 will be downloaded.
        batch_size (int): Batch size for training and validation loaders.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        train_loader, val_loader (DataLoader, DataLoader)
    """

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
