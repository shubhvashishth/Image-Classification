"""
Author: @shubhamvashishth

This module provides a data preprocessing pipeline for the cat vs. dog image classification task.
It leverages PyTorch's torchvision.transforms to prepare the images by resizing, converting to grayscale,
normalizing, and optionally augmenting the data for improved generalization.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(train=True):
    """Returns transforms with augmentation for training, without for validation/test."""
    transform_list = [
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]
    
    if train:  # Add augmentation only for training
        transform_list.insert(1, transforms.RandomHorizontalFlip())
        
    return transforms.Compose(transform_list)

def create_dataloader(data_dir, split='train', batch_size=32, shuffle=True):
    """
    Creates a DataLoader for a specified data split (train, val, or test).
    
    Args:
        data_dir (str): The root directory of your dataset.
                        It should have subdirectories 'train', 'val', and 'test'.
        split (str): Which data split to load ('train', 'val', or 'test').
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset. Typically True for training.
    
    """

    split_dir = os.path.join(data_dir, split)
    
    dataset = datasets.ImageFolder(root=split_dir, transform=get_transforms())
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == '__main__':

    data_directory = 'data'
    
    
    train_loader = create_dataloader(data_directory, split='train', batch_size=32, shuffle=True)
    
    # just checking the pipeline by printing the shape of first batch
    for images, labels in train_loader:
        print("Batch image shape:", images.shape)  
        print("Batch labels:", labels)
        break
