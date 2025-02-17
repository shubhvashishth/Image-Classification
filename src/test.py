"""
Author: @shubhamvashishth
Modified to accept test data and checkpoint as arguments

This script evaluates a trained ResNet model on the test dataset.
It takes the path to the test data and the model checkpoint as command-line arguments,
loads the model from the checkpoint, initializes the data module with the test data,
and then evaluates the model using PyTorch Lightning.
"""
import torch
import pytorch_lightning as pl
import argparse
from train import LitResNet, AnimalDataModule

def evaluate_test_set(test_data, checkpoint_path):
    
    model = LitResNet.load_from_checkpoint(checkpoint_path)
    
    
    data_module = AnimalDataModule(
        batch_size=32,
        data_dir=test_data  
    )
    
    
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger("logs/"),
        accelerator="auto"
    )
    
    results = trainer.test(model, datamodule=data_module)
    print(f"Test Accuracy: {results[0]['test_acc']:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--test_data', type=str, required=True,
                      help='Path to directory containing test data')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint file')
    
    args = parser.parse_args()
    
    evaluate_test_set(
        test_data=args.test_data,
        checkpoint_path=args.checkpoint
    )