"""
author : @shubhamvashishth

"""
import torch
import pytorch_lightning as pl
from train import LitResNet, AnimalDataModule  # Reuse components

def evaluate_test_set(checkpoint_path):
    model = LitResNet.load_from_checkpoint(checkpoint_path)
    data_module = AnimalDataModule(batch_size=32)
    
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger("logs/"),  # Add logger
        accelerator="auto"
    )
    
    # Validate first to ensure metrics are initialized
    trainer.validate(model, datamodule=data_module)
    
    # Now test
    results = trainer.test(model, datamodule=data_module)
    print(f"Test Accuracy: {results[0]['test_acc']:.2%}")

if __name__ == "__main__":
    checkpoint_path = "checkpoints/best_model-epoch=03-val_acc=0.82.ckpt"  
    evaluate_test_set(checkpoint_path)