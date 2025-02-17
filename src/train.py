"""
author : @shubhamvashishth
"""

"""
This script trains the modified ResNet model for the binary classification of cats vs. dogs.
It loads the data using the preprocessing pipeline, defines the model, loss function, optimizer,
and then runs the training loop with validation. Metrics are logged to TensorBoard and the best model
checkpoint is automatically saved.
"""

import os
import torch
import pytorch_lightning as pl
from torch import nn
from torchmetrics import Accuracy
from torch.optim import Adam
from model import ResNet18  
from preprocess import create_dataloader

class LitResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ResNet18()  
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="binary")  
        self.test_accuracy = Accuracy(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, y)  
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

class AnimalDataModule(pl.LightningDataModule):

    """
        Initializes the data module:
        - data_dir: Root directory containing 'train', 'val', and 'test' subdirectories.
        - batch_size: Number of samples per batch.
    """

    def __init__(self, data_dir="data", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def train_dataloader(self):
        return create_dataloader(self.data_dir, split="train", batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return create_dataloader(self.data_dir, split="val", batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return create_dataloader(self.data_dir, split="test", batch_size=self.batch_size, shuffle=False)

def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    model = LitResNet()
    data_module = AnimalDataModule()
  
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints/",
        filename="best_model-{epoch:02d}-{val_acc:.2f}",
    )
   
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("logs/"),
        accelerator="auto",
    )
   
    trainer.fit(model, datamodule=data_module)

    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()