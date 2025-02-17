"""
Author: @shubhamvashishth

This script performs inference using the trained ResNet model for binary classification (cats vs. dogs).
It accepts an image file and a model checkpoint as inputs, processes the image through the same preprocessing
pipeline used during training, and outputs the predicted class along with its confidence.
"""

import torch
from PIL import Image
import pytorch_lightning as pl
from train import LitResNet  
from preprocess import get_transforms

def predict(image_path: str, checkpoint_path: str) -> str:
    
    model = LitResNet.load_from_checkpoint(checkpoint_path)
    model.eval() 
    device = model.device  

    # Preprocess image and move to same device as model
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device) 

    # Predicting
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.softmax(output, dim=1)
    
    
    class_idx = torch.argmax(prob).item()
    confidence = prob[0][class_idx].item()
    return f"Prediction: {'Dog' if class_idx == 1 else 'Cat'} (Confidence: {confidence:.2%})"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    args = parser.parse_args()
    
    print(predict(args.image_path, args.checkpoint_path))