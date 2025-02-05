import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import shutil
from PIL import Image  

mean = [0.6675, 0.6692, 0.6803]
std = [0.1909, 0.1949, 0.2078]

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def set_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def classify(model, image_transform, image_path, classes, threshold=0.91):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    image = Image.open(image_path)
    image = image_transform(image).float()
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    if confidence.item() < threshold:
        print("Predicted Class: Unknown")
        return "Unknown"
    else:
        print(f"Predicted Class: {classes[predicted.item()]} (Confidence: {confidence.item():.2f})")
        return classes[predicted.item()]

train_dataset_path = "./Dataset"
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

def predict(img_path):
    return classify(model=torch.load("./models/best_model.pth"), image_transform=image_transform, image_path=img_path, classes = train_dataset.classes)