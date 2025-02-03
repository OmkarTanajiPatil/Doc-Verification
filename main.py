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

def get_mean_and_std(loader):
    mean = 0
    std = 0
    total_image_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_image_count += image_count_in_a_batch
    mean /= total_image_count
    std /= total_image_count
    return mean, std

def set_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    device = set_device()
    model.to(device)
    best_acc = 0
    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}")
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)
            
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * running_correct / total
        print(f" - Training Accuracy: {epoch_acc:.2f}%. Loss: {epoch_loss:.4f}")
        
        if test_loader:
            test_acc = evaluate_model_on_test_set(model, test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint(model, epoch, optimizer, best_acc)
    
    print("Training Complete")
    return model

def evaluate_model_on_test_set(model, test_loader):
    if not test_loader:
        return 0  # Handle None case
    
    model.eval()
    device = set_device()
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f" - Test Accuracy: {acc:.2f}%")
    return acc

def save_checkpoint(model, epoch, optimizer, best_acc, filename='./models/model_best_checkpoint.pth.tar'):
    os.makedirs("./models", exist_ok=True)  # Ensure directory exists
    state = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'best_accuracy': best_acc,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filename)

def classify(model, image_transform, image_path, classes):
    model.eval()
    device = set_device()
    model.to(device)
    
    image = Image.open(image_path)
    image = image_transform(image).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    print(f"Predicted Class: {classes[predicted.item()]}")
    return classes[predicted.item()]

def add_new_class(model, train_dataset_path, new_class_path, new_class_name, train_transforms, test_loader, n_epochs=10):
    class_dir = os.path.join(train_dataset_path, new_class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    for img_file in os.listdir(new_class_path):
        shutil.copy(os.path.join(new_class_path, img_file), os.path.join(class_dir, img_file))
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    mean, std = get_mean_and_std(train_loader)
    print(f"Updated Mean: {mean}")
    print(f"Updated Std: {std}")
    
    train_transforms.transforms[-1] = transforms.Normalize(mean, std)
    train_dataset.transform = train_transforms
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model.to(set_device())
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.003)
    criterion = nn.CrossEntropyLoss()
    
    model = train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs)
    return model, train_loader

train_dataset_path = "./Dataset"
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

mean, std = get_mean_and_std(train_loader)
print(f"Mean: {mean}, Std: {std}")
train_transforms.transforms.append(transforms.Normalize(mean, std))
train_dataset.transform = train_transforms
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

resnet18_model = models.resnet18(pretrained=True)
resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, len(train_dataset.classes))
resnet18_model.to(set_device())

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.003)
trained_model = train_nn(resnet18_model, train_loader, None, loss_fn, optimizer, 10)

if os.path.exists("./models/model_best_checkpoint.pth.tar"):
    checkpoint = torch.load("./models/model_best_checkpoint.pth.tar")
    resnet18_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best Accuracy: {checkpoint['best_accuracy']}, Epoch: {checkpoint['epoch']}")

torch.save(resnet18_model, "./models/best_model.pth")

classes = train_dataset.classes
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist())
])

classify(torch.load('best_model.pth'), image_transform, "./Dataset/Aadhar_card/my small aadhar.jpg", classes)
