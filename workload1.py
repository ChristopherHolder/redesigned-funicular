import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Check if GPU is available and use it; otherwise, fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data augmentation and normalization for training
# Just normalization for validation
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# Load CIFAR-10 and CIFAR-100 datasets
train_dataset_10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform['train'])
test_dataset_10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform['val'])

train_loader_10 = DataLoader(train_dataset_10, batch_size=64, shuffle=True)
test_loader_10 = DataLoader(test_dataset_10, batch_size=64)

train_dataset_100 = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform['train'])
test_dataset_100 = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform['val'])

train_loader_100 = DataLoader(train_dataset_100, batch_size=64, shuffle=True)
test_loader_100 = DataLoader(test_dataset_100, batch_size=64)

# Adjusting ResNet18 to create a smaller ResNet-like model for demonstration
# Normally, you'd customize the layers to create a true ResNet-10 architecture
def resnet10(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained, **kwargs)
    # Customize the model to fit your requirements
    return model

model_10 = resnet10(pretrained=False).to(device)
model_100 = resnet10(pretrained=False).to(device)

criterion = nn.CrossEntropyLoss()
optimizer_10 = optim.SGD(model_10.parameters(), lr=0.001, momentum=0.9)
optimizer_100 = optim.SGD(model_100.parameters(), lr=0.001, momentum=0.9)

# Train function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, is_inception=False):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)

        print(f'Epoch {epoch}/{num_epochs - 1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# Example training call for CIFAR-10
dataloaders_10 = {'train': train_loader_10, 'val': test_loader_10}
train_model(model_10, dataloaders_10, criterion, optimizer_10, num_epochs=10)

# Repeat the training process for CIFAR-100 by using model_100, train_loader_100, and test_loader_100
