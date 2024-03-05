import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import numpy as np
import time

def evaluate_accuracy(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # No need to track gradients for evaluation
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)  # Move data to the correct device
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()

    accuracy = 100 * total_correct / total_samples
    return accuracy


def train_resnet(model_name, dataset_name, num_epochs, batch_size, learning_rate, use_pretrained=False, data_percentage=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using {model_name} with {dataset_name} dataset with batch size {batch_size}. Pretrained")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.2154, 0.2024))
    ])

    DatasetClass = getattr(torchvision.datasets, dataset_name)
    train_dataset_full = DatasetClass(root='./data', train=True, download=True, transform=transform)
    val_dataset = DatasetClass(root='./data', train=False, download=True, transform=transform)

    if data_percentage != 1.0:
        subset_indices = np.random.choice(len(train_dataset_full), int(len(train_dataset_full) * data_percentage), replace=False)
        train_dataset = Subset(train_dataset_full, subset_indices)
    else:
        train_dataset = train_dataset_full

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ModelClass = getattr(torchvision.models, model_name)
    model = ModelClass(pretrained=use_pretrained)
    num_classes = 10 if dataset_name == 'CIFAR10' else 100
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_accuracy = evaluate_accuracy(model, val_loader, device)
        print(f"Epoch {epoch+1} - Test Accuracy: {test_accuracy:.2f}%")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")

ARGS = [
    {'model_name': 'resnet50', 'dataset_name': 'CIFAR10', 'num_epochs': 15, 'batch_size': 128, 'learning_rate': 0.001, 'use_pretrained': False, 'data_percentage': 1.0},
    {'model_name': 'resnet50', 'dataset_name': 'CIFAR10', 'num_epochs': 15, 'batch_size': 128, 'learning_rate': 0.001, 'use_pretrained': False, 'data_percentage': 0.5},
    {'model_name': 'resnet50', 'dataset_name': 'CIFAR10', 'num_epochs': 15, 'batch_size': 128, 'learning_rate': 0.001, 'use_pretrained': False, 'data_percentage': 0.25}
    {'model_name': 'resnet50', 'dataset_name': 'CIFAR10', 'num_epochs': 15, 'batch_size': 128, 'learning_rate': 0.001, 'use_pretrained': True, 'data_percentage': 1.0},
    {'model_name': 'resnet50', 'dataset_name': 'CIFAR10', 'num_epochs': 15, 'batch_size': 128, 'learning_rate': 0.001, 'use_pretrained': True, 'data_percentage': 0.5},
    {'model_name': 'resnet50', 'dataset_name': 'CIFAR10', 'num_epochs': 15, 'batch_size': 128, 'learning_rate': 0.001, 'use_pretrained': True, 'data_percentage': 0.25}
    {'model_name': 'resnet50', 'dataset_name': 'CIFAR100', 'num_epochs': 15, 'batch_size': 128, 'learning_rate': 0.001, 'use_pretrained': False, 'data_percentage': 1.0},
    {'model_name': 'resnet50', 'dataset_name': 'CIFAR100', 'num_epochs': 15, 'batch_size': 128, 'learning_rate': 0.001, 'use_pretrained': False, 'data_percentage': 0.5},
    {'model_name': 'resnet50', 'dataset_name': 'CIFAR100', 'num_epochs': 15, 'batch_size': 128, 'learning_rate': 0.001, 'use_pretrained': False, 'data_percentage': 0.25}
    {'model_name': 'resnet50', 'dataset_name': 'CIFAR100', 'num_epochs': 15, 'batch_size': 128, 'learning_rate': 0.001, 'use_pretrained': True, 'data_percentage': 1.0},
    {'model_name': 'resnet50', 'dataset_name': 'CIFAR100', 'num_epochs': 15, 'batch_size': 128, 'learning_rate': 0.001, 'use_pretrained': True, 'data_percentage': 0.5},
    {'model_name': 'resnet50', 'dataset_name': 'CIFAR100', 'num_epochs': 15, 'batch_size': 128, 'learning_rate': 0.001, 'use_pretrained': True, 'data_percentage': 0.25
]


for arg in ARGS:
    train_resnet(**arg)
    print()  # Print a new line between experiments