import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

def run_model(model_choice='resnet18', dataset_choice='CIFAR10', optimizer_choice='adam', 
              num_epochs=10, batch_size=128, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the chosen dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

   
    if dataset_choice == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        num_classes = 10
    elif dataset_choice == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                 download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform)
        num_classes = 100
    else:
        print("Invalid dataset choice. Please choose 'CIFAR10' or 'CIFAR100'.")
        return

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize the model
    if model_choice == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif model_choice == 'resnet50':
        model = models.resnet50(pretrained=False)
    else:
        print("Invalid model choice. Please choose 'resnet18' or 'resnet50'.")
        return
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # Define the optimizer
    if optimizer_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        print("Invalid optimizer choice. Please choose 'adam' or 'sgd'.")
        return

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Train the model
    start_time = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"Epoch {epoch+1}", ncols=100):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

    print('Finished Training')
    end_time = time.time()
    total_time = end_time - start_time

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
    print(f'Total training time: {total_time} seconds')

# Example usage:
model = ['resnet18', 'resnet50']
datasets = ['CIFAR10', 'CIFAR100']
lr = [0.01, 0.001]
optimizer = ['sgd', 'adam']
for i in range(len(model)):
    for j in range(len(datasets)):
        for k in range(len(optimizer)):
            for l in range(len(lr)):
                print(model[i], datasets[j], optimizer[k], lr[l])
                run_model(model_choice=model[i], dataset_choice=datasets[j], optimizer_choice=optimizer[k], num_epochs=20, batch_size=128, learning_rate=lr[l])
                print('###########')
    