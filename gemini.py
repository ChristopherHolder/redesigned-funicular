import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

def train_resnet(num_epochs, batch_size, learning_rate):
  """
  Trains a ResNet-18 model on the CIFAR-10 dataset and prints the final accuracy.

  Args:
      num_epochs (int): Number of training epochs.
      batch_size (int): Batch size for training.
      learning_rate (float): Learning rate for the optimizer.
  """

  # Define data transformations
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(32, padding=4),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.2154, 0.2024))
  ])

  # Load datasets
  train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

  # Create data loaders
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  # Define model
  model = torchvision.models.resnet18(pretrained=False)

  # Define loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # Train the model
  for epoch in range(num_epochs):
    # Training loop
    total_loss, total_correct = 0, 0
    with tqdm(train_loader, unit="batch") as pbar:
      for data, target in pbar:
        # Forward pass, calculate loss
        output = model(data)
        loss = criterion(output, target)

        # Update training metrics
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == target).sum().item()

        # Backward pass, optimize parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        pbar.set_postfix({"loss": f"{total_loss/(batch_size*len(train_loader)):.4f}"})

    # Calculate and print accuracy
    accuracy = 100 * total_correct / len(train_dataset)
    print(f"Epoch {epoch+1} - Accuracy: {accuracy:.2f}%")

  # Optional: Save the trained model

# Example usage
train_resnet(num_epochs=10, batch_size=64, learning_rate=0.001)
