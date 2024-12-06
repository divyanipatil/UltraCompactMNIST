import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


torch.manual_seed(10)


class UltraCompactCNN(nn.Module):
    def __init__(self):
        super(UltraCompactCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.05),

            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.05),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 16, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_data_loader(batch_size):
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=15,
            translate=(0.05, 0),
            scale=(0.95, 1.05)
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


    train_dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.MNIST("./data", train=False, transform=test_transform)


    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader


def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loader(batch_size=128)

    # Initialize model
    model = UltraCompactCNN().to(device)

    # Calculate and print parameter count
    total_params = count_parameters(model)
    print("\nParameter count breakdown:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()}")
    print(f"\nTotal trainable parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.2, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.28,
        steps_per_epoch=len(train_loader)*20,
        pct_start=0.06,
        epochs=1,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='cos'
    )

    # Training
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    test_accuracies = []

    for ep in range(20):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {running_loss / (batch_idx + ep*len(train_loader) + 1):.4f}, '
                      f'Accuracy: {100. * correct / total:.2f}%, '
                      f'LR: {scheduler.get_last_lr()[0]:.6f}')

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)

        test_acc = 100. * test_correct / test_total
        test_accuracies.append(test_acc)
        print(f'Epoch: {ep + 1}/20')
        print(f'Test Accuracy: {test_acc:.2f}%')
        if test_acc >= 99.4 and test_acc > max(test_accuracies):
            # Save model checkpoint after training
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_acc
            }

            # Create models directory if it doesn't exist
            if not os.path.exists('models'):
                os.makedirs('models')

            # Save the checkpoint
            torch.save(checkpoint, 'models/mnist_model_test_geq_99.4.pth')
            print(f"\nModel saved to models/mnist_model_test_geq_99.4.pth")

    final_accuracy = 100. * correct / total
    print(f'Final Training Accuracy: {final_accuracy:.2f}%')

    # Save model checkpoint after training
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_accuracy': final_accuracy,
    }

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the checkpoint
    torch.save(checkpoint, 'models/mnist_model.pth')
    print(f"\nModel saved to models/mnist_model.pth")

    return total_params, test_acc


if __name__ == "__main__":
    params, test_acc = train_model()

    # Save results for GitHub Actions
    with open("results.txt", "w") as f:
        f.write(f"Parameters: {params}\nTest Accuracy: {test_acc}")