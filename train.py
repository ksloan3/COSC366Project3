import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_mnist_loaders
from models.lenet import LeNet

def compute_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

def train_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load MNIST
    train_loader, test_loader = get_mnist_loaders(batch_size=32)

    # Initialize model
    model = LeNet().to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            train_acc = compute_accuracy(model, train_loader, device)
            test_acc = compute_accuracy(model, test_loader, device)

            # Compute test loss
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    test_loss += criterion(outputs, y).item()

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            train_loss_list.append(running_loss / len(train_loader))
            test_loss_list.append(test_loss / len(test_loader))

            print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    # Save model
    torch.save(model.state_dict(), "saved_models/lenet_baseline.pth")

    # Save metrics
    torch.save({
        "train_acc": train_acc_list,
        "test_acc": test_acc_list,
        "train_loss": train_loss_list,
        "test_loss": test_loss_list
    }, "saved_models/baseline_metrics.pt")

    print("Baseline training complete. Model + metrics saved.")
if __name__ == "__main__":
    train_baseline()
