import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_mnist_loaders
from models.lenet import LeNet
from models.lenet_dropout import LeNetDropout

def train_model(model, train_loader, device, epochs=10, lr=0.001, weight_decay=0):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(train_loader):.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_mnist_loaders(batch_size=32)

    # --- Dropout model ---
    print("\nTraining LeNet with Dropout...")
    model = LeNetDropout().to(device)
    train_model(model, train_loader, device)
    torch.save(model.state_dict(), "saved_models/lenet_dropout.pth")
    print("Saved lenet_dropout.pth")

    # --- Weight decay models ---
    wd_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    for wd in wd_values:
        print(f"\nTraining LeNet with weight_decay={wd}...")
        model = LeNet().to(device)
        train_model(model, train_loader, device, weight_decay=wd)
        name = f"lenet_wd_{wd:.0e}"
        torch.save(model.state_dict(), f"saved_models/{name}.pth")
        print(f"Saved {name}.pth")

if __name__ == "__main__":
    main()
