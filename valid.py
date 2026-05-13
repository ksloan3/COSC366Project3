import torch
from models.lenet import LeNet
from datasets import get_mnist_loaders

def validate(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = LeNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test data
    _, test_loader = get_mnist_loaders(batch_size=32)

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"Accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    # Change this to test any model you want
    validate("saved_models/lenet_baseline.pth")
