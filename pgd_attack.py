import torch
import torch.nn as nn
from torchvision.utils import save_image
from datasets import get_mnist_loaders
from models.lenet import LeNet
import os

# -----------------------------
# PGD ATTACK FUNCTION
# -----------------------------
def pgd_attack(model, images, labels, eps=0.3, alpha=0.01, iters=40):
    images = images.clone().detach()
    ori_images = images.clone().detach()

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()

        # Gradient sign step
        adv_images = images + alpha * images.grad.sign()

        # Projection step
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()

    return images


# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate(model, loader, device, attack=False):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if attack:
            x = pgd_attack(model, x, y)

        outputs = model(x)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total


# -----------------------------
# MAIN SCRIPT
# -----------------------------
def run_pgd_for_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning PGD attack on {model_name}...")

    # Load model
    model = LeNet().to(device)
    model.load_state_dict(torch.load(f"saved_models/{model_name}.pth", map_location=device))

    # Load data
    _, test_loader = get_mnist_loaders(batch_size=32)

    # Evaluate clean accuracy
    clean_acc = evaluate(model, test_loader, device, attack=False)

    # Evaluate PGD accuracy
    adv_acc = evaluate(model, test_loader, device, attack=True)

    print(f"Clean Accuracy: {clean_acc:.4f}")
    print(f"PGD Accuracy:   {adv_acc:.4f}")

    # Save a few adversarial examples
    os.makedirs("reports/part3_plots", exist_ok=True)

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        adv = pgd_attack(model, x, y)

        save_image(x[:8], f"reports/part3_plots/{model_name}_clean.png")
        save_image(adv[:8], f"reports/part3_plots/{model_name}_adv.png")
        save_image((adv - x)[:8] * 5 + 0.5, f"reports/part3_plots/{model_name}_perturbation.png")
        break  # only save one batch

    return clean_acc, adv_acc


if __name__ == "__main__":
    models = [
        "lenet_baseline",
        "lenet_rotation",
        "lenet_hflip",
        "lenet_rot_hflip"
    ]

    for m in models:
        run_pgd_for_model(m)
