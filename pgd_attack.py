import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import get_mnist_loaders
from models.lenet import LeNet
from models.lenet_dropout import LeNetDropout
from torchvision.utils import save_image
import os

def PGD(x, y, model, loss_fn, niter=5, epsilon=0.3, stepsize=0.01, randinit=True):
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()

    if randinit:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach()

    for _ in range(niter):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(model(x_adv), y)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + stepsize * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def evaluate(model, loader, device, attack=False, **pgd_kwargs):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if attack:
            images = PGD(images, labels, model, loss_fn, **pgd_kwargs)
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total

def task3(device, test_loader):
    print("\n=== Task III: PGD on baseline LeNet ===")
    model = LeNet().to(device)
    model.load_state_dict(torch.load("saved_models/lenet_baseline.pth", map_location=device))
    clean = evaluate(model, test_loader, device, attack=False)
    robust = evaluate(model, test_loader, device, attack=True,
                      niter=5, epsilon=0.3, stepsize=0.01, randinit=True)
    print(f"Clean accuracy:  {clean:.4f}")
    print(f"Robust accuracy: {robust:.4f}")
    os.makedirs("reports/part3_plots", exist_ok=True)
    loss_fn = nn.CrossEntropyLoss()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        adv = PGD(x, y, model, loss_fn, niter=5, epsilon=0.3, stepsize=0.01, randinit=True)
        save_image(x[:8], "reports/part3_plots/clean.png")
        save_image(adv[:8], "reports/part3_plots/adv.png")
        save_image((adv - x)[:8] * 5 + 0.5, "reports/part3_plots/perturbation.png")
        break

def task4_1(device, test_loader):
    print("\n=== Task IV-1: Hyperparameter sweeps ===")
    model = LeNet().to(device)
    model.load_state_dict(torch.load("saved_models/lenet_baseline.pth", map_location=device))
    os.makedirs("reports/part4_plots", exist_ok=True)

    niter_values = [1, 2, 3, 4, 5, 10, 20, 30, 40, 80, 100]
    niter_accs = []
    for n in niter_values:
        acc = evaluate(model, test_loader, device, attack=True,
                       niter=n, epsilon=0.3, stepsize=0.01, randinit=True)
        print(f"  niter={n:3d}  robust_acc={acc:.4f}")
        niter_accs.append(acc)

    plt.figure()
    plt.plot(niter_values, niter_accs, marker='o')
    plt.xlabel("Number of PGD Iterations")
    plt.ylabel("Classification Accuracy")
    plt.title("Robust Accuracy vs. PGD Iterations (epsilon=0.3)")
    plt.grid(True)
    plt.savefig("reports/part4_plots/accuracy_vs_niter.png")
    plt.close()
    print("Saved accuracy_vs_niter.png")

    epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    eps_accs = []
    for eps in epsilon_values:
        acc = evaluate(model, test_loader, device, attack=True,
                       niter=5, epsilon=eps, stepsize=0.01, randinit=True)
        print(f"  epsilon={eps:.2f}  robust_acc={acc:.4f}")
        eps_accs.append(acc)

    plt.figure()
    plt.plot(epsilon_values, eps_accs, marker='o')
    plt.xlabel("Epsilon")
    plt.ylabel("Classification Accuracy")
    plt.title("Robust Accuracy vs. Epsilon (niter=5)")
    plt.grid(True)
    plt.savefig("reports/part4_plots/accuracy_vs_epsilon.png")
    plt.close()
    print("Saved accuracy_vs_epsilon.png")

def task4_2_augmentations(device, test_loader):
    print("\n=== Task IV-2: Augmentation models ===")
    models_to_test = [
        ("lenet_baseline",  "Baseline"),
        ("lenet_rotation",  "Rotation Aug"),
        ("lenet_hflip",     "HFlip Aug"),
        ("lenet_rot_hflip", "Rotation+HFlip Aug"),
    ]
    for filename, label in models_to_test:
        model = LeNet().to(device)
        model.load_state_dict(torch.load(f"saved_models/{filename}.pth", map_location=device))
        clean  = evaluate(model, test_loader, device, attack=False)
        robust = evaluate(model, test_loader, device, attack=True,
                          niter=5, epsilon=0.3, stepsize=0.01, randinit=True)
        print(f"  {label:25s}  clean={clean:.4f}  robust={robust:.4f}")

def task4_2_dropout(device, test_loader):
    print("\n=== Task IV-2: Dropout model ===")
    model = LeNetDropout().to(device)
    model.load_state_dict(torch.load("saved_models/lenet_dropout.pth", map_location=device))
    clean  = evaluate(model, test_loader, device, attack=False)
    robust = evaluate(model, test_loader, device, attack=True,
                      niter=5, epsilon=0.3, stepsize=0.01, randinit=True)
    print(f"  Dropout model:  clean={clean:.4f}  robust={robust:.4f}")
    baseline = LeNet().to(device)
    baseline.load_state_dict(torch.load("saved_models/lenet_baseline.pth", map_location=device))
    b_clean  = evaluate(baseline, test_loader, device, attack=False)
    b_robust = evaluate(baseline, test_loader, device, attack=True,
                        niter=5, epsilon=0.3, stepsize=0.01, randinit=True)
    print(f"  Baseline:       clean={b_clean:.4f}  robust={b_robust:.4f}")

def task4_2_weight_decay(device, test_loader):
    print("\n=== Task IV-2: Weight decay models ===")
    wd_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    results = []
    for wd in wd_values:
        model = LeNet().to(device)
        model.load_state_dict(torch.load(f"saved_models/lenet_wd_{wd:.0e}.pth", map_location=device))
        clean  = evaluate(model, test_loader, device, attack=False)
        robust = evaluate(model, test_loader, device, attack=True,
                          niter=5, epsilon=0.3, stepsize=0.01, randinit=True)
        drop = clean - robust
        print(f"  wd={wd:.0e}  clean={clean:.4f}  robust={robust:.4f}  drop={drop:.4f}")
        results.append((wd, clean, robust))

    os.makedirs("reports/part4_plots", exist_ok=True)
    wds    = [r[0] for r in results]
    cleans = [r[1] for r in results]
    robsts = [r[2] for r in results]
    plt.figure()
    plt.semilogx(wds, cleans, marker='o', label='Clean')
    plt.semilogx(wds, robsts, marker='s', label='Robust')
    plt.xlabel("Weight Decay")
    plt.ylabel("Classification Accuracy")
    plt.title("Clean vs. Robust Accuracy across Weight Decay Values")
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/part4_plots/accuracy_vs_weight_decay.png")
    plt.close()
    print("Saved accuracy_vs_weight_decay.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    _, test_loader = get_mnist_loaders(batch_size=64)
    task3(device, test_loader)
    task4_1(device, test_loader)
    task4_2_augmentations(device, test_loader)
    task4_2_dropout(device, test_loader)
    task4_2_weight_decay(device, test_loader)
