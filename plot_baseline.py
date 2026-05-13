import torch
import matplotlib.pyplot as plt
import os

# Load metrics
metrics = torch.load("saved_models/baseline_metrics.pt")

train_acc = metrics["train_acc"]
test_acc = metrics["test_acc"]
train_loss = metrics["train_loss"]
test_loss = metrics["test_loss"]

# Ensure output directory exists
os.makedirs("reports/part1_plots", exist_ok=True)

# ---- Accuracy Plot ----
plt.figure(figsize=(8, 5))
plt.plot([5, 10], train_acc, label="Train Accuracy", marker="o")
plt.plot([5, 10], test_acc, label="Test Accuracy", marker="o")
plt.title("Baseline LeNet Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("reports/part1_plots/baseline_accuracy.png")
plt.close()

# ---- Loss Plot ----
plt.figure(figsize=(8, 5))
plt.plot([5, 10], train_loss, label="Train Loss", marker="o")
plt.plot([5, 10], test_loss, label="Test Loss", marker="o")
plt.title("Baseline LeNet Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("reports/part1_plots/baseline_loss.png")
plt.close()

print("Saved baseline plots to reports/part1_plots/")
