import torch
import matplotlib.pyplot as plt
import os

# Load all metrics
baseline = torch.load("saved_models/baseline_metrics.pt")
rot = torch.load("saved_models/lenet_rotation_metrics.pt")
hflip = torch.load("saved_models/lenet_hflip_metrics.pt")
rot_hflip = torch.load("saved_models/lenet_rot_hflip_metrics.pt")

# Ensure output directory exists
os.makedirs("reports/part2_plots", exist_ok=True)

# Epochs where metrics were recorded
epochs = [5, 10]

# ------------------ ACCURACY PLOT ------------------
plt.figure(figsize=(10, 6))
plt.plot(epochs, baseline["test_acc"], marker="o", label="Baseline")
plt.plot(epochs, rot["test_acc"], marker="o", label="Rotation")
plt.plot(epochs, hflip["test_acc"], marker="o", label="HFlip")
plt.plot(epochs, rot_hflip["test_acc"], marker="o", label="Rotation + HFlip")

plt.title("Test Accuracy Comparison Across Models")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("reports/part2_plots/accuracy_comparison.png")
plt.close()

# ------------------ LOSS PLOT ------------------
plt.figure(figsize=(10, 6))
plt.plot(epochs, baseline["test_loss"], marker="o", label="Baseline")
plt.plot(epochs, rot["test_loss"], marker="o", label="Rotation")
plt.plot(epochs, hflip["test_loss"], marker="o", label="HFlip")
plt.plot(epochs, rot_hflip["test_loss"], marker="o", label="Rotation + HFlip")

plt.title("Test Loss Comparison Across Models")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("reports/part2_plots/loss_comparison.png")
plt.close()

print("Saved comparison plots to reports/part2_plots/")
