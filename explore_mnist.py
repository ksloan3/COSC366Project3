# explore_mnist.py
# This script just shows what MNIST will look like, not the training

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Download MNIST
print("Downloading MNIST dataset...")
mnist = datasets.MNIST(root='./data', train=True, download=True)

print(f"\nMNIST has {len(mnist)} training images")
print(f"Each image is {mnist[0][0].size}")

# Let's look at the first 10 images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('First 10 MNIST Images')

for i in range(10):
    image, label = mnist[i]
    ax = axes[i // 5, i % 5]
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_examples.png')
print("\nSaved 'mnist_examples.png' - go look at it!")
print("These are the handwritten digits your model will learn to recognize.")