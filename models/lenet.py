import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)     # 1 → 6 channels
        self.pool = nn.MaxPool2d(2, 2)                  # 2×2 pooling
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)    # 6 → 16 channels

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)           # flatten → 120
        self.fc2 = nn.Linear(120, 84)                   # 120 → 84
        self.fc3 = nn.Linear(84, 10)                    # 84 → 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))            # conv1 → relu → pool
        x = self.pool(F.relu(self.conv2(x)))            # conv2 → relu → pool
        x = x.view(-1, 16 * 4 * 4)                      # flatten
        x = F.relu(self.fc1(x))                         # fc1 → relu
        x = F.relu(self.fc2(x))                         # fc2 → relu
        x = self.fc3(x)                                 # output logits
        return x
