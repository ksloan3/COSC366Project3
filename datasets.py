import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Normalize, RandomRotation, RandomHorizontalFlip, Compose
from torch.utils.data import DataLoader
def get_mnist_loaders(batch_size=32, rotation=False, hflip=False):
    transform_list = []
    if rotation:
        transform_list.append(RandomRotation(15))
    if hflip:
        transform_list.append(RandomHorizontalFlip())
    
    transform_list.extend([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
    ])


    transform = Compose(transform_list)
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader