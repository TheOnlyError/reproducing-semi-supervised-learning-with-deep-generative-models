import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import DataLoader

def load_data(batch_size):
    # transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    DATA_FOLDER = './data/MNIST'

    train_data = datasets.MNIST(
        root=DATA_FOLDER,
        train=True,
        download=True,
        transform=transform
    )
    val_data = datasets.MNIST(
        root=DATA_FOLDER,
        train=False,
        download=True,
        transform=transform
    )

    # training and validation data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader