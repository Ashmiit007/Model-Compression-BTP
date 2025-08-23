import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from .config import BATCH_SIZE, INTEL_PATH, DATA_PATH

def get_transforms():
    return transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

def get_intel_loaders(batch_size=BATCH_SIZE):
    transform = get_transforms()
    trainset = datasets.ImageFolder(root=f"{INTEL_PATH}/seg_train/seg_train", transform=transform)
    testset = datasets.ImageFolder(root=f"{INTEL_PATH}/seg_test/seg_test", transform=transform)

    return DataLoader(trainset, batch_size=batch_size, shuffle=True), \
           DataLoader(testset, batch_size=batch_size, shuffle=True), \
           trainset.classes

def get_cifar10_loaders(batch_size=BATCH_SIZE):
    transform = get_transforms()
    trainset = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
    return DataLoader(trainset, batch_size=batch_size, shuffle=True), \
           DataLoader(testset, batch_size=batch_size, shuffle=True), \
           trainset.classes
