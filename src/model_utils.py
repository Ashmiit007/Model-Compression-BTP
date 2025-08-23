
import torch
import torch.nn as nn
from torchprofile import profile_macs
from .config import DEVICE

def load_model(path):
    return torch.load(path, map_location=DEVICE)

def find_accuracy(loader, model, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total

def calculate_flops(model, input_shape, device=DEVICE):
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_shape).to(device)
        flops = profile_macs(model, dummy_input)
    model.train()
    return flops
