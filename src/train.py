
import torch
import torch.nn as nn
from torch.optim import SGD
from .config import LEARNING_RATE, DEVICE

def train_model(model, loader, num_epochs, lr=LEARNING_RATE):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    model.to(DEVICE)
    model.train()

    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 25 == 0:
                acc = (outputs.argmax(1) == labels).float().mean().item() * 100
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{len(loader)}] "
                      f"Loss: {loss:.4f}, Acc: {acc:.2f}%")
        print()
