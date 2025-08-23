
from src.datasets import get_intel_loaders
from src.model_utils import load_model, find_accuracy
from src.config import MODEL_PATH
from src.ppo_conv import ppo_conv
from src.ppo_fc import ppo_fc

if __name__ == "__main__":
    train_loader, test_loader, classes = get_intel_loaders()
    model = load_model(f"{MODEL_PATH}/base_alexnet_intel.pth")
    acc = find_accuracy(test_loader, model)
    print(f"Initial Accuracy: {acc:.2f}%")

    # Run PPO conv & FC iterations here
    # Save weights
