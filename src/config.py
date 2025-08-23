import torch

# General
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EPOCHS = 40
LEARNING_RATE = 0.001

# Paths
DATA_PATH = "./data"
INTEL_PATH = "./data/intel-image-classification"
MODEL_PATH = "./models"
SAVED_WEIGHTS = "./models/saved_weights"

