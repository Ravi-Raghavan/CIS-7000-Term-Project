from build_data import build_dataloaders

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from sklearn.metrics import accuracy_score
from ts2vec import TS2Vec
import time

data_dir = "../Finance Data"

def get_device() -> torch.device:
    """CUDA (NVIDIA) → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

model = TS2Vec(input_dims = 5, device= device, output_dims = 320)

# ── Data ──────────────────────────────────────────────
# seq_len=60 gives the LSTM 2 months of context per sample (up from 30)
train_dl, _, _, _ = build_dataloaders(
    data_dir,
    seq_len=60,
    horizon=1,
    batch_size=32,
)

all_x = []

for x, targets in train_dl:
    # x: (B, S, T, D)
    B, S, T, D = x.shape
    
    x_reshaped = x.view(B * S, T, D)   # (B*S, T, D)
    all_x.append(x_reshaped)

# Concatenate along batch dimension
all_x = torch.cat(all_x, dim=0).cpu().numpy()  # (N, T, D)

# start_time = time.time()
# model.fit(all_x)
# end_time = time.time()

# # Save the Trained Model to a File
# model.save("trained_ts2vec.pth")
# print(f"Training Latency: {end_time - start_time}")

# Practice Inference
model.load("trained_ts2vec.pth")
test = model.encode(all_x)
print(type(test), test.shape)