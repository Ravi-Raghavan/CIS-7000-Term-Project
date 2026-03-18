#!/bin/bash
# setup.sh — cloud environment setup for SPA-MSJF
# Tested on: AWS EC2, GCP Vertex AI, Lambda Labs, RunPod (Ubuntu)
# Usage: bash setup.sh

set -e
echo ""
echo "=== SPA-MSJF Cloud Environment Setup ==="

# --- detect CUDA version ---
WHEEL_URL="https://download.pytorch.org/whl/cpu"
CUDA_LABEL="cpu"

if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[\d.]+")
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    echo "Detected CUDA $CUDA_VERSION"

    if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
        WHEEL_URL="https://download.pytorch.org/whl/cu124"
        CUDA_LABEL="cu124"
    elif [ "$CUDA_MAJOR" -ge 12 ]; then
        WHEEL_URL="https://download.pytorch.org/whl/cu121"
        CUDA_LABEL="cu121"
    elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        WHEEL_URL="https://download.pytorch.org/whl/cu118"
        CUDA_LABEL="cu118"
    else
        echo "CUDA $CUDA_VERSION is older than 11.8 — falling back to CPU-only"
    fi
else
    echo "No GPU detected — installing CPU-only PyTorch"
fi

echo "Using wheel index: $WHEEL_URL"

# --- upgrade pip ---
python3 -m pip install --upgrade pip --quiet

# --- install PyTorch ---
echo ""
echo "Installing PyTorch ($CUDA_LABEL)..."
python3 -m pip install torch torchvision --index-url $WHEEL_URL --no-cache-dir

# --- install dependencies ---
echo ""
echo "Installing remaining dependencies..."
python3 -m pip install -r requirements.txt --no-cache-dir

# --- verify ---
echo ""
echo "=== Verification ==="
python3 -c "
import torch
cuda = torch.cuda.is_available()
gpu  = torch.cuda.get_device_name(0) if cuda else 'N/A'
print(f'PyTorch {torch.__version__}  |  CUDA: {cuda}  |  GPU: {gpu}')
"

echo ""
echo "Setup complete. Run training with:"
echo "  python3 train.py"
