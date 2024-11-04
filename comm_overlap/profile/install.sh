#!/bin/bash

############# zhuohaol customized settings ############
# Set HOME
export HOME="/home/ubuntu"

# Add CUDA to PATH only if not already present
if [[ ":$PATH:" != *":/usr/local/cuda/bin:"* ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

# Set LD_LIBRARY_PATH for CUDA
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Add ~/.local/bin to PATH only if not already present
if [[ ":$PATH:" != *":/home/ubuntu/.local/bin:"* ]]; then
    export PATH="$PATH:/home/ubuntu/.local/bin"
fi

# Update system and install required packages
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y python3-pip vim

# Install Python packages
pip install torch torchvision torchaudio
pip install packaging
pip install cmake
pip install transformer_engine[pytorch]
