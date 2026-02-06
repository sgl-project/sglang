FROM anyscale/ray:2.53.0-py312-cu129

# =============================================================================
# System Dependencies
# =============================================================================
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        libnuma1 \
        libnuma-dev \
        numactl \
        git \
        curl \
        wget \
        netcat \
    && sudo rm -rf /var/lib/apt/lists/*

# =============================================================================
# CUDA Toolkit (nvcc compiler) - CUDA 12.9 to match base image
# =============================================================================
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb && \
    sudo dpkg -i /tmp/cuda-keyring.deb && \
    rm /tmp/cuda-keyring.deb && \
    sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
        cuda-nvcc-12-9 \
        cuda-cudart-dev-12-9 \
        cuda-crt-12-9 \
    && sudo rm -rf /var/lib/apt/lists/*

# Create/update CUDA symlink
RUN sudo rm -rf /usr/local/cuda && \
    sudo ln -s /usr/local/cuda-12.9 /usr/local/cuda

# =============================================================================
# Environment Variables
# =============================================================================
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# =============================================================================
# Python Dependencies
# =============================================================================
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/ray/.local/bin:${PATH}"

RUN uv pip install --system \
    "sglang[all]==0.5.8" \
    numpy \
    transformers \
    accelerate \
    huggingface_hub \
    requests \
    httpx

RUN uv pip install --system "sgl-kernel==0.3.21"

# =============================================================================
# Verification
# =============================================================================
RUN nvcc --version && \
    python -c "import sglang; print(f'SGLang version: {sglang.__version__}')"

WORKDIR /home/ray/default
