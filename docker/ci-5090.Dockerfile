# CI Docker image for 5090 (Blackwell) runners
# Provides CUDA 12.9 with sm_120 support and pre-installed dependencies

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG FLASHINFER_VERSION=0.5.3
ARG SGL_KERNEL_VERSION=0.3.20

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    FLASHINFER_VERSION=${FLASHINFER_VERSION}

# Add CUDA paths
ENV PATH="${PATH}:/usr/local/cuda/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget curl git \
    build-essential cmake ninja-build \
    libopenmpi-dev libnuma-dev \
    libibverbs-dev libibverbs1 ibverbs-providers \
    libssl-dev pkg-config \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch and core dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Install sgl-kernel
RUN pip install sgl-kernel==${SGL_KERNEL_VERSION}

# Install flashinfer with JIT cache
RUN pip install flashinfer-jit-cache==${FLASHINFER_VERSION} --index-url https://flashinfer.ai/whl/cu129

# Pre-download flashinfer cubins to warm up JIT cache
RUN python3 -c "import flashinfer; flashinfer.jit.download_cubins()" || true

# Install other dependencies
RUN pip install \
    transformers accelerate \
    sentencepiece tokenizers \
    aiohttp requests httpx \
    numpy scipy pandas \
    pytest pytest-asyncio \
    uvicorn fastapi pydantic \
    pillow opencv-python-headless \
    tiktoken einops \
    triton \
    packaging setuptools-scm \
    mooncake-transfer-engine==0.3.8.post1 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-nvshmem-cu12==3.4.5 \
    nvidia-cudnn-cu12==9.16.0.29 \
    py-spy scipy huggingface_hub pytest

# Create workspace
WORKDIR /workspace

# Verify installation
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" && \
    python3 -c "import sgl_kernel; print(f'sgl-kernel installed')" && \
    python3 -c "import flashinfer; print(f'flashinfer installed')"

CMD ["/bin/bash"]
