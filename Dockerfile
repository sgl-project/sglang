# SGLang Hathora Deployment Dockerfile
# Use a slim Python image to keep the container size small
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install gcloud CLI
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg ca-certificates dnsutils libnuma1 numactl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install NVIDIA CUDA Toolkit 12.8 (for DeepGEMM JIT and CUDA tools)
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb && \
    apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-8 && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# Safe NCCL defaults and async error handling
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=INIT,ENV,SHM,P2P,NET \
    NCCL_P2P_LEVEL=SYS \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=1 \
    NCCL_SHM_DISABLE=1 \
    NCCL_BUFFSIZE=1048576 \
    NCCL_MIN_NCHANNELS=1 \
    NCCL_MAX_NCHANNELS=1

# Copy the application code
COPY .hathora_build/app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install SGLang from local source
COPY python /opt/sglang-src/python
RUN pip install --no-cache-dir -e /opt/sglang-src/python[all]


COPY .hathora_build/app/* .
RUN chmod +x ./entrypoint.sh

# Expose the port the app runs on
EXPOSE 8000

# Healthcheck for orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Use entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
