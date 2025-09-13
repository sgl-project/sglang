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
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Copy the application code
COPY .hathora_build/app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install SGLang - adjust this line based on your SGLang installation method
# For development, you might want to install from source or a specific version
RUN pip install --no-cache-dir sglang[all]

COPY .hathora_build/app/* .
RUN chmod +x ./entrypoint.sh

# Expose the port the app runs on
EXPOSE 8000

# Use entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
