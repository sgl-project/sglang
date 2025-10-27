FROM python:3.10-slim

WORKDIR /app

# System deps needed by runtime and healthcheck + NVIDIA CUDA Toolkit 12.8
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg ca-certificates dnsutils libnuma1 numactl wget ibutils \
    build-essential gcc g++ python3-dev pkg-config libssl-dev protobuf-compiler libprotobuf-dev && \
    wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb && \
    apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# Safe NCCL defaults and async error handling
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    NCCL_DEBUG=INFO \
    NCCL_SHM_DISABLE=0 \
    NCCL_P2P_DISABLE=0
# Let NCCL auto-tune: do NOT set MIN/MAX_NCHANNELS or BUFFSIZE
# Install deps, breaking this out to optimize cache hits
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY python/pyproject.toml /opt/sglang-src/python/pyproject.toml
RUN pip install --no-cache-dir tomli && \
    python -c "import tomli; d=tomli.load(open('/opt/sglang-src/python/pyproject.toml','rb')); reqs=list(d.get('project',{}).get('dependencies',[])); e=d.get('project',{}).get('optional-dependencies',{}); reqs+=e.get('test',[]); reqs+=e.get('decord',[]); print('\\n'.join(reqs))" > /tmp/sglang-reqs.txt && \
    pip install --no-cache-dir -r /tmp/sglang-reqs.txt

COPY .hathora_build/app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

COPY python /opt/sglang-src/python
RUN pip install --no-cache-dir -e /opt/sglang-src/python[all]

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY sgl-router /opt/sglang-router
RUN cd /opt/sglang-router && pip install --no-cache-dir -e .

COPY .hathora_build/app /app/
RUN chmod +x /app/entrypoint.sh
RUN chmod +x /app/entrypoint_sglang_native.sh
RUN chmod +x /app/kimi_k2/preset.sh

EXPOSE 8000

# Healthcheck for orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Use entrypoint script
ENTRYPOINT ["./entrypoint.sh"]