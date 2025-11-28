FROM ubuntu:24.04
SHELL ["/bin/bash", "-c"]

ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG VER_SGLANG=main

ARG VER_TORCH=2.9.0
ARG VER_TORCHVISION=0.24.0
ARG VER_TRITON=3.5.0

RUN apt-get update && \
    apt-get full-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    git \
    curl \
    wget \
    vim \
    gcc \
    g++ \
    make \
    libsqlite3-dev \
    google-perftools \
    libtbb-dev \
    libnuma-dev \
    numactl

WORKDIR /opt

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12
source .venv/bin/activate

RUN cat > .venv/uv.toml << 'EOF'
[[index]]
name = "torch"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "torchvision"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "triton"
url = "https://download.pytorch.org/whl/cpu"
EOF

ENV UV_CONFIG_FILE=/opt/.venv/uv.toml

WORKDIR /sgl-workspace
RUN git clone ${SGLANG_REPO} sglang && \
    cd sglang && \
    git checkout ${VER_SGLANG} && \
    cd python && \
    cp pyproject_cpu.toml pyproject.toml && \
    uv pip install . && \
    uv pip install torch==${VER_TORCH} torchvision==${VER_TORCHVISION} triton==${VER_TRITON} --force-reinstall && \
    cd ../sgl-kernel && \
    cp pyproject_cpu.toml pyproject.toml && \
    uv pip install .

ENV SGLANG_USE_CPU_ENGINE=1

WORKDIR /sgl-workspace/sglang