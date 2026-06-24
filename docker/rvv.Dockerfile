FROM --platform=linux/riscv64 python:3.13-slim
SHELL ["/bin/bash", "-c"]

ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG VER_SGLANG=main

ARG VER_TORCH=2.8.0+spacemit.1
ARG VER_TORCHVISION=0.23.0
ARG VER_TRITON=3.3.0+spacemit.a0
ARG VER_PYARROW=21.0.0
ARG VER_VLLM=0.11.0.post3+spacemit.0.cpu
ARG VER_LLVM=19
ARG VER_XGRAMMAR=0.1.31

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    SGLANG_USE_CPU_ENGINE=1

# 1. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config gdb lcov \
    ca-certificates curl wget git vim tar gzip unzip \
    libnuma-dev numactl libomp-dev libssl-dev libopenmpi-dev libsleef-dev \
    libsndfile1 \
    clang-${VER_LLVM} lld-${VER_LLVM} llvm-${VER_LLVM} ccache \
    libsqlite3-dev libtbb-dev \
    libgl1 libglib2.0-0 libxrender1 libx11-6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# 2. Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# 3. Setup Workspace & UV VirtualEnv
WORKDIR /sgl-workspace
RUN uv venv /opt/.venv --python 3.13
ENV PATH="/opt/.venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/.venv" \
    UV_HTTP_TIMEOUT=300

# 4. Configure UV (SpacemiT Index)
RUN printf '[[index]]\nname = "spacemit"\nurl = "https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple"\npriority = "default"\n\n[[index]]\nname = "pypi"\nurl = "https://pypi.org/simple"\npriority = "secondary"' > /opt/.venv/uv.toml
ENV UV_CONFIG_FILE=/opt/.venv/uv.toml

# 5. Build Tools & Heavy Dependencies (Pre-install)
RUN uv pip install \
    pip \
    scikit-build-core cmake ninja wheel setuptools \
    "torch==${VER_TORCH}" \
    "torchvision==${VER_TORCHVISION}" \
    "triton==${VER_TRITON}" \
    "pyarrow==${VER_PYARROW}" \
    "xgrammar==${VER_XGRAMMAR}" \
    "vllm==${VER_VLLM}" \
    --index-strategy unsafe-best-match

# 6. Install SGLang Source
WORKDIR /sgl-workspace
RUN git clone ${SGLANG_REPO} sglang && \
    cd sglang && \
    git checkout ${VER_SGLANG}

# 7. Compile sgl-kernel (CLANG REQUIRED for RVV)
WORKDIR /sgl-workspace/sglang/sgl-kernel
RUN cp pyproject_riscv64.toml pyproject.toml && \
    export CC=clang-${VER_LLVM} CXX=clang++-${VER_LLVM} && \
    uv pip install . --no-build-isolation --index-strategy unsafe-best-match
RUN python3 -c "import sgl_kernel; print('sgl_kernel import OK after sgl-kernel build')"

# 8. Install SGLang (GCC REQUIRED for XGrammar compatibility)
WORKDIR /sgl-workspace/sglang/python
RUN cp pyproject_cpu.toml pyproject.toml && \
    # TODO: Remove this line when SpacemiT publishes a cp313 torchaudio wheel.
    sed -i '/torchaudio/d' pyproject.toml
RUN unset CC CXX && \
    export CXXFLAGS="-Wno-error" && \
    export RISCV_OMP_LIB_PATH=/usr/lib/riscv64-linux-gnu/libomp.so.5 && \
    # Override pyproject pins to the preinstalled riscv64 wheel versions.
    # TODO: Remove these overrides when SpacemiT publishes wheels matching sglang's pinned versions.
    printf "torch==${VER_TORCH}\ntorchvision==${VER_TORCHVISION}\ntriton==${VER_TRITON}\nxgrammar==${VER_XGRAMMAR}\n" > /tmp/torch-override.txt && \
    uv pip install . --override /tmp/torch-override.txt --index-strategy unsafe-best-match
RUN python3 -c "import sgl_kernel; print('sgl_kernel import OK in final image')"

# 9. Final Configuration
ENV LD_PRELOAD="/usr/lib/riscv64-linux-gnu/libomp.so.5"
RUN echo 'source /opt/.venv/bin/activate' >> /root/.bashrc

WORKDIR /sgl-workspace/sglang
