# docker build -t sglang:xpu -f xpu.Dockerfile --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg no_proxy=${no_proxy} --no-cache .

# Use Intel deep learning essentials base image with Ubuntu 24.04
FROM intel/deep-learning-essentials:2025.3.2-0-devel-ubuntu24.04

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Define build arguments
ARG PYTHON_VERSION=3.12

ARG SG_LANG_REPO=https://github.com/sgl-project/sglang.git
ARG SG_LANG_BRANCH=main

ARG SG_LANG_KERNEL_REPO=https://github.com/sgl-project/sgl-kernel-xpu.git
ARG SG_LANG_KERNEL_BRANCH=main

USER root

# Install the latest UMD driver for SYCL-TLA
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get update && \
    apt-get install -y \
        libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo \
        libze-dev intel-ocloc && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /sgl-workspace

RUN  pip install --no-cache-dir msgspec blake3 py-cpuinfo compressed_tensors gguf partial_json_parser einops tabulate --root-user-action=ignore && \
    pip install --no-cache-dir torch==2.13.0+xpu torchao==0.17.0+xpu torchvision==0.28.0+xpu torchaudio==2.11.0+xpu --index-url https://download.pytorch.org/whl/xpu

RUN echo "Cloning ${SG_LANG_BRANCH} from ${SG_LANG_REPO}" && \
    git clone --branch ${SG_LANG_BRANCH} --single-branch ${SG_LANG_REPO} sglang && \
    cd sglang && cd python && \
    cp pyproject_xpu.toml pyproject.toml && \
    pip install --no-cache-dir . --extra-index-url https://download.pytorch.org/whl/xpu && \
    pip install --no-cache-dir --no-deps xgrammar==0.1.33

CMD ["bash", "-c", "source /opt/intel/oneapi/setvars.sh --force && exec bash"]
