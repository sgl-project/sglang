# If the device is Battlemage, we need to set UBUNTU_VERSION to 24.10

# Usage: docker build --build-arg UBUNTU_VERSION=24.04 --build-arg PYTHON_VERSION=3.10 -t sglang:xpu_kernel -f  xpu.Dockerfile --no-cache .

# Use Intel deep learning essentials base image with Ubuntu 24.04
FROM intel/deep-learning-essentials:2025.2.2-0-devel-ubuntu24.04

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Define build arguments
ARG PYTHON_VERSION=3.10

ARG SG_LANG_REPO=https://github.com/sgl-project/sglang.git
ARG SG_LANG_BRANCH=main

ARG SG_LANG_KERNEL_REPO=https://github.com/sgl-project/sgl-kernel-xpu.git
ARG SG_LANG_KERNEL_BRANCH=main

RUN useradd -m -d /home/sdp -s /bin/bash sdp && \
    chown -R sdp:sdp /home/sdp

# Switch to non-root user 'sdp'
USER sdp

# Set HOME and WORKDIR to user's home directory
ENV HOME=/home/sdp
WORKDIR /home/sdp

RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda-forge/miniforge/releases/download/25.1.1-0/Miniforge3-Linux-x86_64.sh && \
    bash miniforge.sh -b -p ./miniforge3 && \
    rm miniforge.sh && \
    # Initialize conda environment and install pip
    . ./miniforge3/bin/activate && \
    conda create -y -n py${PYTHON_VERSION} python=${PYTHON_VERSION} && \
    conda activate py${PYTHON_VERSION} && \
    conda install pip && \
    # Append environment activation to .bashrc for interactive shells
    echo ". /home/sdp/miniforge3/bin/activate; conda activate py${PYTHON_VERSION}; . /opt/intel/oneapi/setvars.sh; cd /home/sdp" >> /home/sdp/.bashrc

USER root

# Install the latest UMD driver for SYCL-TLA
RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc && \
    apt-get install -y intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo && \
    apt-get install -y libze-dev intel-ocloc && \
    apt-get update

# Switch back to user sdp
USER sdp

RUN --mount=type=secret,id=github_token \
    cd /home/sdp && \
    . /home/sdp/miniforge3/bin/activate && \
    conda activate py${PYTHON_VERSION} && \
    pip3 install torch==2.9.0+xpu torchao torchvision torchaudio pytorch-triton-xpu==3.5.0 --index-url https://download.pytorch.org/whl/xpu

RUN --mount=type=secret,id=github_token \
    cd /home/sdp && \
    . /home/sdp/miniforge3/bin/activate && \
    conda activate py${PYTHON_VERSION} && \
    echo "Cloning ${SG_LANG_BRANCH} from ${SG_LANG_REPO}" && \
    git clone --branch ${SG_LANG_BRANCH} --single-branch ${SG_LANG_REPO} && \
    cd sglang && cd python && \
    cp pyproject_xpu.toml pyproject.toml && \
    pip install . && \
    pip install xgrammar --no-deps && \
    pip install msgspec blake3 py-cpuinfo compressed_tensors gguf partial_json_parser einops --root-user-action=ignore && \
    conda install libsqlite=3.48.0 -y && \
    # Add environment setup commands to .bashrc again (in case it was overwritten)
    echo ". /home/sdp/miniforge3/bin/activate; conda activate py${PYTHON_VERSION}; cd /home/sdp" >> /home/sdp/.bashrc

# Use bash as default shell with initialization from .bashrc
SHELL ["bash", "-c"]

# Start an interactive bash shell with all environment set up
USER sdp
CMD ["bash", "-c", "source /home/sdp/.bashrc && exec bash"]
