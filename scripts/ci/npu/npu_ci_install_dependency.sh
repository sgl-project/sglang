#!/bin/bash
set -euo pipefail

PIP_INSTALL="python3 -m pip install --no-cache-dir"
UV_PIP_INSTALL="uv pip install "
DEVICE_TYPE=$1


# Install the required dependencies in CI.
apt update -y && apt install -y \
    unzip \
    build-essential \
    cmake \
    wget \
    curl \
    net-tools \
    zlib1g-dev \
    lld \
    clang \
    locales \
    ccache \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    ca-certificates \
    libgl1 \
    libglib2.0-0
update-ca-certificates
${PIP_INSTALL} --upgrade pip
${PIP_INSTALL} uv
export UV_NO_CACHE=true
export UV_SYSTEM_PYTHON=true
export UV_INDEX_STRATEGY=unsafe-best-match

# Install Rust toolchain (needed by crates built via setuptools-rust, e.g. the
# native gRPC extension bundled into the sglang wheel).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/../utils/install_rustup.sh"
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"

# Pin wheel to 0.45.1, REF: https://github.com/pypa/wheel/issues/662
${UV_PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil


### Install MemFabric
${UV_PIP_INSTALL} memfabric-hybrid==1.0.8


### Install PyTorch and PTA
PYTORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"
TORCHAUDIO_VERSION="2.10.0"
${UV_PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url ${TORCH_CACHE_URL:="https://download.pytorch.org/whl/cpu"} --extra-index-url ${PYPI_CACHE_URL:="https://pypi.org/simple/"}
PTA_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.10.0/torch_npu-2.10.0-cp311-cp311-manylinux_2_28_aarch64.whl"
# GitCode does not allow UV downloads.
${PIP_INSTALL} ${PTA_URL}

### Install zbal
${UV_PIP_INSTALL} memfabric-zbal==1.1.1

### Install Triton-Ascend
${PIP_INSTALL} "https://gitcode.com/Ascend/triton-ascend/releases/download/v3.2.1/triton_ascend-3.2.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"


### Install sgl-kernel-npu
SGLANG_KERNEL_NPU_TAG="2026.7.0"
mkdir sgl-kernel-npu
(cd sgl-kernel-npu && wget "${GITHUB_PROXY_URL:=""}https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-py311-cann9.0.0-${DEVICE_TYPE}-$(arch).zip" \
&& unzip ./sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-py311-cann9.0.0-${DEVICE_TYPE}-$(arch).zip \
&& ${UV_PIP_INSTALL} ./deep_ep*.whl ./sgl_kernel_npu*.whl \
&& (cd "$(python3 -m pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so))

### Install custom-ops
mkdir cann-custom-ops
(cd cann-custom-ops && \
wget "${GITHUB_PROXY_URL:=""}https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-cann9.0.0-${DEVICE_TYPE}-$(arch).zip" && \
wget "${GITHUB_PROXY_URL:=""}https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-cann9.0.0-${DEVICE_TYPE}-$(arch).zip" && \
unzip custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-cann9.0.0-${DEVICE_TYPE}-$(arch).zip && \
unzip ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-cann9.0.0-${DEVICE_TYPE}-$(arch).zip && \
chmod +x *.run && \
./CANN-custom_ops-none-linux.$(arch).run --install-path=/usr/local/Ascend/cann-${CANN_VERSION}/opp && \
./cann-ops-transformer-custom_linux-$(arch).run --install-path=/usr/local/Ascend/cann-${CANN_VERSION}/opp && \
${PIP_INSTALL} custom_ops-1.0-cp311-cp311-linux_$(arch).whl)
rm -rf cann-custom-ops


### Install SGLang
rm -rf python/pyproject.toml && mv python/pyproject_npu.toml python/pyproject.toml
${UV_PIP_INSTALL} -v -e "python[dev_npu]"
