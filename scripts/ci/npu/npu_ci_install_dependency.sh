#!/bin/bash
set -euo pipefail

PIP_INSTALL="python3 -m pip install --no-cache-dir"
UV_PIP_INSTALL="uv pip install "
DEVICE_TYPE=$1

CANN_VERSION="${CANN_VERSION:-9.1.0}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.10.0}"
SGLANG_KERNEL_NPU_TAG="${SGLANG_KERNEL_NPU_TAG:-2026.9.0.post9}"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-${CANN_VERSION}}"
export ASCEND_HOME_PATH
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/lib:${ASCEND_HOME_PATH}/$(arch)-linux/devlib/device:/usr/local/Ascend/driver/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"

GITHUB_PROXY_URL="${GITHUB_PROXY_URL:-}"

source_env() {
    if [ -f "$1" ]; then
        # Vendor set_env scripts (e.g. opp/vendors/*/bin/set_env.bash) append to
        # variables like ASCEND_CUSTOM_OPP_PATH via ${VAR} without a default, which
        # aborts under `set -u`. Relax nounset only while sourcing them.
        set +u
        # shellcheck disable=SC1090
        source "$1"
        set -u
    else
        echo "[skip] set_env not found: $1"
    fi
}

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
    ffmpeg \
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/../utils/install_rustup.sh"
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"

${UV_PIP_INSTALL} pybind11 pyyaml decorator scipy attrs psutil

${PIP_INSTALL} memfabric-hybrid==1.2.1
mfcli kernel install
${PIP_INSTALL} memcache-hybrid==1.2.1

### Install memfabric-zbal
if [ "${DEVICE_TYPE}" = "950" ]; then
    ${PIP_INSTALL} memfabric-zbal==1.2.21004.post1
else
    ${PIP_INSTALL} memfabric-zbal==1.1.3
fi

### Install SGLang Model Gateway
${PIP_INSTALL} sglang-router

${UV_PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==0.25.0 torchaudio==2.10.0 --index-url ${TORCH_CACHE_URL:="https://download.pytorch.org/whl/cpu"} --extra-index-url ${PYPI_CACHE_URL:="https://pypi.org/simple/"}
${PIP_INSTALL} torch-npu==2.10.0.post6 --extra-index-url https://ascend.devcloud.huaweicloud.com/pypi/simple/

case "$(arch)" in
aarch64)
    ${PIP_INSTALL} "${GITHUB_PROXY_URL}https://github.com/triton-lang/triton-ascend/releases/download/v3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
    ;;
x86_64)
    ${PIP_INSTALL} "${GITHUB_PROXY_URL}https://github.com/triton-lang/triton-ascend/releases/download/v3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    ;;
*)
    echo "Unsupported architecture: $(arch)" >&2
    exit 1
    ;;
esac

mkdir -p sgl-kernel-npu
(cd sgl-kernel-npu &&
    wget "${GITHUB_PROXY_URL}https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip" &&
    unzip ./sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip &&
    ${UV_PIP_INSTALL} ./deep_ep*.whl ./sgl_kernel_npu*.whl ./attentions*.whl ./torch_memory_saver*.whl &&
    (cd "$(python3 -m pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so))
rm -rf sgl-kernel-npu

mkdir -p cann-custom-ops
(cd cann-custom-ops &&
    source_env "${ASCEND_HOME_PATH}/set_env.sh" &&
    wget "${GITHUB_PROXY_URL}https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip" &&
    wget "${GITHUB_PROXY_URL}https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip" &&
    unzip custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip &&
    unzip ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip &&
    chmod +x *.run &&
    ./CANN-custom_ops-none-linux.$(arch).run --install-path=${ASCEND_HOME_PATH}/opp &&
    ./cann-ops-transformer-custom_linux-$(arch).run --install-path=${ASCEND_HOME_PATH}/opp &&
    source_env "${ASCEND_HOME_PATH}/opp/vendors/customize/bin/set_env.bash" &&
    source_env "${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/bin/set_env.bash" &&
    source_env /usr/local/Ascend/ascend-toolkit/latest/set_env.sh &&
    source_env /usr/local/Ascend/nnal/atb/set_env.sh &&
    ${PIP_INSTALL} custom_ops-1.0-cp312-cp312-linux_$(arch).whl)
rm -rf cann-custom-ops

rm -rf python/pyproject.toml && mv python/pyproject_npu.toml python/pyproject.toml
${UV_PIP_INSTALL} -v -e "python[dev_npu]"
