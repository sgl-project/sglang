#!/bin/bash
set -euo pipefail

PIP_INSTALL="python3 -m pip install --no-cache-dir"
UV_PIP_INSTALL="uv pip install "
DEVICE_TYPE=$1

CANN_VERSION="${CANN_VERSION:-9.2.0}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.13.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.28.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
SGLANG_KERNEL_NPU_TAG="${SGLANG_KERNEL_NPU_TAG:-2026.11.0.post16}"
PYARROW_VERSION="${PYARROW_VERSION:-23.0.1}"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-${CANN_VERSION}}"
export ASCEND_HOME_PATH
DEVICE_ARCH="$(arch)"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/lib:${ASCEND_HOME_PATH}/${DEVICE_ARCH}-linux/devlib/device:/usr/local/Ascend/driver/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"

GITHUB_PROXY_URL="${GITHUB_PROXY_URL:-}"

case "${DEVICE_ARCH}" in
aarch64)
    TORCH_NPU_WHEEL_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.2.0-pytorch2.13.0/torch_npu-2.13.0-cp312-cp312-manylinux_2_28_aarch64.whl"
    TRITON_WHEEL_URL="https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261105160244/triton_ascend-3.6.0%2Bdev20261105160244-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_34_aarch64.whl"
    ;;
x86_64)
    TORCH_NPU_WHEEL_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.2.0-pytorch2.13.0/torch_npu-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    TRITON_WHEEL_URL="https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261105160244/triton_ascend-3.6.0%2Bdev20261105160244-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_34_x86_64.whl"
    ;;
*)
    echo "Unsupported architecture: ${DEVICE_ARCH}" >&2
    exit 1
    ;;
esac

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
    libibverbs-dev \
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

${UV_PIP_INSTALL} pybind11 decorator attrs evalscope

${PIP_INSTALL} cannbot-dsl==0.8.0

case "${DEVICE_TYPE}" in
    950)
        MF_SOC_VERSION="A5"
        ;;
    a3)
        MF_SOC_VERSION="A3"
        ;;
    *)
        echo "Unsupported DEVICE_TYPE for mfcli kernel install: ${DEVICE_TYPE}" >&2
        exit 1
        ;;
esac

${PIP_INSTALL} "memfabric-hybrid==1.5.0.dev20261124" \
    --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/

command -v mfcli
mfcli kernel install --soc-version "${MF_SOC_VERSION}"

${PIP_INSTALL} "https://github.com/kvcache-ai/Mooncake/releases/download/v0.3.14/mooncake_transfer_engine_memfabric-0.3.14-cp312-cp312-manylinux_2_35_${DEVICE_ARCH}.whl" \
    --force --no-deps

${UV_PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url ${TORCH_CACHE_URL:="https://download.pytorch.org/whl/cpu"} --extra-index-url ${PYPI_CACHE_URL:="https://pypi.org/simple/"}
${PIP_INSTALL} "${TORCH_NPU_WHEEL_URL}"
${PIP_INSTALL} "${TRITON_WHEEL_URL}"

#torch_npu patch
sed -Ei -e '/^[[:space:]]*"torch\.npu\.random\.manual_seed(_all)?",[[:space:]]*$/d' -e 's/^([[:space:]]*)"torch\.npu\.set_device",/\1"torch.npu.random.manual_seed_all",\n\1"torch.npu.random.manual_seed",\n\1"torch.npu.set_device",/' /usr/local/python3.12.15/lib/python3.12/site-packages/torch_npu/dynamo/trace_rule.py

case "${DEVICE_ARCH}" in
aarch64)
    LIBDIR="/usr/lib/aarch64-linux-gnu"
    ;;
x86_64)
    LIBDIR="/usr/lib/x86_64-linux-gnu"
    ;;
esac
install -Dm755 "${ASCEND_HOME_PATH}/${DEVICE_ARCH}-linux/lib64/libmmpa.so" "${LIBDIR}/libmmpa.so"
ldconfig
ldconfig -p | grep libmmpa.so

mkdir -p sgl-kernel-npu
(
    cd sgl-kernel-npu
    wget "${GITHUB_PROXY_URL}https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip"
    unzip ./*.zip

    ${UV_PIP_INSTALL} --no-deps ./sgl_kernel_npu*.whl
    ${UV_PIP_INSTALL} \
        ./deep_ep*.whl \
        ./attentions*.whl \
        ./torch_memory_saver*.whl

    cd "$(python3 -m pip show deep-ep | grep -E '^Location:' | awk '{print $2}')"
    ln -sf deep_ep/deep_ep_cpp*.so
)
rm -rf sgl-kernel-npu

rm -f python/pyproject.toml
mv python/pyproject_npu.toml python/pyproject.toml
${UV_PIP_INSTALL} -v -e "python[dev_npu]" "pyarrow==${PYARROW_VERSION}"
