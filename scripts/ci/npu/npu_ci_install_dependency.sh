#!/bin/bash
set -euo pipefail

PIP_INSTALL="python3 -m pip install --no-cache-dir"
UV_PIP_INSTALL="uv pip install "
DEVICE_TYPE=$1

CANN_VERSION="${CANN_VERSION:-9.1.0}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0}"
TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.10.0.post6}"
TORCH_NPU_INDEX_URL="${TORCH_NPU_INDEX_URL:-https://ascend.devcloud.huaweicloud.com/pypi/simple/}"
SGLANG_KERNEL_NPU_TAG="${SGLANG_KERNEL_NPU_TAG:-2026.9.0.post5}"

MF_VERSION="${MF_VERSION:-1.2.1}"
MF_PYPI_VERSION="${MF_PYPI_VERSION:-1.1.5}"
ZBAL_VERSION_950="${ZBAL_VERSION_950:-1.2.21004.post1}"
ZBAL_VERSION_A3="${ZBAL_VERSION_A3:-1.1.3}"

MF_WHEEL_URL_AARCH64="${MF_WHEEL_URL_AARCH64:-https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com:443/memfabric/1.2.1/memfabric_hybrid-1.2.1-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl?AccessKeyId=HPUAAPJN7IAXFCS2GDSQ&Expires=1820732522&Signature=/vRnADjM4r7v392pAygfpiowOMo%3D}"
MF_WHEEL_URL_X86_64="${MF_WHEEL_URL_X86_64:-https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com:443/memfabric/1.2.1/memfabric_hybrid-1.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl?AccessKeyId=HPUAAPJN7IAXFCS2GDSQ&Expires=1820732540&Signature=8TKnsDBAihKWkEGcV5/SLkCXVeM%3D}"
MC_WHEEL_URL_AARCH64="${MC_WHEEL_URL_AARCH64:-https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com:443/memfabric/1.2.1/memcache_hybrid-1.2.1-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl?AccessKeyId=HPUAAPJN7IAXFCS2GDSQ&Expires=1820732457&Signature=xyC5pL2ztyoeIBgsmZ/cB0CFDBU%3D}"
MC_WHEEL_URL_X86_64="${MC_WHEEL_URL_X86_64:-https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com:443/memfabric/1.2.1/memcache_hybrid-1.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl?AccessKeyId=HPUAAPJN7IAXFCS2GDSQ&Expires=1820732498&Signature=0pxMuRqZjSyFaAfTtRBEbKHYmmY%3D}"

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

if [ "${DEVICE_TYPE}" = "950" ]; then
    case "$(arch)" in
    aarch64)
        WHEEL_TAG="cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64"
        MF_URL="${MF_WHEEL_URL_AARCH64}"
        MF_SHA256="27b9c0f18db6260e632f00a2302176bbbd781858d12f3a51d8af6169ac5337c1"
        MC_URL="${MC_WHEEL_URL_AARCH64}"
        MC_SHA256="c578dfa102e1266755c910e701ed557d50d152376799ae84968d07fbb5fa359e"
        ;;
    x86_64)
        WHEEL_TAG="cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64"
        MF_URL="${MF_WHEEL_URL_X86_64}"
        MF_SHA256="b754ee9a511f2a495963eec816001ba34924330441f3d6409e7e5d195c0515a0"
        MC_URL="${MC_WHEEL_URL_X86_64}"
        MC_SHA256="4a684656978880d1cc7b58663036ad13bd966a240067224a93f23ba327bc7370"
        ;;
    *)
        echo "Unsupported architecture: $(arch)" >&2
        exit 1
        ;;
    esac
    MF_WHEEL="/tmp/memfabric_hybrid-${MF_VERSION}-${WHEEL_TAG}.whl"
    MC_WHEEL="/tmp/memcache_hybrid-${MF_VERSION}-${WHEEL_TAG}.whl"
    curl -fL --retry 3 --retry-delay 2 -o "${MF_WHEEL}" "${MF_URL}"
    curl -fL --retry 3 --retry-delay 2 -o "${MC_WHEEL}" "${MC_URL}"
    echo "${MF_SHA256}  ${MF_WHEEL}" | sha256sum -c -
    echo "${MC_SHA256}  ${MC_WHEEL}" | sha256sum -c -
    ${PIP_INSTALL} "${MF_WHEEL}" --force-reinstall
    mfcli kernel install
    ${PIP_INSTALL} "${MC_WHEEL}" --force-reinstall --no-deps
    rm -f "${MF_WHEEL}" "${MC_WHEEL}"
else
    ${PIP_INSTALL} memfabric-hybrid==${MF_PYPI_VERSION}
    ${PIP_INSTALL} memcache-hybrid==${MF_PYPI_VERSION}
fi

### Install memfabric-zbal
if [ "${DEVICE_TYPE}" = "950" ]; then
    ${PIP_INSTALL} memfabric-zbal==${ZBAL_VERSION_950} -i https://pypi.org/simple/
else
    ${PIP_INSTALL} memfabric-zbal==${ZBAL_VERSION_A3} -i https://pypi.org/simple/
fi

### Install SGLang Model Gateway
${PIP_INSTALL} sglang-router

${UV_PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url ${TORCH_CACHE_URL:="https://download.pytorch.org/whl/cpu"} --extra-index-url ${PYPI_CACHE_URL:="https://pypi.org/simple/"}
${PIP_INSTALL} torch-npu==${TORCH_NPU_VERSION} --extra-index-url ${TORCH_NPU_INDEX_URL}

case "$(arch)" in
aarch64)
    ${PIP_INSTALL} https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ta/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
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
# The upstream metadata pins memfabric to single versions (memfabric-hybrid==1.1.4 and
# memfabric-zbal==1.1.2; the latter publishes no cp312 wheel), while this script installs the
# per-device versions above. Strip any memfabric pin so pip neither pulls a different build nor
# fails on a version without a wheel for this Python. No-op once the pins are gone from the file.
sed -i '/"memfabric-/d' python/pyproject.toml
${UV_PIP_INSTALL} -v -e "python[dev_npu]"
