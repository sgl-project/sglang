ARG CANN_VERSION=9.1.0
ARG DEVICE_TYPE=950
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=py3.12
ARG arch

FROM quay.io/ascend/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION

ARG TARGETARCH
ARG CANN_VERSION
ARG DEVICE_TYPE
ARG arch
ARG PIP_INDEX_URL="https://pypi.org/simple/"
ARG APTMIRROR=""
# torch_npu 2.10.0.post6 requires torch==2.10.0, so PYTORCH_VERSION stays at 2.10.0
ARG PYTORCH_VERSION="2.10.0"
ARG TORCHVISION_VERSION="0.25.0"
ARG TORCHAUDIO_VERSION="2.10.0"
ARG TORCH_NPU_VERSION="2.10.0.post6"
ARG TORCH_NPU_INDEX_URL="https://ascend.devcloud.huaweicloud.com/pypi/simple/"
ARG SGLANG_TAG=main
ARG ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit
ARG SGLANG_KERNEL_NPU_TAG=2026.9.0.post5
ARG PIP_INSTALL="python3 -m pip install --no-cache-dir"
ARG DEVICE_TYPE
ARG MODELSCOPE_VERSION=""
ARG EVALSCOPE_VERSION=""

ARG MF_VERSION="1.2.1"
# MemFabric / MemCache 1.2.1 wheels, shared by a3 and 950.
# 1.2.1 is not published on PyPI (PyPI stops at 1.2.0), so the wheels are pulled from the
# sglang-npu OBS bucket. These links are presigned and will expire; when they do, regenerate
# them from the bucket and pass the new values with --build-arg, no Dockerfile edit needed.
ARG MF_WHEEL_URL_AARCH64="https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com:443/memfabric/1.2.1/memfabric_hybrid-1.2.1-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl?AccessKeyId=HPUAAPJN7IAXFCS2GDSQ&Expires=1820732522&Signature=/vRnADjM4r7v392pAygfpiowOMo%3D"
ARG MF_WHEEL_URL_X86_64="https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com:443/memfabric/1.2.1/memfabric_hybrid-1.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl?AccessKeyId=HPUAAPJN7IAXFCS2GDSQ&Expires=1820732540&Signature=8TKnsDBAihKWkEGcV5/SLkCXVeM%3D"
ARG MC_WHEEL_URL_AARCH64="https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com:443/memfabric/1.2.1/memcache_hybrid-1.2.1-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl?AccessKeyId=HPUAAPJN7IAXFCS2GDSQ&Expires=1820732457&Signature=xyC5pL2ztyoeIBgsmZ/cB0CFDBU%3D"
ARG MC_WHEEL_URL_X86_64="https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com:443/memfabric/1.2.1/memcache_hybrid-1.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl?AccessKeyId=HPUAAPJN7IAXFCS2GDSQ&Expires=1820732498&Signature=0pxMuRqZjSyFaAfTtRBEbKHYmmY%3D"

# memfabric-zbal: 950 与 a3 使用不同版本
ARG ZBAL_VERSION_950="1.2.21004.post1"
ARG ZBAL_VERSION_A3="1.1.3"


# Later RUN steps source /etc/environment_new, so make sure it exists
RUN touch /etc/environment_new

WORKDIR /workspace

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

RUN pip config set global.index-url $PIP_INDEX_URL
RUN if [ -n "$APTMIRROR" ];then sed -i "s|.*.ubuntu.com|$APTMIRROR|g" /etc/apt/sources.list ;fi

# Install development tools and utilities
RUN apt-get update -y && apt upgrade -y && apt-get install -y \
    unzip \
    build-essential \
    cmake \
    vim \
    wget \
    curl \
    net-tools \
    zlib1g-dev \
    lld \
    clang \
    locales \
    ccache \
    openssl \
    libssl-dev \
    pkg-config \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    ca-certificates \
    && rm -rf /var/cache/apt/* \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates \
    && locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

### Install MemFabric and MemCache (a3 与 950 统一走 OBS wheel)
# Download both wheels from the OBS bucket first, verify their sha256, then install locally.
# The saved file name has to stay a valid wheel name (distribution-version-python-abi-platform),
# otherwise pip rejects it with "Invalid wheel filename" before it even reads the archive.
# The two wheels must be installed in this order: memcache_hybrid depends on memfabric_hybrid,
# and `mfcli` (provided by memfabric_hybrid) has to exist before `mfcli kernel install` runs.
RUN set -eux; \
    case "$TARGETARCH" in \
      arm64) \
        WHEEL_TAG="cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64"; \
        MF_URL="$MF_WHEEL_URL_AARCH64"; \
        MF_SHA256="27b9c0f18db6260e632f00a2302176bbbd781858d12f3a51d8af6169ac5337c1"; \
        MC_URL="$MC_WHEEL_URL_AARCH64"; \
        MC_SHA256="c578dfa102e1266755c910e701ed557d50d152376799ae84968d07fbb5fa359e"; \
        ;; \
      amd64) \
        WHEEL_TAG="cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64"; \
        MF_URL="$MF_WHEEL_URL_X86_64"; \
        MF_SHA256="b754ee9a511f2a495963eec816001ba34924330441f3d6409e7e5d195c0515a0"; \
        MC_URL="$MC_WHEEL_URL_X86_64"; \
        MC_SHA256="4a684656978880d1cc7b58663036ad13bd966a240067224a93f23ba327bc7370"; \
        ;; \
      *) \
        echo "Unsupported architecture: $TARGETARCH" >&2; \
        exit 1; \
        ;; \
    esac; \
    MF_WHEEL="/tmp/memfabric_hybrid-${MF_VERSION}-${WHEEL_TAG}.whl"; \
    MC_WHEEL="/tmp/memcache_hybrid-${MF_VERSION}-${WHEEL_TAG}.whl"; \
    curl -fL --retry 3 --retry-delay 2 -o "$MF_WHEEL" "$MF_URL"; \
    curl -fL --retry 3 --retry-delay 2 -o "$MC_WHEEL" "$MC_URL"; \
    echo "$MF_SHA256  $MF_WHEEL" | sha256sum -c -; \
    echo "$MC_SHA256  $MC_WHEEL" | sha256sum -c -; \
    ${PIP_INSTALL} "$MF_WHEEL" --force-reinstall; \
    mfcli kernel install; \
    ${PIP_INSTALL} "$MC_WHEEL" --force-reinstall --no-deps; \
    rm -f "$MF_WHEEL" "$MC_WHEEL"

### Install memfabric-zbal
RUN if [ "$DEVICE_TYPE" = "950" ]; then ZBAL_PKG="memfabric-zbal==${ZBAL_VERSION_950}"; \
    else ZBAL_PKG="memfabric-zbal==${ZBAL_VERSION_A3}"; fi; \
    ${PIP_INSTALL} "$ZBAL_PKG" -i https://pypi.org/simple/
### Install SGLang Model Gateway
RUN ${PIP_INSTALL} sglang-router


### Install PyTorch and PTA
RUN . /etc/environment_new && \
    (${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cpu) \
    && (${PIP_INSTALL} torch-npu==${TORCH_NPU_VERSION} --extra-index-url ${TORCH_NPU_INDEX_URL})


### Install ModelScope & EvalScope
# Installed right after torch/torch-npu so their dependencies resolve against the pinned torch.
# MODELSCOPE_VERSION / EVALSCOPE_VERSION are empty by default -> latest release.
RUN . /etc/environment_new && \
    MS_PKG="modelscope" && \
    ES_PKG="evalscope" && \
    if [ -n "${MODELSCOPE_VERSION}" ]; then MS_PKG="modelscope==${MODELSCOPE_VERSION}"; fi && \
    if [ -n "${EVALSCOPE_VERSION}" ]; then ES_PKG="evalscope==${EVALSCOPE_VERSION}"; fi && \
    ${PIP_INSTALL} "${MS_PKG}" "${ES_PKG}"


## Install triton-ascend
RUN . /etc/environment_new && \
    ${PIP_INSTALL} pybind11 && \
    if [ "$TARGETARCH" = "arm64" ]; then \
        ${PIP_INSTALL} https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ta/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl; \
    elif [ "$TARGETARCH" = "amd64" ]; then \
        ${PIP_INSTALL} https://github.com/triton-lang/triton-ascend/releases/download/v3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl; \
    else \
        echo "Unsupported architecture: $TARGETARCH"; \
        exit 1; \
    fi

# Install SGLang
RUN git clone https://github.com/sgl-project/sglang --branch ${SGLANG_TAG} /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang/python && rm -rf pyproject.toml && mv pyproject_npu.toml pyproject.toml && \
    sed -i '/"memfabric-hybrid==1.1.4"/d; /"memfabric-zbal==1.1.2"/d' pyproject.toml && \
    ${PIP_INSTALL} -v -e .[all_npu]

ENV ASCEND_HOME_PATH=/usr/local/Ascend/cann-${CANN_VERSION}

ENV LD_LIBRARY_PATH=/usr/local/Ascend/cann-${CANN_VERSION}/lib64:/usr/local/Ascend/cann-${CANN_VERSION}/lib:/usr/local/Ascend/cann-${CANN_VERSION}/x86_64-linux/devlib/device:/usr/local/Ascend/driver/lib64:/usr/local/lib:${LD_LIBRARY_PATH}


RUN mkdir cann-custom-ops && \
    cd cann-custom-ops && \
    source /usr/local/Ascend/cann-${CANN_VERSION}/set_env.sh && \
    wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    unzip custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    unzip ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    chmod +x *.run && \
    ./CANN-custom_ops-none-linux.$(arch).run --install-path=/usr/local/Ascend/cann-${CANN_VERSION}/opp && \
    ./cann-ops-transformer-custom_linux-$(arch).run --install-path=/usr/local/Ascend/cann-${CANN_VERSION}/opp && \
    source /usr/local/Ascend/cann-${CANN_VERSION}/opp/vendors/customize/bin/set_env.bash && \
    source /usr/local/Ascend/cann-${CANN_VERSION}/opp/vendors/custom_transformer/bin/set_env.bash && \
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    ${PIP_INSTALL} custom_ops-1.0-cp312-cp312-linux_$(arch).whl && \
    cd .. && rm -rf cann-custom-ops

# Install Deep-ep
# pin wheel to 0.45.1 ref: https://github.com/pypa/wheel/issues/662
RUN ${PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil \
    && mkdir sgl-kernel-npu \
    && cd sgl-kernel-npu \
    && wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && unzip sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && ${PIP_INSTALL} deep_ep*.whl sgl_kernel_npu*.whl torch_memory_saver*.whl \
    && cd .. && rm -rf sgl-kernel-npu \
    && cd "$(python3 -m pip show deep-ep | awk '/^Location:/ {print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so

CMD ["/bin/bash"]
