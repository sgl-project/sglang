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
ARG MF_WHEEL_URL_AARCH64="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/mf/v1.2.1/20260923.4/memfabric_hybrid-1.2.1-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl"
ARG MF_WHEEL_URL_X86_64="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/mf/v1.2.1/20260923.4/memfabric_hybrid-1.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
ARG MC_WHEEL_URL_AARCH64="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/memcache/v1.2.1/20260923.4/memcache_hybrid-1.2.1-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl"
ARG MC_WHEEL_URL_X86_64="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/memcache/v1.2.1/20260923.4/memcache_hybrid-1.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

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

### Install MemFabric and MemCache (按 TARGETARCH 直接取对应架构的 OBS wheel)
RUN set -eux; \
    case "$TARGETARCH" in \
      arm64) \
        MF_URL="$MF_WHEEL_URL_AARCH64"; \
        MC_URL="$MC_WHEEL_URL_AARCH64"; \
        ;; \
      amd64) \
        MF_URL="$MF_WHEEL_URL_X86_64"; \
        MC_URL="$MC_WHEEL_URL_X86_64"; \
        ;; \
      *) \
        echo "Unsupported architecture: $TARGETARCH" >&2; \
        exit 1; \
        ;; \
    esac; \
    ${PIP_INSTALL} "$MF_URL" --force-reinstall; \
    mfcli kernel install; \
    ${PIP_INSTALL} "$MC_URL" --force-reinstall --no-deps

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
