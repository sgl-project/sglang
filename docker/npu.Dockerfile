ARG CANN_VERSION=8.3.rc1
ARG DEVICE_TYPE=a3
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=py3.11

FROM quay.io/ascend/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION

# Update pip & apt sources
ARG TARGETARCH
ARG PIP_INDEX_URL="https://pypi.org/simple/"
ARG APTMIRROR=""
ARG PYTORCH_VERSION="2.8.0"
ARG TORCHVISION_VERSION="0.23.0"
ARG PTA_URL_ARM64="https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.8.0/torch_npu-2.8.0-cp311-cp311-manylinux_2_28_aarch64.whl"
ARG PTA_URL_AMD64="https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.8.0/torch_npu-2.8.0-cp311-cp311-manylinux_2_28_x86_64.whl"
ARG BISHENG_NAME_ARM64="Ascend-BiSheng-toolkit_aarch64_20251121.run"
ARG BISHENG_NAME_AMD64=""
ARG BISHENG_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/triton_ascend/${BISHENG_NAME}"
ARG BISHENG_NAME=""
ARG SGLANG_TAG=main
ARG ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit
ARG SGLANG_KERNEL_NPU_TAG=main

ARG PIP_INSTALL="python3 -m pip install --no-cache-dir"
ARG DEVICE_TYPE

RUN if [ "$TARGETARCH" = "amd64" ]; then \
      echo "Using x86_64 dependencies"; \
      echo "PTA_URL=$PTA_URL_AMD64" >> /etc/environment_new; \
      echo "BISHENG_NAME_AMD64=$BISHENG_NAME_AMD64" >> /etc/environment_new; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
      echo "Using aarch64 dependencies"; \
      echo "PTA_URL=$PTA_URL_ARM64" >> /etc/environment_new; \
      echo "BISHENG_NAME=$BISHENG_NAME_ARM64" >> /etc/environment_new; \
    else \
      echo "Unsupported TARGETARCH: $TARGETARCH"; exit 1; \
    fi

WORKDIR /workspace

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

RUN pip config set global.index-url $PIP_INDEX_URL
RUN if [ -n "$APTMIRROR" ];then sed -i "s|.*.ubuntu.com|$APTMIRROR|g" /etc/apt/sources.list ;fi

# Install development tools and utilities
RUN apt-get update -y && apt upgrade -y && apt-get install -y \
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
    ca-certificates \
    && rm -rf /var/cache/apt/* \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates \
    && locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8


### Install MemFabric
RUN ${PIP_INSTALL} mf-adapter==1.0.0
### Install SGLang Model Gateway
RUN ${PIP_INSTALL} sglang-router


### Install PyTorch and PTA
RUN . /etc/environment_new && \
    (${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu) \
    && (${PIP_INSTALL} ${PTA_URL})


## Install triton-ascend
RUN (${PIP_INSTALL} pybind11 triton-ascend)

# Install SGLang
RUN git clone https://github.com/sgl-project/sglang --branch $SGLANG_TAG && \
    (cd sglang/python && rm -rf pyproject.toml && mv pyproject_other.toml pyproject.toml && ${PIP_INSTALL} -v .[srt_npu]) && \
    rm -rf sglang

# Install Deep-ep
# pin wheel to 0.45.1 ref: https://github.com/pypa/wheel/issues/662
RUN ${PIP_INSTALL} wheel==0.45.1 && git clone --branch $SGLANG_KERNEL_NPU_TAG https://github.com/sgl-project/sgl-kernel-npu.git \
    && export LD_LIBRARY_PATH=${ASCEND_CANN_PATH}/latest/runtime/lib64/stub:$LD_LIBRARY_PATH && \
    source ${ASCEND_CANN_PATH}/set_env.sh && \
    cd sgl-kernel-npu && \
    bash build.sh \
    && ${PIP_INSTALL} output/deep_ep*.whl output/sgl_kernel_npu*.whl \
    && cd .. && rm -rf sgl-kernel-npu \
    && cd "$(python3 -m pip show deep-ep | awk '/^Location:/ {print $2}')" && ln -s deep_ep/deep_ep_cpp*.so

# Install CustomOps
# TODO: remove this after sgl-kernel-npu contains custom_ops
RUN wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run && \
    chmod a+x ./CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run && \
    ./CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp && \
    wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/custom_ops-1.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl && \
    ${PIP_INSTALL} ./custom_ops-1.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl

# Install Bisheng
RUN . /etc/environment_new && \
    wget -O "${BISHENG_NAME}" "${BISHENG_URL}" && chmod a+x "${BISHENG_NAME}" && "./${BISHENG_NAME}" --install && rm "${BISHENG_NAME}"

CMD ["/bin/bash"]
