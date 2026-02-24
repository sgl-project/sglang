ARG CANN_VERSION=8.5.0
ARG DEVICE_TYPE=a3
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=py3.11

FROM quay.io/ascend/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION

# Update pip & apt sources
ARG TARGETARCH
ARG CANN_VERSION
ARG DEVICE_TYPE
ARG PIP_INDEX_URL="https://pypi.org/simple/"
ARG APTMIRROR=""
ARG PYTORCH_VERSION="2.8.0"
ARG TORCHVISION_VERSION="0.23.0"
ARG PTA_URL_ARM64="https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl"
ARG PTA_URL_AMD64="https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0.post2-cp311-cp311-manylinux_2_28_x86_64.whl"
ARG SGLANG_TAG=main
ARG ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit
ARG SGLANG_KERNEL_NPU_TAG=main

ARG PIP_INSTALL="python3 -m pip install --no-cache-dir"
ARG DEVICE_TYPE

RUN if [ "$TARGETARCH" = "amd64" ]; then \
      echo "Using x86_64 dependencies"; \
      echo "PTA_URL=$PTA_URL_AMD64" >> /etc/environment_new; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
      echo "Using aarch64 dependencies"; \
      echo "PTA_URL=$PTA_URL_ARM64" >> /etc/environment_new; \
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
    ca-certificates \
    && rm -rf /var/cache/apt/* \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates \
    && locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8


### Install MemFabric
RUN ${PIP_INSTALL} memfabric-hybrid==1.0.5
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
    (cd sglang/python && rm -rf pyproject.toml && mv pyproject_npu.toml pyproject.toml && ${PIP_INSTALL} -v .[all_npu]) && \
    rm -rf sglang

# Install Deep-ep
# pin wheel to 0.45.1 ref: https://github.com/pypa/wheel/issues/662
RUN ${PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil \
    && mkdir sgl-kernel-npu \
    && cd sgl-kernel-npu \
    && wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.8.0-py311-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && unzip sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.8.0-py311-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && ${PIP_INSTALL} deep_ep*.whl sgl_kernel_npu*.whl \
    && cd .. && rm -rf sgl-kernel-npu \
    && cd "$(python3 -m pip show deep-ep | awk '/^Location:/ {print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so

CMD ["/bin/bash"]
