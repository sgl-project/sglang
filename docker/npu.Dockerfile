ARG CANN_VERSION=9.2.0
ARG DEVICE_TYPE=950
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=py3.12
ARG SGLANG_TAG=main

FROM quay.io/ascend/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION AS sgl-router-builder

ARG SGLANG_TAG

ENV DEBIAN_FRONTEND=noninteractive \
    PCRE2_SYS_STATIC=1 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        pkg-config \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --single-branch \
        --branch "${SGLANG_TAG}" \
        https://github.com/sgl-project/sglang.git \
        /src/sglang \
    && curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf \
        https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain none \
    && cd /src/sglang/sgl-router \
    # Install the toolchain pinned by sgl-router/rust-toolchain.toml.
    && rustup toolchain install \
    && cargo build --locked --release --bin sgl-router \
    && install -Dm755 target/release/sgl-router /out/sgl-router \
    && /out/sgl-router --help >/dev/null

FROM quay.io/ascend/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION

ARG CANN_VERSION
ARG DEVICE_TYPE

ARG PYTORCH_VERSION="2.13.0"
ARG TORCHVISION_VERSION="0.28.0"
ARG TORCHAUDIO_VERSION="2.11.0"
ARG TORCH_NPU_VERSION="2.13.0"
ARG TORCH_NPU_WHEEL_URL_AARCH64="https://gitcode.com/Ascend/pytorch/releases/download/v26.2.0-pytorch2.13.0/torch_npu-2.13.0-cp312-cp312-manylinux_2_28_aarch64.whl"
ARG TORCH_NPU_WHEEL_URL_X86_64="https://gitcode.com/Ascend/pytorch/releases/download/v26.2.0-pytorch2.13.0/torch_npu-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl"
ARG TORCH_NPU_INDEX_URL="https://ascend.devcloud.huaweicloud.com/pypi/simple/"

ARG TRITON_AARCH64="https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261105160244/triton_ascend-3.6.0%2Bdev20261105160244-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_34_aarch64.whl"
ARG TRITON_X86_64="https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261105160244/triton_ascend-3.6.0%2Bdev20261105160244-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_34_x86_64.whl"

ARG SGLANG_TAG=main

ARG SGLANG_KERNEL_NPU_TAG=2026.11.0.post15

ARG PIP_INDEX_URL="https://pypi.org/simple/"
ARG PIP_INSTALL="python3 -m pip install --no-cache-dir"
ARG TARGETARCH

COPY --from=sgl-router-builder /out/sgl-router /usr/local/bin/sgl-router

RUN sgl-router --help >/dev/null

WORKDIR /workspace

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

RUN pip config set global.index-url $PIP_INDEX_URL

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
    git \
    ninja-build \
    patchelf \
    libyaml-cpp-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libjsoncpp-dev \
    libunwind-dev \
    libnuma-dev \
    libasio-dev \
    libboost-dev \
    libcurl4-openssl-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler-grpc \
    liburing-dev \
    libmsgpack-dev \
    libzstd-dev \
    libxxhash-dev \
    libcli11-dev \
    libibverbs-dev \
    git-lfs \
    && rm -rf /var/cache/apt/* \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates \
    && locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

RUN git lfs install --system

### Install MemFabric
RUN set -eux; \
    case "$DEVICE_TYPE" in \
      950) MF_SOC_VERSION="A5" ;; \
      a3)  MF_SOC_VERSION="A3" ;; \
      *)   echo "Unsupported DEVICE_TYPE for mfcli kernel install: $DEVICE_TYPE" >&2; \
           exit 1 ;; \
    esac; \
    ${PIP_INSTALL} \
        "memfabric-hybrid==1.5.0.dev20261124" \
        -i https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/; \
    command -v mfcli; \
    mfcli kernel install --soc-version "$MF_SOC_VERSION"

RUN ${PIP_INSTALL} \
    wheel==0.45.1 \
    pybind11 \
    decorator \
    attrs \
    evalscope \
    pytest \
    msgpack \
    cannbot-dsl==0.8.0

RUN ${PIP_INSTALL} https://github.com/kvcache-ai/Mooncake/releases/download/v0.3.14/mooncake_transfer_engine_memfabric-0.3.14-cp312-cp312-manylinux_2_35_$(arch).whl --force --no-deps

### Install PyTorch and PTA
RUN set -eux; \
    case "$TARGETARCH" in \
      arm64) TORCH_NPU_WHEEL="${TORCH_NPU_WHEEL_URL_AARCH64}" ;; \
      amd64) TORCH_NPU_WHEEL="${TORCH_NPU_WHEEL_URL_X86_64}" ;; \
      *) echo "Unsupported architecture: $TARGETARCH" >&2; \
         exit 1 ;; \
    esac; \
    ${PIP_INSTALL} torch==${PYTORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/cpu; \
    ${PIP_INSTALL} "${TORCH_NPU_WHEEL}"

## Install triton-ascend
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        ${PIP_INSTALL} "${TRITON_AARCH64}"; \
    elif [ "$TARGETARCH" = "amd64" ]; then \
        ${PIP_INSTALL} "${TRITON_X86_64}"; \
    else \
        echo "Unsupported architecture: $TARGETARCH"; \
        exit 1; \
    fi

#Install SGLang
RUN git clone https://github.com/sgl-project/sglang --branch ${SGLANG_TAG} /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang/python && rm -rf pyproject.toml && mv pyproject_npu.toml pyproject.toml && \
    ${PIP_INSTALL} -v -e .[all_npu]

ENV ASCEND_HOME_PATH=/usr/local/Ascend/cann-${CANN_VERSION}

ENV LD_LIBRARY_PATH=/usr/local/Ascend/cann-${CANN_VERSION}/lib64:/usr/local/Ascend/cann-${CANN_VERSION}/lib:/usr/local/Ascend/cann-${CANN_VERSION}/x86_64-linux/devlib/device:/usr/local/Ascend/driver/lib64:/usr/local/lib:${LD_LIBRARY_PATH}

RUN sed -Ei -e '/^[[:space:]]*"torch\.npu\.random\.manual_seed(_all)?",[[:space:]]*$/d' -e 's/^([[:space:]]*)"torch\.npu\.set_device",/\1"torch.npu.random.manual_seed_all",\n\1"torch.npu.random.manual_seed",\n\1"torch.npu.set_device",/' /usr/local/python3.12.15/lib/python3.12/site-packages/torch_npu/dynamo/trace_rule.py

RUN set -eux; \
    case "${TARGETARCH}" in \
        arm64) \
            ASCEND_ARCH="aarch64"; \
            LIBDIR="/usr/lib/aarch64-linux-gnu"; \
            ;; \
        amd64) \
            ASCEND_ARCH="x86_64"; \
            LIBDIR="/usr/lib/x86_64-linux-gnu"; \
            ;; \
        *) \
            echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2; \
            exit 1; \
            ;; \
    esac; \
    install -Dm755 \
        "/usr/local/Ascend/cann-${CANN_VERSION}/${ASCEND_ARCH}-linux/lib64/libmmpa.so" \
        "${LIBDIR}/libmmpa.so"; \
    ldconfig; \
    ldconfig -p | grep libmmpa.so

RUN set -eux; \
    mkdir /tmp/sgl-kernel-npu; \
    cd /tmp/sgl-kernel-npu; \
    wget \
      "https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip"; \
    unzip ./*.zip; \
    ${PIP_INSTALL} \
        ./sgl_kernel_npu*.whl \
        ./deep_ep*.whl \
        ./attentions*.whl \
        ./torch_memory_saver*.whl; \
    cd "$(python3 -m pip show deep-ep | awk '/^Location:/ {print $2}')"; \
    ln -sf deep_ep/deep_ep_cpp*.so; \
    cd /; \
    rm -rf /tmp/sgl-kernel-npu

CMD ["/bin/bash"]
