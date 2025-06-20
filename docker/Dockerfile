ARG CUDA_VERSION=12.6.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG BUILD_TYPE=all
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/ \
    NVSHMEM_DIR=/sgl-workspace/nvshmem/install

# Set timezone and install all packages
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
 && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
 && apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    software-properties-common netcat-openbsd kmod unzip openssh-server \
    curl wget lsof zsh ccache tmux htop git-lfs tree \
    python3 python3-pip python3-dev libpython3-dev \
    build-essential cmake \
    libopenmpi-dev libnuma1 libnuma-dev \
    libibverbs-dev libibverbs1 libibumad3 \
    librdmacm1 libnl-3-200 libnl-route-3-200 libnl-route-3-dev libnl-3-dev \
    ibverbs-providers infiniband-diags perftest \
    libgoogle-glog-dev libgtest-dev libjsoncpp-dev libunwind-dev \
    libboost-all-dev libssl-dev \
    libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev libcurl4-openssl-dev \
    libczmq4 libczmq-dev \
    libfabric-dev \
    patchelf \
    nvidia-dkms-550 \
    devscripts debhelper fakeroot dkms check libsubunit0 libsubunit-dev \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# GDRCopy installation
RUN mkdir -p /tmp/gdrcopy && cd /tmp \
 && git clone https://github.com/NVIDIA/gdrcopy.git -b v2.4.4 \
 && cd gdrcopy/packages \
 && CUDA=/usr/local/cuda ./build-deb-packages.sh \
 && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
 && cd / && rm -rf /tmp/gdrcopy

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

# Clone and install SGLang
WORKDIR /sgl-workspace
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel html5lib six \
 && git clone --depth=1 https://github.com/sgl-project/sglang.git \
 && cd sglang \
 && case "$CUDA_VERSION" in \
      12.6.1) CUINDEX=126 ;; \
      12.8.1) CUINDEX=128 ;; \
      *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
 && python3 -m pip install --no-cache-dir -e "python[${BUILD_TYPE}]" --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX} \
 && if [ "$CUDA_VERSION" = "12.8.1" ]; then \
      python3 -m pip install --no-cache-dir nvidia-nccl-cu12==2.27.3 --force-reinstall --no-deps ; \
      python3 -m pip install --no-cache-dir https://github.com/sgl-project/whl/releases/download/v0.1.9/sgl_kernel-0.1.9+cu128-cp39-abi3-manylinux2014_x86_64.whl --force-reinstall --no-deps ; \
    fi

# Build and install NVSHMEM + DeepEP
RUN wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz \
 && git clone https://github.com/deepseek-ai/DeepEP.git \
 && tar -xf nvshmem_src_3.2.5-1.txz && mv nvshmem_src nvshmem \
 && cd nvshmem \
 && git apply /sgl-workspace/DeepEP/third-party/nvshmem.patch \
 && sed -i '1i#include <unistd.h>' examples/moe_shuffle.cu \
 && rm -f /sgl-workspace/nvshmem_src_3.2.5-1.txz \
 && NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=${NVSHMEM_DIR} -DCMAKE_CUDA_ARCHITECTURES=90 \
 && cmake --build build --target install -j \
 && cd /sgl-workspace/DeepEP \
 && NVSHMEM_DIR=${NVSHMEM_DIR} pip install .

# Python tools
RUN python3 -m pip install --no-cache-dir \
    datamodel_code_generator \
    mooncake_transfer_engine==0.3.3.post2 \
    pre-commit \
    pytest \
    black \
    isort \
    icdiff \
    uv \
    wheel \
    scikit-build-core

ENV DEBIAN_FRONTEND=interactive
