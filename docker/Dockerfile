ARG CUDA_VERSION=12.6.1
ARG BUILD_TYPE=all

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# TZ
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections

# Deps
RUN apt-get update &&  apt-get install -y --no-install-recommends software-properties-common netcat-openbsd  \
                        libopenmpi-dev libnuma1 \
                        lsof  zsh ccache tmux htop git-lfs tree \
                        kmod \
                        unzip \
                        rdma-core  \
                        infiniband-diags \
                        openssh-server \
                        perftest \
                        ibverbs-providers \
                        libibumad3 \
                        libibverbs1 \
                        libnl-3-200 \
                        libnl-route-3-200 \
                        librdmacm1  \
                        build-essential \
                        cmake \
                        libibverbs-dev \
                        libgoogle-glog-dev \
                        libgtest-dev \
                        libjsoncpp-dev \
                        libnuma-dev \
                        libibverbs-dev \
                        libunwind-dev \
                        libgoogle-glog-dev \
                        libpython3-dev \
                        libboost-all-dev \
                        libssl-dev \
                        libgrpc-dev \
                        libgrpc++-dev \
                        libprotobuf-dev \
                        protobuf-compiler-grpc \
                        pybind11-dev \
                        libhiredis-dev \
                        pkg-config \
                        patchelf \
                        libcurl4-openssl-dev \
                        curl \
                        libczmq4 \
                        libczmq-dev \
                        libnl-route-3-dev \
                        libnl-3-dev \
                        librdmacm1 \
                        libhiredis-dev  \
                        nvidia-dkms-535 \
                        build-essential \
                        devscripts \
                        debhelper \
                        fakeroot \
                        dkms \
                        check \
                        libsubunit0 \
                        libsubunit-dev \
                        libfabric-dev \
                        python3 \
                        python3-pip \
                        && rm -rf /var/lib/apt/lists/* \
                        && ln -s /usr/bin/python3 /usr/bin/python \
                        && apt clean

# GDR Related
ENV GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/
# GDRCopy
RUN  mkdir -p /tmp \
    && cd /tmp \
    && git clone https://github.com/NVIDIA/gdrcopy.git  -b v2.4.4 \
    && cd /tmp/gdrcopy/packages \
    && CUDA=/usr/local/cuda ./build-deb-packages.sh \
    && dpkg -i gdrdrv-dkms_*.deb \
    && dpkg -i libgdrapi_*.deb \
    && dpkg -i gdrcopy-tests_*.deb \
    && dpkg -i gdrcopy_*.deb \
    && rm -rf /tmp/gdrcopy

# DeepEP RDMA IBGDA Related
RUN ln -sf /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

WORKDIR /sgl-workspace

# Sgl
RUN python3 -m pip install --upgrade pip setuptools wheel html5lib six  --ignore-installed \
    && git clone --depth=1 https://github.com/sgl-project/sglang.git \
    && if [ "$CUDA_VERSION" = "12.1.1" ]; then \
         export CUINDEX=121; \
       elif [ "$CUDA_VERSION" = "12.4.1" ]; then \
         export CUINDEX=124; \
       elif [ "$CUDA_VERSION" = "12.8.1" ]; then \
         export CUINDEX=124; \
       elif [ "$CUDA_VERSION" = "11.8.0" ]; then \
         export CUINDEX=118; \
         python3 -m pip install --no-cache-dir sgl-kernel -i https://docs.sglang.ai/whl/cu118 ; \
       else \
         echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1; \
       fi \
    && if [ "$CUDA_VERSION" = "12.4.1" ]; then \
         python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126 ; \
       else \
         python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu${CUINDEX} ; \
       fi \
    && cd sglang \
    && python3 -m pip --no-cache-dir install -e "python[${BUILD_TYPE}]"  \
    && if [ "$CUDA_VERSION" = "12.8.1" ]; then \
         python3 -m pip install nvidia-nccl-cu12==2.26.2.post1 --force-reinstall --no-deps ; \
       fi

# NVSHMEM
ENV NVSHMEM_DIR=/sgl-workspace/nvshmem/install
RUN wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz \
    && git clone https://github.com/deepseek-ai/DeepEP.git \
    && tar -xf nvshmem_src_3.2.5-1.txz \
    && mv nvshmem_src nvshmem \
    && cd /sgl-workspace/nvshmem \
    && git apply /sgl-workspace/DeepEP/third-party/nvshmem.patch \
    && sed -i '1i#include <unistd.h>' /sgl-workspace/nvshmem/examples/moe_shuffle.cu \
    && cat /sgl-workspace/nvshmem/examples/moe_shuffle.cu \
    && rm -f nvshmem_src_3.2.5-1.txz \
    && cd /sgl-workspace/nvshmem \
    &&  NVSHMEM_SHMEM_SUPPORT=0 \
        NVSHMEM_UCX_SUPPORT=0 \
        NVSHMEM_USE_NCCL=0 \
        NVSHMEM_MPI_SUPPORT=0 \
        NVSHMEM_IBGDA_SUPPORT=1 \
        NVSHMEM_PMIX_SUPPORT=0 \
        NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
        NVSHMEM_USE_GDRCOPY=1 \
        cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/sgl-workspace/nvshmem/install -DCMAKE_CUDA_ARCHITECTURES=90 \
    && cd build \
    && make install -j \
    && cd /sgl-workspace/DeepEP \
    && NVSHMEM_DIR=/sgl-workspace/nvshmem/install pip install .

# PyPi packages
RUN pip3 install --upgrade datamodel_code_generator \
                     mooncake_transfer_engine \
                     pre-commit \
                     pytest \
                     black \
                     isort \
                     icdiff \
                     uv \
                     wheel scikit-build-core

ENV DEBIAN_FRONTEND=interactive
