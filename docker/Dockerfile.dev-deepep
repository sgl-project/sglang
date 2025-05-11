FROM lmsysorg/sglang:latest

# CMake
RUN apt-get update \
&& apt-get install -y --no-install-recommends \
build-essential \
wget \
libssl-dev \
&& wget https://github.com/Kitware/CMake/releases/download/v3.27.4/cmake-3.27.4-linux-x86_64.sh \
&& chmod +x cmake-3.27.4-linux-x86_64.sh \
&& ./cmake-3.27.4-linux-x86_64.sh --skip-license --prefix=/usr/local \
&& rm cmake-3.27.4-linux-x86_64.sh

# Python
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
    && ln -s /usr/bin/python3 /usr/bin/python

# GDRCopy
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/gdrcopy.git
WORKDIR /tmp/gdrcopy
RUN git checkout v2.4.4

RUN apt update
RUN apt install -y nvidia-dkms-535
RUN apt install -y build-essential devscripts debhelper fakeroot pkg-config dkms
RUN apt install -y check libsubunit0 libsubunit-dev

WORKDIR /tmp/gdrcopy/packages
RUN CUDA=/usr/local/cuda ./build-deb-packages.sh
RUN dpkg -i gdrdrv-dkms_*.deb
RUN dpkg -i libgdrapi_*.deb
RUN dpkg -i gdrcopy-tests_*.deb
RUN dpkg -i gdrcopy_*.deb

ENV GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/

# IBGDA dependency
RUN ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so
RUN apt-get install -y libfabric-dev

# DeepEP
WORKDIR /sgl-workspace
RUN git clone https://github.com/deepseek-ai/DeepEP.git

# NVSHMEM
WORKDIR /sgl-workspace
RUN wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz
RUN tar -xf nvshmem_src_3.2.5-1.txz \
    && mv nvshmem_src nvshmem

WORKDIR /sgl-workspace/nvshmem
RUN git apply /sgl-workspace/DeepEP/third-party/nvshmem.patch

WORKDIR /sgl-workspace/nvshmem
ENV CUDA_HOME=/usr/local/cuda
RUN NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/sgl-workspace/nvshmem/install -DCMAKE_CUDA_ARCHITECTURES=90 \
    && cd build \
    && make install -j

WORKDIR /sgl-workspace/DeepEP
ENV NVSHMEM_DIR=/sgl-workspace/nvshmem/install
RUN NVSHMEM_DIR=/sgl-workspace/nvshmem/install pip install .

# Set workspace
WORKDIR /sgl-workspace
