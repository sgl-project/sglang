ARG CUDA_VERSION=12.5.1

FROM nvcr.io/nvidia/tritonserver:24.04-py3-min

ARG BUILD_TYPE=all
ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt update -y \
    && apt install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa -y && apt update \
    && apt install python3.10 python3.10-dev -y \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10 && apt install python3.10-distutils -y \
    && apt install curl git sudo libibverbs-dev -y \
    && apt install -y rdma-core infiniband-diags openssh-server perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py \
    && python3 --version \
    && python3 -m pip --version \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

# For openbmb/MiniCPM models
RUN pip3 install datamodel_code_generator

WORKDIR /sgl-workspace

ARG CUDA_VERSION
RUN python3 -m pip install --upgrade pip setuptools wheel html5lib six \
    && git clone --depth=1 https://github.com/sgl-project/sglang.git \
    && if [ "$CUDA_VERSION" = "12.1.1" ]; then \
         export CUINDEX=121; \
       elif [ "$CUDA_VERSION" = "12.4.1" ]; then \
         export CUINDEX=124; \
       elif [ "$CUDA_VERSION" = "12.5.1" ]; then \
         export CUINDEX=124; \
       elif [ "$CUDA_VERSION" = "11.8.0" ]; then \
         export CUINDEX=118; \
         python3 -m pip install --no-cache-dir sgl-kernel -i https://docs.sglang.ai/whl/cu118; \
       else \
         echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1; \
       fi \
    && python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu${CUINDEX} \
    && cd sglang \
    && python3 -m pip --no-cache-dir install -e "python[${BUILD_TYPE}]" --find-links https://flashinfer.ai/whl/cu${CUINDEX}/torch2.5/flashinfer-python

ENV DEBIAN_FRONTEND=interactive

RUN apt-get update &&  apt-get install -y \
               rdma-core  \
               infiniband-diags \
               openssh-server \
               perftest \
               ibverbs-providers \
               libibumad3 \
               libibverbs1 \
               libnl-3-200  \
               libnl-route-3-200 \
               librdmacm1  \
               build-essential \
               cmake \
               netcat \
               libibverbs-dev \
               libnl-route-3-dev \
               libnl-3-dev \
               librdmacm1 \
               libgoogle-glog-dev \
               libgtest-dev \
               libjsoncpp-dev \
               libcpprest \
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
               libcurl4-openssl-dev \
               libhiredis-dev \
               pkg-config \
               patchelf \
               ccache \
               libcurl4-openssl-dev \
               libhiredis-dev  && rm -rf /var/lib/apt/lists/*
