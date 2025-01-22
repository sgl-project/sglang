#!/bin/bash
set -ex
PYTHON_VERSION=$1
CUDA_VERSION=$2
PYTHON_ROOT_PATH=/opt/python/cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}

docker run --rm \
    -v "$(pwd)":/sgl-kernel \
    pytorch/manylinux-builder:cuda${CUDA_VERSION} \
    bash -c "
    ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.} && \
    export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.9 9.0+PTX' && \
    export CUDA_VERSION=${CUDA_VERSION} && \
    mkdir -p /usr/lib/x86_64-linux-gnu/ && \
    ln -s /usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so && \
    cd /sgl-kernel && \
    ${PYTHON_ROOT_PATH}/bin/python setup.py bdist_wheel
    "
