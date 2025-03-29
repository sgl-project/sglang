#!/bin/bash
set -ex
PYTHON_VERSION=$1
CUDA_VERSION=$2
PYTHON_ROOT_PATH=/opt/python/cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}

if (( ${CUDA_VERSION%.*} < 12 )); then
    ENABLE_SM90A=0
else
    ENABLE_SM90A=1
fi

docker run --rm \
    -v $(pwd):/sgl-kernel \
    pytorch/manylinux-builder:cuda${CUDA_VERSION} \
    bash -c "
    ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.} && \
    ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir ninja setuptools==75.0.0 wheel==0.41.0 numpy uv && \
    export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.9 9.0+PTX' && \
    export CUDA_VERSION=${CUDA_VERSION} && \
    export SGL_KERNEL_ENABLE_BF16=1 && \
    export SGL_KERNEL_ENABLE_FP8=1 && \
    export SGL_KERNEL_ENABLE_SM90A=${ENABLE_SM90A} && \
    mkdir -p /usr/lib/x86_64-linux-gnu/ && \
    ln -s /usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so && \
    cd /sgl-kernel && \
    ls -la ${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages/wheel/ && \
    PYTHONPATH=${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages ${PYTHON_ROOT_PATH}/bin/python -m uv build --wheel -Cbuild-dir=build . --color=always && \
    ./rename_wheels.sh
    "
