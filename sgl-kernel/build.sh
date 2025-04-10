#!/bin/bash
set -ex

PYTHON_VERSION=$1
CUDA_VERSION=$2
PYTHON_ROOT_PATH=/opt/python/cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}
ENABLE_SM90A=$(( ${CUDA_VERSION%.*} >= 12 ? ON : OFF ))

if [ ${CUDA_VERSION} = "12.8" ]; then
   DOCKER_IMAGE="pytorch/manylinux2_28-builder:cuda${CUDA_VERSION}"
   TORCH_INSTALL="pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/nightly/cu${CUDA_VERSION//.}"
else
   DOCKER_IMAGE="pytorch/manylinux-builder:cuda${CUDA_VERSION}"
   TORCH_INSTALL="pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.}"
fi

docker run --rm \
   -v $(pwd):/sgl-kernel \
   ${DOCKER_IMAGE} \
   bash -c "
   ${PYTHON_ROOT_PATH}/bin/${TORCH_INSTALL} && \
   ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir ninja setuptools==75.0.0 wheel==0.41.0 numpy uv && \
   export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.9 9.0+PTX' && \
   export CUDA_VERSION=${CUDA_VERSION} && \
   mkdir -p /usr/lib/x86_64-linux-gnu/ && \
   ln -s /usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so && \
   cd /sgl-kernel && \
   ls -la ${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages/wheel/ && \
   PYTHONPATH=${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages ${PYTHON_ROOT_PATH}/bin/python -m uv build --wheel -Cbuild-dir=build . --color=always && \
   ./rename_wheels.sh
   "
