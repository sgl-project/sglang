#!/bin/bash
set -ex

PYTHON_VERSION=$1
CUDA_VERSION=$2
PYTHON_ROOT_PATH=/opt/python/cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}

if [ -z "$3" ]; then
   ARCH=$(uname -i)
else
   ARCH=$3
fi

echo "ARCH:  $ARCH"
if [ ${ARCH} = "aarch64" ]; then
   LIBCUDA_ARCH="sbsa"
   BUILDER_NAME="pytorch/manylinuxaarch64-builder"
   CMAKE_BUILD_PARALLEL_LEVEL=16
else
   LIBCUDA_ARCH=${ARCH}
   BUILDER_NAME="pytorch/manylinux2_28-builder"
fi

if [ ${CUDA_VERSION} = "12.8" ]; then
   DOCKER_IMAGE="${BUILDER_NAME}:cuda${CUDA_VERSION}"
   TORCH_INSTALL="pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.}"
else
   DOCKER_IMAGE="${BUILDER_NAME}:cuda${CUDA_VERSION}"
   TORCH_INSTALL="pip install --no-cache-dir torch==2.7.1"
fi

docker run --rm \
   -v $(pwd):/sgl-kernel \
   ${DOCKER_IMAGE} \
   bash -c "
   # Install CMake (version >= 3.26) - Robust Installation
   export CMAKE_VERSION_MAJOR=3.31
   export CMAKE_VERSION_MINOR=1
   echo \"Downloading CMake from: https://cmake.org/files/v\${CMAKE_VERSION_MAJOR}/cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-${ARCH}.tar.gz\"
   wget https://cmake.org/files/v\${CMAKE_VERSION_MAJOR}/cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-${ARCH}.tar.gz
   tar -xzf cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-${ARCH}.tar.gz
   mv cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-${ARCH} /opt/cmake
   export PATH=/opt/cmake/bin:\$PATH
   export LD_LIBRARY_PATH=/lib64:\$LD_LIBRARY_PATH

   # Debugging CMake
   echo \"PATH: \$PATH\"
   which cmake
   cmake --version

   yum install numactl-devel -y && \
   yum install libibverbs -y && \
   ln -sv /usr/lib64/libibverbs.so.1 /usr/lib64/libibverbs.so && \
   ${PYTHON_ROOT_PATH}/bin/${TORCH_INSTALL} && \
   ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir ninja setuptools==75.0.0 wheel==0.41.0 numpy uv scikit-build-core && \
   export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.9 9.0+PTX' && \
   export CUDA_VERSION=${CUDA_VERSION} && \
   mkdir -p /usr/lib/${ARCH}-linux-gnu/ && \
   ln -s /usr/local/cuda-${CUDA_VERSION}/targets/${LIBCUDA_ARCH}-linux/lib/stubs/libcuda.so /usr/lib/${ARCH}-linux-gnu/libcuda.so && \

   cd /sgl-kernel && \
   ls -la ${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages/wheel/ && \
   PYTHONPATH=${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages ${PYTHON_ROOT_PATH}/bin/python -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation && \
   ./rename_wheels.sh
   "
