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
else
   LIBCUDA_ARCH=${ARCH}
   BUILDER_NAME="pytorch/manylinux2_28-builder"
fi

if [ ${CUDA_VERSION} = "12.9" ]; then
   DOCKER_IMAGE="${BUILDER_NAME}:cuda${CUDA_VERSION}"
   TORCH_INSTALL="pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129"
elif [ ${CUDA_VERSION} = "12.8" ]; then
   DOCKER_IMAGE="${BUILDER_NAME}:cuda${CUDA_VERSION}"
   TORCH_INSTALL="pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128"
else
   DOCKER_IMAGE="${BUILDER_NAME}:cuda${CUDA_VERSION}"
   TORCH_INSTALL="pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126"
fi

# Create cache directories for persistent build artifacts
CACHE_DIR="${PWD}/.cache"
CMAKE_BUILD_CACHE="${CACHE_DIR}/cmake-build-py${PYTHON_VERSION}-cuda${CUDA_VERSION}-${ARCH}"
CMAKE_DOWNLOAD_CACHE="${CACHE_DIR}/cmake-downloads"

mkdir -p "${CMAKE_BUILD_CACHE}"
mkdir -p "${CMAKE_DOWNLOAD_CACHE}"

echo "Using CMake build cache: ${CMAKE_BUILD_CACHE}"

docker run --rm \
   -v $(pwd):/sgl-kernel \
   -v ${CMAKE_BUILD_CACHE}:/cmake-build-cache \
   -v ${CMAKE_DOWNLOAD_CACHE}:/cmake-downloads \
   ${DOCKER_IMAGE} \
   bash -c "
   # Install CMake (version >= 3.26) - Robust Installation with caching
   export CMAKE_VERSION_MAJOR=3.31
   export CMAKE_VERSION_MINOR=1
   # Setting these flags to reduce OOM chance only on ARM
   export CMAKE_BUILD_PARALLEL_LEVEL=$(( $(nproc)/3 < 48 ? $(nproc)/3 : 48 ))
   if [ \"${ARCH}\" = \"aarch64\" ]; then
      export CUDA_NVCC_FLAGS=\"-Xcudafe --threads=2\"
      export MAKEFLAGS='-j2'
      export CMAKE_BUILD_PARALLEL_LEVEL=2
      export NINJAFLAGS='-j2'
   fi

   CMAKE_TARBALL=\"cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-${ARCH}.tar.gz\"

   # Check if CMake is already cached
   if [ -f \"/cmake-downloads/\${CMAKE_TARBALL}\" ]; then
      echo \"Using cached CMake from /cmake-downloads/\${CMAKE_TARBALL}\"
      cp /cmake-downloads/\${CMAKE_TARBALL} .
   else
      echo \"Downloading CMake from: https://cmake.org/files/v\${CMAKE_VERSION_MAJOR}/\${CMAKE_TARBALL}\"
      wget https://cmake.org/files/v\${CMAKE_VERSION_MAJOR}/\${CMAKE_TARBALL}
      # Cache the downloaded file
      cp \${CMAKE_TARBALL} /cmake-downloads/
   fi

   tar -xzf \${CMAKE_TARBALL}
   mv cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-${ARCH} /opt/cmake
   export PATH=/opt/cmake/bin:\$PATH
   export LD_LIBRARY_PATH=/lib64:\$LD_LIBRARY_PATH

   # Debugging CMake
   echo \"PATH: \$PATH\"
   which cmake
   cmake --version

   yum install numactl-devel -y && \
   yum install libibverbs -y --nogpgcheck && \
   ln -sv /usr/lib64/libibverbs.so.1 /usr/lib64/libibverbs.so && \
   ${PYTHON_ROOT_PATH}/bin/${TORCH_INSTALL} && \
   ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir ninja setuptools==75.0.0 wheel==0.41.0 numpy uv scikit-build-core && \
   export TORCH_CUDA_ARCH_LIST='8.0 8.9 9.0+PTX' && \
   export CUDA_VERSION=${CUDA_VERSION} && \
   mkdir -p /usr/lib/${ARCH}-linux-gnu/ && \
   ln -s /usr/local/cuda-${CUDA_VERSION}/targets/${LIBCUDA_ARCH}-linux/lib/stubs/libcuda.so /usr/lib/${ARCH}-linux-gnu/libcuda.so && \

   cd /sgl-kernel && \
   ls -la ${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages/wheel/ && \

   # Use persistent build cache directory
   # Copy cached build artifacts if they exist
   if [ -d /cmake-build-cache ] && [ \"\$(ls -A /cmake-build-cache 2>/dev/null)\" ]; then
      echo \"Restoring CMake build cache...\"
      mkdir -p build
      cp -r /cmake-build-cache/* build/ || true
   fi

   # Build with cached directory
   PYTHONPATH=${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages ${PYTHON_ROOT_PATH}/bin/python -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation && \

   # Save build artifacts back to cache
   echo \"Saving CMake build cache...\"
   rm -rf /cmake-build-cache/*
   cp -r build/* /cmake-build-cache/ || true

   ./rename_wheels.sh
   "
