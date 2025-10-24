ARG BUILDER_NAME=pytorch/manylinux2_28-builder
ARG CUDA_VERSION=12.6
FROM ${BUILDER_NAME}:cuda${CUDA_VERSION}

ARG PYTHON_VERSION_NO_DOT=311
ARG ARCH=x86_64
ARG LIBCUDA_ARCH=x86_64

# Set environment variables
ENV CMAKE_VERSION_MAJOR=3.31
ENV CMAKE_VERSION_MINOR=1
ENV PATH=/opt/cmake/bin:$PATH
ENV LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH
ENV PYTHON_ROOT_PATH=/opt/python/cp${PYTHON_VERSION_NO_DOT}-cp${PYTHON_VERSION_NO_DOT}

# Set build parallelism based on architecture
RUN if [ "${ARCH}" = "aarch64" ]; then \
      export CUDA_NVCC_FLAGS="-Xcudafe --threads=2" && \
      export MAKEFLAGS='-j2' && \
      export SGL_KERNEL_COMPILE_THREADS=2 && \
      export NINJAFLAGS='-j2'; \
    else \
      export SGL_KERNEL_COMPILE_THREADS=$(( $(nproc)/3 < 48 ? $(nproc)/3 : 48 )); \
    fi

# Install CMake
RUN wget --progress=bar:force:noscroll https://y.y.y/aie-shared-objects/cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}-linux-${ARCH}.tar.gz && \
    tar -xzf cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}-linux-${ARCH}.tar.gz && \
    mv cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}-linux-${ARCH} /opt/cmake && \
    rm cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}-linux-${ARCH}.tar.gz

# Install system dependencies
RUN yum install numactl-devel -y && \
    yum install libibverbs -y --nogpgcheck && \
    ln -sv /usr/lib64/libibverbs.so.1 /usr/lib64/libibverbs.so

# Install PyTorch based on CUDA version
RUN if [ "${CUDA_VERSION}" = "12.9" ]; then \
      ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir torch==2.8.0 --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu129; \
    elif [ "${CUDA_VERSION}" = "12.8" ]; then \
      ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir torch==2.8.0 --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128; \
    else \
      ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir torch==2.8.0 --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu126; \
    fi

# Configure pip and install Python dependencies
RUN ${PYTHON_ROOT_PATH}/bin/pip config set global.index-url https://x.x.x/artifactory/api/pypi/pypi-virtual/simple && \
    ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir ninja setuptools==75.0.0 wheel==0.41.0 numpy uv scikit-build-core

# Set CUDA environment variables
ENV TORCH_CUDA_ARCH_LIST='8.0 8.9 9.0+PTX'
ENV CUDA_VERSION=${CUDA_VERSION}

# Create symbolic link for libcuda.so
RUN mkdir -p /usr/lib/${ARCH}-linux-gnu/ && \
    ln -s /usr/local/cuda-${CUDA_VERSION}/targets/${LIBCUDA_ARCH}-linux/lib/stubs/libcuda.so /usr/lib/${ARCH}-linux-gnu/libcuda.so

WORKDIR /sgl-kernel

# Build command will be run via docker run
CMD ["bash"]
