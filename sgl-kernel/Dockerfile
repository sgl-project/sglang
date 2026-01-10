ARG BASE_IMG=pytorch/manylinux2_28-builder
ARG CUDA_VERSION=12.9

# Dependency stage: install system deps, CMake, ccache, Python deps (including torch)
FROM ${BASE_IMG}:cuda${CUDA_VERSION} AS deps

# Overridable build arguments
ARG ARCH=x86_64
ARG CUDA_VERSION=12.9
ARG PYTHON_VERSION=3.10
# Manylinux python path tag, e.g. cp310-cp310 / cp312-cp312
ARG PYTHON_TAG=cp310-cp310
ARG CMAKE_VERSION_MAJOR=3.31
ARG CMAKE_VERSION_MINOR=1
# Install ccache 4.12.1 from source for CUDA support (yum provides old 3.7.7)
ARG USE_CCACHE=1
ARG CCACHE_VERSION=4.12.1
ARG GITHUB_ARTIFACTORY=github.com
ARG PYTORCH_MIRROR=download.pytorch.org
ARG PIP_DEFAULT_INDEX=https://pypi.python.org/simple

ENV PYTHON_ROOT_PATH=/opt/python/${PYTHON_TAG}
ENV PATH=/opt/cmake/bin:${PATH}
ENV LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH}
ENV NINJA_STATUS="[%f/%t %es] "
ENV FLASHINFER_CUDA_ARCH_LIST="8.0 8.9 9.0a 10.0a 12.0a"
# CUDA headers path
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}
ENV C_INCLUDE_PATH=/usr/local/cuda/include/cccl${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}

# Install build dependencies
RUN yum install gcc gcc-c++ make wget tar numactl-devel libibverbs -y --nogpgcheck \
 && ln -sv /usr/lib64/libibverbs.so.1 /usr/lib64/libibverbs.so \
 && yum clean all && rm -rf /var/cache/yum

# Install CMake (cached download)
RUN --mount=type=cache,id=sgl-kernel-cmake,target=/cmake-downloads \
    set -eux; \
    CMAKE_TARBALL=cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}-linux-${ARCH}.tar.gz; \
    # Check if CMake is already cached
    if [ -f /cmake-downloads/${CMAKE_TARBALL} ]; then \
      echo "Using cached CMake from /cmake-downloads/${CMAKE_TARBALL}"; \
      cp /cmake-downloads/${CMAKE_TARBALL} .; \
    else \
      CMAKE_TARBALL_URL=https://${GITHUB_ARTIFACTORY}/Kitware/CMake/releases/download/v${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}/${CMAKE_TARBALL}; \
      echo "Downloading CMake from: ${CMAKE_TARBALL_URL}"; \
      wget --progress=dot ${CMAKE_TARBALL_URL}; \
      # Cache the downloaded file
      cp ${CMAKE_TARBALL} /cmake-downloads/; \
    fi; \
    tar -xzf ${CMAKE_TARBALL}; \
    mv cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}-linux-${ARCH} /opt/cmake; \
    rm -f ${CMAKE_TARBALL}; \
    cmake --version

# Install ccache
RUN if [ "${USE_CCACHE}" = "1" ]; then \
    set -eux && \
    cd /tmp && \
    wget --progress=dot https://${GITHUB_ARTIFACTORY}/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}.tar.xz && \
    tar -xf ccache-${CCACHE_VERSION}.tar.xz && \
    cd ccache-${CCACHE_VERSION} && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr -D ENABLE_TESTING=OFF -D REDIS_STORAGE_BACKEND=OFF -D HTTP_STORAGE_BACKEND=OFF -D ENABLE_DOCUMENTATION=OFF .. && \
    make -j"$(nproc)" && \
    make install && \
    ccache --version && \
    rm -rf /tmp/ccache-${CCACHE_VERSION}*; \
  else \
    echo "Skipping ccache build (USE_CCACHE=${USE_CCACHE})"; \
  fi

RUN set -eux; \
    if [ "${ARCH}" = "aarch64" ]; then _LIB=sbsa; else _LIB="${ARCH}"; fi; \
    mkdir -p /usr/lib/${ARCH}-linux-gnu/; \
    ln -sf /usr/local/cuda-${CUDA_VERSION}/targets/${_LIB}-linux/lib/stubs/libcuda.so /usr/lib/${ARCH}-linux-gnu/libcuda.so

# Install Python dependencies (torch + build tools)
RUN --mount=type=cache,id=sgl-kernel-pip,target=/root/.cache/pip \
    set -eux; \
    case "${CUDA_VERSION}" in \
      13.0) TORCH_VER=2.9.1; CU_TAG=cu130 ;; \
      12.9) TORCH_VER=2.9.1; CU_TAG=cu128 ;; \
      12.8) TORCH_VER=2.9.1; CU_TAG=cu128 ;; \
      *)    TORCH_VER=2.9.1; CU_TAG=cu126 ;; \
    esac; \
    ${PYTHON_ROOT_PATH}/bin/pip install torch==${TORCH_VER} --index-url https://${PYTORCH_MIRROR}/whl/${CU_TAG}; \
    ${PYTHON_ROOT_PATH}/bin/pip install ninja setuptools==75.0.0 wheel==0.41.0 numpy uv scikit-build-core --index-url ${PIP_DEFAULT_INDEX}

# Build stage: copy source and build wheel
FROM deps AS build
WORKDIR /sgl-kernel
# Only copy sgl-kernel source so code changes only affect later layers
COPY . /sgl-kernel/

# Optional: enable CMake/Ninja profiling (pass non-empty via --build-arg ENABLE_*)
ARG ENABLE_CMAKE_PROFILE
ARG ENABLE_BUILD_PROFILE
# Optional: cmake extra args for aarch64 full kernel build (pass non-empty via --build-arg CMAKE_EXTRA_ARGS)
ARG CMAKE_EXTRA_ARGS
ARG ARCH=x86_64
ARG USE_CCACHE=1

RUN --mount=type=cache,id=sgl-kernel-ccache,target=/ccache \
    --mount=type=cache,id=sgl-kernel-pip,target=/root/.cache/pip \
    set -eux; \
    if [ "${USE_CCACHE}" = "1" ]; then \
      export CCACHE_DIR=/ccache; \
      export CCACHE_BASEDIR=/sgl-kernel; \
      export CCACHE_MAXSIZE=10G; \
      export CCACHE_COMPILERCHECK=content; \
      export CCACHE_COMPRESS=true; \
      export CCACHE_SLOPPINESS=file_macro,time_macros,include_file_mtime,include_file_ctime; \
      export CMAKE_C_COMPILER_LAUNCHER=ccache; \
      export CMAKE_CXX_COMPILER_LAUNCHER=ccache; \
      export CMAKE_CUDA_COMPILER_LAUNCHER=ccache; \
      ccache -sV; \
    fi; \
    # Setting these flags to reduce OOM chance only on ARM
    if [ "${ARCH}" = "aarch64" ]; then \
      export CUDA_NVCC_FLAGS="-Xcudafe --threads=2"; \
      export MAKEFLAGS="-j2"; \
      export CMAKE_BUILD_PARALLEL_LEVEL=2; \
      export NINJAFLAGS="-j2"; \
      echo "ARM detected: Using extra conservative settings (2 parallel jobs)"; \
    else \
      export CMAKE_BUILD_PARALLEL_LEVEL=$(echo "$(( $(nproc) / 3 )) 48" | awk '{print ($1 < $2) ? $1 : $2}'); \
    fi; \
    if [ -n "${ENABLE_CMAKE_PROFILE:-}" ]; then \
      echo "CMake profiling enabled - will save to /sgl-kernel/cmake-profile.json"; \
      export CMAKE_ARGS="--profiling-output=/sgl-kernel/cmake-profile.json --profiling-format=google-trace"; \
    fi; \
    # Merge Extra CMake options into CMAKE_ARGS \
    export CMAKE_ARGS="${CMAKE_ARGS:+${CMAKE_ARGS} }${CMAKE_EXTRA_ARGS:-}"; \
    echo "CMAKE_ARGS= [$CMAKE_ARGS]"; \
    ${PYTHON_ROOT_PATH}/bin/python -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation; \
    ./rename_wheels.sh; \
    if [ -n "${ENABLE_BUILD_PROFILE:-}" ] && [ -f /sgl-kernel/build/.ninja_log ]; then \
      echo "Ninja build profiling enabled - will save to /sgl-kernel/build-trace.json"; \
      wget --progress=dot https://raw.githubusercontent.com/cradleapps/ninjatracing/084212eaf68f25c70579958a2ed67fb4ec2a9ca4/ninjatracing -O /tmp/ninjatracing; \
      if [ -f /tmp/ninjatracing ]; then \
        ${PYTHON_ROOT_PATH}/bin/python /tmp/ninjatracing /sgl-kernel/build/.ninja_log > /sgl-kernel/build-trace.json; \
      fi; \
      if [ -f /sgl-kernel/build-trace.json ]; then \
        gzip -9 -k /sgl-kernel/build-trace.json 2>/dev/null || true; \
        echo "Build trace saved to: build-trace.json"; \
        if [ -f /sgl-kernel/build-trace.json.gz ]; then \
          ORIGINAL_SIZE=$(stat -f%z /sgl-kernel/build-trace.json 2>/dev/null || stat -c%s /sgl-kernel/build-trace.json); \
          COMPRESSED_SIZE=$(stat -f%z /sgl-kernel/build-trace.json.gz 2>/dev/null || stat -c%s /sgl-kernel/build-trace.json.gz); \
          RATIO=$(( (ORIGINAL_SIZE - COMPRESSED_SIZE) * 100 / ORIGINAL_SIZE )); \
          echo "Compressed to: build-trace.json.gz (${RATIO}% smaller)"; \
        fi; \
        echo ""; \
        echo "View in browser:"; \
        echo "  - chrome://tracing (load JSON file)"; \
        echo "  - ui.perfetto.dev (recommended, supports .gz files)"; \
        echo ""; \
        echo "Shows:"; \
        echo "  - Compilation time per file"; \
        echo "  - Parallelism utilization"; \
        echo "  - Critical path (longest dependency chain)"; \
        echo "  - Where the 2-hour build time went"; \
      fi; \
    fi; \
    if [ "${USE_CCACHE}" = "1" ]; then \
      echo "ccache Statistics"; \
      ccache -s; \
    else \
      echo "ccache disabled (USE_CCACHE=${USE_CCACHE})"; \
    fi

# Artifact stage (for --output to export wheel)
FROM scratch AS artifact
COPY --from=build /sgl-kernel/dist/*.whl /
