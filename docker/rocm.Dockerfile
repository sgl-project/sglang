# Usage (to build SGLang ROCm docker image):
#   docker build --build-arg SGL_BRANCH=v0.5.5.post3 --build-arg GPU_ARCH=gfx942 -t v0.5.5.post3-rocm630-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.5.post3 --build-arg GPU_ARCH=gfx942-rocm700 -t v0.5.5.post3-rocm700-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.5.post3 --build-arg GPU_ARCH=gfx950 -t v0.5.5.post3-rocm700-mi35x -f rocm.Dockerfile .


# Default base images
ARG BASE_IMAGE_942="rocm/sgl-dev:vllm20250114"
ARG BASE_IMAGE_942_ROCM700="rocm/sgl-dev:rocm7-vllm-20250904"
ARG BASE_IMAGE_950="rocm/sgl-dev:rocm7-vllm-20250904"

# This is necessary for scope purpose
ARG GPU_ARCH=gfx950

# ===============================
# Base image 942 with rocm630 and args
FROM $BASE_IMAGE_942 AS gfx942
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="1"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT="v0.1.4"
ENV NO_DEPS_FLAG=""
ENV AITER_MXFP4_MOE_SF="0"

# ===============================
# Base image 942 and args
FROM $BASE_IMAGE_942_ROCM700 AS gfx942-rocm700
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="0"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT="v0.1.7.post1"
ENV NO_DEPS_FLAG=""
ENV AITER_MXFP4_MOE_SF="0"

# ===============================
# Base image 950 and args
FROM $BASE_IMAGE_950 AS gfx950
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="0"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT="v0.1.7.post1"
ENV NO_DEPS_FLAG=""
ENV AITER_MXFP4_MOE_SF="1"
# ===============================
# Chosen arch and args
FROM ${GPU_ARCH}

# This is necessary for scope purpose, again
ARG GPU_ARCH=gfx950
ENV GPU_ARCH_LIST=${GPU_ARCH%-*}

ARG SGL_REPO="https://github.com/sgl-project/sglang.git"
ARG SGL_DEFAULT="main"
ARG SGL_BRANCH=${SGL_DEFAULT}

ARG TRITON_REPO="https://github.com/ROCm/triton.git"
ARG TRITON_COMMIT="improve_fa_decode_3.0.0"

ARG AITER_REPO="https://github.com/ROCm/aiter.git"

ARG LLVM_REPO="https://github.com/jrbyrnes/llvm-project.git"
ARG LLVM_BRANCH="MainOpSelV2"
ARG LLVM_COMMIT="6520ace8227ffe2728148d5f3b9872a870b0a560"

ARG MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"
ARG MOONCAKE_COMMIT="b6a841dc78c707ec655a563453277d969fb8f38d"

ARG TILELANG_REPO="https://github.com/HaiShaw/tilelang.git"
ARG TILELANG_BRANCH="dsv32-mi35x"
ARG TILELANG_COMMIT="ae938cf885743f165a19656d1122ad42bb0e30b8"

ARG FHT_REPO="https://github.com/jeffdaily/fast-hadamard-transform.git"
ARG FHT_BRANCH="rocm"
ARG FHT_COMMIT="46efb7d776d38638fc39f3c803eaee3dd7016bd1"
USER root

# Install some basic utilities
RUN python -m pip install --upgrade pip && pip install setuptools_scm
RUN apt-get purge -y sccache; python -m pip uninstall -y sccache; rm -f "$(which sccache)"

WORKDIR /sgl-workspace

# -----------------------
# llvm
RUN if [ "$BUILD_LLVM" = "1" ]; then \
     ENV HIP_CLANG_PATH="/sgl-workspace/llvm-project/build/bin/" \
     git clone --single-branch ${LLVM_REPO} -b ${LLVM_BRANCH} \
     && cd llvm-project \
     && git checkout ${LLVM_COMMIT} \
     && mkdir build \
     && cd build \
     && cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_ENABLE_PROJECTS="clang;lld;" -DLLVM_ENABLE_RUNTIMES="compiler-rt" ../llvm \
     && make -j$(nproc); \
    fi

# -----------------------
# AITER
RUN pip uninstall -y aiter
RUN git clone ${AITER_REPO} \
 && cd aiter \
 && git checkout ${AITER_COMMIT} \
 && git submodule update --init --recursive
RUN cd aiter \
     && if [ "$GPU_ARCH" = "gfx950" ]; then export AITER_MXFP4_MOE_SF=1; fi \
     && echo "[AITER] GPU_ARCH=${GPU_ARCH} AITER_MXFP4_MOE_SF=${AITER_MXFP4_MOE_SF:-unset}" \
     && if [ "$BUILD_AITER_ALL" = "1" ] && [ "$BUILD_LLVM" = "1" ]; then \
          sh -c "HIP_CLANG_PATH=/sgl-workspace/llvm-project/build/bin/ PREBUILD_KERNELS=1 GPU_ARCHS=$GPU_ARCH_LIST python setup.py develop"; \
        elif [ "$BUILD_AITER_ALL" = "1" ]; then \
          sh -c "PREBUILD_KERNELS=1 GPU_ARCHS=$GPU_ARCH_LIST python setup.py develop"; \
        else \
          sh -c "GPU_ARCHS=$GPU_ARCH_LIST python setup.py develop"; \
        fi

# -----------------------
# Triton
RUN if [ "$BUILD_TRITON" = "1" ]; then \
        pip uninstall -y triton \
     && git clone ${TRITON_REPO} \
     && cd triton \
     && git checkout ${TRITON_COMMIT} \
     && cd python \
     && python setup.py install; \
    fi

# -----------------------
# Build vLLM
ARG VLLM_REPO="https://github.com/ROCm/vllm.git"
ARG VLLM_BRANCH="9f6b92db47c3444b7a7d67451ba0c3a2d6af4c2c"
RUN if [ "$BUILD_VLLM" = "1" ]; then \
        git clone ${VLLM_REPO} \
     && cd vllm \
     && git checkout ${VLLM_BRANCH} \
     && python -m pip install -r requirements/rocm.txt \
     && python setup.py clean --all \
     && python setup.py develop; \
    fi

# -----------------------
# Build Mooncake
ENV PATH=$PATH:/usr/local/go/bin

RUN if [ "$BUILD_MOONCAKE" = "1" ]; then \
     apt update && apt install -y zip unzip wget && \
     apt install -y gcc make libtool autoconf  librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool  libibverbs-dev rdma-core && \
     apt install -y openssh-server openmpi-bin openmpi-common libopenmpi-dev && \
     git clone ${MOONCAKE_REPO} && \
     cd Mooncake && \
     git checkout ${MOONCAKE_COMMIT} && \
     git submodule update --init --recursive && \
     bash dependencies.sh -y && \
     rm -rf /usr/local/go && \
     wget https://go.dev/dl/go1.22.2.linux-amd64.tar.gz && \
     tar -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz && \
     rm go1.22.2.linux-amd64.tar.gz && \
     mkdir -p build && \
     cd build && \
     cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON && \
     make -j "$(nproc)" && make install; \
    fi

# -----------------------
# Build SGLang
ARG BUILD_TYPE=all

RUN pip install IPython \
    && pip install orjson \
    && pip install python-multipart \
    && pip install torchao==0.9.0 \
    && pip install pybind11

RUN pip uninstall -y sgl_kernel sglang
RUN git clone ${SGL_REPO} \
    && cd sglang \
    && if [ "${SGL_BRANCH}" = ${SGL_DEFAULT} ]; then \
         echo "Using ${SGL_DEFAULT}, default branch."; \
         git checkout ${SGL_DEFAULT}; \
       else \
         echo "Using ${SGL_BRANCH} branch."; \
         git checkout ${SGL_BRANCH}; \
       fi \
    && cd sgl-kernel \
    && rm -f pyproject.toml \
    && mv pyproject_rocm.toml pyproject.toml \
    && AMDGPU_TARGET=$GPU_ARCH_LIST python setup_rocm.py install \
    && cd .. \
    && rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml \
    && if [ "$BUILD_TYPE" = "srt" ]; then \
         python -m pip --no-cache-dir install -e "python[srt_hip]" ${NO_DEPS_FLAG}; \
       else \
         python -m pip --no-cache-dir install -e "python[all_hip]" ${NO_DEPS_FLAG}; \
       fi

RUN python -m pip cache purge

# Copy config files to support MI300X in virtualized environments (MI300X_VF).  Symlinks will not be created in image build.
RUN find /sgl-workspace/sglang/python/sglang/srt/layers/quantization/configs/ \
         /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/ \
         -type f -name '*MI300X*' | xargs -I {} sh -c 'vf_config=$(echo "$1" | sed "s/MI300X/MI300X_VF/"); cp "$1" "$vf_config"' -- {}

# Install Rust toolchain for sgl-router
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rustc --version && cargo --version

# Build and install sgl-router
RUN python3 -m pip install --no-cache-dir setuptools-rust \
    && cd /sgl-workspace/sglang/sgl-router/bindings/python \
    && cargo build --release \
    && python3 -m pip install --no-cache-dir . \
    && rm -rf /root/.cache

# -----------------------
# TileLang
ENV DEBIAN_FRONTEND=noninteractive
ENV LIBGL_ALWAYS_INDIRECT=1
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

RUN /bin/bash -lc 'set -euo pipefail; \
  # Build TileLang only for gfx950
  if [ "${GPU_ARCH:-}" != "gfx950" ]; then \
    echo "[TileLang] Skipping (GPU_ARCH=${GPU_ARCH:-unset})"; \
    exit 0; \
  fi; \
  echo "[TileLang] Building TileLang for ${GPU_ARCH}"; \
  \
  # System dependencies (NO llvm-dev to avoid llvm-config-16 shadowing)
  apt-get update && apt-get install -y --no-install-recommends \
      build-essential git wget curl ca-certificates gnupg \
      libgtest-dev libgmock-dev \
      libprotobuf-dev protobuf-compiler libgflags-dev libsqlite3-dev \
      python3 python3-dev python3-setuptools python3-pip \
      gcc libtinfo-dev zlib1g-dev libedit-dev libxml2-dev \
      cmake ninja-build pkg-config libstdc++6 \
  && rm -rf /var/lib/apt/lists/*; \
  \
  # Build GoogleTest static libs (Ubuntu package ships sources only)
  cmake -S /usr/src/googletest -B /tmp/build-gtest -DBUILD_GTEST=ON -DBUILD_GMOCK=ON -DCMAKE_BUILD_TYPE=Release && \
  cmake --build /tmp/build-gtest -j"$(nproc)" && \
  cp -v /tmp/build-gtest/lib/*.a /usr/lib/x86_64-linux-gnu/ && \
  rm -rf /tmp/build-gtest; \
  \
  # Keep setuptools < 80 (compat with base image)
  python3 -m pip install --upgrade "setuptools>=77.0.3,<80" wheel cmake ninja && \
  python3 -m pip cache purge || true; \
  \
  # Locate ROCm llvm-config; fallback to installing LLVM 18 if missing
  LLVM_CONFIG_PATH=""; \
  for p in /opt/rocm/llvm/bin/llvm-config /opt/rocm/llvm-*/bin/llvm-config /opt/rocm-*/llvm*/bin/llvm-config; do \
    if [ -x "$p" ]; then LLVM_CONFIG_PATH="$p"; break; fi; \
  done; \
  if [ -z "$LLVM_CONFIG_PATH" ]; then \
    echo "[TileLang] ROCm llvm-config not found; installing LLVM 18..."; \
    curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh; \
    chmod +x /tmp/llvm.sh; \
    /tmp/llvm.sh 18; \
    LLVM_CONFIG_PATH="$(command -v llvm-config-18)"; \
    if [ -z "$LLVM_CONFIG_PATH" ]; then echo "ERROR: llvm-config-18 not found after install"; exit 1; fi; \
  fi; \
  echo "[TileLang] Using LLVM_CONFIG at: $LLVM_CONFIG_PATH"; \
  export PATH="$(dirname "$LLVM_CONFIG_PATH"):/usr/local/bin:${PATH}"; \
  export LLVM_CONFIG="$LLVM_CONFIG_PATH"; \
  \
  # Optional shim for tools that expect llvm-config-16
  mkdir -p /usr/local/bin && \
  printf "#!/usr/bin/env bash\nexec \"%s\" \"\$@\"\n" "$LLVM_CONFIG_PATH" > /usr/local/bin/llvm-config-16 && \
  chmod +x /usr/local/bin/llvm-config-16; \
  \
  # TVM Python bits need Cython
  python3 -m pip install --no-cache-dir "cython>=0.29.36,<3.0"; \
  \
  # Clone + pin TileLang (bundled TVM), then build
  git clone --recursive --branch "${TILELANG_BRANCH}" "${TILELANG_REPO}" /opt/tilelang && \
  cd /opt/tilelang && \
  git fetch --depth=1 origin "${TILELANG_COMMIT}" || true && \
  git checkout -f "${TILELANG_COMMIT}" && \
  git submodule update --init --recursive && \
  export CMAKE_ARGS="-DLLVM_CONFIG=${LLVM_CONFIG} ${CMAKE_ARGS:-}" && \
  bash ./install_rocm.sh'

# -----------------------
# Hadamard-transform (HIP build)
RUN /bin/bash -lc 'set -euo pipefail; \
    git clone --branch "${FHT_BRANCH}" "${FHT_REPO}" fast-hadamard-transform; \
    cd fast-hadamard-transform; \
    git checkout -f "${FHT_COMMIT}"; \
    python setup.py install'

# -----------------------
# Python tools
RUN python3 -m pip install --no-cache-dir \
    py-spy \
    pre-commit

# -----------------------
# Performance environment variable.
RUN echo "AITER_MXFP4_MOE_SF=${AITER_MXFP4_MOE_SF}" >> /etc/environment

ENV HIP_FORCE_DEV_KERNARG=1
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
ENV SGLANG_INT4_WEIGHT=0
ENV SGLANG_MOE_PADDING=1
ENV SGLANG_ROCM_DISABLE_LINEARQUANT=0
ENV SGLANG_ROCM_FUSED_DECODE_MLA=1
ENV SGLANG_SET_CPU_AFFINITY=1
ENV SGLANG_USE_AITER=1
ENV SGLANG_USE_ROCM700A=1

ENV NCCL_MIN_NCHANNELS=112
ENV VLLM_FP8_PADDING=1
ENV VLLM_FP8_ACT_PADDING=1
ENV VLLM_FP8_WEIGHT_PADDING=1
ENV VLLM_FP8_REDUCE_CONV=1
ENV TORCHINDUCTOR_MAX_AUTOTUNE=1
ENV TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1

CMD ["/bin/bash"]
