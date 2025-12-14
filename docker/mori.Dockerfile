ARG GPU_ARCH=gfx950

ARG BASE_IMAGE=rocm/dev-ubuntu-22.04:7.1.1-complete
ARG TRITON_REPO="https://github.com/ROCm/triton.git"
ARG PYTORCH_BRANCH="1c57644d"
ARG PYTORCH_VISION_BRANCH="v0.23.0"
ARG PYTORCH_REPO="https://github.com/ROCm/pytorch.git"
ARG PYTORCH_VISION_REPO="https://github.com/pytorch/vision.git"
ARG PYTORCH_AUDIO_BRANCH="v2.9.0"
ARG PYTORCH_AUDIO_REPO="https://github.com/pytorch/audio.git"
ARG FA_BRANCH="0e60e394"
ARG FA_REPO="https://github.com/Dao-AILab/flash-attention.git"
ARG AITER_BRANCH="v0.1.7.post5"
ARG AITER_REPO="https://github.com/ROCm/aiter.git"

FROM ${BASE_IMAGE} AS base

ARG GPU_ARCH=gfx950
ENV PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:
ARG PYTORCH_ROCM_ARCH=${GPU_ARCH}
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV AITER_ROCM_ARCH=${GPU_ARCH}

# Required for RCCL in ROCm7.1
ENV HSA_NO_SCRATCH_RECLAIM=1

ARG PYTHON_VERSION=3.12
ENV PYTHON_VERSION=${PYTHON_VERSION}

RUN mkdir -p /app
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update -y \
    && apt-get install -y software-properties-common git curl sudo vim less libgfortran5 \
    && for i in 1 2 3; do \
        add-apt-repository -y ppa:deadsnakes/ppa && break || \
        { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
    done \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
       python${PYTHON_VERSION}-lib2to3 python-is-python3  \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

RUN pip install -U packaging 'cmake<4' ninja wheel 'setuptools<80' pybind11 Cython
RUN apt-get update && apt-get install -y libjpeg-dev libsox-dev libsox-fmt-all sox && rm -rf /var/lib/apt/lists/*

FROM base AS build_triton
ARG TRITON_BRANCH
ARG TRITON_REPO
RUN git clone ${TRITON_REPO}
RUN cd triton \
    && git checkout ${TRITON_BRANCH} \
    && if [ ! -f setup.py ]; then cd python; fi \
    && python3 setup.py bdist_wheel --dist-dir=dist \
    && mkdir -p /app/install && cp dist/*.whl /app/install
RUN if [ -d triton/python/triton_kernels ]; then pip install build && cd triton/python/triton_kernels \
    && python3 -m build --wheel && cp dist/*.whl /app/install; fi

FROM base AS build_amdsmi
RUN cd /opt/rocm/share/amd_smi \
    && pip wheel . --wheel-dir=dist
RUN mkdir -p /app/install && cp /opt/rocm/share/amd_smi/dist/*.whl /app/install

FROM base AS build_pytorch
ARG PYTORCH_BRANCH
ARG PYTORCH_VISION_BRANCH
ARG PYTORCH_AUDIO_BRANCH
ARG PYTORCH_REPO
ARG PYTORCH_VISION_REPO
ARG PYTORCH_AUDIO_REPO

RUN git clone ${PYTORCH_REPO} pytorch
RUN cd pytorch && git checkout ${PYTORCH_BRANCH} \
    && pip install -r requirements.txt && git submodule update --init --recursive \
    && python3 tools/amd_build/build_amd.py \
    && CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)') python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl
RUN git clone ${PYTORCH_VISION_REPO} vision
RUN cd vision && git checkout ${PYTORCH_VISION_BRANCH} \
    && python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl
RUN git clone ${PYTORCH_AUDIO_REPO} audio
RUN cd audio && git checkout ${PYTORCH_AUDIO_BRANCH} \
    && git submodule update --init --recursive \
    && pip install -r requirements.txt \
    && python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl
RUN mkdir -p /app/install && cp /app/pytorch/dist/*.whl /app/install \
    && cp /app/vision/dist/*.whl /app/install \
    && cp /app/audio/dist/*.whl /app/install

FROM base AS build_fa
ARG FA_BRANCH
ARG FA_REPO
RUN --mount=type=bind,from=build_pytorch,src=/app/install/,target=/install \
    pip install /install/*.whl
RUN git clone ${FA_REPO}
RUN cd flash-attention \
    && git checkout ${FA_BRANCH} \
    && git submodule update --init \
    && GPU_ARCHS=$(echo ${PYTORCH_ROCM_ARCH} | sed -e 's/;gfx1[0-9]\{3\}//g') python3 setup.py bdist_wheel --dist-dir=dist
RUN mkdir -p /app/install && cp /app/flash-attention/dist/*.whl /app/install

FROM base AS build_aiter
ARG AITER_BRANCH
ARG AITER_REPO
RUN --mount=type=bind,from=build_pytorch,src=/app/install/,target=/install \
    pip install /install/*.whl
RUN git clone --recursive ${AITER_REPO}
RUN cd aiter \
    && git checkout ${AITER_BRANCH} \
    && git submodule update --init --recursive \
    && pip install -r requirements.txt
RUN pip install pyyaml && cd aiter && PREBUILD_KERNELS=1 GPU_ARCHS=${AITER_ROCM_ARCH} python3 setup.py bdist_wheel --dist-dir=dist && ls /app/aiter/dist/*.whl
RUN mkdir -p /app/install && cp /app/aiter/dist/*.whl /app/install

FROM base AS debs
RUN mkdir /app/debs
RUN --mount=type=bind,from=build_triton,src=/app/install/,target=/install \
    cp /install/*.whl /app/debs
RUN --mount=type=bind,from=build_fa,src=/app/install/,target=/install \
    cp /install/*.whl /app/debs
RUN --mount=type=bind,from=build_amdsmi,src=/app/install/,target=/install \
    cp /install/*.whl /app/debs
RUN --mount=type=bind,from=build_pytorch,src=/app/install/,target=/install \
    cp /install/*.whl /app/debs
RUN --mount=type=bind,from=build_aiter,src=/app/install/,target=/install \
    cp /install/*.whl /app/debs

FROM base AS final
ARG GPU_ARCH=gfx950
RUN --mount=type=bind,from=debs,src=/app/debs,target=/install \
    pip install /install/*.whl


ENV BUILD_VLLM="1"
ENV BUILD_MOONCAKE="1"
ENV NO_DEPS_FLAG=""

# This is necessary for scope purpose, again
ENV GPU_ARCH_LIST=${GPU_ARCH%-*}

ARG SGL_REPO="https://github.com/sgl-project/sglang.git"
ARG SGL_DEFAULT="main"
# ARG SGL_BRANCH="amd_mori"
ARG SGL_BRANCH="1dec9e80b8d8d0b86180dc5cb3202528145cddc7"

ARG MORI_REPO="https://github.com/ROCm/mori.git"
# ARG MORI_COMMIT="ionic_new_950_1128"
ARG MORI_COMMIT="84a2881d46bce5161f2c5ece346df8d747b03361"

ARG MOONCAKE_REPO="https://github.com/Duyi-Wang/Mooncake.git"
ARG MOONCAKE_COMMIT="amd_mori"

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
# Build vLLM
ARG VLLM_REPO="https://github.com/vllm-project/vllm.git"
ARG VLLM_BRANCH="v0.12.0"
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

# Install Rust toolchain for sgl-model-gateway
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rustc --version && cargo --version

# Build and install sgl-model-gateway
RUN python3 -m pip install --no-cache-dir setuptools-rust \
    && cd /sgl-workspace/sglang/sgl-model-gateway/bindings/python \
    && cargo build --release -j 8 \
    && python3 -m pip install --no-cache-dir . \
    && rm -rf /root/.cache

# -----------------------
# TileLang
ENV DEBIAN_FRONTEND=noninteractive
ENV LIBGL_ALWAYS_INDIRECT=1
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

RUN /bin/bash -lc 'set -euo pipefail; \
  # Build TileLang only for gfx950
  rm -f /etc/alternatives/python3 && ln -s /usr/bin/python3.10 /etc/alternatives/python3 && \
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
      cmake ninja-build pkg-config libstdc++6 python3-six \
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
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py --force-reinstall && /usr/local/bin/pip3.10 install six; \
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
  bash ./install_rocm.sh;  \
  if [ "${GPU_ARCH}" == "gfx950" ]; then \
    rm -f /etc/alternatives/python3 && ln -s /usr/bin/python3.12 /etc/alternatives/python3; \
    cp /usr/local/bin/pip3.12 /usr/local/bin/pip; \
    cp /usr/local/bin/pip3.12 /usr/local/bin/pip3; \
  fi;'

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
# MORI
ENV MORI_GPU_ARCHS=${GPU_ARCH}
RUN apt-get install initramfs-tools jq libopenmpi-dev libpci-dev -y
RUN apt-get install -y g++-12 libstdc++-12-dev -y

# AINIC library
ARG AINIC_VERSION=1.117.5
ARG UBUNTU_CODENAME=jammy


RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg apt-transport-https \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /etc/apt/keyrings


RUN curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor > /etc/apt/keyrings/amdainic.gpg


RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdainic.gpg] \
https://repo.radeon.com/amdainic/pensando/ubuntu/${AINIC_VERSION} ${UBUNTU_CODENAME} main" \
    > /etc/apt/sources.list.d/amdainic.list

    
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libionic-dev \
      ionic-common \
      ionic-dkms \
      pds-dkms \
 && rm -rf /var/lib/apt/lists/*

#  Enable AINIC in mori
ENV USE_IONIC=ON

RUN git clone ${MORI_REPO} \
    && cd mori \
    && git checkout ${MORI_COMMIT} \
    && git submodule update --init --recursive \
    && python3 setup.py develop

# -----------------------
# Triton
RUN pip uninstall -y triton \
     && pip install triton==3.5.1

# -----------------------
# Performance environment variable.

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