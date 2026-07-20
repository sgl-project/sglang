# Usage (to build SGLang ROCm docker image):
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx950-rocm7_14 -t v0.5.10.post1-rocm714-mi35x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx942 -t v0.5.10.post1-rocm700-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx942-rocm720 -t v0.5.10.post1-rocm720-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx950 -t v0.5.10.post1-rocm700-mi35x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx950-rocm720 -t v0.5.10.post1-rocm720-mi35x -f rocm.Dockerfile .

# Usage (to build SGLang ROCm + Mori docker image):
# remove --build-arg NIC_BACKEND=ainic since new MoRI JIT will do NIC auto detection on target
# Keep the build-arg for user to select the desired nic support, current choice: [ainic, bxnt]
# if no set this arg, it will support nic auto detection. On a target with more than 1 type of
# RDMA NICs installed (rare), overwrite w. runtime env MORI_DEVICE_NIC = "bnxt"|"ionic"|"mlx5"
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx950-rocm7_14 --build-arg ENABLE_MORI=1 -t v0.5.10.post1-rocm714-mi35x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx942 --build-arg ENABLE_MORI=1 -t v0.5.10.post1-rocm700-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx942-rocm720 --build-arg ENABLE_MORI=1 -t v0.5.10.post1-rocm720-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx950 --build-arg ENABLE_MORI=1 -t v0.5.10.post1-rocm700-mi35x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx950-rocm720 --build-arg ENABLE_MORI=1 -t v0.5.10.post1-rocm720-mi35x -f rocm.Dockerfile .

# Usage (to build SGLang ROCm + NIXL docker image, for prefill/decode disaggregation):
# Builds UCX (--with-rocm) and upstream ai-dynamo/nixl from source by default.
# Set ENABLE_NIXL=0 to skip NIXL.
# At runtime use --disaggregation-transfer-backend nixl (env is wired via /etc/bash.bashrc).
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx950-rocm720 -t v0.5.10.post1-rocm720-mi35x -f rocm.Dockerfile .

# Default base images
ARG BASE_IMAGE_950_ROCM7_14="ubuntu:24.04"
ARG BASE_IMAGE_942="rocm/sgl-dev:rocm7-vllm-20250904"
ARG BASE_IMAGE_942_ROCM720="rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1"
ARG BASE_IMAGE_950="rocm/sgl-dev:rocm7-vllm-20250904"
ARG BASE_IMAGE_950_ROCM720="rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1"

# This is necessary for scope purpose
ARG GPU_ARCH=gfx950

# ===============================
# Base image 950 with rocm7_14 and args
FROM $BASE_IMAGE_950_ROCM7_14 AS gfx950-rocm7_14

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg \
        build-essential \
        python3 python3-dev python3-pip python-is-python3 python3.12-venv \
        wget git \
        ca-certificates \
        libstdc++-12-dev \
    && rm -rf /var/lib/apt/lists/*
# ---- python 3.14 ----
# RUN apt update \
#     && apt install -y --no-install-recommends software-properties-common \
#     && add-apt-repository -y ppa:deadsnakes/ppa \
#     && apt update \
#     && apt install -y python3.14 python3.14-venv python3.14-dev \
#     && rm /usr/bin/python3 && ln -s /usr/bin/python3.14 /usr/bin/python3
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip tooling a bit
RUN python3 -m pip install --no-cache-dir -U pip setuptools setuptools_scm wheel

# ROCm SDK and PyTorch dependencies
ARG PIP_EXTRA_INDEX_URL="https://repo.amd.com/rocm/whl-multi-arch/"
ARG ROCM_SDK_VERSION="7.14.0"
ARG TORCH_VERSION="2.11.0"
ARG TORCHVISION_VERSION="0.26.0"
ARG TORCHAUDIO_VERSION="2.11.0"

# Install ROCm SDK
RUN ROCM_SPEC="${ROCM_SDK_VERSION:+==${ROCM_SDK_VERSION}}" \
    && python3 -m pip install --no-cache-dir \
         --index-url "${PIP_EXTRA_INDEX_URL}" \
         "rocm[libraries,devel,device-gfx950]${ROCM_SPEC}" \
         "torch[device-gfx950]==${TORCH_VERSION}" \
         "torchvision[device-gfx950]==${TORCHVISION_VERSION}" \
         "torchaudio==${TORCHAUDIO_VERSION}"

# Initialize ROCm SDK
RUN rocm-sdk init && rocm-sdk targets
ENV ROCM_HOME=$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_devel
ENV CPATH=$ROCM_HOME/include
ENV LIBRARY_PATH=$ROCM_HOME/lib
ENV LD_LIBRARY_PATH=$ROCM_HOME/lib
ENV ROCM_PATH=$ROCM_HOME
RUN echo 'export PATH=$ROCM_HOME/llvm/bin:$ROCM_HOME/bin:$PATH' >> /etc/bash.bashrc

# Workaround: ROCm SDK hsakmtTargets.cmake contains hardcoded /usr/lib64/libc.so from
# the upstream build system, but Ubuntu uses /lib/x86_64-linux-gnu/. Create symlink to
# avoid "ninja: error: /usr/lib64/libc.so missing and no known rule to make it"
RUN mkdir -p /usr/lib64 && ln -sf /lib/x86_64-linux-gnu/libc.so /usr/lib64/libc.so

# Workaround: At runtime, AITER determines `DEFAULT_GPU_ARCH` by calling
# `/opt/rocm/llvm/bin/amdgpu-arch`.
RUN ln -s ${ROCM_HOME} /opt/rocm

ENV BUILD_VLLM="0"
ENV BUILD_TRITON="1"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="9127c94a18e4398e1eba91f6639e910f0994ad02"
ENV TRITON_COMMIT_DEFAULT="5f3f125e8f63c24613f1f73b937442864f263f94"

# ===============================
# Base image 942 with rocm700 and args
FROM $BASE_IMAGE_942 AS gfx942
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="0"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="9127c94a18e4398e1eba91f6639e910f0994ad02"

# ===============================
# Base image 942 with rocm720 and args
FROM $BASE_IMAGE_942_ROCM720 AS gfx942-rocm720
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="1"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="9127c94a18e4398e1eba91f6639e910f0994ad02"
ENV TRITON_COMMIT_DEFAULT="42270451990532c67e69d753fbd026f28fcc4840"

# ===============================
# Base image 950 and args
FROM $BASE_IMAGE_950 AS gfx950
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="0"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="9127c94a18e4398e1eba91f6639e910f0994ad02"

# ===============================
# Base image 950 with rocm720 and args
FROM $BASE_IMAGE_950_ROCM720 AS gfx950-rocm720
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="1"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="9127c94a18e4398e1eba91f6639e910f0994ad02"
ENV TRITON_COMMIT_DEFAULT="42270451990532c67e69d753fbd026f28fcc4840"

# Local source stage: with BRANCH_TYPE=local the build context is copied here and
# used instead of git clone (mirrors docker/Dockerfile's local_src stage).
FROM scratch AS local_src
COPY . /src

# ===============================
# Chosen arch and args
FROM ${GPU_ARCH}

# This is necessary for scope purpose, again
ARG GPU_ARCH=gfx950
ENV GPU_ARCH_LIST=${GPU_ARCH%-*}
ENV PYTORCH_ROCM_ARCH=gfx942;gfx950

ARG SGL_REPO="https://github.com/sgl-project/sglang.git"
ARG SGL_DEFAULT="main"
ARG SGL_BRANCH=${SGL_DEFAULT}
ARG BRANCH_TYPE=remote

# Version override for setuptools_scm (used in nightly builds)
ARG SETUPTOOLS_SCM_PRETEND_VERSION=""

ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
ENV TRITON_COMMIT="${TRITON_COMMIT:-${TRITON_COMMIT_DEFAULT}}"

ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT=""
ENV AITER_COMMIT="${AITER_COMMIT:-${AITER_COMMIT_DEFAULT}}"

ARG LLVM_REPO="https://github.com/jrbyrnes/llvm-project.git"
ARG LLVM_BRANCH="MainOpSelV2"
ARG LLVM_COMMIT="6520ace8227ffe2728148d5f3b9872a870b0a560"

ARG MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"
ARG MOONCAKE_COMMIT="01d1eb2a7ec37fd5e20a88573e9b4956e7846e9a"

ARG TILELANG_REPO="https://github.com/tile-ai/tilelang.git"
ARG TILELANG_COMMIT="a55a82302bf7f3c5af635b5c9146f728185cc900"

ARG FHT_REPO="https://github.com/jeffdaily/fast-hadamard-transform.git"
ARG FHT_BRANCH="rocm"
ARG FHT_COMMIT="46efb7d776d38638fc39f3c803eaee3dd7016bd1"

ARG ENABLE_MORI=0
ARG NIC_BACKEND=none

ARG MORI_REPO="https://github.com/ROCm/mori.git"
ARG MORI_COMMIT="f7e6ac6863c53821bc7afb91a578cc6ce38fcad0"

# NIXL (upstream ai-dynamo/nixl) — KV transfer backend for prefill/decode disaggregation.
# Built from source for ROCm; needs UCX built --with-rocm (built here from openucx).
# Enabled by default; disable with --build-arg ENABLE_NIXL=0.
ARG ENABLE_NIXL=1
ARG UCX_REPO="https://github.com/openucx/ucx.git"
ARG UCX_BRANCH="v1.19.x"
ARG NIXL_REPO="https://github.com/ai-dynamo/nixl.git"
ARG NIXL_COMMIT="c28061f9782e099f975bcc79198b7b5a1a36cc40"

# AMD AINIC apt repo settings
ARG AINIC_VERSION=1.117.5-a-38
ARG UBUNTU_CODENAME=jammy

# Optional Ubuntu mirror override + apt hardening.
# - UBUNTU_MIRROR is empty by default (no behaviour change for local builds).
#   When set (typically in CI), all http://*archive.ubuntu.com and
#   http://*security.ubuntu.com entries in /etc/apt/sources.list are rewritten
#   to point at the given base URL, e.g.
#     --build-arg UBUNTU_MIRROR=https://archive.ubuntu.com
#     --build-arg UBUNTU_MIRROR=https://tw.archive.ubuntu.com
#     --build-arg UBUNTU_MIRROR=http://internal-cache.example.com
#   This mirrors the pattern already used in docker/Dockerfile (NVIDIA) and
#   docker/npu.Dockerfile, and lets CI runners that cannot reach Canonical's
#   port-80 mirror IPs still complete `apt-get update`.
# - The 80-net-hardening apt config adds retries + per-request timeout so that
#   transient mirror flakes don't immediately fail a build (apt's default is 0
#   retries).
ARG UBUNTU_MIRROR=
USER root

RUN if [ -n "$UBUNTU_MIRROR" ]; then \
        sed -i "s|http://[^[:space:]/]*archive.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list && \
        sed -i "s|http://[^[:space:]/]*security.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list; \
    fi && \
    printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
        > /etc/apt/apt.conf.d/80-net-hardening

# Fix hipDeviceGetName returning empty string in ROCm 7.0 docker images.
# The ROCm 7.0 base image is missing libdrm-amdgpu-common which provides the
# amdgpu.ids device-ID-to-marketing-name mapping file.
# ROCm 7.2 base images already ship these packages, so this step is skipped.
# See https://github.com/ROCm/ROCm/issues/5992
RUN set -eux; \
    case "${GPU_ARCH}" in \
      *rocm7_14*) \
        ;; \
      *rocm720*) \
        echo "ROCm 7.2 (GPU_ARCH=${GPU_ARCH}): libdrm-amdgpu packages already present, skipping"; \
        ;; \
      *) \
        echo "ROCm 7.0 (GPU_ARCH=${GPU_ARCH}): installing libdrm-amdgpu packages"; \
        curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
          | gpg --dearmor -o /etc/apt/keyrings/amdgpu-graphics.gpg \
        && echo 'deb [arch=amd64,i386 signed-by=/etc/apt/keyrings/amdgpu-graphics.gpg] https://repo.radeon.com/graphics/7.0/ubuntu jammy main' \
          > /etc/apt/sources.list.d/amdgpu-graphics.list \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
             libdrm-amdgpu-common \
             libdrm-amdgpu-amdgpu1 \
             libdrm2-amdgpu \
        && rm -rf /var/lib/apt/lists/* \
        && cp /opt/amdgpu/share/libdrm/amdgpu.ids /usr/share/libdrm/amdgpu.ids; \
        ;; \
    esac


# Install some basic utilities
RUN python -m pip install --upgrade pip && pip install setuptools_scm
RUN apt-get purge -y sccache; python -m pip uninstall -y sccache; rm -f "$(which sccache)"

# Install AMD SMI Python package from ROCm distribution.
# The ROCm 7.2 base image (rocm/pytorch) does not pre-install this package.
RUN set -eux; \
    case "${GPU_ARCH}" in \
      *rocm7_14*) \
        # Should install it properly, however it seems there are race
        # conditions between torch and amdsmi module initialization code.
        # keep the following section commented before it is fixed.
        # cd $ROCM_HOME/share/amd_smi \
        # && python3 -m pip install --no-cache-dir . \
        ;; \
      *rocm720*) \
        echo "ROCm 7.2 flavor detected from GPU_ARCH=${GPU_ARCH}"; \
        cd /opt/rocm/share/amd_smi \
        && python3 -m pip install --no-cache-dir . \
        ;; \
      *) \
        echo "Not rocm720 (GPU_ARCH=${GPU_ARCH}), skip amdsmi installation"; \
        ;; \
    esac

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
# FlyDSL
# The double-sed patch was to workaround TheRock#2484. As TheRock#3250 has fixed
# the PATH issue of LLVM, this can be removed for ROCm versions later than
# 0a20260625.  Keep this commented because we may end up pinning earlier versions.
# RUN if [ "${GPU_ARCH}" = "gfx950-rocm7_14" ]; then \
#       apt-get update && apt-get install -y ninja-build patchelf cmake \
#       && git clone https://github.com/ROCm/FlyDSL.git --branch v0.2.0; \
#     fi
# RUN if [ "${GPU_ARCH}" = "gfx950-rocm7_14" ]; then \
#       cd FlyDSL \
#       && sed -i '/-DMLIR_ENABLE_ROCM_RUNNER=ON/a\    -DROCM_TEST_CHIPSET="gfx942" \\' scripts/build_llvm.sh \
#       && sed -i scripts/build_llvm.sh -e '51i\ls && sed -i mlir/lib/Target/LLVM/ROCDL/Target.cpp -e "s|{\\"ld.lld\\"|{\\"/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/llvm/bin/ld.lld\\"|"' \
#       && bash -lc 'unset LLVM_COMMIT && source /opt/venv/bin/activate \
#         && CMAKE_PREFIX_PATH=${ROCM_HOME}/lib/cmake bash scripts/build_llvm.sh -j64 \
#         && CMAKE_PREFIX_PATH=${ROCM_HOME}/lib/cmake LLVM_DIR=/sgl-workspace/llvm-project/mlir_install/lib/cmake/llvm MLIR_PATH=/sgl-workspace/llvm-project/mlir_install bash scripts/build.sh -j64 \
#         && FLYDSL_RELEASE_TYPE=release pip install . \
#         && rm -fr /sgl-workspace/llvm-project;'; \
#     fi

# -----------------------
# AITER
# Unset setuptools_scm override so AITER gets its own version (AITER_COMMIT), not SGLang's
# (SETUPTOOLS_SCM_PRETEND_VERSION is set later for SGLang nightly builds and would otherwise
# leak into AITER's version when AITER uses setuptools_scm)

ENV SETUPTOOLS_SCM_PRETEND_VERSION=
# Keep the base image's Torch-compatible Triton by default. Override with
# AITER_USE_SYSTEM_TRITON=0 when intentionally testing aiter-managed Triton.
ENV AITER_USE_SYSTEM_TRITON=1
RUN pip uninstall -y aiter
# Use `checkout -f` so the smudge-filter-induced "dirty" working tree from
# AITER's .gitattributes (*.csv text eol=lf, added in ROCm/aiter#3370) does not
# block switching to commits that predate that rule (e.g. the current default
# AITER_COMMIT_DEFAULT). The working tree was just produced by a fresh
# `git clone` above, so there are no real user changes to preserve.
RUN git clone ${AITER_REPO} \
 && cd aiter \
 && git checkout -f ${AITER_COMMIT} \
 && git submodule update --init --recursive \
 && pip install -r requirements.txt

RUN cd aiter \
     && echo "[AITER] GPU_ARCH=${GPU_ARCH}" \
     && echo "[AITER] AITER_USE_SYSTEM_TRITON=${AITER_USE_SYSTEM_TRITON}" \
     && if [ "${GPU_ARCH}" = "gfx950-rocm7_14" ]; then \
         PATH=$PATH:$ROCM_HOME/llvm/bin PREBUILD_KERNELS=1 GPU_ARCHS="${GPU_ARCH_LIST}" python setup.py build_ext --inplace \
          && PATH=$PATH:$ROCM_HOME/llvm/bin GPU_ARCHS="${GPU_ARCH_LIST}" pip install --no-build-isolation -e .; \
        elif [ "$BUILD_AITER_ALL" = "1" ] && [ "$BUILD_LLVM" = "1" ]; then \
          sh -c "HIP_CLANG_PATH=/sgl-workspace/llvm-project/build/bin/ PREBUILD_KERNELS=1 GPU_ARCHS=$GPU_ARCH_LIST python setup.py build_ext --inplace" \
          && sh -c "HIP_CLANG_PATH=/sgl-workspace/llvm-project/build/bin/ GPU_ARCHS=$GPU_ARCH_LIST pip install --config-settings editable_mode=compat -e ."; \
        elif [ "$BUILD_AITER_ALL" = "1" ]; then \
          sh -c "PREBUILD_KERNELS=1 GPU_ARCHS=$GPU_ARCH_LIST python setup.py build_ext --inplace" \
          && sh -c "GPU_ARCHS=$GPU_ARCH_LIST pip install --config-settings editable_mode=compat -e ."; \
        else \
          sh -c "GPU_ARCHS=$GPU_ARCH_LIST pip install --config-settings editable_mode=compat -e ."; \
        fi \
      && echo "export PYTHONPATH=/sgl-workspace/aiter:\${PYTHONPATH}" >> /etc/bash.bashrc

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
     cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON -DENABLE_MULTI_PROTOCOL=ON -DWITH_STORE=ON -DBUILD_UNIT_TESTS=OFF && \
     make -j "$(nproc)" && make install; \
    fi

# -----------------------
# Build SGLang
ARG BUILD_TYPE=all

# Set version for setuptools_scm if provided (for nightly builds). Only pass in the SGLang
# pip install RUN so it does not affect AITER, sgl-model-gateway, TileLang, FHT, MORI, etc.
ARG SETUPTOOLS_SCM_PRETEND_VERSION

RUN pip install IPython \
    && pip install orjson \
    && pip install python-multipart \
    && pip install torchao==0.9.0 \
    && pip install pybind11

# Rust toolchain — needed by setuptools-rust to build the sglang-mm extension
# (sglang.srt.multimodal._core) during the sglang pip install below, and later by
# sgl-model-gateway. Must precede the sglang install.
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rustc --version && cargo --version
ENV CARGO_BUILD_JOBS=4

RUN pip uninstall -y sgl_kernel sglang

# Obtain sglang source: copied from the build context (BRANCH_TYPE=local) or git clone.
COPY --from=local_src /src /tmp/local_src
RUN if [ "$BRANCH_TYPE" = "local" ]; then \
         echo "Using local source (BRANCH_TYPE=local)."; \
         cp -r /tmp/local_src sglang; \
       else \
         git clone ${SGL_REPO} sglang \
         && cd sglang \
         && if [ "${SGL_BRANCH}" = ${SGL_DEFAULT} ]; then \
              echo "Using ${SGL_DEFAULT}, default branch."; \
              git checkout ${SGL_DEFAULT}; \
            else \
              echo "Using ${SGL_BRANCH} branch."; \
              git checkout ${SGL_BRANCH}; \
            fi \
         && cd ..; \
       fi \
    && rm -rf /tmp/local_src \
    && cd sglang \
    && cd sgl-kernel \
    && rm -f pyproject.toml \
    && mv pyproject_rocm.toml pyproject.toml \
    && AMDGPU_TARGET=$GPU_ARCH_LIST python setup_rocm.py install \
    && cd .. \
    && rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml \
    && if [ "$BUILD_TYPE" = "srt" ]; then \
         export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION}" && python -m pip --no-cache-dir install -e "python[srt_hip,diffusion_hip]"; \
       else \
         export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION}" && python -m pip --no-cache-dir install -e "python[all_hip]"; \
       fi

RUN python -m pip cache purge

# Copy config files to support MI300X in virtualized environments (MI300X_VF).  Symlinks will not be created in image build.
RUN find /sgl-workspace/sglang/python/sglang/srt/layers/quantization/configs/ \
         /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/ \
         -type f -name '*MI300X*' | xargs -I {} sh -c 'vf_config=$(echo "$1" | sed "s/MI300X/MI300X_VF/"); cp "$1" "$vf_config"' -- {}

# Rust toolchain already installed above (before the sglang install).

# Build and install sgl-model-gateway
RUN python3 -m pip install --no-cache-dir "maturin<1.14" \
    && sed -i -E 's|^(smg-[a-zA-Z-]+)\s*=\s*"~1\.0\.0"|\1 = "=1.0.0"|' \
           /sgl-workspace/sglang/sgl-model-gateway/Cargo.toml \
    && grep -E '^smg-' /sgl-workspace/sglang/sgl-model-gateway/Cargo.toml \
    && cd /sgl-workspace/sglang/sgl-model-gateway/bindings/python \
    && ulimit -n 65536 && maturin build --release --features vendored-openssl --out dist \
    && python3 -m pip install --force-reinstall dist/*.whl \
    && rm -rf /root/.cache

# -----------------------
# TileLang
ENV DEBIAN_FRONTEND=noninteractive
ENV LIBGL_ALWAYS_INDIRECT=1
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

RUN /bin/bash -lc 'set -euo pipefail; \
  echo "[TileLang] Building TileLang for ${GPU_ARCH}"; \
  # System dependencies (NO llvm-dev to avoid llvm-config-16 shadowing)
  apt-get update && apt-get install -y --no-install-recommends \
      build-essential git wget curl ca-certificates gnupg \
      libgtest-dev libgmock-dev \
      libprotobuf-dev protobuf-compiler libgflags-dev libsqlite3-dev \
      python3 python3-dev python3-setuptools python3-pip python3-apt \
      gcc libtinfo-dev zlib1g-dev libedit-dev libxml2-dev vim \
      cmake ninja-build pkg-config libstdc++6 software-properties-common \
  && rm -rf /var/lib/apt/lists/*; \
  \
  # Prefer the container venv
  VENV_PY="/opt/venv/bin/python"; \
  VENV_PIP="/opt/venv/bin/pip"; \
  if [ ! -x "$VENV_PY" ]; then VENV_PY="python3"; fi; \
  if [ ! -x "$VENV_PIP" ]; then VENV_PIP="pip3"; fi; \
  \
  # Build GoogleTest static libs (Ubuntu package ships sources only)
  cmake -S /usr/src/googletest -B /tmp/build-gtest -DBUILD_GTEST=ON -DBUILD_GMOCK=ON -DCMAKE_BUILD_TYPE=Release && \
  cmake --build /tmp/build-gtest -j"$(nproc)" && \
  cp -v /tmp/build-gtest/lib/*.a /usr/lib/x86_64-linux-gnu/ && \
  rm -rf /tmp/build-gtest; \
  \
  # Keep setuptools < 80 (compat with base image). Pin cmake to the last known-good
  # 4.3.4: cmake 4.4's gtest_discover_tests breaks the (pinned) MoRI build with a
  # JSON parse error. This image is rebuilt daily, so pin the exact version for
  # reproducible builds rather than letting cmake drift.
  "$VENV_PIP" install --upgrade "setuptools>=77.0.3,<80" wheel "cmake==4.3.4" ninja scikit-build-core && \
  "$VENV_PIP" cache purge || true; \
  \
  # Locate ROCm llvm-config; fallback to installing LLVM 18 if missing
  LLVM_CONFIG_PATH=""; \
  for p in /opt/rocm/llvm/bin/llvm-config /opt/rocm/llvm-*/bin/llvm-config /opt/rocm-*/llvm*/bin/llvm-config; do \
    if [ -x "$p" ]; then LLVM_CONFIG_PATH="$p"; break; fi; \
  done; \
  if [ -z "$LLVM_CONFIG_PATH" ]; then \
    echo "[TileLang] ROCm llvm-config not found; installing LLVM 18..."; \
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /etc/apt/keyrings/llvm.gpg; \
    echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" > /etc/apt/sources.list.d/llvm.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends llvm-18; \
    rm -rf /var/lib/apt/lists/*; \
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
  # TVM Python bits need Cython + z3 before configure.
  # Pin z3-solver==4.15.4.0: 4.15.4.0 has a manylinux wheel; 4.15.5.0 has no wheel and builds from source (fails: C++20 <format> needs GCC 14+, image has GCC 11).
  "$VENV_PIP" install --no-cache-dir "cython>=0.29.36,<3.0" "apache-tvm-ffi @ git+https://github.com/apache/tvm-ffi.git@37d0485b2058885bf4e7a486f7d7b2174a8ac1ce" "z3-solver==4.15.4.0"; \
  \
  # Clone + pin TileLang (bundled TVM), then build
  git clone --recursive "${TILELANG_REPO}" /opt/tilelang && \
  cd /opt/tilelang && \
  git fetch --depth=1 origin "${TILELANG_COMMIT}" || true && \
  git checkout -f "${TILELANG_COMMIT}" && \
  git submodule update --init --recursive && \
  if [ "${GPU_ARCH}" = "gfx950-rocm7_14" ]; then \
    export ROCM_PATH=${ROCM_HOME}; \
  else \
    export ROCM_PATH=/opt/rocm; \
  fi; \
  export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=ON -DROCM_PATH=${ROCM_PATH} -DLLVM_CONFIG=${LLVM_CONFIG} -DSKBUILD_SABI_VERSION= ${CMAKE_ARGS:-}" && \
  "$VENV_PIP" install -e . -v --no-build-isolation --no-deps; \
  if [ -f pyproject.toml ]; then sed -i "/^[[:space:]]*\"torch/d" pyproject.toml || true; fi; \
  "$VENV_PIP" cache purge || true; \
  "$VENV_PY" -c "import tilelang; print(tilelang.__version__)"'

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
    pre-commit \
    tabulate

# -----------------------
# MORI (optional)
RUN /bin/bash -lc 'set -euo pipefail; \
  if [ "${ENABLE_MORI}" != "1" ]; then \
    echo "[MORI] Skipping (ENABLE_MORI=${ENABLE_MORI})"; \
    exit 0; \
  fi; \
  echo "[MORI] Enabling MORI (NIC_BACKEND=${NIC_BACKEND})"; \
  \
  # Base deps for MORI build
  apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      g++ \
      jq \
      libopenmpi-dev \
      libpci-dev \
      libdrm-dev \
      initramfs-tools \
      librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool \
      libibverbs-dev rdma-core \
      openssh-server openmpi-bin openmpi-common libopenmpi-dev \
      libgrpc++-dev protobuf-compiler-grpc \
  && rm -rf /var/lib/apt/lists/*; \
  \
  # NIC backend deps — mori auto-detects NIC at runtime (MORI_DEVICE_NIC env var override).
  # Only vendor packages are installed here for dlopen (e.g. libionic.so); no compile-time flags needed.
  case "${NIC_BACKEND}" in \
    # default: install ainic and bxnt driver
    none) \
      apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg apt-transport-https && \
      rm -rf /var/lib/apt/lists/* && mkdir -p /etc/apt/keyrings; \
      curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/amdainic.gpg; \
      echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdainic.gpg] https://repo.radeon.com/amdainic/pensando/ubuntu/${AINIC_VERSION} ${UBUNTU_CODENAME} main" \
        > /etc/apt/sources.list.d/amdainic.list; \
      apt-get update && apt-get install -y --no-install-recommends \
          libionic-dev \
          ionic-common \
      ; \
      rm -rf /var/lib/apt/lists/*; \
      install -m 0755 -d /etc/apt/keyrings \
      && curl -fsSL https://packages.broadcom.com/artifactory/api/security/keypair/PackagesKey/public -o /etc/apt/keyrings/broadcom-nic.asc \
      && chmod a+r /etc/apt/keyrings/broadcom-nic.asc \
      && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/broadcom-nic.asc] https://packages.broadcom.com/artifactory/ethernet-nic-debian-public jammy main" > /etc/apt/sources.list.d/broadcom-nic.list \
      && apt-get update \
      && apt-get install -y ibverbs-utils bnxt-rocelib=235.2.86.0 \
      && cp /usr/local/lib/x86_64-linux-gnu/libbnxt_re* /usr/local/lib/. \
      ;; \
    # AMD NIC
    ainic) \
      apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg apt-transport-https && \
      rm -rf /var/lib/apt/lists/* && mkdir -p /etc/apt/keyrings; \
      curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/amdainic.gpg; \
      echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdainic.gpg] https://repo.radeon.com/amdainic/pensando/ubuntu/${AINIC_VERSION} ${UBUNTU_CODENAME} main" \
        > /etc/apt/sources.list.d/amdainic.list; \
      apt-get update && apt-get install -y --no-install-recommends \
          libionic-dev \
          ionic-common \
      ; \
      rm -rf /var/lib/apt/lists/*; \
      ;; \
     bnxt) \
       echo "[MORI] Enabling Broadcom BNXT backend"; \
       apt-get update \
       && apt-get install -y --no-install-recommends ca-certificates curl \
       && install -m 0755 -d /etc/apt/keyrings \
       && curl -fsSL https://packages.broadcom.com/artifactory/api/security/keypair/PackagesKey/public -o /etc/apt/keyrings/broadcom-nic.asc \
       && chmod a+r /etc/apt/keyrings/broadcom-nic.asc \
       && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/broadcom-nic.asc] https://packages.broadcom.com/artifactory/ethernet-nic-debian-public jammy main" > /etc/apt/sources.list.d/broadcom-nic.list \
       && apt-get update \
       && apt-get install -y ibverbs-utils bnxt-rocelib=235.2.86.0 \
       && cp /usr/local/lib/x86_64-linux-gnu/libbnxt_re* /usr/local/lib/. \
       ;; \
    *) \
      echo "ERROR: unknown NIC_BACKEND=${NIC_BACKEND}. Use one of: none, ainic"; \
      exit 2; \
      ;; \
  esac; \
  \
  # Build/install MORI
  export MORI_GPU_ARCHS="${GPU_ARCH_LIST}"; \
  echo "[MORI] MORI_GPU_ARCHS=${MORI_GPU_ARCHS} NIC_BACKEND=${NIC_BACKEND}"; \
  rm -rf /sgl-workspace/mori; \
  git clone "${MORI_REPO}" /sgl-workspace/mori; \
  cd /sgl-workspace/mori; \
  git checkout "${MORI_COMMIT}"; \
  git submodule update --init --recursive; \
  \
  if [ "${GPU_ARCH}" = "gfx950-rocm7_14" ]; then \
    # Fix for ROCm SDK: add find_package(NUMA) before hsakmt
    sed -i "/find_package(hsa-runtime64 REQUIRED)/i find_package(NUMA REQUIRED)" src/application/CMakeLists.txt; \
    \
    export ROCM_PATH=${ROCM_HOME}; \
    # Build with proper CMAKE_PREFIX_PATH to find NUMA and other ROCm SDK dependencies
    PATH=${ROCM_HOME}/bin:$PATH \
    CMAKE_PREFIX_PATH=${ROCM_HOME}/lib/rocm_sysdeps/lib/cmake:${ROCM_HOME}/lib/cmake:${ROCM_HOME} \
    pip install -e . --no-build-isolation; \
  else \
    python3 setup.py develop;  \
  fi; \
  python3 -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))" > /etc/ld.so.conf.d/torch.conf; \
  ldconfig; \
  echo "export PYTHONPATH=/sgl-workspace/mori:\${PYTHONPATH}" >> /etc/bash.bashrc; \
  echo "[MORI] Done."'

# -----------------------
# NIXL — upstream ai-dynamo/nixl KV transfer backend for PD disaggregation on ROCm.
# Builds UCX (--with-rocm) + nixl from source by default; skip with ENABLE_NIXL=0.
# --no-build-isolation reuses the image's ROCm torch (nixl pins torch==2.11.* as a build dep,
# which would otherwise pull a multi-GB CUDA torch); --no-deps keeps CUDA runtime deps out.
# wheel_variant=rocm names the pkg nixl_rocm, so symlink `nixl` since SGLang imports plain nixl.
# taskflow (header-only) is provided via pkg-config so meson skips its broken upstream wrap
# download (GitHub regenerated the v3.10.0 tarball, breaking the pinned source_hash).
RUN /bin/bash -lc 'set -euo pipefail; \
  [ "${ENABLE_NIXL}" = "1" ] || { echo "[NIXL] skip (ENABLE_NIXL=${ENABLE_NIXL})"; exit 0; }; \
  apt-get update && apt-get install -y --no-install-recommends \
      build-essential autoconf automake libtool pkg-config git \
      libibverbs-dev librdmacm-dev rdma-core && rm -rf /var/lib/apt/lists/*; \
  apt-get remove -y libabsl-dev libabsl20220623 || true; \
  pip install --no-cache-dir meson ninja pybind11 meson-python patchelf pyyaml; \
  git clone --depth=1 -b "${UCX_BRANCH}" "${UCX_REPO}" /sgl-workspace/ucx; \
  cd /sgl-workspace/ucx && ./autogen.sh && mkdir build && cd build && \
  ../configure --prefix=/opt/ucx --enable-shared --disable-static --disable-doxygen-doc \
      --enable-optimizations --enable-devel-headers \
      --with-rocm=/opt/rocm --with-verbs --with-dm --enable-mt && \
  make -j"$(nproc)" && make install; \
  git clone --depth=1 -b v3.10.0 https://github.com/taskflow/taskflow.git /sgl-workspace/taskflow; \
  cp -r /sgl-workspace/taskflow/taskflow /usr/local/include/; \
  mkdir -p /usr/local/lib/pkgconfig; \
  printf "Name: taskflow\nDescription: Taskflow\nVersion: 3.10.0\nCflags: -I/usr/local/include\n" > /usr/local/lib/pkgconfig/taskflow.pc; \
  git clone "${NIXL_REPO}" /sgl-workspace/nixl && cd /sgl-workspace/nixl && git checkout -f "${NIXL_COMMIT}"; \
  CXXFLAGS="-Wno-error" LD_LIBRARY_PATH="/opt/ucx/lib:/opt/rocm/lib" PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" \
  pip install . --no-deps --no-build-isolation \
      --config-settings=setup-args="-Ducx_path=/opt/ucx" \
      --config-settings=setup-args="-Dwheel_variant=rocm" \
      --config-settings=setup-args="-Denable_plugins=UCX,POSIX"; \
  SITE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])"); \
  ln -sfn nixl_rocm "$SITE/nixl"; \
  echo "export LD_LIBRARY_PATH=/opt/ucx/lib:\${LD_LIBRARY_PATH}" >> /etc/bash.bashrc'

# -----------------------
# Hot patch: torch-ROCm
# The artifact hardcoded the supported triton version to be 3.5.1.
# Rewrite the restriction directly.
ARG TORCH_ROCM_FILE="torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp310-cp310-linux_x86_64.whl"
RUN mkdir /tmp/whl && cd /tmp/whl \
     && export TORCH_ROCM_FILE="${TORCH_ROCM_FILE}" \
     && cat > hack.py <<"PY"
import zipfile, csv, os, re
from pathlib import Path

fname = os.environ["TORCH_ROCM_FILE"]
in_whl  = Path("/")   / fname
out_whl = Path("/tmp")/ fname
work = Path("/tmp/whl")

# 1) Extract
with zipfile.ZipFile(in_whl, "r") as z:
    z.extractall(work)

# 2) Locate dist-info and patch METADATA (edit this logic to match your exact line)
dist_info = next(work.glob("*.dist-info"))
meta = dist_info / "METADATA"
txt = meta.read_text(encoding="utf-8")

# Example: replace one exact requirement form.
# Adjust the string to match what you actually see.
pat = r"^Requires-Dist:\s*triton==3.5.1[^\s]*;"
txt2, n = re.subn(pat, r"triton>=3.5.1;", txt, flags=re.MULTILINE)
if txt2 == txt:
    raise SystemExit("Did not find expected Requires-Dist line to replace in METADATA")
meta.write_text(txt2, encoding="utf-8")

# 3) Hacky step: blank hash/size columns in RECORD
record = dist_info / "RECORD"
rows = []
with record.open(newline="", encoding="utf-8") as f:
    for r in csv.reader(f):
        if not r:
            continue
        # keep filename, blank out hash and size
        rows.append([r[0], "", ""])
with record.open("w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

# 4) Re-zip as a wheel
with zipfile.ZipFile(out_whl, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in work.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(work).as_posix())

print("Wrote", out_whl)
PY

RUN cd /tmp/whl \
    && case "${GPU_ARCH}" in \
      *rocm720*) \
        echo "ROCm 7.2 flavor detected from GPU_ARCH=${GPU_ARCH}"; \
        python hack.py \
        && python3 -m pip install --force --no-deps /tmp/${TORCH_ROCM_FILE} \
        && rm -fr /tmp/whl /tmp/${TORCH_ROCM_FILE} \
        ;; \
      *) \
        echo "Not rocm720 (GPU_ARCH=${GPU_ARCH}), skip patch"; \
        ;; \
    esac


# -----------------------
# Hot patch: Triton
# For ROCm 7.2, this custom build breaks pip dependency management,
# so future `pip install` will break the ROCm stack.
# A workaround for this is to reinstall the default triton
# wheel with the `rocm/pytorch` image in the root directory.
# For ROCm 7.14, it is a different story:
# https://github.com/ROCm/rocm-systems/issues/7643
# Rebuilding tag 3.7.0 seems to workaround this issue without
# sacrificing accuracy. The previous section to rewrite the metadata
# of the wheel becomes unnecessary once we apply the trick to fake
# the version string as we do here.
RUN if [ "$BUILD_TRITON" = "1" ]; then \
        TRITON_INSTALLED_VERSION=$(pip show triton 2>/dev/null | grep '^Version:' | cut -d' ' -f2 || echo "") \
     && TRITON_BASE_VERSION=$(echo "$TRITON_INSTALLED_VERSION" | cut -d'+' -f1) \
     && TRITON_VERSION_SUFFIX=$(echo "$TRITON_INSTALLED_VERSION" | grep -o '+.*' || echo "") \
     && echo "Captured Triton version: $TRITON_INSTALLED_VERSION (base: $TRITON_BASE_VERSION, suffix: $TRITON_VERSION_SUFFIX)" \
     && pip uninstall -y triton \
     && apt install -y cmake \
     && git clone ${TRITON_REPO} triton-custom \
     && cd triton-custom \
     && git checkout ${TRITON_COMMIT} \
     && if [ -n "$TRITON_BASE_VERSION" ]; then \
            TRITON_SOURCE_VERSION=$(grep -oP 'TRITON_VERSION = "\K[^"]+' setup.py || echo "") \
         && if [ -n "$TRITON_SOURCE_VERSION" ]; then \
                sed -i "s/TRITON_VERSION = \"$TRITON_SOURCE_VERSION\"/TRITON_VERSION = \"$TRITON_BASE_VERSION\"/" setup.py \
             && sed -i "s/__version__ = '$TRITON_SOURCE_VERSION'/__version__ = '$TRITON_BASE_VERSION'/" python/triton/__init__.py; \
            fi \
         && sed -i '/^def get_git_version_suffix():/,/^def get_triton_version_suffix():/{ /^def get_triton_version_suffix():/!{ /^def get_git_version_suffix():/!d; }; }' setup.py \
         && sed -i '/^def get_git_version_suffix():/a\    return ""' setup.py; \
        fi \
     && pip install -r python/requirements.txt \
     && if [ -n "$TRITON_VERSION_SUFFIX" ]; then \
            TRITON_WHEEL_VERSION_SUFFIX="$TRITON_VERSION_SUFFIX" pip install -e .; \
        else \
            pip install -e .; \
        fi \
     && if [ -d python/triton_kernels ]; then pip install -e python/triton_kernels --no-deps; fi; \
    fi

# -----------------------
# Hot patch: transformers dynamic_module_utils symlink bug (v5.12.1).
# _compute_local_source_files_hash calls Path(...).resolve() on custom-code
# module files, following the HF-cache snapshots/<hash>/x.py -> blobs/<blob>
# symlink. trust_remote_code models whose custom code uses relative imports
# (e.g. Kimi-K2.6's kimi_k25_vision_processing.py: `from .media_utils import`)
# then crash with FileNotFoundError: .../blobs/<name>.py at processor init.
# Mirrors upstream transformers PR #46618 (merged, not yet released): drop the
# .resolve() on the module file and its relative-import sources so the snapshot
# .py names (not the blob targets) are used. Self-skips once transformers ships
# the fix; fails the build loudly if the pattern is present but unpatched.
RUN python3 - <<'PY'
import pathlib
import transformers.dynamic_module_utils as m

MARKS = ["Path(resolved_module_file).resolve()", "Path(source_file).resolve()"]
path = pathlib.Path(m.__file__)
src = path.read_text()
if not any(mark in src for mark in MARKS):
    print("transformers dynamic_module_utils already fixed; no patch needed")
else:
    patched = (
        src.replace("Path(resolved_module_file).resolve()", "Path(resolved_module_file)")
           .replace("Path(source_file).resolve()", "Path(source_file)")
    )
    assert patched != src, "FATAL: transformers symlink patch matched nothing"
    path.write_text(patched)
    print("patched transformers dynamic_module_utils.py (symlink hash fix)")
PY

# -----------------------
# Performance environment variable.

# Skip CuDNN compatibility check - not applicable for ROCm (uses MIOpen instead)
ENV SGLANG_DISABLE_CUDNN_CHECK=1
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
ENV ROCM_QUICK_REDUCE_QUANTIZATION=INT8
ENV TORCHINDUCTOR_MAX_AUTOTUNE=1
ENV TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1

CMD ["/bin/bash"]
