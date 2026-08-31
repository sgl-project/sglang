# Usage (to build SGLang ROCm docker image):
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx942 -t v0.5.17-rocm700-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx942-rocm720 -t v0.5.17-rocm720-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx942-rocm724 -t v0.5.17-rocm724-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx950 -t v0.5.17-rocm700-mi35x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx950-rocm720 -t v0.5.17-rocm720-mi35x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx950-rocm724 -t v0.5.17-rocm724-mi35x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx942-rocm1000 -t v0.5.17-rocm1000-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx950-rocm1000 -t v0.5.17-rocm1000-mi35x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx1250-rocm1000 -t v0.5.17-rocm1000-mi45x -f rocm.Dockerfile .
#
# Flavor notes:
#   GPU_ARCH=*-rocm724 is built on a Python 3.12 base and upgrades the stack to
#   torch 2.11 (+torchvision 0.26 / torchaudio 2.11) and Triton 3.7.
#   The ROCm 7.2.0 flavors remain on Python 3.10 and torch 2.9.1.
#   GPU_ARCH=*-rocm1000 is Python 3.12 + torch 2.11, and takes the
#   whole ROCm stack from AMD's stable wheel channel rather than an apt
#   ROCm base image; see the rocm1000-base stage for what that changes.
#   GPU_ARCH=gfx1250-rocm1000 carries the gfx1250 bring-up onto that same GA
#   wheel channel. The gfx1250 workarounds key off GPU_ARCH_LIST=gfx1250 rather
#   than the flavor name.

# Usage (to build SGLang ROCm + Mori docker image):
# remove --build-arg NIC_BACKEND=ainic since new MoRI JIT will do NIC auto detection on target
# Keep the build-arg for user to select the desired nic support, current choice: [ainic, bxnt]
# if no set this arg, it will support nic auto detection. On a target with more than 1 type of
# RDMA NICs installed (rare), overwrite w. runtime env MORI_DEVICE_NIC = "bnxt"|"ionic"|"mlx5"
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx942 --build-arg ENABLE_MORI=1 -t v0.5.17-rocm700-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx942-rocm720 --build-arg ENABLE_MORI=1 -t v0.5.17-rocm720-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx950 --build-arg ENABLE_MORI=1 -t v0.5.17-rocm700-mi35x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx950-rocm720 --build-arg ENABLE_MORI=1 -t v0.5.17-rocm720-mi35x -f rocm.Dockerfile .

# Usage (to build SGLang ROCm + NIXL docker image, for prefill/decode disaggregation):
# Builds UCX (--with-rocm) and upstream ai-dynamo/nixl from source by default.
# Set ENABLE_NIXL=0 to skip NIXL.
# At runtime use --disaggregation-transfer-backend nixl (env is wired via /etc/bash.bashrc).
#   docker build --build-arg SGL_BRANCH=v0.5.17 --build-arg GPU_ARCH=gfx950-rocm720 -t v0.5.17-rocm720-mi35x -f rocm.Dockerfile .

# Default base images
ARG BASE_IMAGE_942="rocm/sgl-dev:rocm7-vllm-20250904"
ARG BASE_IMAGE_942_ROCM720="rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1"
ARG BASE_IMAGE_942_ROCM724="rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"
ARG BASE_IMAGE_950="rocm/sgl-dev:rocm7-vllm-20250904"
ARG BASE_IMAGE_950_ROCM720="rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1"
ARG BASE_IMAGE_950_ROCM724="rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"
# The ROCm 10.0.0 flavors default to the rocm1000-base stage below rather
# than a published image; point these at one to build on a prebuilt base.
ARG BASE_IMAGE_942_ROCM1000="rocm1000-base"
ARG BASE_IMAGE_950_ROCM1000="rocm1000-base"
ARG BASE_IMAGE_1250_ROCM1000="rocm1000-base"
ARG BASE_IMAGE_ROCM1000="ubuntu:24.04"

# This is necessary for scope purpose
ARG GPU_ARCH=gfx950

# ===============================
# Base image 942 with rocm700 and args
FROM $BASE_IMAGE_942 AS gfx942
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="0"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="c16d44b93a528b2a4bfd6d8d3409116d465872a9"

# ===============================
# Base image 942 with rocm720 and args
FROM $BASE_IMAGE_942_ROCM720 AS gfx942-rocm720
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="1"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="c16d44b93a528b2a4bfd6d8d3409116d465872a9"
ENV TRITON_COMMIT_DEFAULT="42270451990532c67e69d753fbd026f28fcc4840"

# ===============================
# Base image 942 with rocm724 and args (Python 3.12 + torch 2.11)
FROM $BASE_IMAGE_942_ROCM724 AS gfx942-rocm724
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="1"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="c16d44b93a528b2a4bfd6d8d3409116d465872a9"
# Pin the ROCm torch stack for every pip invocation in this flavor. The file is
# filled in after the torch 2.11 upgrade below; it must already exist (empty is
# valid) because pip reads PIP_CONSTRAINT from the first pip call onwards.
# Deliberately still set in the shipped image, not just during the build: a later
# `pip install` that resolves torch would otherwise pull the PyPI CUDA build over
# this ROCm one, and the constraint turns that into a resolution error instead.
# It names only torch / torchvision / torchaudio, so nothing else is constrained.
ENV PIP_CONSTRAINT="/etc/sglang/constraints/torch-rocm.txt"
RUN mkdir -p /etc/sglang/constraints && : > /etc/sglang/constraints/torch-rocm.txt
# Work around ROCM-21485: the CUDA/ROCm IPC path leaks GPU memory (a freed IPC
# block is not returned to the driver). Legacy IPC mode releases it. Verified on
# ROCm 7.2.1 and 7.2.4; scoped to this flavor so rocm700 / rocm720 keep current
# IPC behavior.
ENV HSA_ENABLE_IPC_MODE_LEGACY=1

# ===============================
# Base image 950 and args
FROM $BASE_IMAGE_950 AS gfx950
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="0"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="c16d44b93a528b2a4bfd6d8d3409116d465872a9"

# ===============================
# Base image 950 with rocm720 and args
FROM $BASE_IMAGE_950_ROCM720 AS gfx950-rocm720
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="1"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="c16d44b93a528b2a4bfd6d8d3409116d465872a9"
ENV TRITON_COMMIT_DEFAULT="42270451990532c67e69d753fbd026f28fcc4840"

# ===============================
# Base image 950 with rocm724 and args (Python 3.12 + torch 2.11)
FROM $BASE_IMAGE_950_ROCM724 AS gfx950-rocm724
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="1"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="c16d44b93a528b2a4bfd6d8d3409116d465872a9"
# Pin the ROCm torch stack for every pip invocation in this flavor. The file is
# filled in after the torch 2.11 upgrade below; it must already exist (empty is
# valid) because pip reads PIP_CONSTRAINT from the first pip call onwards.
# Deliberately still set in the shipped image, not just during the build: a later
# `pip install` that resolves torch would otherwise pull the PyPI CUDA build over
# this ROCm one, and the constraint turns that into a resolution error instead.
# It names only torch / torchvision / torchaudio, so nothing else is constrained.
ENV PIP_CONSTRAINT="/etc/sglang/constraints/torch-rocm.txt"
RUN mkdir -p /etc/sglang/constraints && : > /etc/sglang/constraints/torch-rocm.txt
# Work around ROCM-21485: the CUDA/ROCm IPC path leaks GPU memory (a freed IPC
# block is not returned to the driver). Legacy IPC mode releases it. Verified on
# ROCm 7.2.1 and 7.2.4; scoped to this flavor so rocm700 / rocm720 keep current
# IPC behavior.
ENV HSA_ENABLE_IPC_MODE_LEGACY=1

# ===============================
# Shared ROCm 10.0.0 base for gfx942 and gfx950. Assemble the stack from
# AMD's stable wheels on a plain Ubuntu base so each output image carries only
# its own GPU device payload.
# The SDK lands in site-packages instead of /opt/rocm, which the rest of this
# Dockerfile and AITER both assume, hence the path fixups below.
#
# This is deliberately AMD's stable channel, not a prerelease or nightly.
# Every ROCm/PyTorch artifact below is pinned to the 10.0.0 release.
#
# Python 3.12 (the Ubuntu 24.04 default) rather than 3.13/3.14: st_attn==0.0.7,
# vsa==0.0.4, petit_kernel==0.0.2 and wave-lang==3.8.2 publish wheels only up to
# cp313 and no sdist, so pip has no candidate at all for srt_hip on 3.14.
FROM $BASE_IMAGE_ROCM1000 AS rocm1000-base

# Redeclare the global selector inside this stage so each matrix build installs
# only the device payload for its target image (gfx942 or gfx950).
ARG GPU_ARCH

# ROCM_TRITON_VERSION rather than TRITON_VERSION: the final stage declares a
# TRITON_VERSION of its own for the ROCm 7.2 wheel, and a --build-arg would
# otherwise land on both.
ARG ROCM_SDK_VERSION="10.0.0"
ARG ROCM_TORCH_VERSION="2.11.0"
ARG ROCM_TORCHVISION_VERSION="0.26.0"
ARG ROCM_TORCHAUDIO_VERSION="2.11.0"
ARG ROCM_TRITON_VERSION="3.8.0+git4cff872c"
ARG ROCM_INDEX_URL="https://stable.repo.amd.com/rocm/whl-next/"
# Keep device targets data-driven: adding a new image should require one list
# entry here, not another pairwise OTHER_ROCM_DEVICE_ARCH mapping.
ARG ROCM_DEVICE_ARCH_LIST="gfx942 gfx950 gfx1250"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        gnupg \
        libstdc++-12-dev \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        python3.12-venv \
        wget \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m pip install --no-cache-dir -U pip setuptools setuptools_scm wheel

# The two release jobs invoke separate Docker builds. Derive the device target
# from GPU_ARCH so the MI300 image carries only gfx942 wheels and the MI350
# image carries only gfx950 wheels. Keeping the packages as explicit specs also
# makes the intended per-image device payload explicit to the resolver.
RUN set -eux; \
    ROCM_DEVICE_ARCH="${GPU_ARCH%%-*}"; \
    case " ${ROCM_DEVICE_ARCH_LIST} " in \
      *" ${ROCM_DEVICE_ARCH} "*) ;; \
      *) echo "Unsupported ROCm 10.0.0 GPU_ARCH=${GPU_ARCH}"; exit 1 ;; \
    esac; \
    python3 -m pip install --no-cache-dir \
        --index-url ${ROCM_INDEX_URL} \
        "rocm-sdk-core==${ROCM_SDK_VERSION}" \
        "rocm-sdk-libraries==${ROCM_SDK_VERSION}" \
        "rocm-sdk-devel==${ROCM_SDK_VERSION}" \
        "rocm-sdk-device-${ROCM_DEVICE_ARCH}==${ROCM_SDK_VERSION}" \
        "torch==${ROCM_TORCH_VERSION}+rocm${ROCM_SDK_VERSION}" \
        "torchvision==${ROCM_TORCHVISION_VERSION}+rocm${ROCM_SDK_VERSION}" \
        "torchaudio==${ROCM_TORCHAUDIO_VERSION}+rocm${ROCM_SDK_VERSION}" \
        "amd-torch-device-${ROCM_DEVICE_ARCH}==${ROCM_TORCH_VERSION}+rocm${ROCM_SDK_VERSION}" \
        "amd-torchvision-device-${ROCM_DEVICE_ARCH}==${ROCM_TORCHVISION_VERSION}+rocm${ROCM_SDK_VERSION}" \
        "triton==${ROCM_TRITON_VERSION}.rocm${ROCM_SDK_VERSION}"; \
    for package in \
        "rocm-sdk-device-${ROCM_DEVICE_ARCH}" \
        "amd-torch-device-${ROCM_DEVICE_ARCH}" \
        "amd-torchvision-device-${ROCM_DEVICE_ARCH}"; do \
      if ! python3 -m pip show "${package}" >/dev/null 2>&1; then \
        echo "Missing target ROCm device package: ${package}"; \
        exit 1; \
      fi; \
    done; \
    for candidate_arch in ${ROCM_DEVICE_ARCH_LIST}; do \
      [ "${candidate_arch}" = "${ROCM_DEVICE_ARCH}" ] && continue; \
      for package in \
          "rocm-sdk-device-${candidate_arch}" \
          "amd-torch-device-${candidate_arch}" \
          "amd-torchvision-device-${candidate_arch}"; do \
        if python3 -m pip show "${package}" >/dev/null 2>&1; then \
          echo "Unexpected non-target ROCm device package: ${package}"; \
          exit 1; \
        fi; \
      done; \
    done

RUN rocm-sdk init && rocm-sdk targets

# rocm-sdk init expands a devel tree that carries its own copy of libamd_smi,
# byte-identical to the one in _rocm_sdk_core that HIP loads through its RPATH.
# Since ROCM_HOME below puts the devel tree on LD_LIBRARY_PATH, the amdsmi
# python package binds that second copy while torch already holds the first,
# and two independent copies in one process each keep their own global state:
# whichever initialises second enumerates no devices. torch asks amdsmi for the
# device count before HIP, so `torch.cuda.device_count()` comes back 0 on a
# machine where hipGetDeviceCount() says 1. Collapse the duplicate so both land
# on the same library. Idempotent when the SDK already ships a symlink here.
RUN set -eux; \
    SP="$VIRTUAL_ENV/lib/python3.12/site-packages"; \
    CORE=$(ls "$SP"/_rocm_sdk_core/lib/libamd_smi.so.* 2>/dev/null | head -1); \
    DEVEL="$SP/_rocm_sdk_devel/lib/libamd_smi.so"; \
    if [ -n "${CORE}" ] && [ -e "${DEVEL}" ] && [ ! -L "${DEVEL}" ]; then \
      ln -sf "${CORE}" "${DEVEL}"; \
      echo "linked ${DEVEL} -> ${CORE}"; \
    fi

ENV ROCM_HOME=$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_devel
ENV ROCM_PATH=$ROCM_HOME
ENV CPATH=$ROCM_HOME/include
ENV LIBRARY_PATH=$ROCM_HOME/lib
ENV LD_LIBRARY_PATH=$ROCM_HOME/lib
RUN echo 'export PATH=$ROCM_HOME/llvm/bin:$ROCM_HOME/bin:$PATH' >> /etc/bash.bashrc

# The SDK's hsakmtTargets.cmake hardcodes /usr/lib64/libc.so from its own build
# host; Ubuntu keeps libc in /lib/x86_64-linux-gnu, so cmake would otherwise
# fail with "ninja: error: /usr/lib64/libc.so missing and no known rule to make it".
RUN mkdir -p /usr/lib64 && ln -sf /lib/x86_64-linux-gnu/libc.so /usr/lib64/libc.so

# ROCm lives in site-packages here, but AITER shells out to
# /opt/rocm/llvm/bin/amdgpu-arch at runtime to pick DEFAULT_GPU_ARCH, and the
# rest of this Dockerfile (TileLang, UCX, amd_smi) refers to /opt/rocm throughout.
RUN ln -s ${ROCM_HOME} /opt/rocm

# ===============================
# Base image 942 with ROCm 10.0.0 and args (Python 3.12 + torch 2.11)
# BUILD_TRITON=0 keeps the Triton installed above, which is the build AMD ships
# with this SDK; the BUILD_TRITON=1 path installs a ROCm 7.2 wheel instead.
FROM $BASE_IMAGE_942_ROCM1000 AS gfx942-rocm1000
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="0"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="c16d44b93a528b2a4bfd6d8d3409116d465872a9"
# Same reasoning as the rocm724 stages: keep pip from resolving the image's
# ROCm torch away to a PyPI CUDA build. Populated after the stack is in place.
ENV PIP_CONSTRAINT="/etc/sglang/constraints/torch-rocm.txt"
RUN mkdir -p /etc/sglang/constraints && : > /etc/sglang/constraints/torch-rocm.txt

# ===============================
# Base image 950 with ROCm 10.0.0 and args (Python 3.12 + torch 2.11)
FROM $BASE_IMAGE_950_ROCM1000 AS gfx950-rocm1000
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="0"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="c16d44b93a528b2a4bfd6d8d3409116d465872a9"
ENV PIP_CONSTRAINT="/etc/sglang/constraints/torch-rocm.txt"
RUN mkdir -p /etc/sglang/constraints && : > /etc/sglang/constraints/torch-rocm.txt

# ===============================
# Base image 1250 with ROCm 10.0.0 GA and args (Python 3.12 + torch 2.11)
# The gfx1250 build paths are keyed on GPU_ARCH_LIST=gfx1250 rather than the
# flavor name, so they apply here unchanged.
FROM $BASE_IMAGE_1250_ROCM1000 AS gfx1250-rocm1000
ENV BUILD_VLLM="0"
# Unlike the gfx942/gfx950 images, this one replaces the SDK's Triton: the
# revision below is what the MI45x bring-up ran on, and it carries a fix the
# SDK build does not have yet.
ENV BUILD_TRITON="1"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
# gfx1250 tracks a different AITER than the gfx942/gfx950 baseline: this commit
# plus the four reverts applied at clone time are what the gfx1250 kernels were
# brought up against.
ENV AITER_COMMIT_DEFAULT="a6d2b564fd671724a3720b8edf70e8d674e4d694"
# The upstream Triton the gfx1250 bring-up was validated against, carried over
# from the ROCm 7.14 flavor this image replaced. Built from source below.
ENV TRITON_COMMIT_DEFAULT="76940ad348795521b3dc9f6c79acd7309ff924e3"
ENV PIP_CONSTRAINT="/etc/sglang/constraints/torch-rocm.txt"
RUN mkdir -p /etc/sglang/constraints && : > /etc/sglang/constraints/torch-rocm.txt

# Local source stage: with BRANCH_TYPE=local the build context is copied here and
# used instead of git clone (mirrors docker/Dockerfile's local_src stage).
FROM scratch AS local_src
COPY . /src

# ===============================
# Chosen arch and args
FROM ${GPU_ARCH}

# This is necessary for scope purpose, again
ARG GPU_ARCH=gfx950
RUN echo GPU_ARCH="${GPU_ARCH}" \
    && echo "GPU_ARCH_LIST=${GPU_ARCH%%-*}" >> /etc/environment \
    && echo "export GPU_ARCH_LIST=${GPU_ARCH%%-*}" >> /etc/bash.bashrc
# ARG is build-time only. Stamp the stage name (gfx950-rocm724, gfx942, ...)
# so CI can read which AITER_COMMIT_DEFAULT block to use instead of guessing
# from torch or HIP — 720 may also ship torch 2.11 later, and both 7.2 flavors
# report HIP 7.2*.
ENV GPU_ARCH=${GPU_ARCH}
ENV GPU_ARCH_LIST=${GPU_ARCH%-*}
ENV PYTORCH_ROCM_ARCH="gfx942;gfx950;gfx1250"

ARG SGL_REPO="https://github.com/sgl-project/sglang.git"
ARG SGL_DEFAULT="main"
ARG SGL_BRANCH=${SGL_DEFAULT}
ARG BRANCH_TYPE=remote

# Version override for setuptools_scm (used in nightly builds)
ARG SETUPTOOLS_SCM_PRETEND_VERSION=""

ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
ENV TRITON_COMMIT="${TRITON_COMMIT:-${TRITON_COMMIT_DEFAULT}}"

# ROCm 7.2 Triton (BUILD_TRITON=1 stages only). Both wheels are the same
# upstream revision, triton-lang/triton@89002410. AITER only requires
# triton>=3.6.0 and treats the base image as the owner of the version, so the
# choice is ours; bump these together after checking the index.
ARG TRITON_INDEX_URL="https://pypi.amd.com/triton/release/rocm-7.2.0/simple/"
ARG TRITON_VERSION="3.7.0+amd.rocm7.2.0.git89002410"
ARG TRITON_KERNELS_VERSION="1.0.0+amd.rocm7.2.0.git89002410"

# ROCm 7.2.4 torch upgrade pins (Python 3.12). Torch 2.11 for ROCm 7.2 is only
# published on the PyTorch Foundation index; AMD's repo.radeon.com wheels top
# out at torch 2.10.
ARG TORCH_ROCM_INDEX_URL="https://download.pytorch.org/whl/rocm7.2"
ARG TORCH_ROCM_VERSION="2.11.0+rocm7.2"
ARG TORCHVISION_ROCM_VERSION="0.26.0+rocm7.2"
ARG TORCHAUDIO_ROCM_VERSION="2.11.0+rocm7.2"

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
ARG MORI_COMMIT="12d1bc32d0c93dcd5062e74f4e0f772e36e1aac4"

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
#   http://*security.ubuntu.com entries in every /etc/apt source file are
#   rewritten to point at the given base URL, e.g.
#     --build-arg UBUNTU_MIRROR=https://archive.ubuntu.com
#     --build-arg UBUNTU_MIRROR=https://tw.archive.ubuntu.com
#     --build-arg UBUNTU_MIRROR=http://internal-cache.example.com
#   This mirrors the pattern already used in docker/Dockerfile (NVIDIA) and
#   docker/npu.Dockerfile, and lets CI runners that cannot reach Canonical's
#   port-80 mirror IPs still complete `apt-get update`. Every file, not just
#   sources.list: the noble base used by rocm724 keeps its URIs in the deb822
#   /etc/apt/sources.list.d/ubuntu.sources instead.
# - The 80-net-hardening apt config adds retries + per-request timeout so that
#   transient mirror flakes don't immediately fail a build (apt's default is 0
#   retries).
ARG UBUNTU_MIRROR=
USER root

RUN if [ -n "$UBUNTU_MIRROR" ]; then \
        find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) \
          -exec sed -i \
            -e "s|http://[^[:space:]/]*archive.ubuntu.com|$UBUNTU_MIRROR|g" \
            -e "s|http://[^[:space:]/]*security.ubuntu.com|$UBUNTU_MIRROR|g" \
            {} + ; \
    fi && \
    printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
        > /etc/apt/apt.conf.d/80-net-hardening

# Fix hipDeviceGetName returning empty / generic names.
# amdgpu.ids maps PCI IDs to marketing names. The ROCm 7.0 base is missing it.
# The 7.2.0 base was built with amdgpu-install and already has it. The 7.2.4
# ubuntu24.04 base installs the `rocm` apt metapackage and does not; noble's
# distro table has MI300 (74A*) but no MI355X (75A3), so gfx950-rocm724 would
# otherwise report "AMD Radeon Graphics" and miss every name-keyed config.
# The ROCm 10.0.0 flavors need nothing here: their libdrm comes from the pip SDK, which
# links the ids table into libdrm_amdgpu.so itself (the .so carries the MI300X /
# MI325X / MI355X names and never opens share/libdrm/amdgpu.ids).
# See https://github.com/ROCm/ROCm/issues/5992
RUN set -eux; \
    case "${GPU_ARCH}" in \
      *rocm1000*) \
        echo "ROCm 10.0.0 (GPU_ARCH=${GPU_ARCH}): pip SDK libdrm has the ids table built in, skipping"; \
        ;; \
      *rocm724*) \
        echo "ROCm 7.2.4 (GPU_ARCH=${GPU_ARCH}): installing libdrm-amdgpu from graphics/7.2.4 noble"; \
        curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors https://repo.radeon.com/rocm/rocm.gpg.key \
          | gpg --dearmor -o /etc/apt/keyrings/amdgpu-graphics.gpg \
        && echo 'deb [arch=amd64,i386 signed-by=/etc/apt/keyrings/amdgpu-graphics.gpg] https://repo.radeon.com/graphics/7.2.4/ubuntu noble main' \
          > /etc/apt/sources.list.d/amdgpu-graphics.list \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
             libdrm-amdgpu-common \
             libdrm-amdgpu-amdgpu1 \
             libdrm2-amdgpu \
        && rm -rf /var/lib/apt/lists/* \
        && cp /opt/amdgpu/share/libdrm/amdgpu.ids /usr/share/libdrm/amdgpu.ids; \
        ;; \
      *rocm720*) \
        echo "ROCm 7.2.0 (GPU_ARCH=${GPU_ARCH}): libdrm-amdgpu packages already present, skipping"; \
        ;; \
      *) \
        echo "ROCm 7.0 (GPU_ARCH=${GPU_ARCH}): installing libdrm-amdgpu packages"; \
        curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors https://repo.radeon.com/rocm/rocm.gpg.key \
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
# Neither the ROCm 7.2 base image (rocm/pytorch) nor the pip-installed ROCm 10.0.0
# SDK pre-installs this package.
RUN set -eux; \
    case "${GPU_ARCH}" in \
      *rocm720*|*rocm724*|*rocm1000*) \
        echo "ROCm 7.2 / 10.0.0 flavor detected from GPU_ARCH=${GPU_ARCH}"; \
        cd /opt/rocm/share/amd_smi \
        && python3 -m pip install --no-cache-dir . \
        ;; \
      *) \
        echo "Not rocm720/rocm724/rocm1000 (GPU_ARCH=${GPU_ARCH}), skip amdsmi installation"; \
        ;; \
    esac

# -----------------------
# ROCm 7.2.4: upgrade torch 2.10 -> 2.11 (+ vision/audio), which pulls triton-rocm 3.6.0.
# Done here, before AITER / sgl-kernel, so those extensions build against torch 2.11's ABI.
RUN case "${GPU_ARCH}" in \
      *-rocm724) \
        python3 -m pip install --no-cache-dir --index-url "${TORCH_ROCM_INDEX_URL}" \
            "torch==${TORCH_ROCM_VERSION}" \
            "torchvision==${TORCHVISION_ROCM_VERSION}" \
            "torchaudio==${TORCHAUDIO_ROCM_VERSION}" \
        ;; \
      *) \
        echo "Not a ROCm 7.2.4 flavor (GPU_ARCH=${GPU_ARCH}), keep base torch/triton"; \
        ;; \
    esac

# Populate the PIP_CONSTRAINT file, which only the explicitly upgraded torch
# stages define, so resolving AITER and SGLang cannot replace the torch stack.
# Triton is left out: on rocm724 the BUILD_TRITON step installs it later, and on
# rocm1000 it came from the ROCm SDK alongside torch.
RUN case "${GPU_ARCH}" in \
      *-rocm724|*-rocm1000) \
        python3 -m pip freeze \
          | grep -E '^(torch|torchvision|torchaudio)(==| @ )' \
          > /etc/sglang/constraints/torch-rocm.txt \
        && cat /etc/sglang/constraints/torch-rocm.txt \
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
# AITER
# Clear any inherited setuptools_scm override so AITER gets its own version
# from AITER_COMMIT rather than SGLang's nightly version.

ENV SETUPTOOLS_SCM_PRETEND_VERSION=
# Compile AITER against the base image's Triton; the Triton step at the end of
# this file installs the pinned one afterwards.
ENV AITER_USE_SYSTEM_TRITON=1
RUN pip uninstall -y aiter
# Use `checkout -f` so the smudge-filter-induced "dirty" working tree from
# AITER's .gitattributes (*.csv text eol=lf, added in ROCm/aiter#3370) does not
# block AITER_COMMIT overrides that predate that rule. The working tree was just
# produced by a fresh `git clone` above, so there are no real user changes to
# preserve.
# cherry pick 8578af1 commit for v4 fp4 indexer kv-cache fix, may be removed in next aiter upgrade
RUN git clone ${AITER_REPO} \
 && cd aiter \
 && git checkout -f ${AITER_COMMIT} \
 && git cherry-pick --no-commit 8578af153f4fa1e007fede7e3c1e1b373f07af4c \
 && git submodule update --init --recursive \
 && pip install -r requirements.txt \
 && if [ "${GPU_ARCH_LIST}" = "gfx1250" ]; then \
    git revert --no-edit --no-commit 8d581370b; \
    git revert --no-edit --no-commit 79c11b677; \
    git revert --no-edit --no-commit 1ecb760a5; \
    git revert --no-edit --no-commit e708f6c15; \
 fi

RUN cd aiter \
     && echo "[AITER] GPU_ARCH=${GPU_ARCH}" \
     && echo "[AITER] AITER_USE_SYSTEM_TRITON=${AITER_USE_SYSTEM_TRITON}" \
     && if [ "${GPU_ARCH_LIST}" = "gfx1250" ]; then \
          PATH=$PATH:$ROCM_HOME/llvm/bin ENABLE_CK=0 GPU_ARCHS="${GPU_ARCH_LIST}" python setup.py build_ext --inplace \
          && PATH=$PATH:$ROCM_HOME/llvm/bin ENABLE_CK=0 GPU_ARCHS="${GPU_ARCH_LIST}" pip install --no-build-isolation -e .; \
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

# torch 2.11 Dynamo may pass a base torch.Stream; drop after ROCm/aiter#4817.
RUN python3 <<'PY'
from pathlib import Path
p = Path("/sgl-workspace/aiter/csrc/cpp_itfs/torch_utils.py")
s = p.read_text()
old = """        elif isinstance(arg, torch.cuda.Stream):
            c_args.append(ctypes.cast(arg.cuda_stream, ctypes.c_void_p))
"""
new = """        elif isinstance(arg, torch.Stream):
            handle = getattr(arg, "cuda_stream", None)
            if handle is None:
                handle = torch.cuda.Stream(
                    stream_id=arg.stream_id,
                    device_index=arg.device_index,
                    device_type=arg.device_type,
                ).cuda_stream
            c_args.append(ctypes.cast(handle, ctypes.c_void_p))
"""
if old in s:
    p.write_text(s.replace(old, new))
PY

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
    && pip install pybind11

# Rust toolchain — needed by setuptools-rust to build the sglang-mm extension
# (sglang.srt.rust_extensions._multimodal) during the sglang pip install below
# and later by sgl-model-gateway. Must precede the sglang install.
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 --retry 5 --retry-delay 3 --retry-all-errors -sSf https://sh.rustup.rs | sh -s -- -y \
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
    && cd python/sglang/kernels/aot \
    && rm -f pyproject.toml \
    && mv pyproject_rocm.toml pyproject.toml \
    && AMDGPU_TARGET=$GPU_ARCH_LIST python setup_rocm.py install
RUN pip list --format=freeze | grep -E '^(torch|triton)' > /tmp/constraints.txt

# srt_hip pins compressed-tensors==0.15.0, which requires torch<2.11 and so
# cannot be satisfied on the ROCm 7.2.4/1000 torch 2.11 stack. The *_rocm724
# extras carry a 0.16.0 pin instead; all other flavors keep the extras they used before.
RUN cd sglang \
    && cp python/pyproject_other.toml python/pyproject.toml \
    && case "${GPU_ARCH}" in \
         *-rocm1000) srt_extras="srt_hip_rocm724,diffusion_hip"; \
                     all_extras="all_hip_rocm724" ; \
                     CONS="-c /tmp/constraints.txt" ; \
                     ;; \
         *-rocm724) srt_extras="srt_hip_rocm724,diffusion_hip"; \
                    all_extras="all_hip_rocm724" ; \
                    CONS="-c /tmp/constraints.txt" ; \
                    ;; \
         *-rocm720) srt_extras="srt_hip,diffusion_hip"; \
                    all_extras="all_hip" ; \
                    CONS="-c /tmp/constraints.txt" ; \
                    ;; \
         *) srt_extras="srt_hip,diffusion_hip"; \
            all_extras="all_hip" ; \
            CONS="-c /tmp/constraints.txt" ; \
            echo 'diffusers==0.37.0' >> /tmp/constraints.txt ; \
            echo 'transformers==5.12.1' >> /tmp/constraints.txt ; \
            echo 'tokenizers==0.22.2' >> /tmp/constraints.txt ; \
            echo 'huggingface_hub==1.27.0' >> /tmp/constraints.txt ; \
            ;; \
       esac \
    && if [ "$BUILD_TYPE" = "srt" ]; then \
         export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION}" && python -m pip --no-cache-dir install $CONS -e "python[${srt_extras}]"; \
       else \
         export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION}" && python -m pip --no-cache-dir install $CONS -e "python[${all_extras}]"; \
       fi

RUN python -m pip cache purge

RUN case "${GPU_ARCH##*-}" in \
      rocm724) expected_torch="2.11."; expected_audio="2.11."; expected_vision="0.26." ;; \
      rocm1000) expected_torch="2.11."; expected_audio="2.11."; expected_vision="0.26." ;; \
      *) exit 0 ;; \
    esac \
    && python3 -m pip check \
    && python3 -c "import torch, torchaudio, torchvision, triton; expected={'torch':'${expected_torch}','torchaudio':'${expected_audio}','torchvision':'${expected_vision}'}; actual={'torch':torch.__version__,'torchaudio':torchaudio.__version__,'torchvision':torchvision.__version__,'triton':triton.__version__}; assert torch.version.hip, actual; assert all(actual[name].startswith(version) for name, version in expected.items()), actual; print('Validated ROCm stack:', actual, 'HIP', torch.version.hip)" \
    && if pip list --format=freeze | grep -Eq '^nvidia-.*-cu[0-9]+'; then \
         echo "ERROR: NVIDIA CUDA runtime packages were installed into the ROCm image"; \
         exit 1; \
       fi

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
    curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /etc/apt/keyrings/llvm.gpg; \
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
  if [ "${GPU_ARCH_LIST}" = "gfx1250" ]; then \
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
    PYTORCH_ROCM_ARCH=${GPU_ARCH_LIST} python setup.py install'

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
      initramfs-tools \
  && rm -rf /var/lib/apt/lists/*; \
  \
  # gfx1250 brought its own MORI build up against a wider set of RDMA, MPI and
  # gRPC packages. Scoped to that arch so the CDNA images keep the package set
  # they were released with.
  if [ "${GPU_ARCH_LIST}" = "gfx1250" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
        libdrm-dev \
        librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool \
        libibverbs-dev rdma-core \
        openssh-server openmpi-bin openmpi-common libopenmpi-dev \
        libgrpc++-dev protobuf-compiler-grpc \
    && rm -rf /var/lib/apt/lists/*; \
  fi; \
  \
  # NIC backend deps — mori auto-detects NIC at runtime (MORI_DEVICE_NIC env var override).
  # Only vendor packages are installed here for dlopen (e.g. libionic.so); no compile-time flags needed.
  case "${NIC_BACKEND}" in \
    # default: install ainic and bxnt driver
    none) \
      apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg apt-transport-https && \
      rm -rf /var/lib/apt/lists/* && mkdir -p /etc/apt/keyrings; \
      curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/amdainic.gpg; \
      echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdainic.gpg] https://repo.radeon.com/amdainic/pensando/ubuntu/${AINIC_VERSION} ${UBUNTU_CODENAME} main" \
        > /etc/apt/sources.list.d/amdainic.list; \
      apt-get update && apt-get install -y --no-install-recommends \
          libionic-dev \
          ionic-common \
      ; \
      rm -rf /var/lib/apt/lists/*; \
      install -m 0755 -d /etc/apt/keyrings \
      && curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors https://packages.broadcom.com/artifactory/api/security/keypair/PackagesKey/public -o /etc/apt/keyrings/broadcom-nic.asc \
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
      curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/amdainic.gpg; \
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
       && curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors https://packages.broadcom.com/artifactory/api/security/keypair/PackagesKey/public -o /etc/apt/keyrings/broadcom-nic.asc \
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
  # The pip ROCm SDK vendors NUMA and libdrm under lib/rocm_sysdeps, which is on
  # none of the three search paths the MORI build needs: hsakmt-config.cmake
  # calls find_dependency(NUMA), rocm_smi.h reaches for <libdrm/drm.h>, and
  # mori_application links -ldrm/-ldrm_amdgpu. The SDK's own libraries find
  # these through an $ORIGIN/rocm_sysdeps/lib RPATH that MORI does not inherit,
  # hence the ldconfig entry; every soname in there is librocm_sysdeps_*-prefixed,
  # so it shadows nothing system-wide. Scope this explicitly to ROCm 10 so the
  # ROCm 7.2 and 7.2.4 MORI build paths remain byte-for-byte equivalent here.
  ROCM_SYSDEPS="${ROCM_HOME:-/opt/rocm}/lib/rocm_sysdeps"; \
  if [ "${GPU_ARCH_LIST}" = "gfx1250" ] && [ -d "${ROCM_SYSDEPS}" ]; then \
    # gfx1250 was brought up needing the SDK's own cmake trees on the prefix
    # path as well: that is what lets hsakmt-config.cmake resolve
    # find_dependency(NUMA) without patching MORI's CMakeLists.
    export PATH="${ROCM_HOME}/bin:${PATH}"; \
    export CMAKE_PREFIX_PATH="${ROCM_SYSDEPS}/lib/cmake:${ROCM_SYSDEPS}:${ROCM_HOME}/lib/cmake:${ROCM_HOME}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"; \
    export CPATH="${ROCM_SYSDEPS}/include${CPATH:+:${CPATH}}"; \
    export LIBRARY_PATH="${ROCM_SYSDEPS}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"; \
    echo "${ROCM_SYSDEPS}/lib" > /etc/ld.so.conf.d/rocm-sysdeps.conf; \
    ldconfig; \
    echo "[MORI] rocm_sysdeps prefix: ${ROCM_SYSDEPS}"; \
  elif [ "${GPU_ARCH##*-}" = "rocm1000" ] && [ -d "${ROCM_SYSDEPS}" ]; then \
    export CMAKE_PREFIX_PATH="${ROCM_SYSDEPS}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"; \
    export CPATH="${ROCM_SYSDEPS}/include${CPATH:+:${CPATH}}"; \
    export LIBRARY_PATH="${ROCM_SYSDEPS}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"; \
    echo "${ROCM_SYSDEPS}/lib" > /etc/ld.so.conf.d/rocm-sysdeps.conf; \
    ldconfig; \
    echo "[MORI] rocm_sysdeps prefix: ${ROCM_SYSDEPS}"; \
  fi; \
  python3 setup.py develop; \
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
  # Mooncake's dependencies.sh apt-installs Ubuntu's libabsl-dev (20220623 on
  # the noble base used by rocm724 and rocm1000). NIXL's meson then
  # finds absl_base but no absl_log and refuses to fall back to its bundled
  # Abseil -- "that would result in a mix of Abseil versions at runtime" -- so
  # nixl fails at metadata generation. Drop just the -dev package (headers and
  # pkg-config files); the runtime library that already-built components link
  # against stays in place.
  #
  # gfx1250 was brought up dropping the runtime package as well, so keep that
  # for gfx1250 only rather than changing what the CDNA images were released with.
  if [ "${GPU_ARCH_LIST}" = "gfx1250" ]; then \
    apt-get remove -y libabsl-dev libabsl20220623 || true; \
  else \
    case "${GPU_ARCH}" in *-rocm724|*-rocm1000) apt-get remove -y libabsl-dev ;; esac; \
  fi; \
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

# transformers 5.12.1: don't follow HF-cache symlinks when hashing custom modules
# (transformers#46618, not yet released).
RUN python3 -c "from pathlib import Path; import transformers.dynamic_module_utils as m; p=Path(m.__file__); t=p.read_text(); p.write_text(t.replace('Path(resolved_module_file).resolve()','Path(resolved_module_file)').replace('Path(source_file).resolve()','Path(source_file)'))"

# -----------------------
# Install AMD's ROCm Triton, replacing the base image's. The local version is
# part of the pin: `==3.7.0` alone would accept any revision the index later
# publishes under that number, and pip would choose between them by lexical
# order of the git hash rather than by date.
#
# Keep this last. Base ROCm Torch pins triton==3.5.1 and the torch patch above
# is what drops that pin, so installing Triton any earlier lets the next pip
# install pull CUDA torch instead. The hip check below is the tripwire.
# torch 2.11 names this `triton-rocm`; uninstall it so the pin is the only Triton.
RUN if [ "$BUILD_TRITON" = "1" ]; then \
        case "${GPU_ARCH}" in \
          gfx1250-rocm1000) \
            echo "[Triton] gfx1250: building ${TRITON_COMMIT} from source"; \
            pip uninstall -y triton triton-rocm || true \
            && apt-get update && apt-get install -y --no-install-recommends cmake && rm -rf /var/lib/apt/lists/* \
            && git clone ${TRITON_REPO} triton-custom \
            && cd triton-custom \
            && git checkout ${TRITON_COMMIT} \
            && pip install -r python/requirements.txt \
            && pip install -e . \
            && if [ -d python/triton_kernels ]; then pip install -e python/triton_kernels --no-deps; fi; \
            ;; \
          *rocm72*) \
            echo "[Triton] ROCm 7.2: installing pinned wheels from ${TRITON_INDEX_URL}"; \
            pip uninstall -y triton-rocm || true && \
            PIP_NO_CACHE_DIR=1 pip install --extra-index-url ${TRITON_INDEX_URL} \
                "triton==${TRITON_VERSION}" "triton-kernels==${TRITON_KERNELS_VERSION}"; \
            ;; \
        esac \
     && python3 -c "import torch; from importlib.metadata import version; v = version('triton'); k = version('triton-kernels'); assert torch.version.hip is not None, torch.__version__; print(f'[Triton] ROCm Torch {torch.__version__}, Triton {v}, triton-kernels {k}')"; \
    fi

# torch 2.11 still Requires-Dist: triton-rocm after the swap above.
RUN case "${GPU_ARCH}" in *-rocm724) python3 -c "import pathlib,re,importlib.metadata as m; p=pathlib.Path(m.distribution('torch')._path)/'METADATA'; v=m.version('triton'); t,n=re.subn(r'^Requires-Dist: (?:triton|triton-rocm)==[^ ;]+', 'Requires-Dist: triton=='+v, p.read_text(), count=1, flags=re.M); assert n==1, n; p.write_text(t)" ;; esac

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
