# Usage (to build SGLang ROCm docker image):
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx942 -t v0.5.10.post1-rocm700-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx942-rocm720 -t v0.5.10.post1-rocm720-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx950 -t v0.5.10.post1-rocm700-mi35x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx950-rocm720 -t v0.5.10.post1-rocm720-mi35x -f rocm.Dockerfile .

# Usage (to build SGLang ROCm + Mori docker image):
# remove --build-arg NIC_BACKEND=ainic since new MoRI JIT will do NIC auto detection on target
# Keep the build-arg for user to select the desired nic support, current choice: [ainic, bxnt]
# if no set this arg, it will support nic auto detection. On a target with more than 1 type of
# RDMA NICs installed (rare), overwrite w. runtime env MORI_DEVICE_NIC = "bnxt"|"ionic"|"mlx5"
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx942 --build-arg ENABLE_MORI=1 -t v0.5.10.post1-rocm700-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx942-rocm720 --build-arg ENABLE_MORI=1 -t v0.5.10.post1-rocm720-mi30x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx950 --build-arg ENABLE_MORI=1 -t v0.5.10.post1-rocm700-mi35x -f rocm.Dockerfile .
#   docker build --build-arg SGL_BRANCH=v0.5.10.post1 --build-arg GPU_ARCH=gfx950-rocm720 --build-arg ENABLE_MORI=1 -t v0.5.10.post1-rocm720-mi35x -f rocm.Dockerfile .

# Default base images (slim ROCm dev images; PyTorch is installed below).
ARG BASE_IMAGE_ROCM700="rocm/dev-ubuntu-22.04:7.0-complete"
ARG BASE_IMAGE_ROCM720="rocm/dev-ubuntu-22.04:7.2-complete"

# AMD-published ROCm torch/triton/torchvision wheels (Python 3.10, "lw" lightweight variants).
# torchvision is required because the SGLang `timm` dep pulls it transitively;
# without a ROCm wheel installed up front, pip's resolver fetches the PyPI CUDA
# build of torchvision and cascades a CUDA torch replacement that breaks GPU access.
ARG ROCM700_TORCH_WHL="torch-2.8.0+rocm7.0.0.lw.git64359f59-cp310-cp310-linux_x86_64.whl"
ARG ROCM700_TRITON_WHL="pytorch_triton_rocm-3.4.0+rocm7.0.0.gitf9e5bf54-cp310-cp310-linux_x86_64.whl"
ARG ROCM700_TORCHVISION_WHL="torchvision-0.23.0+rocm7.0.0.git824e8c87-cp310-cp310-linux_x86_64.whl"
ARG ROCM700_WHL_INDEX="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0"

ARG ROCM720_TORCH_WHL="torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp310-cp310-linux_x86_64.whl"
ARG ROCM720_TRITON_WHL="triton-3.5.1+rocm7.2.0.gita272dfa8-cp310-cp310-linux_x86_64.whl"
ARG ROCM720_TORCHVISION_WHL="torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp310-cp310-linux_x86_64.whl"
ARG ROCM720_WHL_INDEX="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2"

# This is necessary for scope purpose
ARG GPU_ARCH=gfx950

# ===============================
# Shared base for ROCm 7.0 flavors: install Python toolchain + AMD torch/triton wheels.
FROM $BASE_IMAGE_ROCM700 AS base-rocm700
ARG ROCM700_TORCH_WHL
ARG ROCM700_TRITON_WHL
ARG ROCM700_TORCHVISION_WHL
ARG ROCM700_WHL_INDEX
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv \
        git curl wget gnupg ca-certificates build-essential \
    && mkdir -p /etc/apt/keyrings \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && python -m pip install --no-cache-dir --upgrade pip wheel setuptools \
    && cd /tmp \
    && curl -fLO "${ROCM700_WHL_INDEX}/${ROCM700_TORCH_WHL}" \
    && curl -fLO "${ROCM700_WHL_INDEX}/${ROCM700_TRITON_WHL}" \
    && curl -fLO "${ROCM700_WHL_INDEX}/${ROCM700_TORCHVISION_WHL}" \
    && pip install --no-cache-dir \
        "/tmp/${ROCM700_TORCH_WHL}" \
        "/tmp/${ROCM700_TRITON_WHL}" \
        "/tmp/${ROCM700_TORCHVISION_WHL}" \
    && rm -f "/tmp/${ROCM700_TORCH_WHL}" "/tmp/${ROCM700_TRITON_WHL}" "/tmp/${ROCM700_TORCHVISION_WHL}" \
    # Alias pytorch-triton-rocm dist-info as `triton` so pip considers SGLang's
    # transitive `triton` requirement (via xgrammar) satisfied; otherwise pip
    # pulls triton 3.7.0 from PyPI (CUDA build) and breaks the inductor path.
    # The pytorch_triton_rocm wheel already installs the `triton` Python module;
    # this just adds the package-name registration that pip's resolver needs.
    && SITE=$(python -c 'import site; print(site.getsitepackages()[0])') \
    && SRC_DIST=$(ls -d "$SITE"/pytorch_triton_rocm-*.dist-info | head -1) \
    && DST_DIST=$(echo "$SRC_DIST" | sed 's|/pytorch_triton_rocm-|/triton-|') \
    && cp -r "$SRC_DIST" "$DST_DIST" \
    && sed -i 's/^Name: pytorch[-_]triton[-_]rocm$/Name: triton/' "$DST_DIST/METADATA" \
    && python -c 'import importlib.metadata as m; print("pip sees triton ==", m.version("triton"))'

# ===============================
# Shared base for ROCm 7.2.2 flavors.
FROM $BASE_IMAGE_ROCM720 AS base-rocm720
ARG ROCM720_TORCH_WHL
ARG ROCM720_TRITON_WHL
ARG ROCM720_TORCHVISION_WHL
ARG ROCM720_WHL_INDEX
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv \
        git curl wget gnupg ca-certificates build-essential \
    && mkdir -p /etc/apt/keyrings \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && python -m pip install --no-cache-dir --upgrade pip wheel setuptools \
    && cd /tmp \
    && curl -fLO "${ROCM720_WHL_INDEX}/${ROCM720_TORCH_WHL}" \
    && curl -fLO "${ROCM720_WHL_INDEX}/${ROCM720_TRITON_WHL}" \
    && curl -fLO "${ROCM720_WHL_INDEX}/${ROCM720_TORCHVISION_WHL}" \
    && pip install --no-cache-dir \
        "/tmp/${ROCM720_TORCH_WHL}" \
        "/tmp/${ROCM720_TRITON_WHL}" \
        "/tmp/${ROCM720_TORCHVISION_WHL}" \
    && rm -f "/tmp/${ROCM720_TORCH_WHL}" "/tmp/${ROCM720_TRITON_WHL}" "/tmp/${ROCM720_TORCHVISION_WHL}"

# ===============================
# Per-arch base aliases (so the rest of the file can FROM ${GPU_ARCH} into the right base).
FROM base-rocm700 AS gfx942
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="0"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="32e1e6d76988e4fbc67cabd9eb72a45a3c6a1bab"

FROM base-rocm720 AS gfx942-rocm720
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="1"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="32e1e6d76988e4fbc67cabd9eb72a45a3c6a1bab"

FROM base-rocm700 AS gfx950
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="0"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="32e1e6d76988e4fbc67cabd9eb72a45a3c6a1bab"

FROM base-rocm720 AS gfx950-rocm720
ENV BUILD_VLLM="0"
ENV BUILD_TRITON="1"
ENV BUILD_LLVM="0"
ENV BUILD_AITER_ALL="1"
ENV BUILD_MOONCAKE="1"
ENV AITER_COMMIT_DEFAULT="32e1e6d76988e4fbc67cabd9eb72a45a3c6a1bab"

# ===============================
# Builder stage: sgl-model-gateway
FROM ${GPU_ARCH} AS builder-gateway

ARG GPU_ARCH=gfx950
ARG SGL_REPO="https://github.com/sgl-project/sglang.git"
ARG SGL_DEFAULT="main"
ARG SGL_BRANCH=${SGL_DEFAULT}

ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        protobuf-compiler libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && pip install --no-cache-dir maturin \
    && git clone ${SGL_REPO} /tmp/sglang \
    && cd /tmp/sglang \
    && git checkout ${SGL_BRANCH} \
    && sed -i -E 's|^(smg-[a-zA-Z-]+)\s*=\s*"~1\.0\.0"|\1 = "=1.0.0"|' \
           sgl-model-gateway/Cargo.toml \
    && cd sgl-model-gateway/bindings/python \
    && ulimit -n 65536 && CARGO_BUILD_JOBS=4 maturin build \
        --release --features vendored-openssl --out /tmp/gateway-wheel

# ===============================
# Builder stage: Mooncake
FROM ${GPU_ARCH} AS builder-mooncake

ARG GPU_ARCH=gfx950
ARG MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"
ARG MOONCAKE_COMMIT="b6a841dc78c707ec655a563453277d969fb8f38d"

ENV PATH=$PATH:/usr/local/go/bin

# Always create the staging dir so the unconditional COPY --from in the
# final stage is a safe no-op when BUILD_MOONCAKE != 1.
RUN mkdir -p /mooncake-install/usr/local \
    && if [ "$BUILD_MOONCAKE" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            zip unzip wget gcc make libtool autoconf \
            librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool \
            libibverbs-dev rdma-core \
            openssh-server openmpi-bin openmpi-common libopenmpi-dev \
        && rm -rf /var/lib/apt/lists/* \
        && git clone ${MOONCAKE_REPO} /tmp/Mooncake \
        && cd /tmp/Mooncake \
        && git checkout ${MOONCAKE_COMMIT} \
        && git submodule update --init --recursive \
        && bash dependencies.sh -y \
        && rm -rf /usr/local/go \
        && wget -q https://go.dev/dl/go1.22.2.linux-amd64.tar.gz \
        && tar -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz \
        && rm go1.22.2.linux-amd64.tar.gz \
        && mkdir -p build && cd build \
        && cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON \
        && make -j "$(nproc)" \
        && DESTDIR=/mooncake-install make install; \
    fi

# ===============================
# Builder stage: fast-hadamard-transform
FROM ${GPU_ARCH} AS builder-fht

ARG FHT_REPO="https://github.com/jeffdaily/fast-hadamard-transform.git"
ARG FHT_BRANCH="rocm"
ARG FHT_COMMIT="46efb7d776d38638fc39f3c803eaee3dd7016bd1"

RUN pip install --no-cache-dir wheel \
    && git clone --branch "${FHT_BRANCH}" "${FHT_REPO}" /tmp/fht \
    && cd /tmp/fht \
    && git checkout -f "${FHT_COMMIT}" \
    && FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE python setup.py bdist_wheel -d /tmp/fht-wheel

# ===============================
# Builder stage: TileLang
FROM ${GPU_ARCH} AS builder-tilelang

ARG GPU_ARCH=gfx950
ARG TILELANG_REPO="https://github.com/tile-ai/tilelang.git"
ARG TILELANG_COMMIT="a55a82302bf7f3c5af635b5c9146f728185cc900"

ENV DEBIAN_FRONTEND=noninteractive

RUN /bin/bash -lc 'set -euo pipefail; \
  echo "[TileLang] Building TileLang wheel for ${GPU_ARCH}"; \
  apt-get update && apt-get install -y --no-install-recommends \
      build-essential git wget curl ca-certificates gnupg \
      libgtest-dev libgmock-dev \
      libprotobuf-dev protobuf-compiler libgflags-dev libsqlite3-dev \
      python3 python3-dev python3-setuptools python3-pip python3-apt \
      gcc libtinfo-dev zlib1g-dev libedit-dev libxml2-dev \
      cmake ninja-build pkg-config libstdc++6 software-properties-common \
  && rm -rf /var/lib/apt/lists/*; \
  \
  VENV_PIP="/opt/venv/bin/pip"; \
  if [ ! -x "$VENV_PIP" ]; then VENV_PIP="pip3"; fi; \
  \
  # Build GoogleTest static libs (Ubuntu package ships sources only)
  cmake -S /usr/src/googletest -B /tmp/build-gtest -DBUILD_GTEST=ON -DBUILD_GMOCK=ON -DCMAKE_BUILD_TYPE=Release && \
  cmake --build /tmp/build-gtest -j"$(nproc)" && \
  cp -v /tmp/build-gtest/lib/*.a /usr/lib/x86_64-linux-gnu/ && \
  rm -rf /tmp/build-gtest; \
  \
  "$VENV_PIP" install --no-cache-dir --upgrade "setuptools>=77.0.3,<80" wheel cmake ninja scikit-build-core; \
  \
  # Locate ROCm llvm-config; fallback to LLVM 18 if missing
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
  # TVM bits need Cython + z3 before configure.
  # Pin z3-solver==4.15.4.0: 4.15.4.0 has a manylinux wheel; 4.15.5.0 builds from source (needs GCC 14+).
  "$VENV_PIP" install --no-cache-dir "cython>=0.29.36,<3.0" \
      "apache-tvm-ffi @ git+https://github.com/apache/tvm-ffi.git@37d0485b2058885bf4e7a486f7d7b2174a8ac1ce" \
      "z3-solver==4.15.4.0"; \
  \
  git clone --recursive "${TILELANG_REPO}" /opt/tilelang && \
  cd /opt/tilelang && \
  git fetch --depth=1 origin "${TILELANG_COMMIT}" || true && \
  git checkout -f "${TILELANG_COMMIT}" && \
  git submodule update --init --recursive && \
  if [ -f pyproject.toml ]; then sed -i "/^[[:space:]]*\"torch/d" pyproject.toml || true; fi && \
  export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=ON -DROCM_PATH=/opt/rocm -DLLVM_CONFIG=${LLVM_CONFIG} -DSKBUILD_SABI_VERSION= ${CMAKE_ARGS:-}" && \
  "$VENV_PIP" wheel -w /tmp/tilelang-wheel . -v --no-build-isolation --no-deps'

# ===============================
# Builder stage: Triton custom build
FROM ${GPU_ARCH} AS builder-triton

ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
ARG TRITON_COMMIT="42270451990532c67e69d753fbd026f28fcc4840"

# BUILD_TRITON is inherited as ENV from the selected base stage.
# Always create the output dir so the COPY --from in the final stage is a
# safe no-op when BUILD_TRITON != 1.
RUN mkdir -p /tmp/triton-wheel \
    && if [ "$BUILD_TRITON" = "1" ]; then \
        pip uninstall -y triton 2>/dev/null || true \
     && apt-get update && apt-get install -y --no-install-recommends cmake \
     && rm -rf /var/lib/apt/lists/* \
     && git clone ${TRITON_REPO} /tmp/triton-custom \
     && cd /tmp/triton-custom \
     && git checkout ${TRITON_COMMIT} \
     && pip install --no-cache-dir -r python/requirements.txt \
     && pip wheel --no-cache-dir --no-deps -w /tmp/triton-wheel .; \
    fi

# ===============================
# Final stage
FROM ${GPU_ARCH}

# This is necessary for scope purpose, again
ARG GPU_ARCH=gfx950
ENV GPU_ARCH_LIST=${GPU_ARCH%-*}
ENV PYTORCH_ROCM_ARCH=gfx942;gfx950

ARG SGL_REPO="https://github.com/sgl-project/sglang.git"
ARG SGL_DEFAULT="main"
ARG SGL_BRANCH=${SGL_DEFAULT}

# Version override for setuptools_scm (used in nightly builds)
ARG SETUPTOOLS_SCM_PRETEND_VERSION=""

ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT=""
ENV AITER_COMMIT="${AITER_COMMIT:-${AITER_COMMIT_DEFAULT}}"

ARG LLVM_REPO="https://github.com/jrbyrnes/llvm-project.git"
ARG LLVM_BRANCH="MainOpSelV2"
ARG LLVM_COMMIT="6520ace8227ffe2728148d5f3b9872a870b0a560"

ARG ENABLE_MORI=1
ARG NIC_BACKEND=none

ARG MORI_REPO="https://github.com/ROCm/mori.git"
ARG MORI_COMMIT="96ffa169710f214e76e07abe5008d686fe54522b"

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

# Install basic utilities + common pip dependencies in a single layer.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        setuptools_scm IPython orjson python-multipart torchao==0.9.0 pybind11 \
    && (apt-get purge -y sccache 2>/dev/null || true) \
    && (python -m pip uninstall -y sccache 2>/dev/null || true) \
    && rm -f "$(which sccache 2>/dev/null)" || true

# Install AMD SMI Python package from ROCm distribution.
# dev-ubuntu-22.04:*-complete images ship the sources under /opt/rocm/share/amd_smi
# but no Python wheel; build & install in-place. Skip if the dir is absent.
RUN set -eux; \
    if [ -d /opt/rocm/share/amd_smi ]; then \
        cd /opt/rocm/share/amd_smi && python3 -m pip install --no-cache-dir .; \
    else \
        echo "amd_smi sources not found at /opt/rocm/share/amd_smi, skipping"; \
    fi

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
     && make -j$(nproc) \
     && find /sgl-workspace/llvm-project -name '*.o' -delete; \
    fi

# -----------------------
# AITER
# Unset setuptools_scm override so AITER gets its own version (AITER_COMMIT), not SGLang's
# (SETUPTOOLS_SCM_PRETEND_VERSION is set later for SGLang nightly builds and would otherwise
# leak into AITER's version when AITER uses setuptools_scm)

# cherry pick b639cb6 commit for aiter_mhc_pre fix, may be removed in next aiter upgrade
ENV SETUPTOOLS_SCM_PRETEND_VERSION=
# Keep the base image's Torch-compatible Triton by default. Override with
# AITER_USE_SYSTEM_TRITON=0 when intentionally testing aiter-managed Triton.
ENV AITER_USE_SYSTEM_TRITON=1
RUN pip uninstall -y aiter 2>/dev/null || true \
 && git clone ${AITER_REPO} \
 && cd aiter \
 && git checkout ${AITER_COMMIT} \
 && git cherry-pick --no-commit b639cb63bcac4672dce33a731fad042a65cb3649 \
 && git submodule update --init --recursive \
 && pip install --no-cache-dir -r requirements.txt \
 && echo "[AITER] GPU_ARCH=${GPU_ARCH}" \
 && echo "[AITER] AITER_USE_SYSTEM_TRITON=${AITER_USE_SYSTEM_TRITON}" \
 && if [ "$BUILD_AITER_ALL" = "1" ] && [ "$BUILD_LLVM" = "1" ]; then \
      sh -c "HIP_CLANG_PATH=/sgl-workspace/llvm-project/build/bin/ PREBUILD_KERNELS=1 GPU_ARCHS=$GPU_ARCH_LIST python setup.py build_ext --inplace" \
      && sh -c "HIP_CLANG_PATH=/sgl-workspace/llvm-project/build/bin/ GPU_ARCHS=$GPU_ARCH_LIST pip install --no-cache-dir --config-settings editable_mode=compat -e ."; \
    elif [ "$BUILD_AITER_ALL" = "1" ]; then \
      sh -c "PREBUILD_KERNELS=1 GPU_ARCHS=$GPU_ARCH_LIST python setup.py build_ext --inplace" \
      && sh -c "GPU_ARCHS=$GPU_ARCH_LIST pip install --no-cache-dir --config-settings editable_mode=compat -e ."; \
    else \
      sh -c "GPU_ARCHS=$GPU_ARCH_LIST pip install --no-cache-dir --config-settings editable_mode=compat -e ."; \
    fi \
 && echo "export PYTHONPATH=/sgl-workspace/aiter:\${PYTHONPATH}" >> /etc/bash.bashrc \
 # Drop .o files and shrink .git; keep the source tree and .so kernels.
 && find /sgl-workspace/aiter -name '*.o' -delete 2>/dev/null || true \
 && (cd /sgl-workspace/aiter && git reflog expire --expire=now --all && git gc --prune=now --quiet) || true

# -----------------------
# Mooncake: install runtime deps; build artifacts come from builder-mooncake.
RUN if [ "$BUILD_MOONCAKE" = "1" ]; then \
     apt-get update && apt-get install -y --no-install-recommends \
         librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool \
         libibverbs-dev rdma-core \
         openssh-server openmpi-bin openmpi-common libopenmpi-dev \
         libgoogle-glog-dev libjsoncpp-dev libunwind-dev libnuma-dev \
         libboost-all-dev libssl-dev libyaml-cpp-dev libgflags-dev \
         libgrpc-dev libgrpc++-dev libprotobuf-dev \
     && rm -rf /var/lib/apt/lists/*; \
    fi

COPY --from=builder-mooncake /mooncake-install/ /

# -----------------------
# Build SGLang
ARG BUILD_TYPE=all

# Set version for setuptools_scm if provided (for nightly builds). Only pass in the SGLang
# pip install RUN so it does not affect AITER, sgl-model-gateway, TileLang, FHT, MORI, etc.
ARG SETUPTOOLS_SCM_PRETEND_VERSION

RUN pip uninstall -y sgl_kernel sglang 2>/dev/null || true \
    && git clone ${SGL_REPO} \
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
         export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION}" && python -m pip --no-cache-dir install -e "python[srt_hip,diffusion_hip]"; \
       else \
         export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION}" && python -m pip --no-cache-dir install -e "python[all_hip]"; \
       fi \
    && find /sgl-workspace/sglang -name '*.o' -delete 2>/dev/null || true \
    && (cd /sgl-workspace/sglang && git reflog expire --expire=now --all && git gc --prune=now --quiet) || true \
    && python -m pip cache purge

# Copy config files to support MI300X in virtualized environments (MI300X_VF).  Symlinks will not be created in image build.
RUN find /sgl-workspace/sglang/python/sglang/srt/layers/quantization/configs/ \
         /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/ \
         -type f -name '*MI300X*' | xargs -I {} sh -c 'vf_config=$(echo "$1" | sed "s/MI300X/MI300X_VF/"); cp "$1" "$vf_config"' -- {}

# -----------------------
# Install sgl-model-gateway from builder-gateway wheel.
COPY --from=builder-gateway /tmp/gateway-wheel/ /tmp/gateway-wheel/
RUN pip install --no-cache-dir --force-reinstall /tmp/gateway-wheel/*.whl \
    && rm -rf /tmp/gateway-wheel /root/.cache

# -----------------------
# Install TileLang from builder-tilelang wheel; tvm_ffi and z3-solver are
# runtime deps of the bundled TVM.
ENV DEBIAN_FRONTEND=noninteractive
ENV LIBGL_ALWAYS_INDIRECT=1
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

COPY --from=builder-tilelang /tmp/tilelang-wheel/ /tmp/tilelang-wheel/
RUN /bin/bash -lc 'set -euo pipefail; \
  VENV_PIP="/opt/venv/bin/pip"; \
  VENV_PY="/opt/venv/bin/python"; \
  if [ ! -x "$VENV_PIP" ]; then VENV_PIP="pip3"; fi; \
  if [ ! -x "$VENV_PY" ]; then VENV_PY="python3"; fi; \
  "$VENV_PIP" install --no-cache-dir \
      "apache-tvm-ffi @ git+https://github.com/apache/tvm-ffi.git@37d0485b2058885bf4e7a486f7d7b2174a8ac1ce" \
      "z3-solver==4.15.4.0" \
  && "$VENV_PIP" install --no-cache-dir --no-deps --force-reinstall /tmp/tilelang-wheel/*.whl \
  && rm -rf /tmp/tilelang-wheel; \
  "$VENV_PY" -c "import tilelang; print(tilelang.__version__)"'

# -----------------------
# Install fast-hadamard-transform from builder-fht wheel. --no-deps is
# required because the wheel declares torch as a runtime dep, and pip would
# otherwise resolve it to PyPI CUDA torch and replace the ROCm torch from
# the base image.
COPY --from=builder-fht /tmp/fht-wheel/ /tmp/fht-wheel/
RUN pip install --no-cache-dir --no-deps --force-reinstall /tmp/fht-wheel/*.whl \
    && rm -rf /tmp/fht-wheel

# -----------------------
# Python tools
RUN python3 -m pip install --no-cache-dir \
    py-spy \
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
      echo "ERROR: unknown NIC_BACKEND=${NIC_BACKEND}. Use one of: none, ainic, bnxt"; \
      exit 2; \
      ;; \
  esac; \
  \
  # Build/install MORI (editable)
  export MORI_GPU_ARCHS="${GPU_ARCH_LIST}"; \
  echo "[MORI] MORI_GPU_ARCHS=${MORI_GPU_ARCHS} NIC_BACKEND=${NIC_BACKEND}"; \
  rm -rf /sgl-workspace/mori; \
  git clone "${MORI_REPO}" /sgl-workspace/mori; \
  cd /sgl-workspace/mori; \
  git checkout "${MORI_COMMIT}"; \
  git submodule update --init --recursive; \
  python3 setup.py develop; \
  python3 -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))" > /etc/ld.so.conf.d/torch.conf; \
  ldconfig; \
  echo "export PYTHONPATH=/sgl-workspace/mori:\${PYTHONPATH}" >> /etc/bash.bashrc; \
  find /sgl-workspace/mori -name "*.o" -delete 2>/dev/null || true; \
  ( cd /sgl-workspace/mori && git reflog expire --expire=now --all && git gc --prune=now --quiet ) || true; \
  echo "[MORI] Done."'

# -----------------------
# Install custom Triton from builder-triton wheel (opt-in via BUILD_TRITON=1).
# When BUILD_TRITON=0 the builder produced no wheel and this is a no-op; the
# AMD-published triton installed in the base stage is used instead.
COPY --from=builder-triton /tmp/triton-wheel/ /tmp/triton-wheel/
RUN set -eux; \
    if ls /tmp/triton-wheel/*.whl 1>/dev/null 2>&1; then \
        pip uninstall -y triton 2>/dev/null || true; \
        pip install --no-cache-dir --force-reinstall /tmp/triton-wheel/*.whl; \
    fi; \
    rm -rf /tmp/triton-wheel

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
