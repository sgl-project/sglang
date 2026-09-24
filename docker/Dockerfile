ARG CUDA_VERSION=13.0.3
# Declared before the first FROM so it is a global ARG: the `runtime` stage
# starts from a fresh FROM and therefore does not inherit ARGs from `base`,
# but the `runtime-efa` target needs this version to pin the EFA Mooncake wheel
# to the same release the `framework` stage installed.
ARG MOONCAKE_VERSION=0.3.13
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG TARGETARCH
ARG BUILD_TYPE=all
ARG BRANCH_TYPE=remote
ARG SGL_KERNEL_VERSION=0.4.7
ARG SGL_VERSION
ARG USE_LATEST_SGLANG=0
ARG GDRCOPY_VERSION=2.5.1
ARG SGL_NCCL_VERSION=2.30.7
ARG PIP_DEFAULT_INDEX
ARG UBUNTU_MIRROR
ARG GITHUB_ARTIFACTORY=github.com
ARG INSTALL_FLASHINFER_JIT_CACHE=0
ARG FLASHINFER_VERSION=0.6.18
ARG MSCCLPP_VERSION=sglang-v0.9.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/ \
    FLASHINFER_VERSION=${FLASHINFER_VERSION}

# Add GKE default lib and bin locations
ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Replace Ubuntu sources if specified
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    sed -i "s|http://.*archive.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list && \
    sed -i "s|http://.*security.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list; \
fi

# Python setup (combined with apt update to reduce layers)
# Ubuntu 24.04 ships Python 3.12 in main, so we no longer need the deadsnakes
# PPA. Dropping it avoids transient Launchpad 504s in `add-apt-repository`.
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt update && apt install -y --no-install-recommends wget software-properties-common \
    && apt install -y --no-install-recommends python3.12-full python3.12-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    # Fix for apt-add-repository
    && cd /usr/lib/python3/dist-packages/ \
    && ln -s apt_pkg.cpython-312-*-linux-gnu.so apt_pkg.so

# create virtual env for sglang to avoid conflict with system python packages
RUN python3 -m venv /opt/sglang
ENV PATH="/opt/sglang/bin:${PATH}"

# Install system dependencies (organized by category for better caching)
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    # Core system utilities
    ca-certificates \
    software-properties-common \
    netcat-openbsd \
    kmod \
    unzip \
    openssh-server \
    curl \
    wget \
    lsof \
    locales \
    # Build essentials (needed for framework stage)
    build-essential \
    cmake \
    perl \
    patchelf \
    ccache \
    git-lfs \
    # MPI and NUMA
    libopenmpi-dev \
    libnuma1 \
    libnuma-dev \
    numactl \
    # transformers multimodal VLM
    ffmpeg \
    # InfiniBand/RDMA
    libibverbs-dev \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libnl-route-3-dev \
    libnl-3-dev \
    ibverbs-providers \
    infiniband-diags \
    perftest \
    # Development libraries
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libunwind-dev \
    libboost-all-dev \
    libssl-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev \
    libcurl4-openssl-dev \
    libczmq4 \
    libczmq-dev \
    libfabric-dev \
    linux-libc-dev \
    # Package building tools
    devscripts \
    debhelper \
    fakeroot \
    dkms \
    check \
    libsubunit0 \
    libsubunit-dev \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Replace pip global cache if specified
RUN if [ -n "${PIP_DEFAULT_INDEX}" ]; then \
    python3 -m pip config set global.index-url ${PIP_DEFAULT_INDEX}; \
fi

# GDRCopy installation
RUN mkdir -p /tmp/gdrcopy && cd /tmp \
    && curl --retry 3 --retry-delay 2 -fsSL -o v${GDRCOPY_VERSION}.tar.gz \
        https://${GITHUB_ARTIFACTORY}/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz \
    && tar -xzf v${GDRCOPY_VERSION}.tar.gz && rm v${GDRCOPY_VERSION}.tar.gz \
    && cd gdrcopy-${GDRCOPY_VERSION}/packages \
    && CUDA=/usr/local/cuda ./build-deb-packages.sh \
    && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
    && cd / && rm -rf /tmp/gdrcopy

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

########################################################
########## PARALLEL BUILDER STAGES ####################
########################################################
#
# These stages run IN PARALLEL via BuildKit:
#
#   base
#     |
#     +-- torch_deps ------> flashinfer_cache (needs flashinfer)
#     |                  \-> hpc_ops_builder (cmake-only build)
#     |
#     +-- devtools_builder (independent)
#     +-- gateway_builder  (independent, only needs gateway source)
#     |
#     v
#   framework (combines all artifacts)
#

########################################################
# PARALLEL STAGE 1: Torch/Deps Builder (starts from base)
########################################################
FROM base AS torch_deps

ARG CUDA_VERSION
ARG BUILD_TYPE
ARG SGL_KERNEL_VERSION
ARG GITHUB_ARTIFACTORY

WORKDIR /sgl-workspace

# Rust toolchain for setuptools-rust extensions (e.g. sglang-grpc).
# Requires >= 1.85 (edition 2024). Inherited by framework via FROM torch_deps.
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
    && rustc --version && cargo --version

# Install sgl-kernel (from pre-built wheel)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel html5lib six \
    && case "$CUDA_VERSION" in \
        13.0.3) ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    # --no-deps prevents pip from pulling torch from default PyPI
    && python3 -m pip install sglang-kernel==${SGL_KERNEL_VERSION} --force-reinstall --no-deps

# Copy dep spec + Rust crate source + proto files. setuptools-rust compiles the
# Rust extension during the stub wheel build; the crate's build.rs references
# ../../proto for tonic_build. Split from the pip install so source changes to
# these paths invalidate the dep-install layer, but Python source changes don't.
COPY python/pyproject.toml /tmp/sglang_deps/python/pyproject.toml
COPY rust/sglang-grpc      /tmp/sglang_deps/rust/sglang-grpc
COPY rust/sglang-mm        /tmp/sglang_deps/rust/sglang-mm
COPY proto                 /tmp/sglang_deps/proto

# Install sglang dependencies (torch, transformers, etc.).
# Generate constraints.txt to prevent reinstalling these deps in later stages.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cargo/registry \
    case "$CUDA_VERSION" in \
        13.0.3) CUINDEX=130 ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && cd /tmp/sglang_deps/python \
    && mkdir -p sglang \
    && touch sglang/__init__.py \
    && echo '__version__ = "0.0.0"' > sglang/version.py \
    && touch README.md \
    && touch LICENSE \
    && python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX} ".[${BUILD_TYPE}]" \
    && cd /sgl-workspace \
    && rm -rf /tmp/sglang_deps \
    && pip freeze | grep -v "^sglang==" > /sgl-workspace/constraints.txt

# distro resolves to the apt python3-distro under /usr/lib/python3, which the runtime
# stage does not COPY; force a pip copy into /usr/local so it survives the stage split.
RUN python3 -m pip install --ignore-installed --no-deps distro

# Cosmos3 guardrails default ON: sglang enables them whenever `cosmos_guardrail`
# is importable (Cosmos3Pipeline.create_pipeline_stages); opt out at runtime with
# SGLANG_DISABLE_COSMOS3_GUARDRAILS=1. Installed with --no-deps because the
# package metadata conflicts with our pins in ways that don't matter at runtime
# (imageio>=2.37 vs our ==2.36) or would be harmful (opencv-python — also pulled
# by retinaface-py — would shadow our opencv-python-headless cv2). The deps not
# already in the image are installed explicitly; the find_spec check mirrors
# sglang's availability probe plus cosmos_guardrail's own imports.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-deps \
        cosmos-guardrail==0.3.1 \
        retinaface-py==0.0.2 \
    && python3 -m pip install \
        "better-profanity==0.7.0" \
        "nltk==3.9.1" \
        "peft==0.18.1" \
    && python3 -c "import importlib.util as u; missing = [m for m in ('cosmos_guardrail', 'retinaface', 'better_profanity', 'nltk', 'cv2', 'peft', 'sentencepiece') if u.find_spec(m) is None]; assert not missing, f'guardrail deps missing: {missing}'"

########################################################
# PARALLEL STAGE 2: HPC-Ops Builder (needs torch_deps)
########################################################
FROM torch_deps AS hpc_ops_builder

# HPC-Ops (https://github.com/Tencent/hpc-ops, MIT): fused attention / MoE /
# RoPE kernels from the Tencent Hunyuan AI Infra team, consumed by the opt-in
# hpc_ops attention and MoE runner backends.
ARG HPC_OPS_COMMIT=ab1a402724635507037426068f6cddc3d30dc0a8

WORKDIR /build

# The kernels target Hopper (sm90a) only, so skip non-x86_64 images.
# setup.py derives the version from `git rev-parse`, so keep the .git dir
# (a source zip archive would not build).
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /wheels && \
    if [ "$(uname -m)" = "x86_64" ]; then \
        git clone https://github.com/Tencent/hpc-ops.git && \
        cd hpc-ops && \
        git checkout ${HPC_OPS_COMMIT} && \
        python3 setup.py bdist_wheel -d /wheels; \
    fi

########################################################
# PARALLEL STAGE 3: FlashInfer Cache (needs torch_deps)
########################################################
FROM torch_deps AS flashinfer_cache

ARG CUDA_VERSION
ARG INSTALL_FLASHINFER_JIT_CACHE
ARG FLASHINFER_VERSION

# Stage jit-cache/cubin artifacts into /flashinfer_jit_output for clean COPY later
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$CUDA_VERSION" in \
        13.0.3) CUINDEX=130 ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && mkdir -p /flashinfer_jit_output \
    # flashinfer-cubin is CUDA-version-agnostic, unlike jit-cache, so its index-url has no cu${CUINDEX} suffix
    && python3 -m pip install flashinfer-cubin==${FLASHINFER_VERSION} --index-url https://flashinfer.ai/whl \
    && cp -r /opt/sglang/lib/python3.12/site-packages/flashinfer_cubin /flashinfer_jit_output/ \
    && cp -r /opt/sglang/lib/python3.12/site-packages/flashinfer_cubin-*.dist-info /flashinfer_jit_output/ \
    && if [ "$INSTALL_FLASHINFER_JIT_CACHE" = "1" ]; then \
        python3 -m pip install flashinfer-jit-cache==${FLASHINFER_VERSION} --index-url https://flashinfer.ai/whl/cu${CUINDEX} \
        && cp -r /opt/sglang/lib/python3.12/site-packages/flashinfer_jit_cache /flashinfer_jit_output/ \
        && cp -r /opt/sglang/lib/python3.12/site-packages/flashinfer_jit_cache-*.dist-info /flashinfer_jit_output/ ; \
    fi

########################################################
# PARALLEL STAGE 4: Dev Tools Builder (starts from base)
########################################################
FROM base AS devtools_builder

ARG GITHUB_ARTIFACTORY

WORKDIR /tools

# Minimal apt deps needed for oh-my-zsh install in this stage
# Full dev apt packages (gdb, vim, tmux, nsight, etc.) are installed in the framework stage
RUN --mount=type=cache,target=/var/cache/apt,id=devtools-apt \
    apt-get update && apt-get install -y --no-install-recommends zsh git \
    && rm -rf /var/lib/apt/lists/*

# Download CLI tools (each in its own layer for parallel downloads)
RUN curl --retry 3 --retry-delay 2 -LSso /tools/diff-so-fancy \
        https://${GITHUB_ARTIFACTORY}/so-fancy/diff-so-fancy/releases/download/v1.4.4/diff-so-fancy \
    && chmod +x /tools/diff-so-fancy

RUN curl --retry 3 --retry-delay 2 -LSso /tools/clang-format \
        https://${GITHUB_ARTIFACTORY}/muttleyxd/clang-tools-static-binaries/releases/download/master-32d3ac78/clang-format-16_linux-amd64 \
    && chmod +x /tools/clang-format

RUN curl --retry 3 --retry-delay 2 -fsSL -o /tmp/clangd.zip \
        https://${GITHUB_ARTIFACTORY}/clangd/clangd/releases/download/18.1.3/clangd-linux-18.1.3.zip \
    && unzip -q /tmp/clangd.zip -d /tmp \
    && cp /tmp/clangd_18.1.3/bin/* /tools/ \
    && mkdir -p /tools/lib && cp -r /tmp/clangd_18.1.3/lib/* /tools/lib/ \
    && rm -rf /tmp/clangd.zip /tmp/clangd_18.1.3

RUN CMAKE_VERSION=3.31.1 \
    && ARCH=$(uname -m) \
    && CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}" \
    && curl --retry 3 --retry-delay 2 -fsSL -o "/tmp/${CMAKE_INSTALLER}.tar.gz" \
        "https://${GITHUB_ARTIFACTORY}/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}.tar.gz" \
    && tar -xzf "/tmp/${CMAKE_INSTALLER}.tar.gz" -C /tmp \
    && cp -r "/tmp/${CMAKE_INSTALLER}/bin/"* /tools/ \
    && mkdir -p /tools/share && cp -r "/tmp/${CMAKE_INSTALLER}/share/"* /tools/share/ \
    && rm -rf "/tmp/${CMAKE_INSTALLER}" "/tmp/${CMAKE_INSTALLER}.tar.gz"

RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://just.systems/install.sh | \
    sed "s|https://github.com|https://${GITHUB_ARTIFACTORY}|g" | \
    bash -s -- --tag 1.42.4 --to /tools

# Install oh-my-zsh and plugins
RUN sh -c "$(curl --retry 3 --retry-delay 2 -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone --depth 1 https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

########################################################
# PARALLEL STAGE 5: Gateway Builder (starts from base)
########################################################
# Builds sgl-model-gateway in isolation so Python-only changes
# don't trigger a full Rust recompilation.
FROM base AS gateway_builder

ARG GITHUB_ARTIFACTORY
ARG BRANCH_TYPE
ARG SGL_VERSION
ARG USE_LATEST_SGLANG

WORKDIR /build

# Copy ONLY the gateway source (not the full repo)
COPY sgl-model-gateway /build/sgl-model-gateway

# Install Rust, build gateway binary and Python bindings, then clean up Rust toolchain
RUN --mount=type=cache,target=/root/.cache/pip \
    curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && python3 -m pip install maturin \
    && cd /build/sgl-model-gateway/bindings/python \
    && ulimit -n 65536 && maturin build --release --features vendored-openssl --out /build/gateway_wheels \
    && cd /build/sgl-model-gateway \
    && cargo build --release --bin sgl-model-gateway --features vendored-openssl \
    && cp target/release/sgl-model-gateway /build/sgl-model-gateway-bin \
    && rm -rf /root/.cargo /root/.rustup /build/sgl-model-gateway/target /build/sgl-model-gateway/bindings/python/target

########################################################
########## Final Framework Image ######################
########################################################
#
# Combines all artifacts from parallel builder stages
#
FROM torch_deps AS framework

ARG BRANCH_TYPE
ARG BUILD_TYPE
ARG CUDA_VERSION
ARG SGL_VERSION
ARG USE_LATEST_SGLANG
ARG GITHUB_ARTIFACTORY
ARG MOONCAKE_VERSION
ARG MSCCLPP_VERSION
# Renamed from NCCL_VERSION: the nvidia/cuda base image sets ENV NCCL_VERSION
# (Debian apt format, e.g. 2.28.3-1), which shadows a same-named ARG inside RUN.
ARG SGL_NCCL_VERSION

WORKDIR /sgl-workspace

# =============================================================================
# Copy artifacts from parallel builders
# =============================================================================

# Copy HPC-Ops wheel and install (empty on non-x86_64; kernels are sm90a-only)
COPY --from=hpc_ops_builder /wheels /tmp/wheels/hpc-ops
RUN --mount=type=cache,target=/root/.cache/pip \
    if ls /tmp/wheels/hpc-ops/*.whl >/dev/null 2>&1; then \
        pip install --no-deps /tmp/wheels/hpc-ops/*.whl; \
    fi && rm -rf /tmp/wheels/hpc-ops

# Copy flashinfer cubin (always) and jit-cache (if installed) packages
COPY --from=flashinfer_cache /flashinfer_jit_output/ /opt/sglang/lib/python3.12/site-packages/

# Copy dev tools
COPY --from=devtools_builder /tools/diff-so-fancy /usr/local/bin/
COPY --from=devtools_builder /tools/clang-format /usr/local/bin/
COPY --from=devtools_builder /tools/clangd /usr/local/bin/
COPY --from=devtools_builder /tools/lib /usr/local/lib/
COPY --from=devtools_builder /tools/cmake /usr/local/bin/
COPY --from=devtools_builder /tools/ctest /usr/local/bin/
COPY --from=devtools_builder /tools/cpack /usr/local/bin/
COPY --from=devtools_builder /tools/share/cmake-3.31 /usr/local/share/cmake-3.31
COPY --from=devtools_builder /tools/just /usr/local/bin/
COPY --from=devtools_builder /root/.oh-my-zsh /root/.oh-my-zsh

# Install dev apt packages (need to re-run since we're in a different stage)
RUN --mount=type=cache,target=/var/cache/apt,id=framework-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    gdb \
    ninja-build \
    vim \
    tmux \
    htop \
    zsh \
    tree \
    silversearcher-ag \
    cloc \
    pkg-config \
    bear \
    less \
    rdma-core \
    openssh-server \
    gnuplot \
    infiniband-diags \
    perftest \
    ibverbs-providers \
    libibumad3 \
    libibverbs1 \
    libnl-3-200 \
    libnl-route-3-200 \
    librdmacm1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install NVIDIA development tools
RUN --mount=type=cache,target=/var/cache/apt,id=framework-apt \
    apt update -y \
    && apt install -y --no-install-recommends gnupg \
    && echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2004/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "x86_64"; fi)/7fa2af80.pub \
    && apt update -y \
    && apt install -y --no-install-recommends nsight-systems-cli \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python packages and tools (before source copy for better caching)
# =============================================================================

# Install Mooncake
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}

# Install MSCCL++ Python dependencies and package (builds extension via CMake through pip)
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth=1 --branch ${MSCCLPP_VERSION} https://${GITHUB_ARTIFACTORY}/microsoft/mscclpp.git /tmp/mscclpp \
    && case "${CUDA_VERSION}" in \
           13.*) \
               CMAKE_ARGS="-DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_CUDA=ON -DMSCCLPP_GPU_ARCHS=80,90,100,100a,103,103a" \
               python3 -m pip install "/tmp/mscclpp[cuda13]"; \
               ;; \
           *) \
               echo "Unsupported CUDA version for MSCCL++: ${CUDA_VERSION}" && exit 1; \
               ;; \
       esac \
    && rm -rf /tmp/mscclpp

# Install essential Python packages (use constraints to prevent conflicts)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -c /sgl-workspace/constraints.txt \
    datamodel_code_generator \
    pre-commit \
    pytest \
    black \
    isort \
    icdiff \
    uv \
    wheel \
    scikit-build-core \
    py-spy \
    cubloaty \
    google-cloud-storage \
    pandas \
    matplotlib \
    tabulate \
    termplotlib \
    "runai-model-streamer[s3,gcs,azure]>=0.15.7"

# PyTorch 2.13 requires nvidia-nccl-cu13==2.29.7, so keep that version while
# resolving dependencies and generating constraints.txt. DeepEP needs 2.30.7
# at runtime; apply the override only after the constrained dependency solve.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --force-reinstall --no-deps \
        "nvidia-nccl-cu13==${SGL_NCCL_VERSION}"

# The `nixl` stub package is needed (it owns the `nixl` import path) but
# unconditionally requires nixl-cu12, so install it with --no-deps and pair it
# with the nixl-cu13 binary to avoid shipping wrong-CUDA libs.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install nixl nixl-cu13 --no-deps \
    && python3 -m pip install "cuda-python>=13,<14"

# Install the latest available AI Dynamo prerelease from NVIDIA's package index.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo

# Add yank script
COPY --chown=root:root --chmod=755 docker/configs/yank /usr/local/bin/yank

# These configs are optional; users can override them by mounting their own files
COPY docker/configs/opt/.vimrc /opt/sglang/.vimrc
COPY docker/configs/opt/.tmux.conf /opt/sglang/.tmux.conf
COPY docker/configs/opt/.gitconfig /opt/sglang/.gitconfig

# Configure development environment
COPY docker/configs/.zshrc /root/.zshrc

# Fix Trivy-reported CVEs
# pip:             urllib3 (CVE-2025-43859), pillow (CVE-2026-25990)
# binutils family: CVE-2025-{1147,1148,3198,5244,5245,7545,7546,8225,11082,11083,11412,11413,11414,11494,11839,11840}
# libgnutls30t64:  CVE-2025-{9820,14831}
# libpam:          CVE-2024-10963
# libsqlite3-0:    CVE-2025-{6965,7709}
# libtasn1-6:      CVE-2025-13151
# dpkg:            CVE-2025-6297
RUN python3 -m pip install --upgrade "urllib3>=2.6.3" "pillow>=12.1.1"
RUN --mount=type=cache,target=/var/cache/apt,id=framework-apt \
    apt-get update && apt-get install -y --only-upgrade \
    binutils binutils-common binutils-x86-64-linux-gnu libbinutils \
    libctf0 libctf-nobfd0 libgprofng0 libsframe1 \
    libgnutls30t64 \
    libpam-modules libpam-modules-bin libpam-runtime libpam0g \
    libsqlite3-0 libtasn1-6 \
    dpkg dpkg-dev libdpkg-perl \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Copy sglang source and do editable install (LAST for better caching)
# =============================================================================

# Copy local source if building from local
FROM scratch AS local_src
COPY . /src

FROM framework AS framework_final

ARG BRANCH_TYPE
ARG BUILD_TYPE
ARG CUDA_VERSION
ARG SGL_VERSION
ARG USE_LATEST_SGLANG

WORKDIR /sgl-workspace

COPY --from=local_src /src /tmp/local_src
RUN if [ "$BRANCH_TYPE" = "local" ]; then \
        cp -r /tmp/local_src /sgl-workspace/sglang; \
    elif [ "$USE_LATEST_SGLANG" = "1" ]; then \
        git clone --depth=1 https://github.com/sgl-project/sglang.git /sgl-workspace/sglang; \
    elif [ -z "$SGL_VERSION" ]; then \
        echo "ERROR: SGL_VERSION must be set when USE_LATEST_SGLANG=0 and BRANCH_TYPE!=local" && exit 1; \
    else \
        git clone --depth=1 --branch v${SGL_VERSION} https://github.com/sgl-project/sglang.git /sgl-workspace/sglang; \
    fi \
    && rm -rf /tmp/local_src

# Editable install (fast - dependencies already installed via constraints)
# Clean up __pycache__/tests/pyc in same RUN to avoid writing ~28k files to layer
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /sgl-workspace/sglang \
    && python3 -m pip install --no-deps -e "python[${BUILD_TYPE}]" \
    && kernels lock python \
    && ( success=0; \
         # aarch64: kernels-community/sgl-flash-attn3 ships no arm variants; JIT-compile at runtime.
         # Remove this branch once arm cubins are published upstream.
         if [ "$(uname -m)" = "aarch64" ]; then \
             echo "Skipping kernels-community/sgl-flash-attn3 cubin download on aarch64 (no variants published upstream); kernels will be JIT-compiled at runtime"; \
             success=1; \
         else \
             for i in 1 2 3; do \
                 echo "Attempt $i/3: downloading sgl-kernel cubins..." && \
                 kernels download python && \
                 success=1 && break; \
                 echo "sgl-kernel cubin download failed, retrying in 30s..." && sleep 30; \
             done; \
             # x86: if no prebuilt sgl-flash-attn3 variant matches this torch+CUDA \
             # combo, fall back to runtime JIT instead of failing the build, \
             # mirroring the aarch64 branch. \
             if [ "$success" != "1" ]; then \
                 echo "WARNING: no matching sgl-flash-attn3 cubin variant for this torch+CUDA; kernels will be JIT-compiled at runtime"; \
                 success=1; \
             fi; \
         fi; \
         [ "$success" = "1" ] ) \
    && mkdir -p /root/.cache/huggingface /root/.cache/sglang \
    && ( if [ -f python/kernels.lock ]; then mv python/kernels.lock /root/.cache/sglang/; fi ) \
    && ( find /opt/sglang/lib/python3.12/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true )

ENV SGLANG_RUST_BUILD_MODE=never


# Install pre-built gateway artifacts from parallel builder
COPY --from=gateway_builder /build/sgl-model-gateway-bin /opt/sglang/bin/sgl-model-gateway
COPY --from=gateway_builder /build/gateway_wheels /tmp/gateway_wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --force-reinstall /tmp/gateway_wheels/*.whl \
    && rm -rf /tmp/gateway_wheels

# Set workspace directory
WORKDIR /sgl-workspace/sglang

# Keep build provenance at the end so metadata changes do not invalidate build layers.
ARG SGLANG_BUILD_COMMIT=unknown
ARG SGLANG_BUILD_URL=
ARG SGLANG_IMAGE_TAG=local/sglang:dev
ENV SGLANG_BUILD_COMMIT=${SGLANG_BUILD_COMMIT:-unknown} \
    SGLANG_BUILD_URL=${SGLANG_BUILD_URL:-} \
    SGLANG_IMAGE_TAG=${SGLANG_IMAGE_TAG:-local/sglang:dev}
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.revision="${SGLANG_BUILD_COMMIT}" \
      org.opencontainers.image.version="${SGLANG_IMAGE_TAG}" \
      org.opencontainers.image.url="${SGLANG_BUILD_URL}" \
      ai.sglang.build.commit="${SGLANG_BUILD_COMMIT}" \
      ai.sglang.build.url="${SGLANG_BUILD_URL}" \
      ai.sglang.image.tag="${SGLANG_IMAGE_TAG}"

########################################################
########## Runtime Image ##############################
########################################################
#
# PURPOSE: Production runtime environment with JIT support
#
# This stage creates a production-ready image containing:
# - Pre-installed SGLang and CUDA dependencies
# - Full CUDA toolchain for JIT compilation (DeepGEMM, Triton, FlashInfer)
# - Optimized for inference workloads and deployment
# - Smaller than framework (no dev tools like vim, tmux, nsight, etc.)
#
# Use this stage when you need:
# - Production deployment of SGLang
# - JIT compilation support for FP8/microscaling kernels
# - Ready-to-run inference server environment
#
# Note: Uses devel base for complete NVCC toolchain required by DeepGEMM JIT
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS runtime

ARG CUDA_VERSION
ARG TARGETARCH
ARG GDRCOPY_VERSION=2.5.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/ \
    SGLANG_RUST_BUILD_MODE=never

# Add GKE default lib and bin locations + CUDA compiler paths for FlashInfer JIT
ENV PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Install runtime dependencies (devel base provides gcc/g++/build tools)
# Python 3.12 ships in Ubuntu 24.04 main, so no deadsnakes PPA needed.
RUN --mount=type=cache,target=/var/cache/apt,id=runtime-apt \
    apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    # Python runtime
    python3.12-full \
    python3.12-dev \
    wget \
    # Core system utilities
    ca-certificates \
    netcat-openbsd \
    curl \
    ffmpeg \
    git \
    # Runtime libraries
    libopenmpi3 \
    libnuma1 \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    ibverbs-providers \
    libgoogle-glog0v6t64 \
    libunwind8 \
    libboost-system1.83.0 \
    libboost-thread1.83.0 \
    libboost-filesystem1.83.0 \
    libgrpc++1.51t64 \
    libprotobuf32t64 \
    libhiredis1.1.0 \
    libcurl4 \
    libczmq4 \
    libfabric1 \
    libssl-dev \
    # RDMA runtime
    rdma-core \
    infiniband-diags \
    perftest \
    # Build tools for JIT compilation
    ninja-build \
    # NCCL packages needed for pynccl_allocator JIT compilation (-lnccl)
    libnccl2 \
    libnccl-dev \
    # GPG key verification
    gnupg2 \
    linux-libc-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# create virtual env for sglang to avoid conflict with system python packages
RUN python3 -m venv /opt/sglang
ENV PATH="/opt/sglang/bin:${PATH}"

# Set up locale
RUN apt-get update && apt-get install -y --no-install-recommends locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Fix Trivy-reported CVEs (see framework stage for full CVE list)
RUN --mount=type=cache,target=/var/cache/apt,id=runtime-apt \
    apt-get update && apt-get install -y --only-upgrade \
    binutils binutils-common binutils-x86-64-linux-gnu libbinutils \
    libctf0 libctf-nobfd0 libgprofng0 libsframe1 \
    libgnutls30t64 \
    libpam-modules libpam-modules-bin libpam-runtime libpam0g \
    libsqlite3-0 libtasn1-6 \
    dpkg dpkg-dev libdpkg-perl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python site-packages from framework (already cleaned of __pycache__/tests/pyc files)
COPY --from=framework_final /opt/sglang/lib/python3.12/site-packages /opt/sglang/lib/python3.12/site-packages

# Copy SGLang workspace
COPY --from=framework_final /sgl-workspace /sgl-workspace

# Copy sgl-model-gateway binary
COPY --from=framework_final /opt/sglang/bin/sgl-model-gateway /opt/sglang/bin/sgl-model-gateway

# Copy sglang binary
COPY --from=framework_final /opt/sglang/bin/sglang /opt/sglang/bin/sglang

# Copy py-spy binary
COPY --from=framework_final /opt/sglang/bin/py-spy /opt/sglang/bin/py-spy

# Copy cache for kernels from kernels community
COPY --from=framework_final /root/.cache/huggingface /root/.cache/huggingface
COPY --from=framework_final /root/.cache/sglang /root/.cache/sglang

# Copy GDRCopy runtime libraries (but not the build artifacts)
COPY --from=framework_final /usr/lib/libgdrapi.so* /usr/lib/
COPY --from=framework_final /usr/bin/gdrcopy_* /usr/bin/
COPY --from=framework_final /usr/src/gdrdrv-2.5.1 /usr/src/gdrdrv-2.5.1

# Fix DeepEP IBGDA symlink in runtime
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

WORKDIR /sgl-workspace/sglang

# Keep build provenance at the end so metadata changes do not invalidate build layers.
ARG SGLANG_BUILD_COMMIT=unknown
ARG SGLANG_BUILD_URL=
ARG SGLANG_IMAGE_TAG=local/sglang:dev
ENV SGLANG_BUILD_COMMIT=${SGLANG_BUILD_COMMIT:-unknown} \
    SGLANG_BUILD_URL=${SGLANG_BUILD_URL:-} \
    SGLANG_IMAGE_TAG=${SGLANG_IMAGE_TAG:-local/sglang:dev}
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.revision="${SGLANG_BUILD_COMMIT}" \
      org.opencontainers.image.version="${SGLANG_IMAGE_TAG}" \
      org.opencontainers.image.url="${SGLANG_BUILD_URL}" \
      ai.sglang.build.commit="${SGLANG_BUILD_COMMIT}" \
      ai.sglang.build.url="${SGLANG_BUILD_URL}" \
      ai.sglang.image.tag="${SGLANG_IMAGE_TAG}"

# Default command
CMD ["/bin/bash"]

# =============================================================================
# AWS EFA runtime (opt-in target)
# =============================================================================
# Extends `runtime` with the pieces needed to drive AWS EFA from inside the
# container, so no host-side EFA installation is required:
#
#   1. The AWS EFA userspace stack (libfabric plus its EFA provider).
#   2. The EFA build of the Mooncake transfer engine wheel.
#
#   docker build --target runtime-efa -t sglang:efa      -f docker/Dockerfile .
#
# NOTE: this stage is last, so it is what a bare `docker build` with no
# --target produces. Always pass --target explicitly.
#
# Kernel modules and host limits remain the responsibility of the container
# host.
FROM runtime AS runtime-efa

ARG TARGETARCH
ARG MOONCAKE_VERSION
ARG EFA_INSTALLER_VERSION=1.50.0
ARG EFA_INSTALLER_SHA256=fa6dff8593d866866c13cb4640d9059835cd4efa427971f100ab40c97bef2841

# Install the AWS EFA userspace stack.
#
# `set -eux` is required: the body is a `;`-separated sequence run by /bin/sh,
# so without it only the last command decides the RUN exit status and a failed
# install would silently produce an empty layer.
#
# The distro libfabric1 stays installed on purpose -- libopenmpi3t64 depends on
# it, so removing it would break that package. Both libfabric.so.1 copies
# coexist; the ENV below makes the AWS one win deterministically.
RUN set -eux; \
    EFA_INSTALLER_ARCHIVE="/tmp/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"; \
    curl -fsSL --retry 3 --retry-all-errors \
        "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
        -o "${EFA_INSTALLER_ARCHIVE}"; \
    echo "${EFA_INSTALLER_SHA256}  ${EFA_INSTALLER_ARCHIVE}" | sha256sum -c -; \
    tar -xzf "${EFA_INSTALLER_ARCHIVE}" -C /tmp; \
    # efa_installer.sh apt-installs pciutils/environment-modules/tcl, and the
    # runtime stage purges /var/lib/apt/lists, so refresh the lists first.
    apt-get update -y; \
    # --minimal is not usable here: it installs rdma-core but skips libfabric,
    # which is the only piece needed for the EFA data path.
    (cd /tmp/aws-efa-installer && \
        ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify); \
    test -x /opt/amazon/efa/bin/fi_info; \
    /opt/amazon/efa/bin/fi_info --version; \
    # Assert the EFA provider is actually compiled into this libfabric. `-l`
    # lists providers without needing EFA hardware, so it works during a build.
    /opt/amazon/efa/bin/fi_info -l | grep -q '^efa:'; \
    # Assert the AWS libfabric precedes the distro 1.17.0 in the loader cache.
    # Match the first libfabric.so.1 entry, since `ldconfig -p` prints
    # "libfabric.so.1 (libc6,x86-64) => /path" and order decides the winner.
    ldconfig -p | grep -m1 'libfabric\.so\.1' | grep -q '/opt/amazon/efa/lib/'; \
    rm -rf "${EFA_INSTALLER_ARCHIVE}" /tmp/aws-efa-installer \
        /var/lib/apt/lists/*

# Prefer the AWS libfabric over the distro libfabric1 that libopenmpi3t64 pulls
# in, and expose fi_info/fi_pingpong for debugging EFA from inside the image.
ENV PATH="/opt/amazon/efa/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"

# Swap the standard Mooncake wheel for the EFA build. Both wheels install the
# same `mooncake` module, so the standard one must be removed first rather than
# installed over. --no-deps preserves the dependency stack the image was built
# against, matching docs/advanced_features/pd_disaggregation.mdx.
#
# The EFA wheel is published for x86_64 only (manylinux_2_35_x86_64), so on other
# architectures this step is skipped and the standard wheel is kept.
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    if [ "${TARGETARCH}" = "amd64" ]; then \
        python3 -m pip uninstall -y mooncake-transfer-engine-cuda13; \
        python3 -m pip install --no-deps \
            "mooncake-transfer-engine-efa-cuda13==${MOONCAKE_VERSION}"; \
        # Verify by distribution metadata rather than importing the module: the
        # extension links libcuda.so.1, absent from a GPU-less builder.
        python3 -c "import importlib.metadata as m; \
m.version('mooncake-transfer-engine-efa-cuda13')"; \
        ! python3 -c "import importlib.metadata as m; \
m.version('mooncake-transfer-engine-cuda13')" 2>/dev/null; \
    else \
        echo "TARGETARCH=${TARGETARCH}: mooncake-transfer-engine-efa-cuda13 is \
x86_64-only, keeping the standard wheel (Mooncake EFA protocol unavailable)"; \
    fi

# Mooncake protocol, derived from the architecture. amd64 carries the EFA wheel
# so it can use the `efa` protocol; other architectures kept the standard wheel,
# which speaks `rdma` over the AWS rdma-core installed above. Setting `efa` there
# would name a protocol the installed wheel does not implement.
#
# ENV cannot be made conditional and cannot read anything a RUN produced, so the
# value is derived with ARG substitution. `#amd64` strips that exact prefix, so
# ARCH_IF_NOT_AMD64 is empty on amd64 and holds the architecture name everywhere
# else; `:+` maps that to rdma, and `:-` supplies efa for the empty case. The
# ENV line reads as: rdma if not amd64, otherwise efa.
#
# The test is deliberately "not amd64" rather than "is amd64" so it matches the
# `[ "${TARGETARCH}" = "amd64" ]` test used by the Mooncake swap above, keeping a
# third architecture on rdma alongside the standard wheel. Writing it positively
# would require stripping `arm64`, which actually means "not arm64" and would
# wrongly select efa on any future architecture.
#
# Nested expansion (${PROTO_${TARGETARCH}}) is not an option -- BuildKit rejects
# it with "unsupported modifier ($) in substitution" -- and concatenating two
# complementary `:+` expansions silently yields "rdmaefa" on a third
# architecture, so keep the explicit `:+` then `:-` pair.
ARG ARCH_IF_NOT_AMD64=${TARGETARCH#amd64}
ARG PROTOCOL_IF_NOT_AMD64=${ARCH_IF_NOT_AMD64:+rdma}
ENV MOONCAKE_PROTOCOL=${PROTOCOL_IF_NOT_AMD64:-efa}
