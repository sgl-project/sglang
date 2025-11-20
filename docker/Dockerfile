ARG CUDA_VERSION=12.9.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04 AS base

ARG TARGETARCH
ARG BUILD_TYPE=all
ARG BRANCH_TYPE=remote
ARG GRACE_BLACKWELL=0

ARG GRACE_BLACKWELL_DEEPEP_BRANCH=gb200_blog_part_2
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG TRITON_LANG_COMMIT=4caa0328bf8df64896dd5f6fb9df41b0eb2e750a
ARG BUILD_AND_DOWNLOAD_PARALLEL=8
ARG SGL_KERNEL_VERSION=0.3.17.post2
ARG SGL_VERSION=0.5.5.post3
ARG USE_LATEST_SGLANG=0
ARG GDRCOPY_VERSION=2.5.1
ARG NVSHMEM_VERSION=3.4.5
ARG PIP_DEFAULT_INDEX
ARG UBUNTU_MIRROR
ARG GITHUB_ARTIFACTORY=github.com
ARG INSTALL_FLASHINFER_JIT_CACHE=0
ARG FLASHINFER_VERSION=0.5.2

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/ \
    NVSHMEM_DIR=/sgl-workspace/nvshmem/install \
    FLASHINFER_VERSION=${FLASHINFER_VERSION}
# Add GKE default lib and bin locations.
ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Replace Ubuntu sources if it is specified
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    sed -i "s|http://.*archive.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list && \
    sed -i "s|http://.*security.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list; \
fi

RUN --mount=type=cache,target=/var/cache/apt apt update && apt install wget -y && apt install software-properties-common -y \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt install python3.12-full python3.12-dev python3.10-venv -y \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
 && update-alternatives --set python3 /usr/bin/python3.12 \
 && wget https://bootstrap.pypa.io/get-pip.py \
 && python3 get-pip.py \
 # Fix for `apt-add-repository`
 && cd /usr/lib/python3/dist-packages/ \
 && ln -s apt_pkg.cpython-310-*-linux-gnu.so apt_pkg.so

# Set timezone and install all packages
RUN --mount=type=cache,target=/var/cache/apt echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
 && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
 && apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    software-properties-common netcat-openbsd kmod unzip openssh-server \
    curl wget lsof zsh ccache tmux htop git-lfs tree \
    build-essential cmake perl \
    libopenmpi-dev libnuma1 libnuma-dev numactl \
    libibverbs-dev libibverbs1 libibumad3 \
    librdmacm1 libnl-3-200 libnl-route-3-200 libnl-route-3-dev libnl-3-dev \
    ibverbs-providers infiniband-diags perftest \
    libgoogle-glog-dev libgtest-dev libjsoncpp-dev libunwind-dev \
    libboost-all-dev libssl-dev \
    libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev libcurl4-openssl-dev \
    libczmq4 libczmq-dev \
    libfabric-dev \
    patchelf \
    nvidia-dkms-550 \
    devscripts debhelper fakeroot dkms check libsubunit0 libsubunit-dev \
 && ln -sf /usr/bin/python3.12 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# Replace pip global cache if it is specified
RUN if [ -n "${PIP_DEFAULT_INDEX}" ]; then \
    python3 -m pip config set global.index-url ${PIP_DEFAULT_INDEX}; \
fi

# GDRCopy installation
RUN mkdir -p /tmp/gdrcopy && cd /tmp \
 && wget -q https://${GITHUB_ARTIFACTORY}/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz \
 && tar -xzf v${GDRCOPY_VERSION}.tar.gz && rm v${GDRCOPY_VERSION}.tar.gz \
 && cd gdrcopy-${GDRCOPY_VERSION}/packages \
 && CUDA=/usr/local/cuda ./build-deb-packages.sh \
 && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
 && cd / && rm -rf /tmp/gdrcopy

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

FROM scratch AS local_src
COPY . /src

FROM base AS build-image
# Install SGLang
# Until torch 2.9 and cu13 are stable we manually update torch if you are on CUDA 13
WORKDIR /sgl-workspace
ARG BRANCH_TYPE
COPY --from=local_src /src /tmp/local_src
RUN if [ "$BRANCH_TYPE" = "local" ]; then \
        cp -r /tmp/local_src /sgl-workspace/sglang; \
    elif [ "$USE_LATEST_SGLANG" = "1" ]; then \
        git clone --depth=1 https://github.com/sgl-project/sglang.git /sgl-workspace/sglang; \
    else \
        git clone --depth=1 --branch v${SGL_VERSION} https://github.com/sgl-project/sglang.git /sgl-workspace/sglang; \
    fi \
 && rm -rf /tmp/local_src
RUN --mount=type=cache,target=/root/.cache/pip  python3 -m pip install --upgrade pip setuptools wheel html5lib six \
 && cd sglang \
 && case "$CUDA_VERSION" in \
      12.6.1) CUINDEX=126 ;; \
      12.8.1) CUINDEX=128 ;; \
      12.9.1) CUINDEX=129 ;; \
      13.0.1) CUINDEX=130 ;; \
      *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
 && if [ "$CUDA_VERSION" = "12.6.1" ]; then \
      python3 -m pip install https://${GITHUB_ARTIFACTORY}/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu124-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps \
   ; \
   elif [ "$CUDA_VERSION" = "12.8.1" ] || [ "$CUDA_VERSION" = "12.9.1" ]; then \
      python3 -m pip install sgl-kernel==${SGL_KERNEL_VERSION} \
   ; \
   elif [ "$CUDA_VERSION" = "13.0.1" ]; then \
      python3 -m pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps \
   ; \
   else \
      echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 \
   ; \
   fi \
 && python3 -m pip install -e "python[${BUILD_TYPE}]" --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX} \
 && if [ "$INSTALL_FLASHINFER_JIT_CACHE" = "1" ]; then \
      python3 -m pip install flashinfer-jit-cache==${FLASHINFER_VERSION} --index-url https://flashinfer.ai/whl/cu${CUINDEX} ; \
   fi \
 && if [ "${CUDA_VERSION%%.*}" = "12" ]; then \
      python3 -m pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps ; \
    elif [ "${CUDA_VERSION%%.*}" = "13" ]; then \
      python3 -m pip install nvidia-nccl-cu13==2.28.3 --force-reinstall --no-deps ; \
      python3 -m pip uninstall -y torch torchaudio torchvision ; \
      python3 -m pip install torch==2.9.0 torchaudio==2.9.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX} ; \
    else \
      echo "No NCCL mapping for CUDA_VERSION=${CUDA_VERSION}" && exit 1 ; \
    fi \
 && FLASHINFER_CUBIN_DOWNLOAD_THREADS=${BUILD_AND_DOWNLOAD_PARALLEL} FLASHINFER_LOGGING_LEVEL=warning python3 -m flashinfer --download-cubin

# Download NVSHMEM source files
# We use Tom's DeepEP fork for GB200 for now; the 1fd57b0276311d035d16176bb0076426166e52f3 commit is https://github.com/fzyzcjy/DeepEP/tree/gb200_blog_part_2
RUN set -eux; \
    if [ "${CUDA_VERSION%%.*}" = "13" ]; then \
      wget -q https://${GITHUB_ARTIFACTORY}/NVIDIA/nvshmem/releases/download/v${NVSHMEM_VERSION}-0/nvshmem_src_cuda-all-all-${NVSHMEM_VERSION}.tar.gz; \
      NVSHMEM_TARBALL="nvshmem_src_cuda-all-all-${NVSHMEM_VERSION}.tar.gz"; \
    else \
      wget -q https://developer.download.nvidia.com/compute/redist/nvshmem/${NVSHMEM_VERSION}/source/nvshmem_src_cuda12-all-all-${NVSHMEM_VERSION}.tar.gz; \
      NVSHMEM_TARBALL="nvshmem_src_cuda12-all-all-${NVSHMEM_VERSION}.tar.gz"; \
    fi && \
    if [ "$GRACE_BLACKWELL" = "1" ]; then \
      git clone https://github.com/fzyzcjy/DeepEP.git && \
      cd DeepEP && \
      git checkout ${GRACE_BLACKWELL_DEEPEP_BRANCH} && \
      sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh && \
      cd .. ; \
    else \
      wget -q https://${GITHUB_ARTIFACTORY}/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip && \
      unzip ${DEEPEP_COMMIT}.zip && rm ${DEEPEP_COMMIT}.zip && mv DeepEP-${DEEPEP_COMMIT} DeepEP && cd DeepEP && \
      sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh && \
      cd .. ; \
    fi && \
    tar -xf "${NVSHMEM_TARBALL}" && \
    mv nvshmem_src nvshmem && \
    rm -f "/sgl-workspace/${NVSHMEM_TARBALL}"

# Build and install NVSHMEM
RUN cd /sgl-workspace/nvshmem && \
    if [ "$GRACE_BLACKWELL" = "1" ]; then CUDA_ARCH="90;100;103;120"; else CUDA_ARCH="90"; fi && \
    NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=${NVSHMEM_DIR} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} && \
    cmake --build build --target install -j${BUILD_AND_DOWNLOAD_PARALLEL}

# Install DeepEP
# CTK13 requires the cccl include
RUN --mount=type=cache,target=/root/.cache/pip cd /sgl-workspace/DeepEP && \
    case "$CUDA_VERSION" in \
      12.6.1) \
        CHOSEN_TORCH_CUDA_ARCH_LIST='9.0' \
        ;; \
      12.8.1|12.9.1|13.0.1) \
        CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3' \
        ;; \
      *) \
        echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 \
        ;; \
    esac && \
    if [ "${CUDA_VERSION%%.*}" = "13" ]; then \
      sed -i "/^    include_dirs = \['csrc\/'\]/a\    include_dirs.append('${CUDA_HOME}/include/cccl')" setup.py; \
    fi && \
    NVSHMEM_DIR=${NVSHMEM_DIR} TORCH_CUDA_ARCH_LIST="${CHOSEN_TORCH_CUDA_ARCH_LIST}" MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL} pip install --no-build-isolation .

# In order to use flashinfer_cutedsl without IMA for WideEP configs we must install
# latest flashinfer_cutedsl. Once 0.4.3 is officially released, remove this
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install --upgrade --pre "nvidia-cutlass-dsl==4.3.0.dev0" --extra-index-url https://pypi.org/simple/

# For cuda 13, we install triton from source to fix some sm103 issues
# This can be reverted after >3.4.5 is released
# See the conversation in: https://github.com/triton-lang/triton/pull/8536
RUN --mount=type=cache,target=/root/.cache/pip if [ "$CUDA_VERSION" = "13.0.1" ]; then \
    wget -q https://${GITHUB_ARTIFACTORY}/triton-lang/triton/archive/${TRITON_LANG_COMMIT}.zip && \
    unzip -q ${TRITON_LANG_COMMIT}.zip && rm ${TRITON_LANG_COMMIT}.zip && mv triton-${TRITON_LANG_COMMIT} triton && \
    cd triton && pip install --break-system-packages -r python/requirements.txt && \
    MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL} pip install --break-system-packages -e .; \
fi

# Python tools
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install \
    datamodel_code_generator \
    mooncake-transfer-engine==0.3.7.post2 \
    pre-commit \
    pytest \
    black \
    isort \
    icdiff \
    uv \
    wheel \
    scikit-build-core \
    nixl \
    py-spy

# Install development tools and utilities
RUN --mount=type=cache,target=/var/cache/apt apt-get update && apt-get install -y \
    gdb \
    ninja-build \
    vim \
    tmux \
    htop \
    wget \
    curl \
    locales \
    lsof \
    git \
    git-lfs \
    zsh \
    tree \
    silversearcher-ag \
    cloc \
    unzip \
    pkg-config \
    libssl-dev \
    bear \
    ccache \
    less \
    && apt install -y rdma-core infiniband-diags openssh-server perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN --mount=type=cache,target=/var/cache/apt apt update -y \
    && apt install -y --no-install-recommends gnupg \
    && echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2004/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "x86_64"; fi)/7fa2af80.pub \
    && apt update -y \
    && apt install nsight-systems-cli -y

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install minimal Python packages
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install --break-system-packages \
    pytest \
    black \
    isort \
    icdiff \
    scikit-build-core \
    uv \
    pre-commit \
    pandas \
    matplotlib \
    tabulate

# Install diff-so-fancy
RUN curl -LSso /usr/local/bin/diff-so-fancy https://${GITHUB_ARTIFACTORY}/so-fancy/diff-so-fancy/releases/download/v1.4.4/diff-so-fancy \
    && chmod +x /usr/local/bin/diff-so-fancy

# Install clang-format
RUN curl -LSso /usr/local/bin/clang-format https://${GITHUB_ARTIFACTORY}/muttleyxd/clang-tools-static-binaries/releases/download/master-32d3ac78/clang-format-16_linux-amd64 \
    && chmod +x /usr/local/bin/clang-format

# Install clangd
RUN curl -L https://${GITHUB_ARTIFACTORY}/clangd/clangd/releases/download/18.1.3/clangd-linux-18.1.3.zip -o clangd.zip \
    && unzip clangd.zip \
    && cp -r clangd_18.1.3/bin/* /usr/local/bin/ \
    && cp -r clangd_18.1.3/lib/* /usr/local/lib/ \
    && rm -rf clangd_18.1.3 clangd.zip

# Install CMake
RUN CMAKE_VERSION=3.31.1 \
    && ARCH=$(uname -m) \
    && CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}" \
    && wget -q "https://${GITHUB_ARTIFACTORY}/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}.tar.gz" \
    && tar -xzf "${CMAKE_INSTALLER}.tar.gz" \
    && cp -r "${CMAKE_INSTALLER}/bin/"* /usr/local/bin/ \
    && cp -r "${CMAKE_INSTALLER}/share/"* /usr/local/share/ \
    && rm -rf "${CMAKE_INSTALLER}" "${CMAKE_INSTALLER}.tar.gz"

# Build and install sgl-router (Rust toolchain removed after build to save space)
RUN --mount=type=cache,target=/root/.cache/pip curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && rustc --version && cargo --version \
    && python3 -m pip install maturin \
    && cd /sgl-workspace/sglang/sgl-router/bindings/python \
    && ulimit -n 65536 && maturin build --release --features vendored-openssl --out dist \
    && python3 -m pip install --force-reinstall dist/*.whl \
    && rm -rf /root/.cargo /root/.rustup target dist ~/.cargo \
    && sed -i '/\.cargo\/env/d' /root/.profile /root/.bashrc 2>/dev/null || true


# Add yank script
COPY --chown=root:root --chmod=755 docker/configs/yank /usr/local/bin/yank

# Install oh-my-zsh and plugins
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Configure Vim and tmux
COPY docker/configs/.vimrc /root/.vimrc
COPY docker/configs/.tmux.conf /root/.tmux.conf

# Configure Git
COPY docker/configs/.gitconfig /tmp/.gitconfig
RUN cat /tmp/.gitconfig >> /root/.gitconfig && rm /tmp/.gitconfig

# Configure zsh
COPY docker/configs/.zshrc /root/.zshrc

RUN set -euxo ; \
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | \
    sed "s|https://github.com|https://${GITHUB_ARTIFACTORY}|g" | \
    bash -s -- --tag 1.42.4 --to /usr/local/bin

# Set workspace directory
WORKDIR /sgl-workspace/sglang
