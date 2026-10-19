# docker build -t sglang:xpu -f xpu.Dockerfile --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg no_proxy=${no_proxy} --no-cache .

# Use Intel deep learning essentials base image with Ubuntu 24.04
FROM intel/deep-learning-essentials:2026.1.2-devel-ubuntu24.04

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Define build arguments
ARG PYTHON_VERSION=3.12

ARG SG_LANG_REPO=https://github.com/sgl-project/sglang.git
ARG SG_LANG_BRANCH=main

ARG SG_LANG_KERNEL_REPO=https://github.com/sgl-project/sgl-kernel-xpu.git
# Branch, tag or commit SHA; only used when SG_LANG_KERNEL_SOURCE=source.
ARG SG_LANG_KERNEL_BRANCH=main
# wheel: prebuilt sglang-kernel-xpu pinned in pyproject_xpu.toml; source: build SG_LANG_KERNEL_BRANCH.
ARG SG_LANG_KERNEL_SOURCE=wheel

USER root

# Pin Level-Zero UMD + IGC (rolling PPA once faulted libze on B580; see sgl-kernel-xpu#296).
# Keep in lockstep with the host xe KMD; override via --build-arg.
ARG COMPUTE_RUNTIME_VERSION=26.27.39122.11
ARG IGC_VERSION=2.38.2+22051
ARG GMM_VERSION=22.10.0

RUN apt-get update && apt-get install -y software-properties-common curl && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get update && \
    # Loader + media/metrics from the PPA; the GPU driver is pinned below.
    apt-get install -y \
        libze1 intel-metrics-discovery clinfo intel-gsc \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo \
        libze-dev && \
    cd /tmp && \
    igc_url="https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION%%+*}" && \
    cr_url="https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}" && \
    # IGC first: libze-intel-gpu1 / intel-opencl-icd depend on its exact version.
    curl -fsSL -O "${igc_url}/intel-igc-core-2_${IGC_VERSION}_amd64.deb" && \
    curl -fsSL -O "${igc_url}/intel-igc-opencl-2_${IGC_VERSION}_amd64.deb" && \
    curl -fsSL -O "${cr_url}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb" && \
    curl -fsSL -O "${cr_url}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb" && \
    curl -fsSL -O "${cr_url}/intel-ocloc_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb" && \
    curl -fsSL -O "${cr_url}/libigdgmm12_${GMM_VERSION}_amd64.deb" && \
    apt-get install -y --allow-downgrades \
        ./intel-igc-core-2_${IGC_VERSION}_amd64.deb \
        ./intel-igc-opencl-2_${IGC_VERSION}_amd64.deb \
        ./libigdgmm12_${GMM_VERSION}_amd64.deb \
        ./libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb \
        ./intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb \
        ./intel-ocloc_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
    rm -f /tmp/*.deb && \
    # Hold so later apt upgrades can't pull the rolling PPA version back.
    apt-mark hold libze-intel-gpu1 intel-opencl-icd intel-ocloc libigdgmm12 \
        intel-igc-core-2 intel-igc-opencl-2 && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
RUN curl --proto '=https' --retry 3 --retry-delay 2 --tlsv1.2 -sSf https://sh.rustup.rs \
| sh -s -- -y --no-modify-path --profile minimal && rustc --version && cargo --version
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /sgl-workspace

RUN pip install --no-cache-dir torch==2.14.0+xpu torchvision==0.29.0+xpu torchaudio==2.11.0+xpu --index-url https://download.pytorch.org/whl/xpu && \
    pip install --no-cache-dir msgspec blake3 py-cpuinfo compressed_tensors gguf partial_json_parser einops tabulate --root-user-action=ignore

# SG_LANG_KERNEL_SOURCE=source points the sglang-kernel-xpu dependency at the kernel repo.
RUN echo "Cloning ${SG_LANG_BRANCH} from ${SG_LANG_REPO}" && \
    git clone --branch ${SG_LANG_BRANCH} --single-branch ${SG_LANG_REPO} sglang && \
    git -C sglang fetch --tags --force origin && \
    cd sglang && cd python && \
    if [ "${SG_LANG_KERNEL_SOURCE}" = "source" ]; then \
        echo "Building sgl-kernel-xpu ${SG_LANG_KERNEL_BRANCH} from ${SG_LANG_KERNEL_REPO}" && \
        sed -i -E "s|\"sglang-kernel-xpu @ [^\"]*\"|\"sglang-kernel-xpu @ git+${SG_LANG_KERNEL_REPO}@${SG_LANG_KERNEL_BRANCH}\"|" pyproject_xpu.toml && \
        grep -q "sglang-kernel-xpu @ git+" pyproject_xpu.toml; \
    elif [ "${SG_LANG_KERNEL_SOURCE}" != "wheel" ]; then \
        echo "Invalid SG_LANG_KERNEL_SOURCE=${SG_LANG_KERNEL_SOURCE} (expected wheel or source)" && exit 1; \
    fi && \
    cp pyproject_xpu.toml pyproject.toml && \
    pip install --no-cache-dir ".[dev,diffusion]" --extra-index-url https://download.pytorch.org/whl/xpu && \
    pip install --no-cache-dir --no-deps xgrammar==0.1.33

# Install torch_memory_saver for release/resume_memory_occupation ("memory saver").
# XPU ships no prebuilt wheel: it is built from source against the local oneAPI +
# torch-XPU runtime (the .so links libsycl.so.<N>, which must match the installed
# intel-sycl-rt). TMS_PLATFORM=xpu forces the XPU backend; --no-build-isolation
# lets the build import the installed torch (above) so it can match the libsycl
# major to it -- under build isolation torch is absent and the match is skipped.
# Pinned (v0.0.10b2) so image builds are reproducible; bump via --build-arg.
ARG TORCH_MEMORY_SAVER_REF=a5c99f11b18ebb8e9fda71a68812e476ae49e417
# Base image already applies setvars.sh in its own layers (SETVARS_COMPLETED=1,
# icpx on PATH, LIBRARY_PATH/CPATH populated), so re-sourcing here is redundant.
RUN TMS_PLATFORM=xpu pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/fzyzcjy/torch_memory_saver.git@${TORCH_MEMORY_SAVER_REF}

# Mooncake transfer engine for PD disaggregation, built with icpx (the XPU TUs
# use -fsycl). TENT carries the XPU path: on one node it copies KV card to card
# through Level Zero IPC (kvcache-ai/Mooncake#4379), so no RDMA NIC is needed.
# WITH_STORE=OFF: the store does not build with icpx at this commit.
ARG MOONCAKE_REPO=https://github.com/kvcache-ai/Mooncake.git
ARG MOONCAKE_COMMIT=c67a88f381f69e869045f353539e90c76e738830
RUN git clone ${MOONCAKE_REPO} /tmp/Mooncake && \
    cd /tmp/Mooncake && \
    git checkout ${MOONCAKE_COMMIT} && \
    # Mooncake uses Boost headers only (as the script's SHCA path assumes), and
    # libboost-all-dev pulls OpenMPI, libfabric and a second UCX (about 500 MB).
    # The grep, anchored to the package entry, fails the build if a
    # MOONCAKE_COMMIT bump renames or drops it.
    grep -qE '^ +libboost-all-dev \\$' dependencies.sh && \
    sed -i 's/libboost-all-dev/libboost-dev/' dependencies.sh && \
    bash dependencies.sh -y && \
    cmake -S . -B build -G Ninja \
        -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_BUILD_TYPE=Release -DENABLE_DEBUG_SYMBOLS=OFF \
        -DUSE_XPU=ON -DUSE_TENT=ON \
        -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF && \
    cmake --build build -j "$(nproc)" && \
    cmake --install build && \
    cd / && rm -rf /tmp/Mooncake /usr/local/go && \
    sed -i '\|/usr/local/go/bin|d' /root/.bashrc && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python -c "import mooncake.engine"

# UCX with the Level Zero (ze) and mlx5 transports, then NIXL on top of it, for
# the NIXL PD transfer backend. --with-ze=yes builds ZE copy and ZE IPC.
# A master commit, not a release: ZE IPC for Intel GPUs (openucx/ucx#11218) is in no
# release yet (not v1.22.0, v1.23.0-rc1 or the v1.23.x branch). Move to the first
# release tag that contains it.
ARG UCX_REPO=https://github.com/openucx/ucx.git
ARG UCX_COMMIT=d76cd54b62b64d263e5e588e91871085dc4da7b2
ARG UCX_INSTALL_DIR=/opt/ucx
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf automake libtool pkg-config libibverbs-dev && \
    git clone ${UCX_REPO} /tmp/ucx && \
    cd /tmp/ucx && \
    git checkout ${UCX_COMMIT} && \
    ./autogen.sh && \
    ./configure --prefix=${UCX_INSTALL_DIR} --with-mlx5=yes --with-ze=yes --enable-mt \
        --disable-static --enable-optimizations --disable-doxygen-doc && \
    make -j "$(nproc)" && \
    make -j install-strip && \
    cd / && rm -rf /tmp/ucx && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    test -f ${UCX_INSTALL_DIR}/lib/ucx/libuct_ze.so && \
    test -f ${UCX_INSTALL_DIR}/lib/ucx/libuct_ib_mlx5.so

# NIXL needs Abseil 20240116 or newer (older fails with VLOG not declared), but
# Mooncake's dependencies.sh pulls in Ubuntu's libabsl-dev 20220623. Build a static
# copy and put it first on PKG_CONFIG_PATH so meson resolves every absl module to it.
ARG ABSEIL_VERSION=20250814.1
ARG ABSEIL_PREFIX=/opt/abseil
# Record which NIXL build tools the venv lacks, so only those are removed after the
# NIXL build; ones sglang already depends on (ninja, pyyaml) stay.
RUN for pkg in meson ninja pybind11 meson-python patchelf pyyaml tomlkit; do \
        pip show -q "$pkg" >/dev/null 2>&1 || echo "$pkg"; \
    done > /tmp/nixl-build-only.txt && \
    pip install --no-cache-dir meson ninja pybind11 && \
    curl -fsSL https://github.com/abseil/abseil-cpp/releases/download/${ABSEIL_VERSION}/abseil-cpp-${ABSEIL_VERSION}.tar.gz \
        | tar xz -C /tmp && \
    cmake -S /tmp/abseil-cpp-${ABSEIL_VERSION} -B /tmp/absl_build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${ABSEIL_PREFIX} \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DABSL_PROPAGATE_CXX_STD=ON \
        -DBUILD_TESTING=OFF && \
    ninja -C /tmp/absl_build install && \
    rm -rf /tmp/abseil-cpp-${ABSEIL_VERSION} /tmp/absl_build

# NIXL's documented source install: the nixl-cu12 Python package, the C++ libraries
# and plugins, then the nixl meta package. The Python package keeps a RUNPATH to the
# meson prefix for libnixl_common and friends, so both builds must share NIXL_PREFIX.
# --no-build-isolation reuses the image's XPU torch (pyproject pins a CUDA torch).
ARG NIXL_REPO=https://github.com/ai-dynamo/nixl.git
# Includes ai-dynamo/nixl#1536: the UCX plugin passes the VRAM memory type to UCX at
# registration (auto-selects ze-device on a ZE-only UCX), so XPU memory is not
# misclassified as host and no UCX_MEMTYPE_CACHE override is needed.
ARG NIXL_COMMIT=466ebb1ccc5a6e4e8cbaf954c94020400c9ae876
ARG NIXL_PREFIX=/opt/nixl
RUN export PKG_CONFIG_PATH=${ABSEIL_PREFIX}/lib/pkgconfig:${UCX_INSTALL_DIR}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}} && \
    pip install --no-cache-dir meson-python patchelf pyyaml tomlkit && \
    git clone ${NIXL_REPO} /tmp/nixl && \
    cd /tmp/nixl && \
    git checkout ${NIXL_COMMIT} && \
    pip install --no-cache-dir --no-deps --no-build-isolation . \
        --config-settings=setup-args="-Dprefix=${NIXL_PREFIX}" \
        --config-settings=setup-args="-Ducx_path=${UCX_INSTALL_DIR}" \
        --config-settings=setup-args="-Denable_plugins=UCX,POSIX" \
        --config-settings=setup-args="-Dbuild_examples=false" && \
    meson setup build --prefix=${NIXL_PREFIX} --buildtype=release \
        -Dinstall_headers=true -Ducx_path=${UCX_INSTALL_DIR} \
        -Denable_plugins=UCX,POSIX -Dbuild_examples=false && \
    ninja -C build install && \
    pip install --no-cache-dir --no-deps build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl && \
    xargs -r pip uninstall -y < /tmp/nixl-build-only.txt && \
    cd / && rm -rf /tmp/nixl /tmp/nixl-build-only.txt
RUN python -c "import nixl; nixl.nixl_agent('agent1')"

# The PD tests launch a sglang-router between the prefill and decode servers.
ARG SGLANG_ROUTER_VERSION=0.3.2
RUN pip install --no-cache-dir sglang-router==${SGLANG_ROUTER_VERSION}

CMD ["bash"]
