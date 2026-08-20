# docker build -t sglang:xpu -f xpu.Dockerfile --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg no_proxy=${no_proxy} --no-cache .

# Use Intel deep learning essentials base image with Ubuntu 24.04
FROM intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Define build arguments
ARG PYTHON_VERSION=3.12

ARG SG_LANG_REPO=https://github.com/sgl-project/sglang.git
ARG SG_LANG_BRANCH=main

ARG SG_LANG_KERNEL_REPO=https://github.com/sgl-project/sgl-kernel-xpu.git
ARG SG_LANG_KERNEL_BRANCH=main

USER root

# Pin Level-Zero UMD + IGC (rolling PPA once faulted libze on B580; see sgl-kernel-xpu#296).
# Keep in lockstep with the host xe KMD; override via --build-arg.
ARG COMPUTE_RUNTIME_VERSION=26.18.38308.1
ARG IGC_VERSION=2.34.4+21428
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
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /sgl-workspace

RUN pip install --no-cache-dir torch==2.13.0+xpu torchvision==0.28.0+xpu torchaudio==2.11.0+xpu --index-url https://download.pytorch.org/whl/xpu && \
    pip install --no-cache-dir msgspec blake3 py-cpuinfo compressed_tensors gguf partial_json_parser einops tabulate --root-user-action=ignore

RUN echo "Cloning ${SG_LANG_BRANCH} from ${SG_LANG_REPO}" && \
    git clone --branch ${SG_LANG_BRANCH} --single-branch ${SG_LANG_REPO} sglang && \
    git -C sglang fetch --tags --force origin && \
    cd sglang && cd python && \
    cp pyproject_xpu.toml pyproject.toml && \
    pip install --no-cache-dir ".[dev,diffusion]" --extra-index-url https://download.pytorch.org/whl/xpu && \
    pip install --no-cache-dir --no-deps xgrammar==0.1.33

CMD ["bash", "-c", "source /opt/intel/oneapi/setvars.sh --force && exec bash"]
