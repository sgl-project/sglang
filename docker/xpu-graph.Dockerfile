# Build from the SGLang repository root:
#   docker build --network=host \
#       -t sglang-xpu:local \
#       -f docker/xpu-graph.Dockerfile .

ARG BASE_IMAGE=intel/deep-learning-essentials:2026.1.2-devel-ubuntu24.04@sha256:a0e75ccb976a29cf3afb087487867b9f1f05d037886920ae64dfe63d9acfc698
FROM ${BASE_IMAGE} AS runtime-base

ARG PYTHON_VERSION
ARG PYTORCH_INDEX
ARG TRITON_XPU_INDEX
ARG TRITON_XPU_VERSION
ARG COMPUTE_RUNTIME_VERSION
ARG COMPUTE_RUNTIME_BASE_URL
ARG COMPUTE_RUNTIME_DEB_SUFFIX
ARG IGC_VERSION
ARG GMM_VERSION
ARG LEVEL_ZERO_VERSION
ARG LEVEL_ZERO_DEB_SUFFIX

ENV DEBIAN_FRONTEND=noninteractive

ENV HTTP_PROXY= \
    HTTPS_PROXY= \
    ALL_PROXY= \
    http_proxy= \
    https_proxy= \
    all_proxy= \
    NO_PROXY=* \
    no_proxy=*

ARG PYTHON_VERSION=3.12
ARG PYTORCH_INDEX=https://download.pytorch.org/whl/xpu
ARG TRITON_XPU_INDEX=https://download.pytorch.org/whl/nightly/xpu
ARG TRITON_XPU_VERSION=3.8.0+git1e2d42a0

ARG COMPUTE_RUNTIME_VERSION=26.22.38646.4
ARG COMPUTE_RUNTIME_BASE_URL=https://github.com/intel/compute-runtime/releases/download
ARG COMPUTE_RUNTIME_DEB_SUFFIX=-0
ARG IGC_VERSION=2.36.3+21719
ARG GMM_VERSION=22.10.0
ARG LEVEL_ZERO_VERSION=1.28.6
ARG LEVEL_ZERO_DEB_SUFFIX=+u24.04

WORKDIR /workspace
SHELL ["/bin/bash", "-c"]

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH} \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential cmake curl ffmpeg git \
        libgl1 libsndfile1 libsm6 libxext6 lsb-release ninja-build \
        numactl python${PYTHON_VERSION} python${PYTHON_VERSION}-dev \
        unzip vim wget && \
    apt-get update && \
    rm -rf /var/cache/apt/archives/* && \
    apt-get install -y --no-install-recommends \
        intel-metrics-discovery clinfo intel-gsc \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools \
        libva-glx2 va-driver-all vainfo && \
    : "${COMPUTE_RUNTIME_BASE_URL:?Pass --build-arg COMPUTE_RUNTIME_BASE_URL=<compute-runtime mirror>}" && \
    igc_url="https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION%%+*}" && \
    cr_url="${COMPUTE_RUNTIME_BASE_URL}/${COMPUTE_RUNTIME_VERSION}" && \
    lz_url="https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}" && \
    cd /tmp && \
    curl -fsSL -O "${igc_url}/intel-igc-core-2_${IGC_VERSION}_amd64.deb" && \
    curl -fsSL -O "${igc_url}/intel-igc-opencl-2_${IGC_VERSION}_amd64.deb" && \
    curl -fsSL -O "${cr_url}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}${COMPUTE_RUNTIME_DEB_SUFFIX}_amd64.deb" && \
    curl -fsSL -O "${cr_url}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}${COMPUTE_RUNTIME_DEB_SUFFIX}_amd64.deb" && \
    curl -fsSL -O "${cr_url}/intel-ocloc_${COMPUTE_RUNTIME_VERSION}${COMPUTE_RUNTIME_DEB_SUFFIX}_amd64.deb" && \
    curl -fsSL -O "${cr_url}/libigdgmm12_${GMM_VERSION}_amd64.deb" && \
    curl -fsSL -O "${lz_url}/libze1_${LEVEL_ZERO_VERSION}${LEVEL_ZERO_DEB_SUFFIX}_amd64.deb" && \
    curl -fsSL -O "${lz_url}/libze-dev_${LEVEL_ZERO_VERSION}${LEVEL_ZERO_DEB_SUFFIX}_amd64.deb" && \
    apt-get install -y --allow-downgrades \
        ./intel-igc-core-2_${IGC_VERSION}_amd64.deb \
        ./intel-igc-opencl-2_${IGC_VERSION}_amd64.deb \
        ./libigdgmm12_${GMM_VERSION}_amd64.deb \
        ./libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}${COMPUTE_RUNTIME_DEB_SUFFIX}_amd64.deb \
        ./intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}${COMPUTE_RUNTIME_DEB_SUFFIX}_amd64.deb \
        ./intel-ocloc_${COMPUTE_RUNTIME_VERSION}${COMPUTE_RUNTIME_DEB_SUFFIX}_amd64.deb \
        ./libze1_${LEVEL_ZERO_VERSION}${LEVEL_ZERO_DEB_SUFFIX}_amd64.deb \
        ./libze-dev_${LEVEL_ZERO_VERSION}${LEVEL_ZERO_DEB_SUFFIX}_amd64.deb && \
    test "$(dpkg-query -W -f='${Version}' intel-igc-core-2)" = "${IGC_VERSION%%+*}" && \
    test "$(dpkg-query -W -f='${Version}' intel-igc-opencl-2)" = "${IGC_VERSION%%+*}" && \
    test "$(dpkg-query -W -f='${Version}' libze1)" = "${LEVEL_ZERO_VERSION}" && \
    test "$(dpkg-query -W -f='${Version}' libze-dev)" = "${LEVEL_ZERO_VERSION}" && \
    apt-mark hold libze1 libze-dev libze-intel-gpu1 intel-opencl-icd intel-ocloc libigdgmm12 \
        intel-igc-core-2 intel-igc-opencl-2 && \
    rm -f /tmp/*.deb && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv venv --python ${PYTHON_VERSION} ${VIRTUAL_ENV} && \
    uv pip install --no-cache-dir \
        scikit-build-core setuptools setuptools-scm wheel cmake ninja

FROM runtime-base AS sglang-builder
ARG PYTORCH_INDEX
ARG TRITON_XPU_INDEX
ARG TRITON_XPU_VERSION
COPY . /workspace/sglang
RUN cd /workspace/sglang/python && \
    cp pyproject_xpu.toml pyproject.toml && \
    python -c 'import os, subprocess, tomllib; dependencies = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]; torch_dependencies = [dependency.split(";", 1)[0].strip() for dependency in dependencies if dependency.split(";", 1)[0].strip().lower().startswith(("torch", "torchaudio", "torchcodec", "torchvision"))]; subprocess.check_call(["uv", "pip", "install", "--no-deps", "--index-url", os.environ["PYTORCH_INDEX"], *torch_dependencies])' && \
    export CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}" && \
    env -u UV_EXTRA_INDEX_URL -u PIP_EXTRA_INDEX_URL \
    uv pip install --no-build-isolation -v \
        --index-url https://pypi.org/simple \
        --extra-index-url ${PYTORCH_INDEX} \
        --extra-index-url ${TRITON_XPU_INDEX} \
        . && \
    uv pip install --no-cache-dir --no-deps \
        xgrammar==0.1.33 apache-tvm-ffi && \
    expected_triton="${TRITON_XPU_VERSION}" && \
    current_triton=$(python -c "import importlib.metadata as metadata; print(metadata.version('triton-xpu'))" 2>/dev/null || echo "<missing>") && \
    if [[ "${current_triton}" != "${expected_triton}" ]]; then \
        echo "Switching triton-xpu from ${current_triton} to ${expected_triton}"; \
        env -u UV_EXTRA_INDEX_URL -u PIP_EXTRA_INDEX_URL \
        uv pip install --upgrade --no-deps \
            --index-url "${TRITON_XPU_INDEX}" \
            "triton-xpu==${expected_triton}"; \
    fi

FROM sglang-builder AS final

ARG TRITON_XPU_VERSION

RUN expected_triton="${TRITON_XPU_VERSION}" && \
    python -c "import importlib.metadata as metadata, torch, sglang, sgl_kernel; assert torch.__version__ == '2.13.0+xpu', torch.__version__; print('torch:', torch.__version__); print('triton-xpu:', metadata.version('triton-xpu')); print('sglang:', sglang.__file__); print('sgl_kernel:', sgl_kernel.__file__)" && \
    test "$(python -c "import importlib.metadata as metadata; print(metadata.version('triton-xpu'))")" = "${expected_triton}" && \
    test "$(dpkg-query -W -f='${Version}' intel-igc-core-2)" = "2.36.3" && \
    test "$(dpkg-query -W -f='${Version}' intel-igc-opencl-2)" = "2.36.3" && \
    test "$(dpkg-query -W -f='${Version}' intel-ocloc)" = "26.22.38646.4-0" && \
    test "$(dpkg-query -W -f='${Version}' intel-opencl-icd)" = "26.22.38646.4-0" && \
    test "$(dpkg-query -W -f='${Version}' libigdgmm12)" = "22.10.0" && \
    test "$(dpkg-query -W -f='${Version}' libze-intel-gpu1)" = "26.22.38646.4-0" && \
    test "$(dpkg-query -W -f='${Version}' libze1)" = "1.28.6" && \
    test "$(dpkg-query -W -f='${Version}' libze-dev)" = "1.28.6"

WORKDIR /workspace
CMD ["bash"]