# syntax=docker/dockerfile:1.7

# Usage:
#   docker build -f docker/musa.Dockerfile -t sglang:main-musa .
#   docker run --rm -it --network=host \
#     --env MTHREADS_VISIBLE_DEVICES=0 \
#     --env MTHREADS_DRIVER_CAPABILITIES=all \
#     --shm-size=32g \
#     sglang:main-musa

ARG BASE_IMAGE=ubuntu:22.04

FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG MUSA_APT_SOURCE=https://dl.mthreads.com/repo/repository/ubuntu2204/
ARG MUSA_PIP_INDEX_URL=https://dl.mthreads.com/repo/api/pypi/pypi/simple
ARG PYPI_INDEX_URL=https://pypi.org/simple

ENV MUSA_HOME=/usr/local/musa \
    MATE_MUSA_ARCH_LIST=3.1 \
    PATH=/usr/local/musa/bin:/usr/local/musa/mudnn/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/musa/mudnn/lib:/usr/local/mtshmem/lib:/usr/local/lib \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
    TRITON_CACHE_DIR=/root/.triton/cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        g++ \
        gcc \
        git \
        libmkl-core \
        libmkl-def \
        libmkl-gnu-thread \
        libmkl-intel-lp64 \
        libmkl-vml-def \
        libomp-dev \
        libopenmpi3 \
        libsndfile1 \
        ninja-build \
        pkg-config \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        sox \
    && true

RUN echo "deb [trusted=true] ${MUSA_APT_SOURCE} jammy main" \
        > /etc/apt/sources.list.d/mthreads-musa.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libmthreads-compute \
        libmthreads-mtml \
        libmublas-5-2 \
        libmublaslt-5-2 \
        libmudnn3-dev-musa-5-2 \
        libmudnn3-musa-5-2 \
        libmufft-5-2 \
        libmupp-5-2 \
        libmurand-5-2 \
        libmusolver-5-2 \
        libmusparse-5-2 \
        mccl-s5000 \
        mccl-s5000-dev \
        mtcc-5-2 \
        musa-mualg-5-2 \
        musa-mupti-5-2 \
        musa-musart-5-2 \
        musa-muthrust-5-2 \
        musa-toolkit-5-2 \
        musa-toolkit-5-2-config-common \
    && printf '%s\n' \
        "${MUSA_HOME}/lib" \
        "${MUSA_HOME}/mudnn/lib" \
        "/usr/local/mtshmem/lib" \
        "/usr/lib/x86_64-linux-gnu" \
        > /etc/ld.so.conf.d/musa-runtime.conf \
    && ln -sf /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so.2 \
    && ln -sf /usr/lib/x86_64-linux-gnu/libmkl_core.so /usr/lib/x86_64-linux-gnu/libmkl_core.so.2 \
    && ln -sf /usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so /usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so.2 \
    && ldconfig

RUN python -m pip install --upgrade pip "setuptools<82" wheel

WORKDIR /workspace/sglang
COPY . .

# Pip does not prioritize --index-url over --extra-index-url for equal-version
# candidates. Reinstall Triton from the MUSA index in this same layer.
RUN cp python/pyproject_other.toml python/pyproject.toml \
    && python -m pip install -e "python[all_musa]" \
        --index-url "${MUSA_PIP_INDEX_URL}" \
        --extra-index-url "${PYPI_INDEX_URL}" \
        --trusted-host dl.mthreads.com \
        --no-build-isolation \
    && python -m pip install --no-cache-dir --force-reinstall --no-deps \
        --index-url "${MUSA_PIP_INDEX_URL}" \
        --trusted-host dl.mthreads.com \
        "triton==3.2.0" \
    && python -c "import triton.backends.mtgpu" \
    && ! python -m pip freeze | grep -E '^(nvidia-|cuda-)'

RUN cd python/sglang/kernels/aot \
    && cp pyproject_musa.toml pyproject.toml \
    && MTGPU_TARGET=mp_31 python setup_musa.py install

# Keep this check in a single shell command: the legacy Docker builder does not
# reliably preserve Dockerfile heredocs and can turn the check into a silent
# `python -` EOF success.
RUN python -c \
    "import torch; \
assert getattr(torch.version, 'musa', None), torch.__version__; \
assert hasattr(torch, 'musa'); \
import torchada, triton, tilelang, sglang; \
import triton.backends.mtgpu"

CMD ["/bin/bash"]
