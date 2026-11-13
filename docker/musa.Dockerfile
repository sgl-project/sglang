# syntax=docker/dockerfile:1.7

# Usage:
#   docker build -f docker/musa.Dockerfile -t sglang:main-musa .
#   docker run --rm -it --network=host \
#     --env MTHREADS_VISIBLE_DEVICES=0 \
#     --env MTHREADS_DRIVER_CAPABILITIES=all \
#     --shm-size=32g \
#     sglang:main-musa

ARG BASE_IMAGE=ubuntu:22.04
ARG PYTHON_VERSION=3.12
ARG DEADSNAKES_MIRROR_URL=
ARG DEADSNAKES_GPGKEY_URL=
ARG GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py
ARG PIP_BOOTSTRAP_INDEX_URL=https://pypi.org/simple

FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION
ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL
ARG GET_PIP_URL
ARG PIP_BOOTSTRAP_INDEX_URL
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
        gnupg \
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
        software-properties-common \
        sox \
    && rm -rf /var/lib/apt/lists/*

# MUSA 5.2 remains on jammy, whose default archive does not provide Python 3.12.
# Keep the PPA and bootstrap endpoints configurable for isolated build networks.
RUN if [[ "${PYTHON_VERSION}" != "3.12" ]]; then \
        echo "This image supports only PYTHON_VERSION=3.12 (got ${PYTHON_VERSION})" >&2; \
        exit 1; \
    fi \
    && if [[ -n "${DEADSNAKES_MIRROR_URL}" ]]; then \
        if [[ -z "${DEADSNAKES_GPGKEY_URL}" ]]; then \
            echo "DEADSNAKES_GPGKEY_URL is required with DEADSNAKES_MIRROR_URL" >&2; \
            exit 1; \
        fi; \
        install -d -m 0755 /etc/apt/keyrings; \
        curl -fsSL "${DEADSNAKES_GPGKEY_URL}" \
            | gpg --dearmor -o /etc/apt/keyrings/deadsnakes.gpg; \
        chmod 0644 /etc/apt/keyrings/deadsnakes.gpg; \
        printf 'deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] %s jammy main\n' \
            "${DEADSNAKES_MIRROR_URL}" \
            > /etc/apt/sources.list.d/deadsnakes.list; \
    else \
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
            add-apt-repository -y ppa:deadsnakes/ppa; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
    && update-alternatives --install /usr/bin/python3 python3 \
        /usr/bin/python${PYTHON_VERSION} 100 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sfn /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && rm -f /usr/lib/python${PYTHON_VERSION}/EXTERNALLY-MANAGED \
    && curl -fsSL --retry 5 --retry-all-errors --connect-timeout 10 \
        "${GET_PIP_URL}" -o /tmp/get-pip.py \
    && /usr/bin/python${PYTHON_VERSION} /tmp/get-pip.py \
        --index-url "${PIP_BOOTSTRAP_INDEX_URL}" \
    && rm -f /tmp/get-pip.py \
    && ln -sfn /usr/local/bin/pip${PYTHON_VERSION} /usr/local/bin/pip \
    && ln -sfn /usr/local/bin/pip${PYTHON_VERSION} /usr/local/bin/pip3 \
    && python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version" \
    && rm -rf /var/lib/apt/lists/*

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

ARG GITHUB_MIRROR_URL=

# torchada 0.1.109 ports project-owned CUDA headers in place and intentionally
# rejects symlinks. Materialize SGLang's shared GGUF headers inside the image so
# neither their links nor the AOT targets outside the JIT tree are overwritten.
RUN jit_gguf_dir=python/sglang/kernels/jit/include/sgl_kernel/quantization/gguf \
    && for shared_header in "${jit_gguf_dir}"/*; do \
        if [[ -L "${shared_header}" ]]; then \
            materialized_header="$(mktemp)"; \
            cp --dereference "${shared_header}" "${materialized_header}"; \
            rm "${shared_header}"; \
            mv "${materialized_header}" "${shared_header}"; \
        fi; \
    done \
    && if [[ -n "${GITHUB_MIRROR_URL}" ]]; then \
        git config --global url."${GITHUB_MIRROR_URL}".insteadOf \
            https://github.com/; \
    fi \
    && cd python/sglang/kernels/aot \
    && cp pyproject_musa.toml pyproject.toml \
    && MTGPU_TARGET=mp_31 python setup_musa.py install \
    && if [[ -n "${GITHUB_MIRROR_URL}" ]]; then \
        git config --global --unset-all \
            url."${GITHUB_MIRROR_URL}".insteadOf; \
    fi

# Keep this check in a single shell command: the legacy Docker builder does not
# reliably preserve Dockerfile heredocs and can turn the check into a silent
# `python -` EOF success.
RUN python -c \
    "import sys; from importlib.metadata import version; import torchada, torch; \
assert sys.version_info[:2] == (3, 12), sys.version; \
assert version('torch') == '2.13.0.post1+musa5.2.0', version('torch'); \
assert getattr(torch.version, 'musa', None), torch.__version__; \
assert hasattr(torch, 'musa'); \
import torch_musa, torchada, torchvision, torchaudio, triton, tilelang, sglang; \
assert version('torch-musa') == '2.13.0.post1+musa5.2.0', version('torch-musa'); \
assert version('torchvision') == '0.28.0.post1+musa5.2.0', version('torchvision'); \
assert version('torchaudio') == '2.11.0.post2+musa5.2.0', version('torchaudio'); \
assert version('torchada') == '0.1.109', version('torchada'); \
import triton.backends.mtgpu; \
print('Validated MUSA image stack:', sys.version, version('torch'))"

CMD ["/bin/bash"]
