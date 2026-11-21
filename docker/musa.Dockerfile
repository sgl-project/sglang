# syntax=docker/dockerfile:1.7

# Usage:
#   docker build -f docker/musa.Dockerfile -t sglang:main-musa .
#   docker run --rm -it --network=host \
#     --env MTHREADS_VISIBLE_DEVICES=0 \
#     --env MTHREADS_DRIVER_CAPABILITIES=all \
#     --shm-size=32g \
#     sglang:main-musa
#
# A prebuilt MUSA base image can skip the apt SDK layer with:
#   --build-arg INSTALL_MUSA_STACK=0
# A network-isolated build can use uv-managed CPython instead of deadsnakes:
#   --build-arg PYTHON_INSTALL_METHOD=uv
# A base image with Python 3.12 and development headers can reuse them with:
#   --build-arg PYTHON_INSTALL_METHOD=existing

ARG BASE_IMAGE=ubuntu:22.04
ARG PYTHON_VERSION=3.12
ARG PYTHON_INSTALL_METHOD=apt
ARG UBUNTU_APT_MIRROR=
ARG DEADSNAKES_MIRROR_URL=
ARG DEADSNAKES_GPGKEY_URL=
ARG GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py
ARG PIP_BOOTSTRAP_INDEX_URL=https://pypi.org/simple
ARG UV_INSTALL_URL=https://astral.sh/uv/install.sh
ARG UV_PYTHON_MIRROR=

FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION
ARG PYTHON_INSTALL_METHOD
ARG UBUNTU_APT_MIRROR
ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL
ARG GET_PIP_URL
ARG PIP_BOOTSTRAP_INDEX_URL
ARG UV_INSTALL_URL
ARG UV_PYTHON_MIRROR
ARG MUSA_APT_SOURCE=https://dl.mthreads.com/repo/repository/ubuntu2204/
ARG INSTALL_MUSA_STACK=1
ARG MUSA_PIP_INDEX_URL=https://dl.mthreads.com/repo/api/pypi/pypi/simple
ARG PYPI_INDEX_URL=https://pypi.org/simple

ENV MUSA_HOME=/usr/local/musa \
    MATE_MUSA_ARCH_LIST=3.1 \
    PATH=/usr/local/musa/bin:/usr/local/musa/mudnn/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/musa/mudnn/lib:/usr/local/mtshmem/lib:/usr/local/lib \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
    TRITON_CACHE_DIR=/root/.triton/cache

RUN rm -f /etc/apt/sources.list.d/mthreads-musa.list \
    && if [[ -n "${UBUNTU_APT_MIRROR}" ]]; then \
        sed -Ei \
            "s#https?://[^ ]+/ubuntu/?#${UBUNTU_APT_MIRROR%/}/#g" \
            /etc/apt/sources.list; \
    fi \
    && apt-get update \
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
# Keep both the PPA and uv endpoints configurable for isolated build networks.
RUN case "${PYTHON_INSTALL_METHOD}" in apt|uv|existing) ;; *) \
        echo "PYTHON_INSTALL_METHOD must be apt, uv, or existing (got ${PYTHON_INSTALL_METHOD})" >&2; \
        exit 1 ;; \
    esac \
    && if [[ "${PYTHON_VERSION}" != "3.12" ]]; then \
        echo "This image supports only PYTHON_VERSION=3.12 (got ${PYTHON_VERSION})" >&2; \
        exit 1; \
    elif [[ "${PYTHON_INSTALL_METHOD}" == "existing" ]]; then \
        python_bin="$(command -v python${PYTHON_VERSION} || true)"; \
        [[ -n "${python_bin}" ]] || { \
            echo "PYTHON_INSTALL_METHOD=existing requires python${PYTHON_VERSION}" >&2; \
            exit 1; \
        }; \
        "${python_bin}" -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"; \
        python_config="$(command -v python${PYTHON_VERSION}-config || true)"; \
        [[ -n "${python_config}" ]] || { \
            echo "PYTHON_INSTALL_METHOD=existing requires python${PYTHON_VERSION}-config" >&2; \
            exit 1; \
        }; \
        python_include="$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_path("include"))')"; \
        [[ -f "${python_include}/Python.h" ]] || { \
            echo "PYTHON_INSTALL_METHOD=existing requires ${python_include}/Python.h" >&2; \
            exit 1; \
        }; \
        ln -sfn "${python_bin}" "/usr/local/bin/python${PYTHON_VERSION}"; \
        ln -sfn "${python_bin}" /usr/local/bin/python3; \
        ln -sfn "${python_bin}" /usr/local/bin/python; \
        ln -sfn "${python_config}" "/usr/local/bin/python${PYTHON_VERSION}-config"; \
        ln -sfn "${python_config}" /usr/local/bin/python3-config; \
    elif [[ "${PYTHON_INSTALL_METHOD}" == "uv" ]]; then \
        install -d -m 0755 /opt/uv /opt/python; \
        curl -fsSL --retry 5 --retry-all-errors --connect-timeout 10 \
            "${UV_INSTALL_URL}" \
            | UV_INSTALL_DIR=/opt/uv sh; \
        uv_args=(--install-dir /opt/python --default); \
        if [[ -n "${UV_PYTHON_MIRROR}" ]]; then \
            uv_args+=(--mirror "${UV_PYTHON_MIRROR}"); \
        fi; \
        /opt/uv/uv python install "${PYTHON_VERSION}" "${uv_args[@]}"; \
        python_bin="$(find /opt/python -type f -path "*/bin/python${PYTHON_VERSION}" -print -quit)"; \
        [[ -n "${python_bin}" ]] || { echo "uv did not install Python ${PYTHON_VERSION}" >&2; exit 1; }; \
        python_root="$(dirname "$(dirname "${python_bin}")")"; \
        for python_link in python python3 python${PYTHON_VERSION} python${PYTHON_VERSION}-config python3-config; do \
            if [[ -e "${python_root}/bin/${python_link}" ]]; then \
                ln -sfn "${python_root}/bin/${python_link}" "/usr/local/bin/${python_link}"; \
            fi; \
        done; \
        ln -sfn "${python_root}/include/python${PYTHON_VERSION}" \
            "/usr/include/python${PYTHON_VERSION}"; \
        printf '%s\n' "${python_root}/lib" > /etc/ld.so.conf.d/python-uv.conf; \
        ldconfig; \
    else \
        if [[ -n "${DEADSNAKES_MIRROR_URL}" ]]; then \
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
        fi; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            python${PYTHON_VERSION} python${PYTHON_VERSION}-dev; \
        python_bin="/usr/bin/python${PYTHON_VERSION}"; \
        update-alternatives --install /usr/bin/python3 python3 \
            /usr/bin/python${PYTHON_VERSION} 100; \
        update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}; \
        ln -sfn /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config; \
        rm -f /usr/lib/python${PYTHON_VERSION}/EXTERNALLY-MANAGED; \
    fi \
    && if [[ "${PYTHON_INSTALL_METHOD}" == "existing" ]] \
        && "${python_bin}" -m pip --version >/dev/null 2>&1; then \
        :; \
    else \
        curl -fsSL --retry 5 --retry-all-errors --connect-timeout 10 \
            "${GET_PIP_URL}" -o /tmp/get-pip.py; \
        pip_args=(--index-url "${PIP_BOOTSTRAP_INDEX_URL}"); \
        if [[ "${PYTHON_INSTALL_METHOD}" != "apt" ]]; then \
            pip_args+=(--break-system-packages); \
        fi; \
        "${python_bin}" /tmp/get-pip.py "${pip_args[@]}"; \
        rm -f /tmp/get-pip.py; \
    fi \
    && if [[ "${PYTHON_INSTALL_METHOD}" == "uv" ]]; then \
        for pip_link in pip pip3 pip${PYTHON_VERSION}; do \
            if [[ -e "${python_root}/bin/${pip_link}" ]]; then \
                ln -sfn "${python_root}/bin/${pip_link}" "/usr/local/bin/${pip_link}"; \
            fi; \
        done; \
    fi \
    && export PATH="/usr/local/bin:${PATH}" \
    && ln -sfn /usr/local/bin/pip${PYTHON_VERSION} /usr/local/bin/pip \
    && ln -sfn /usr/local/bin/pip${PYTHON_VERSION} /usr/local/bin/pip3 \
    && python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version" \
    && rm -rf /var/lib/apt/lists/*

# Keep managed Python and pip links ahead of a base image's pre-existing venv.
ENV PATH=/usr/local/bin:${PATH}
# uv-managed Python marks its prefix externally managed; this image intentionally
# owns that prefix and installs its runtime dependencies into it.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN case "${INSTALL_MUSA_STACK}" in 0|1) ;; *) \
        echo "INSTALL_MUSA_STACK must be 0 or 1 (got ${INSTALL_MUSA_STACK})" >&2; \
        exit 1 ;; \
    esac \
    && if [[ "${INSTALL_MUSA_STACK}" == "1" ]]; then \
        echo "deb [trusted=true] ${MUSA_APT_SOURCE} jammy main" \
            > /etc/apt/sources.list.d/mthreads-musa.list; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
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
            musa-toolkit-5-2-config-common; \
    else \
        command -v mcc >/dev/null || { \
            echo "INSTALL_MUSA_STACK=0 requires mcc in the base image" >&2; \
            exit 1; \
        }; \
        [[ -d "${MUSA_HOME}/lib" ]] || { \
            echo "INSTALL_MUSA_STACK=0 requires ${MUSA_HOME}/lib in the base image" >&2; \
            exit 1; \
        }; \
    fi \
    && rm -rf /var/lib/apt/lists/* \
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

RUN python -m pip install --upgrade pip "setuptools<82" wheel \
        --index-url "${PYPI_INDEX_URL}"

WORKDIR /workspace/sglang
COPY . .

# Install MUSA-only wheels before public dependencies. This prevents ambiguous
# public project names and broad transitive torch requirements from selecting
# CUDA packages. Versions remain single-sourced in pyproject_other.toml.
RUN cp python/pyproject_other.toml python/pyproject.toml \
    && python -c 'import pathlib, tomllib; \
project = tomllib.loads(pathlib.Path("python/pyproject.toml").read_text()); \
requirements = project["project"]["optional-dependencies"]["srt_musa"]; \
private = ("apache-tvm-ffi==", "deep_ep==", "deep-gemm==", "flash_attn_3==", \
"flash_mla==", "flashinfer-python==", "mate==", "mate-mubin==", \
"sageattention==", "tilelang_musa==", "torch==", "torch_musa==", \
"torchaudio==", "torchvision==", "torch-c-dlpack-ext==", "triton=="); \
selected = [item for item in requirements if item.startswith(private)]; \
assert len(selected) == len(private), selected; \
torch = [item for item in selected if item.startswith(("torch==", "torch_musa==", "torchvision==", "torchaudio=="))]; \
pathlib.Path("/tmp/musa-private-requirements.txt").write_text("\n".join(selected) + "\n"); \
pathlib.Path("/tmp/musa-torch-requirements.txt").write_text("\n".join(torch) + "\n")'
RUN python -m pip install -r /tmp/musa-private-requirements.txt \
        --no-deps \
        --only-binary=:all: \
        --index-url "${MUSA_PIP_INDEX_URL}" \
        --trusted-host dl.mthreads.com \
    && python -m pip install -e "python[all_musa]" \
        --constraint /tmp/musa-torch-requirements.txt \
        --index-url "${PYPI_INDEX_URL}" \
        --no-build-isolation \
    && python -c "import triton.backends.mtgpu" \
    && ! python -m pip freeze | grep -E '^(nvidia-|cuda-)' \
    && python -m pip cache purge

ARG GITHUB_MIRROR_URL=
ARG MUTLASS_ARCHIVE_URL=
ARG FLASHINFER_ARCHIVE_URL=

RUN if [[ -n "${GITHUB_MIRROR_URL}" ]]; then \
        git config --global url."${GITHUB_MIRROR_URL}".insteadOf \
            https://github.com/; \
    fi \
    && source_root="$(pwd)" \
    && sampling_header="${source_root}/python/sglang/kernels/jit/include/sgl_kernel/speculative/sampling.cuh" \
    && cp "${sampling_header}" /tmp/sampling.cuh \
    && if [[ -n "${MUTLASS_ARCHIVE_URL}" || -n "${FLASHINFER_ARCHIVE_URL}" ]]; then \
        if [[ -z "${MUTLASS_ARCHIVE_URL}" || -z "${FLASHINFER_ARCHIVE_URL}" ]]; then \
            echo "MUTLASS_ARCHIVE_URL and FLASHINFER_ARCHIVE_URL must be set together" >&2; \
            exit 1; \
        fi; \
        third_party=python/sglang/kernels/aot/build/_deps; \
        mkdir -p "${third_party}/mutlass" "${third_party}/flashinfer"; \
        curl -fsSL --retry 5 --retry-all-errors --connect-timeout 10 \
            "${MUTLASS_ARCHIVE_URL}" -o /tmp/mutlass.tar.gz; \
        curl -fsSL --retry 5 --retry-all-errors --connect-timeout 10 \
            "${FLASHINFER_ARCHIVE_URL}" -o /tmp/flashinfer.tar.gz; \
        tar -xzf /tmp/mutlass.tar.gz --strip-components=1 \
            -C "${third_party}/mutlass"; \
        tar -xzf /tmp/flashinfer.tar.gz --strip-components=1 \
            -C "${third_party}/flashinfer"; \
        rm -f /tmp/mutlass.tar.gz /tmp/flashinfer.tar.gz; \
        export SGLANG_MUSA_SKIP_THIRD_PARTY=1; \
    fi \
    && cd python/sglang/kernels/aot \
    && cp pyproject_musa.toml pyproject.toml \
    && MTGPU_TARGET=mp_31 python setup_musa.py install \
    && cp /tmp/sampling.cuh "${sampling_header}" \
    && rm -f /tmp/sampling.cuh \
    && if [[ -n "${GITHUB_MIRROR_URL}" ]]; then \
        git config --global --unset-all \
            url."${GITHUB_MIRROR_URL}".insteadOf; \
    fi \
    && rm -rf build dist ./*.egg-info pyproject.toml

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
assert version('torchada') >= '0.1.112', version('torchada'); \
import triton.backends.mtgpu; \
print('Validated MUSA image stack:', sys.version, version('torch'))"

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

CMD ["/bin/bash"]
