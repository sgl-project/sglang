# Usage (to build SGLang ROCm RDNA docker image for Navi GPUs):
#   docker build --build-arg SGL_BRANCH=v0.5.15.post1 --build-arg GPU_ARCH=gfx1201 -t v0.5.15.post1-rocm721-navi48 -f rocm.rdna.Dockerfile .

# NOTE: Building this Dockerfile requires gfx1201 support in sgl-kernel/setup_rocm.py.
# The target GPU arch must be listed in the supported architectures check.
# Use SGL_BRANCH to point to a branch/commit that includes this support.

# Default base image — ROCm 7.2.1 Navi-optimised image with PyTorch + vLLM
ARG BASE_IMAGE="rocm/vllm-dev:rocm7.2.1_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.16.0"

FROM $BASE_IMAGE

ARG GPU_ARCH=gfx1201
ENV GPU_ARCH_LIST=${GPU_ARCH}

ARG SGL_REPO="https://github.com/sgl-project/sglang.git"
ARG SGL_DEFAULT="main"
ARG SGL_BRANCH=${SGL_DEFAULT}

# Version override for setuptools_scm (used in nightly builds)
ARG SETUPTOOLS_SCM_PRETEND_VERSION=""

ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT=""
ENV AITER_COMMIT_DEFAULT="9127c94a18e4398e1eba91f6639e910f0994ad02"
ENV AITER_COMMIT="${AITER_COMMIT:-${AITER_COMMIT_DEFAULT}}"

# Optional Ubuntu mirror override + apt hardening.
ARG UBUNTU_MIRROR=
USER root

RUN if [ -n "$UBUNTU_MIRROR" ]; then \
        sed -i "s|http://[^[:space:]/]*archive.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list && \
        sed -i "s|http://[^[:space:]/]*security.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list; \
    fi && \
    printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
        > /etc/apt/apt.conf.d/80-net-hardening

# Install some basic utilities
RUN python -m pip install --upgrade pip && pip install setuptools_scm

WORKDIR /sgl-workspace

# -----------------------
# AITER (minimal install — no prebuild kernels for RDNA)
ENV SETUPTOOLS_SCM_PRETEND_VERSION=
ENV AITER_USE_SYSTEM_TRITON=1
RUN pip uninstall -y aiter
RUN git clone ${AITER_REPO} \
    && cd aiter \
    && git checkout -f ${AITER_COMMIT} \
    && git submodule update --init --recursive \
    && pip install -r requirements.txt \
    && echo "[AITER] GPU_ARCH=${GPU_ARCH} (RDNA — minimal install)" \
    && sh -c "GPU_ARCHS=$GPU_ARCH_LIST pip install --config-settings editable_mode=compat -e ." \
    && echo "export PYTHONPATH=/sgl-workspace/aiter:\${PYTHONPATH}" >> /etc/bash.bashrc

# -----------------------
# Build SGLang
ARG BUILD_TYPE=all
ARG SETUPTOOLS_SCM_PRETEND_VERSION

RUN pip uninstall -y sgl_kernel sglang
RUN git clone ${SGL_REPO} \
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
       fi

ENV SGLANG_USE_AITER=0
ENV SGLANG_USE_AITER_AR=0

CMD ["/bin/bash"]
