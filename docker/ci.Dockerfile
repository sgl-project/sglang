# syntax=docker/dockerfile:1

# Job container for CUDA CI on externally managed runners: the environment
# scripts/ci/cuda/ci_install_dependency.sh builds, so a job's install step only
# installs the PR's checkout. It is not a runner; the host provides one.

ARG CUDA_VERSION=13.0.3

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# Must match the CU_VERSION the rest of the fleet installs, or these runners test
# a different torch and kernel stack.
ARG CU_VERSION=cu130
# 1 for SM100+ (B200, B300, GB300), 0 for older GPUs; at job time the install
# script detects the GPU, and a mismatch reinstalls the dependency set.
ARG IS_BLACKWELL=1

# A job container's HOME is per job (/github/home), so everything the install
# script would keep under HOME lives in /opt instead. Compiled kernels get their
# own directory, empty in the image, so a runner can mount a persistent volume
# there without hiding anything baked in. Model weights come from the runner's
# HF_HOME / HF_HUB_CACHE.
ENV CU_VERSION=${CU_VERSION} \
    XDG_CACHE_HOME=/opt/sglang-ci/cache \
    SGLANG_CACHE_DIR=/opt/sglang-ci/kernels \
    FLASHINFER_WORKSPACE_BASE=/opt/sglang-ci/kernels \
    TRITON_CACHE_DIR=/opt/sglang-ci/kernels/triton \
    CARGO_HOME=/opt/sglang-ci/cargo \
    RUSTUP_HOME=/opt/sglang-ci/rustup \
    UV_PYTHON_INSTALL_DIR=/opt/sglang-ci/python

# The install script's apt list is skipped once these are present; the rest are
# what its bootstrap and the artifact download steps need.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git jq sudo python3 python3-pip python3-venv zstd \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# The source is bind-mounted, not copied: each job installs its own checkout, so
# baking one in would only go stale. The build has no GPU, so GDRCopy is forced
# and a libcuda stub lets verify_imports load deep_ep.
# The job user owns the environment, and group 0 gets the same rights for a
# runner that picks another UID; done in the same step to avoid a second copy.
RUN --mount=type=bind,source=.,target=/sgl-src,rw \
    mkdir -p /tmp/cuda-stub \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 \
    && cd /sgl-src \
    && IS_BLACKWELL=${IS_BLACKWELL} INSTALL_GDRCOPY=1 LD_LIBRARY_PATH=/tmp/cuda-stub \
        bash scripts/ci/cuda/ci_install_dependency.sh \
    && : > /opt/sglang-ci-py3.12/env.sh \
    && rm -rf /tmp/cuda-stub "${XDG_CACHE_HOME}/uv" /root/.cache "${SGLANG_CACHE_DIR}" \
    && mkdir -p "${SGLANG_CACHE_DIR}" \
    && chown -R 1001:0 /opt/sglang-ci /opt/sglang-ci-py3.12 \
    && chmod -R g=u /opt/sglang-ci /opt/sglang-ci-py3.12 \
    && chmod g+w /opt

# UID 1001 is ARC's default runner UID and owns the runners' model cache. Its
# primary group is 0, not its own, so it also keeps group 0's rights.
RUN useradd --uid 1001 --gid 0 --create-home --shell /bin/bash runner
USER 1001
