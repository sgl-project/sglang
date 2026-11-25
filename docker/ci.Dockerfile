# syntax=docker/dockerfile:1

# Job container for CUDA CI on externally managed runners: the environment
# scripts/ci/cuda/ci_install_dependency.sh builds, so a job's install step only
# installs the PR's checkout. It is not a runner; the host provides one.

ARG CUDA_VERSION=13.0.3

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# Must match the CU_VERSION the rest of the fleet installs, or these runners test
# a different torch and kernel stack.
ARG CU_VERSION=cu130
# 1 for SM100+ (B200, B300, GB300); a job on the other family adds its gated
# extras at install time (MSA on Blackwell, lmms_eval otherwise).
ARG IS_BLACKWELL=1

# A job container's HOME changes per job (/github/home), so the script's caches
# and toolchains live under /opt. Kernels compile into an empty directory a
# runner can mount a volume on; model weights come from the runner's HF_HOME.
# SGLang derives FLASHINFER_WORKSPACE_BASE and TRITON_CACHE_DIR from
# SGLANG_CACHE_DIR; they are repeated for processes that never import SGLang.
ENV CU_VERSION=${CU_VERSION} \
    XDG_CACHE_HOME=/opt/sglang-ci/cache \
    SGLANG_CACHE_DIR=/opt/sglang-ci/kernels \
    FLASHINFER_WORKSPACE_BASE=/opt/sglang-ci/kernels \
    TRITON_CACHE_DIR=/opt/sglang-ci/kernels/triton \
    CARGO_HOME=/opt/sglang-ci/cargo \
    RUSTUP_HOME=/opt/sglang-ci/rustup \
    UV_PYTHON_INSTALL_DIR=/opt/sglang-ci/python

# Just what the script needs to start; it installs its own apt list below. No
# sudo: the job user has no sudo rights, and the CI scripts skip a missing sudo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git jq python3 python3-pip python3-venv zstd \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# The checkout is bind-mounted, not baked in: each job installs its own. With no
# GPU in the build, GDRCopy is forced and a libcuda stub lets verify_imports load
# deep_ep; env.sh is emptied because the build wrote that stub into it.
# The venv path and the removed caches belong to the install script, and no job
# reads the removed ones (CI stages download prebuilt Rust extensions).
# Group 0 shares UID 1001's rights for runners that pick another UID, and /opt
# stays group-writable so an interrupted install can rebuild the venv.
RUN --mount=type=bind,source=.,target=/sgl-src,rw \
    mkdir -p /tmp/cuda-stub \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 \
    && cd /sgl-src \
    && IS_BLACKWELL=${IS_BLACKWELL} INSTALL_GDRCOPY=1 LD_LIBRARY_PATH=/tmp/cuda-stub \
        bash scripts/ci/cuda/ci_install_dependency.sh \
    && : > /opt/sglang-ci-py3.12/env.sh \
    && rm -rf /tmp/cuda-stub "${XDG_CACHE_HOME}/uv" /root/.cache "${SGLANG_CACHE_DIR}" \
        "${XDG_CACHE_HOME}/flashinfer-wheels" "${XDG_CACHE_HOME}"/sglang-cargo-target* \
        /opt/gdrcopy /var/lib/apt/lists/* \
    && mkdir -p "${SGLANG_CACHE_DIR}" \
    && chown -R 1001:0 /opt/sglang-ci /opt/sglang-ci-py3.12 \
    && chmod -R g=u /opt/sglang-ci /opt/sglang-ci-py3.12 \
    && chmod g+w /opt

# UID 1001 is ARC's default runner UID and owns the runners' model cache. Its
# primary group is 0, not its own, so it also keeps group 0's rights.
RUN useradd --uid 1001 --gid 0 --create-home --shell /bin/bash runner
USER 1001
