# syntax=docker/dockerfile:1

# Self-contained CUDA CI runner: the GitHub Actions runner plus the environment
# scripts/ci/cuda/ci_install_dependency.sh builds. Models and compiled kernels live
# in a volume mounted at /root/.cache; usage is in docker/ci-runner-entrypoint.sh.

ARG CUDA_VERSION=13.0.3

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG TARGETARCH
ARG RUNNER_VERSION=2.338.2

# The install script's apt list is skipped once these are present; the rest are
# what the runner and the install script's bootstrap need first.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git jq sudo libicu-dev python3 python3-pip python3-venv \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN case "${TARGETARCH}" in \
        amd64) runner_arch=x64 ;; \
        arm64) runner_arch=arm64 ;; \
        *) echo "unsupported TARGETARCH=${TARGETARCH}" && exit 1 ;; \
    esac \
    && mkdir -p /actions-runner \
    && curl -fsSL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-${runner_arch}-${RUNNER_VERSION}.tar.gz" \
        | tar -xz -C /actions-runner \
    && /actions-runner/bin/installdependencies.sh \
    && rm -rf /var/lib/apt/lists/*

# 1 selects the Blackwell (SM100+) dependency set, matching what the install
# script detects on a B200, B300 or GB300 runner; the build has no GPU to detect.
ARG IS_BLACKWELL=1

# The source is bind-mounted, not copied: each job installs its own checkout, so
# baking one in would only go stale. The libcuda stub lets verify_imports load
# deep_ep without a driver and is gone once this step ends.
RUN --mount=type=bind,source=.,target=/sgl-src,rw \
    mkdir -p /tmp/cuda-stub \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 \
    && cd /sgl-src \
    && IS_BLACKWELL=${IS_BLACKWELL} LD_LIBRARY_PATH=/tmp/cuda-stub \
        bash scripts/ci/cuda/ci_install_dependency.sh \
    && : > /opt/sglang-ci-py3.12/env.sh \
    && rm -rf /tmp/cuda-stub /root/.cache/uv /root/.cache/pip

# Triton defaults to ~/.triton; keep its kernels in the mounted cache with the rest.
ENV TRITON_CACHE_DIR=/root/.cache/triton \
    RUNNER_ALLOW_RUNASROOT=1

COPY docker/ci-runner-entrypoint.sh /usr/local/bin/ci-runner-entrypoint.sh

WORKDIR /actions-runner
ENTRYPOINT ["/usr/local/bin/ci-runner-entrypoint.sh"]
