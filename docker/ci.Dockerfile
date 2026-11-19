# syntax=docker/dockerfile:1

# Self-contained CUDA CI runner for Blackwell GPUs (B200, B300, GB300): the GitHub
# Actions runner plus the environment scripts/ci/cuda/ci_install_dependency.sh
# builds. Models and compiled kernels live in volumes; see docker/ci-runner.md.

ARG CUDA_VERSION=13.0.3

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG TARGETARCH
ARG RUNNER_VERSION=2.338.2
# Must match the CU_VERSION the rest of the fleet installs, or partner runners
# test a different torch and kernel stack.
ARG CU_VERSION=cu130

# The install script's apt list is skipped once these are present; the rest are
# what the runner, the install script's bootstrap and the artifact steps need.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git jq sudo libicu-dev python3 python3-pip python3-venv zstd \
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

# The source is bind-mounted, not copied: each job installs its own checkout, so
# baking one in would only go stale. The build has no GPU, so the Blackwell set and
# GDRCopy are forced, and a libcuda stub lets verify_imports load deep_ep.
RUN --mount=type=bind,source=.,target=/sgl-src,rw \
    mkdir -p /tmp/cuda-stub \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 \
    && cd /sgl-src \
    && CU_VERSION=${CU_VERSION} IS_BLACKWELL=1 INSTALL_GDRCOPY=1 LD_LIBRARY_PATH=/tmp/cuda-stub \
        bash scripts/ci/cuda/ci_install_dependency.sh \
    && : > /opt/sglang-ci-py3.12/env.sh \
    && mkdir -p /opt/sglang-ci /opt/sglang-ci-models \
    && cp python/sglang/cli/killall.py /opt/sglang-ci/killall.py \
    && rm -rf /tmp/cuda-stub /root/.cache/uv /root/.cache/pip

COPY docker/ci-runner-entrypoint.sh docker/ci-runner-healthcheck.sh \
     docker/ci-runner-job-started.sh docker/ci-runner-job-completed.sh /usr/local/bin/

# Models go in a volume shared by every runner on the host; the rest of
# /root/.cache is a per-runner volume. Triton defaults to ~/.triton, outside both.
ENV CU_VERSION=${CU_VERSION} \
    HF_HOME=/opt/sglang-ci-models \
    TRITON_CACHE_DIR=/root/.cache/triton \
    RUNNER_ALLOW_RUNASROOT=1 \
    ACTIONS_RUNNER_HOOK_JOB_STARTED=/usr/local/bin/ci-runner-job-started.sh \
    ACTIONS_RUNNER_HOOK_JOB_COMPLETED=/usr/local/bin/ci-runner-job-completed.sh

WORKDIR /actions-runner
ENTRYPOINT ["/usr/local/bin/ci-runner-entrypoint.sh"]
