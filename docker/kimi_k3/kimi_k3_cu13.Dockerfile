# Kimi-K3 serving image (aarch64 / sm_90 + sm_100a + sm_103a).
#
# Base ships stock SGLang (editable at /sgl-workspace/sglang), DeepEP source
# (deepseek-ai@d28bd67 at /sgl-workspace/DeepEP), the deep_gemm pip package,
# and the CUDA 13 toolchain (nvcc + /usr/local/cuda/include/cccl).
#
# This image adds the three Kimi-K3-specific pieces that stock lacks:
#   1. the Kimi-K3 SGLang code (this repo), editable-installed
#   2. DeepEP patch + rebuild:
#        topk 11->16, SWITCH_HIDDEN += 3584, EP>8 SourceMeta alignment,
#        cross-node timeout headroom, CUDA-13 cccl include; rebuilt for
#        sm_90, sm_100a, and sm_103a
#   3. DeepGEMM upgrade to 0.1.5.post2:
#        official MegaMoE runtime-JIT header with Kimi-K3 SiTU support
#
# Build (on/for aarch64; nvcc cross-compiles the DeepEP cubin, no GPU needed):
#   docker build -f docker/kimi_k3/kimi_k3_cu13.Dockerfile \
#     --build-arg 'TORCH_CUDA_ARCH_LIST=9.0;10.0a;10.3a' -t kimi-k3 .
#
# The FlashInfer MXFP4 MoE runner cubins are installed in the image below.
# The runner is auto-selected on SM100/103; the remaining kernel sources
# JIT-compile from the installed FlashInfer wheel on first launch and are
# cached.

FROM lmsysorg/sglang:v0.5.16 AS base

ARG SGL_DEEP_GEMM_VERSION="0.1.5.post2"
ARG NVIMGCODEC_VERSION="0.9.0.20"

# Current Kimi-K3 source auto-discovers and builds its PyO3 extensions.
ARG RUST_VERSION="1.90.0"
ENV RUSTUP_HOME="/usr/local/rustup" \
    CARGO_HOME="/usr/local/cargo" \
    PATH="/usr/local/cargo/bin:${PATH}"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --no-modify-path --profile minimal \
        --default-toolchain "${RUST_VERSION}" && \
    cargo --version && \
    rustc --version && \
    rm -rf /var/lib/apt/lists/*

# Build one DeepEP wheel with native cubins for Hopper, B200, and GB300.
ARG TORCH_CUDA_ARCH_LIST="9.0;10.0a;10.3a"

# --- 1. Kimi-K3 SGLang code (replaces the base's stock sglang, editable) ---
# Keep the installed extension modules, but discard Rust and pip build
# artifacts that are not used at runtime.
RUN rm -rf /sgl-workspace/sglang && \
    git clone --branch main \
      https://github.com/sgl-project/sglang.git /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang && \
    rm -rf .git && \
    test ! -e .git && \
    pip install -e python --no-deps && \
    rm -rf \
      rust/target \
      rust/sglang-grpc/target \
      rust/sglang-mm/target \
      rust/sglang-server/target \
      /usr/local/cargo/registry \
      /root/.cache/pip

# --- 2. DeepEP: patch (topk16 / hidden3584 / SourceMeta / cccl) + multi-arch rebuild ---
RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    bash /sgl-workspace/sglang/docker/kimi_k3/apply_deepep_k3_patch.sh && \
    rm -rf /sgl-workspace/DeepEP/build /sgl-workspace/DeepEP/dist

# --- 3. DeepGEMM: upgrade to the first release with Kimi-K3 SiTU ---
# The v0.5.16 base contains DeepGEMM 0.1.4.post1.
RUN python3 -m pip install --no-deps --force-reinstall \
    "sgl-deep-gemm==${SGL_DEEP_GEMM_VERSION}"

# High-fidelity GPU JPEG decode. The K3 processor enables nvJPEG interpolated
# chroma upsampling through nvImageCodec and zero-copy DLPack handoff to Torch.
RUN python3 -m pip install \
      "nvidia-nvimgcodec-cu13[all]==${NVIMGCODEC_VERSION}" && \
    rm -rf /root/.cache/pip

# Install the matching official FlashInfer package trio. A mixed
# Python/cubin/JIT-cache installation fails at import time.
RUN python3 -m pip uninstall -y \
      flashinfer-python flashinfer-cubin flashinfer-jit-cache && \
    rm -rf /root/.cache/flashinfer /root/.cache/pip && \
    python3 -m pip install --no-deps \
      "flashinfer-python==0.6.17" && \
    python3 -m pip install --no-deps \
      "flashinfer-cubin==0.6.17" \
      --index-url https://flashinfer.ai/whl && \
    python3 -m pip install --no-deps \
      "flashinfer-jit-cache==0.6.17" \
      --index-url https://flashinfer.ai/whl/cu130 && \
    python3 -c 'from importlib.metadata import version; expected = "0.6.17"; packages = ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"); actual = {package: version(package).split("+", 1)[0] for package in packages}; assert all(value == expected for value in actual.values()), actual' && \
    rm -rf /root/.cache/pip

ENV FLASHINFER_VERSION="0.6.17"

WORKDIR /sgl-workspace/sglang
