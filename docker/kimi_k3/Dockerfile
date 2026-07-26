# Kimi-K3 serving image (aarch64 / sm_90 + sm_100a + sm_103a).
#
# Base ships stock SGLang (editable at /sgl-workspace/sglang), DeepEP source
# (deepseek-ai@d28bd67 at /sgl-workspace/DeepEP), the deep_gemm pip package,
# and the CUDA 13 toolchain (nvcc + /usr/local/cuda/include/cccl).
#
# This image adds the three Kimi-K3-specific pieces that stock lacks:
#   1. the Kimi-K3 SGLang code (this repo), editable-installed
#   2. DeepEP patch + rebuild   [k3-track p0-wideep/scripts/v1]:
#        topk 11->16, SWITCH_HIDDEN += 3584, EP>8 SourceMeta alignment,
#        cross-node timeout headroom, CUDA-13 cccl include; rebuilt for
#        sm_90, sm_100a, and sm_103a
#   3. DeepGEMM mega-MoE SiTU patch [k3-track p0-wideep/scripts/v2]:
#        JIT-header sentinel (activation_clamp==0.03125 -> K3 SiTU); no rebuild
#   4. FlashInfer CuTeDSL MLA DCP patch:
#        apply the seven runtime-file diffs; exclude tests absent from the wheel
#
# Build (on/for aarch64; nvcc cross-compiles the DeepEP cubin, no GPU needed):
#   docker build -f docker/kimi_k3/Dockerfile \

#     --build-arg 'TORCH_CUDA_ARCH_LIST=9.0;10.0a;10.3a' -t kimi-k3 .
#
# FlashInfer MXFP4 MoE runner (trtllm-gen SiTU cubins) — runtime opt-in, no
# image change needed: download the SiTU cubin pool archive (link published
# with the release), unzip it, and run the container with
#   -v /host/path/trtllm_gen_moe_cubin_pool:/opt/trtllm_gen_moe_cubin_pool \
#   -e SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL=/opt/trtllm_gen_moe_cubin_pool
# The runner is auto-selected on SM100/103 when the pool is present; the
# remaining kernel sources JIT-compile from the installed flashinfer wheel on
# first launch and are cached.

FROM lmsysorg/sglang:v0.5.16 AS base

# Current Kimi-K3 source auto-discovers and builds its PyO3 extensions.
ARG RUST_VERSION="1.90.0"
ENV RUSTUP_HOME="/usr/local/rustup" \
    CARGO_HOME="/usr/local/cargo" \
    PATH="/usr/local/cargo/bin:${PATH}"
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --no-modify-path --profile minimal \
        --default-toolchain "${RUST_VERSION}" && \
    cargo --version && \
    rustc --version && \
    rm -rf /var/lib/apt/lists/*

# Build one DeepEP wheel with native cubins for Hopper, B200, and GB300.
ARG TORCH_CUDA_ARCH_LIST="9.0;10.0a;10.3a"

# --- 1. Kimi-K3 SGLang code (replaces the base's stock sglang, editable) ---
COPY . /sgl-workspace/sglang
RUN cd /sgl-workspace/sglang && pip install -e python --no-deps

# --- 2. DeepEP: patch (topk16 / hidden3584 / SourceMeta / cccl) + multi-arch rebuild ---
RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    bash /sgl-workspace/sglang/docker/kimi_k3/apply_deepep_k3_patch.sh

# --- 3. DeepGEMM mega-MoE: SiTU JIT-header patch (runtime-JIT, no rebuild) ---
RUN python3 /sgl-workspace/sglang/docker/kimi_k3/apply_deepgemm_situ_patch.py

# TODO(kimi-k3): Download and install the FlashInfer MXFP4 MoE kernel cubin
# in this image once its distributable artifact is available.

# Reinstall the matching FlashInfer package trio before patching its Python
# sources. A mixed Python/cubin/JIT-cache installation fails at import time.
RUN python3 -m pip install --no-deps --force-reinstall \
      "flashinfer-python==0.6.15.post1" && \
    python3 -m pip install --no-deps --force-reinstall \
      "flashinfer-cubin==0.6.15.post1" \
      --index-url https://flashinfer.ai/whl && \
    python3 -m pip install --no-deps --force-reinstall \
      "flashinfer-jit-cache==0.6.15.post1" \
      --index-url https://flashinfer.ai/whl/cu130 && \
    python3 -c 'from importlib.metadata import version; expected = "0.6.15.post1"; packages = ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"); actual = {package: version(package).split("+", 1)[0] for package in packages}; assert all(value == expected for value in actual.values()), actual'

# --- 4. FlashInfer: CuTeDSL MLA decode-context-parallel runtime patch ---
RUN FLASHINFER_DCP_PATCH=/sgl-workspace/sglang/docker/kimi_k3/flashinfer-perkz-dcp-0.6.15.txt && \
    FLASHINFER_SITE_PACKAGES="$(python3 -c 'from pathlib import Path; import flashinfer; print(Path(flashinfer.__file__).resolve().parent.parent)')" && \
    sed '/^diff --git a\/tests\//,$d' "${FLASHINFER_DCP_PATCH}" | \
      patch --dry-run --batch --forward --strip=1 --directory="${FLASHINFER_SITE_PACKAGES}" && \
    sed '/^diff --git a\/tests\//,$d' "${FLASHINFER_DCP_PATCH}" | \
      patch --batch --forward --strip=1 --directory="${FLASHINFER_SITE_PACKAGES}"

WORKDIR /sgl-workspace/sglang
