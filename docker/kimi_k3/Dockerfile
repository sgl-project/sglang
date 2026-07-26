# Kimi-K3 serving image (GB300 / aarch64 / sm_103a).
#
# Base ships stock SGLang (editable at /sgl-workspace/sglang), DeepEP source
# (deepseek-ai@d28bd67 at /sgl-workspace/DeepEP), the deep_gemm pip package,
# and the CUDA 13 toolchain (nvcc + /usr/local/cuda/include/cccl).
#
# This image adds the three Kimi-K3-specific pieces that stock lacks:
#   1. the Kimi-K3 SGLang code (this repo), editable-installed
#   2. DeepEP patch + rebuild   [k3-track p0-wideep/scripts/v1]:
#        topk 11->16, SWITCH_HIDDEN += 3584, EP>8 SourceMeta alignment,
#        cross-node timeout headroom, CUDA-13 cccl include; rebuilt for sm_103a
#   3. DeepGEMM mega-MoE SiTU patch [k3-track p0-wideep/scripts/v2]:
#        JIT-header sentinel (activation_clamp==0.03125 -> K3 SiTU); no rebuild
#
# Build (on/for aarch64; nvcc cross-compiles the DeepEP cubin, no GPU needed):
#   docker build -f docker/kimi_k3/Dockerfile \
#     --build-arg TORCH_CUDA_ARCH_LIST=10.3a -t kimi-k3 .
#
# FlashInfer MXFP4 MoE runner (trtllm-gen SiTU cubins) — runtime opt-in, no
# image change needed: download the SiTU cubin pool archive (link published
# with the release), unzip it, and run the container with
#   -v /host/path/trtllm_gen_moe_cubin_pool:/opt/trtllm_gen_moe_cubin_pool \
#   -e SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL=/opt/trtllm_gen_moe_cubin_pool
# The runner is auto-selected on SM100/103 when the pool is present; the
# remaining kernel sources JIT-compile from the installed flashinfer wheel on
# first launch and are cached.
FROM lmsysorg/sglang:v0.5.15.post1 AS base

# GB300 = sm_103a. Override for other Blackwell (e.g. 10.0a for B200).
ARG TORCH_CUDA_ARCH_LIST=10.3a

# --- 1. Kimi-K3 SGLang code (replaces the base's stock sglang, editable) ---
COPY . /sgl-workspace/sglang
RUN cd /sgl-workspace/sglang && pip install -e python --no-deps

# --- 2. DeepEP: patch (topk16 / hidden3584 / SourceMeta / cccl) + rebuild for sm_103a ---
RUN TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    bash /sgl-workspace/sglang/docker/kimi_k3/apply_deepep_k3_patch.sh

# --- 3. DeepGEMM mega-MoE: SiTU JIT-header patch (runtime-JIT, no rebuild) ---
RUN python3 /sgl-workspace/sglang/docker/kimi_k3/apply_deepgemm_situ_patch.py

WORKDIR /sgl-workspace/sglang
