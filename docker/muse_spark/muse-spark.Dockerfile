# syntax=docker/dockerfile:1@sha256:ecfaec9ed6d810b56388c508f4121597bfbba70d41a6dfeee4d8cad5f295fc32
# Muse Spark serving image (CUDA targets: 9.0, 10.0a, 10.3a).
# Build from the repository root; application source is cloned from upstream:
#   docker buildx build --platform linux/amd64 --load \
#     -f docker/muse_spark/muse-spark.Dockerfile -t muse-spark:local-test .
FROM lmsysorg/sglang:v0.5.21@sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75 AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Override to narrow the CUDA compilation targets.
ARG TORCH_CUDA_ARCH_LIST="9.0;10.0a;10.3a"
ARG MAX_JOBS=16

# Replace the stock SGLang with the upstream muse-spark branch.
# This branch must include the Muse Spark Docker patch helpers used below.
# Align the runtime with Muse Spark's PyTorch pin before compiling its extensions.
# The base has Torch 2.13 and matching older TorchVision/DeepEP binaries.
WORKDIR /sgl-workspace
RUN git clone --depth 1 --single-branch --branch muse-spark \
    https://github.com/sgl-project/sglang.git /sgl-workspace/sglang-muse-spark
RUN python3 -m pip uninstall -y sglang \
    && python3 -m pip install --no-cache-dir \
        torch==2.14.1 torchvision==0.29.1 sgl-deep-ep==0.1.2.post1 \
    && python3 -m pip install --no-cache-dir --no-deps -e sglang-muse-spark/python

# Match python/pyproject.toml; --no-deps leaves the base's older kernel installed.
RUN python3 -m pip install --no-cache-dir --no-deps sglang-kernel==0.4.9

# FlashInfer 0.7 contains the per-token NVFP4 scale-layout/cubin fix
# (flashinfer-ai/flashinfer#4741). Use matching stock Python, cubin and
# architecture-specific JIT-cache providers; no Muse native replacement.
# The v0.5.21 SGLang base still ships FlashInfer 0.6.18, so upgrading
# the base alone cannot replace this explicit dependency upgrade.
ARG FLASHINFER_VERSION=0.7.0.post1
RUN python3 -m pip install --no-cache-dir \
        "flashinfer-python[cu13]==${FLASHINFER_VERSION}" \
        "nvidia-cutlass-dsl[cu13]==4.8.0" quack-kernels==0.6.5 \
        sentencepiece==0.2.1 tokenizers==0.23.2 transformers==5.19.0 \
    && python3 -m pip install --no-cache-dir \
        "flashinfer-cubin==${FLASHINFER_VERSION}" --index-url https://flashinfer.ai/whl \
    && python3 -m pip install --no-cache-dir \
        "flashinfer-jit-cache==${FLASHINFER_VERSION}" --index-url https://flashinfer.ai/whl/cu130 \
    && python3 -c "from importlib.metadata import version; assert version('flashinfer-python') == '${FLASHINFER_VERSION}'; assert version('flashinfer-cubin') == '${FLASHINFER_VERSION}'; assert version('flashinfer-jit-cache').split('+')[0] == '${FLASHINFER_VERSION}'"

# DeepJIT in DeepGEMM requires elfutils/libdwfl.h at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libdw-dev \
    && rm -rf /var/lib/apt/lists/*

# DeepGEMM dev includes PR #92: TRT-LLM weights and scales are shared upstream.
# Retain Muse post-expert norm and tuning on top of that API; the wheel version
# 0.2.1.post1+muse.cbe1727 satisfies the SGLang pin and identifies this source.
# Use the patch and checksum helper from the same muse-spark checkout as SGLang.
# Pin the source revision required by the helper; rebuild the extension and JIT headers.
RUN git clone https://github.com/sgl-project/DeepGEMM.git /tmp/muse-deep-gemm \
    && cd /tmp/muse-deep-gemm \
    && git checkout --detach cbe1727eacd9ad704795584ce1421dd1f5a8d67e \
    && bash /sgl-workspace/sglang-muse-spark/docker/muse_spark/apply_deep_gemm_patches.sh /tmp/muse-deep-gemm \
    && TVM_FFI_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST//;/ }" MAX_JOBS="${MAX_JOBS}" \
        bash build_sgl_deep_gemm.sh \
    && python3 -m pip install --no-cache-dir --no-deps --force-reinstall dist/sgl_deep_gemm-*.whl \
    && cd /sgl-workspace \
    && python3 -c "from importlib.metadata import version; from sglang.srt.layers.moe.muse_spark_megamoe import require_muse_megamoe_patch; assert version('sgl-deep-gemm') == '0.2.1.post1+muse.cbe1727'; require_muse_megamoe_patch()" \
    && rm -rf /tmp/muse-deep-gemm

# The muse-spark branch also requires NVSHMEM Python bindings absent from the base.
RUN python3 -m pip install --no-cache-dir nvshmem4py-cu13==0.3.1

WORKDIR /sgl-workspace/sglang-muse-spark
