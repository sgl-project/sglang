# Muse Spark Open AgentX image (refactored tree): the day0 image plus this SGLang tree and aiter patch 05.
# All Muse optimizations are on by default; no environment variable is needed at runtime.
#
# Build context: produced by prepare_context.sh from one commit of this repo (git archive), never the repo
# directory itself (the repo's .dockerignore is a symlink to .gitignore, which drops *.csv).
#   docker/muse_spark_image/prepare_context.sh <commit> <context dir>
#   docker build --build-arg MUSE_TREE_COMMIT=<commit> -t <tag> <context dir>
FROM lmsysorg/sglang-rocm:muse-spark-open@sha256:72065dfc0b87818ab415f96c0fc08848233d25f22b002ac8f28c33fad6ec82a4

ARG MUSE_TREE_COMMIT=unknown
LABEL org.opencontainers.image.title="sglang-rocm muse-spark-open AgentX (refactor)" \
      org.opencontainers.image.revision="${MUSE_TREE_COMMIT}" \
      org.opencontainers.image.base.name="lmsysorg/sglang-rocm:muse-spark-open" \
      org.opencontainers.image.base.digest="sha256:72065dfc0b87818ab415f96c0fc08848233d25f22b002ac8f28c33fad6ec82a4"

# The day0 image installs SGLang editable: site-packages/__editable___sglang_0_5_21_gc722c0a1_finder.py maps
# "sglang" -> /sgl-workspace/sglang/python/sglang. Swap that package directory in place; the mapping, the
# dist-info and every other package (sgl_kernel and aiter included) stay as shipped. The old directory's build
# leftovers (_version.py, rust_extensions/_multimodal*.so, kernels/aot/build) go with it, exactly as when the
# tree is mounted through PYTHONPATH for validation.
RUN rm -rf /sgl-workspace/sglang/python/sglang
COPY python/sglang /sgl-workspace/sglang/python/sglang

# aiter patch 05 (3rdparty/amd/muse_spark): opt-in unified-attention tiles. The day0 aiter already carries
# patches 01-04; 05 touches two Python/Triton files only, so nothing is rebuilt.
COPY 05-aiter-unified-attention-muse.patch /opt/muse_aiter_patches/05-aiter-unified-attention-muse.patch
RUN git -C /sgl-workspace/aiter apply --check /opt/muse_aiter_patches/05-aiter-unified-attention-muse.patch \
 && git -C /sgl-workspace/aiter apply /opt/muse_aiter_patches/05-aiter-unified-attention-muse.patch

# aiter merges every configs/model_configs/*bf16_tuned_gemm*.csv into its bf16 GEMM table at startup
# ("merge tuned file under model_configs/ and configs/"). These 16 Muse rows collide with no existing key.
COPY aiter_model_configs/muse_spark_bf16_tuned_gemm.csv /sgl-workspace/aiter/aiter/configs/model_configs/muse_spark_bf16_tuned_gemm.csv

COPY muse_image_manifest.json /opt/muse_image_manifest.json
COPY verify_manifest.py /opt/muse_image_verify_manifest.py

# Check the swap (editable mapping resolves to the new tree; run from / because the default WORKDIR /sgl-workspace
# holds the repo dir sglang/ that would resolve as a namespace package), byte-compile the package and the patched
# aiter files, verify every manifest sha256.
RUN cd / && python3 -c "import importlib.util as u; s = u.find_spec('sglang'); assert s.origin == '/sgl-workspace/sglang/python/sglang/__init__.py', s.origin" \
 && python3 -m compileall -q -j 0 /sgl-workspace/sglang/python/sglang \
 && python3 -m py_compile /sgl-workspace/aiter/aiter/ops/triton/attention/unified_attention.py \
      /sgl-workspace/aiter/aiter/ops/triton/_triton_kernels/attention/unified_attention.py \
 && python3 /opt/muse_image_verify_manifest.py /opt/muse_image_manifest.json
