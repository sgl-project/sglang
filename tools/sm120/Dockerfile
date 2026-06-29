# Dockerfile to run the SM120 tilelang paged-MQA-logits validator on an
# RTX PRO 6000 / Blackwell-desktop (sm_120) GPU.
#
# It bases on an SGLang runtime image (which already ships CUDA 12.8+, a torch
# build with sm_120 support, tilelang, deep_gemm and sglang), checks out your
# fork branch with the SM120 fix, and runs the validator that compares the
# tilelang kernel against the pure-torch reference.
#
# Build (from the folder containing this file and validate_tilelang_paged_logits.py):
#   docker build -f Dockerfile.sm120-validator -t sm120-validator .
#
# Run (must expose the SM120 GPU):
#   docker run --rm --gpus all sm120-validator
#
# Override the base image to match what you run on the box, e.g.:
#   docker build -f Dockerfile.sm120-validator \
#     --build-arg SGLANG_IMAGE=lmsysorg/sglang:v0.5.5-cu128 \
#     --build-arg SGLANG_BRANCH=sm120-glm5.2-fp8 \
#     -t sm120-validator .

# --- Base: an SGLang image that already works on your SM120 box ---
ARG SGLANG_IMAGE=lmsysorg/sglang:latest
FROM ${SGLANG_IMAGE}

# --- Pull the fork branch containing the SM120 changes (editable source install) ---
ARG SGLANG_REPO=https://github.com/shivajid/sglang.git
ARG SGLANG_BRANCH=sm120-glm5.2-fp8
ARG SGLANG_DIR=/sgl-workspace/sglang

# The official image installs sglang editable from ${SGLANG_DIR}; updating the
# checkout there updates the imported code. Fall back gracefully if the layout
# differs.
RUN if [ -d "${SGLANG_DIR}/.git" ]; then \
        cd "${SGLANG_DIR}" && \
        git config --global --add safe.directory "${SGLANG_DIR}" && \
        git fetch --depth=1 "${SGLANG_REPO}" "${SGLANG_BRANCH}" && \
        git checkout -B "${SGLANG_BRANCH}" FETCH_HEAD && \
        git log --oneline -3 ; \
    else \
        echo "WARNING: ${SGLANG_DIR}/.git not found; using sglang as installed in the base image." ; \
    fi

# --- Validator + tilelang patch scripts (from build context) ---
WORKDIR /workspace
COPY validate_tilelang_paged_logits.py /workspace/validate_tilelang_paged_logits.py
COPY patch_tilelang_wait_wgmma.py /workspace/patch_tilelang_wait_wgmma.py

# Patch the installed tilelang gemm_sm120.h with a no-op tl::wait_wgmma so the DSA
# attention kernels compile on SM120. tilelang JIT-compiles with these headers at
# runtime, so applying it at build time means the in-container validator (and any
# kernel compile) sees it. Idempotent.
RUN python3 /workspace/patch_tilelang_wait_wgmma.py

# tilelang JIT needs a writable cache dir at runtime.
ENV TILELANG_CACHE_DIR=/tmp/tilelang_cache \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "validate_tilelang_paged_logits.py"]
