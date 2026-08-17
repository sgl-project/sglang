# Qwen38 serving image (x86_64 / CUDA 12.9 / sm_90a + sm_100a).
#
# Base ships stock SGLang (editable at /sgl-workspace/sglang), DeepEP source
# (at /sgl-workspace/DeepEP), a released FlashInfer trio, the Rust toolchain at
# /root/.cargo, and the CUDA 12.9 toolchain.
#
# This image adds the two Qwen38-specific pieces that stock lacks:
#   1. FlashInfer, replacing the base's release: the pinned nightly trio
#      (python + cubin + jit-cache), which ships the standalone BF16 direct
#      GEMM kernel (PR #4266) and the communication infrastructure the
#      in-tree MNNVL CuTe DSL all-reduce kernel builds on (the kernel itself
#      lives at python/sglang/kernels/ops/communication/mnnvl_cutedsl*, a
#      port of flashinfer#4358 pending a FlashInfer release that ships it).
#      nvidia-cutlass-dsl is floored at 4.7.0 because the CuTe DSL kernels call
#      PipelineTmaAsync.create(enable_multicast_signaling=..)
#   2. this repo's SGLang code, copied from the build context and editable-installed
#
# Unlike the CUDA 13 recipe this image keeps the base's DeepEP: the DeepEP v2
# source build and its NCCL 2.30.7 preload exist for GB300 on CUDA 13 only.
#
# The build refuses to start unless the base's torch is the one
# python/pyproject.toml pins, reinstalls the CUDA-tagged wheels this tree pins
# above what the base ships, and asserts those pins before finishing. All three
# live in docker/qwen38/cuda_pins.sh, which explains why each is needed.
#
# Build (no GPU needed). The context must be the repo root -- the SGLang source
# is taken from it rather than cloned:
#   docker build -f docker/qwen38/qwen38_cu12.Dockerfile -t qwen38-cu129 .

# The nightly, not a release tag: v0.5.17-cu129 ships torch 2.11.0 /
# sglang-kernel 0.4.5, which the gate below rejects against this tree's pins.
# `dev-cu12` is the CUDA 12.9 nightly and carries torch 2.13.0+cu129,
# sglang-kernel 0.4.6.post1+cu129 and sgl-deep-gemm 0.1.5.post2+cu129 -- exactly
# what python/pyproject.toml asks for, already in the +cu129 builds.
#
# `dev-cu12` MOVES every night. Builds are therefore not reproducible from this
# tag alone; pin `dev-cu12@sha256:<digest>` when a build has to be repeatable.
FROM lmsysorg/sglang:dev-cu12 AS base

# --- 0. Base/tree compatibility gate, FIRST ---
# Whether this base can host this tree is decided entirely by the base's torch,
# and it is knowable before any work happens. Run it here so a wrong base costs
# two seconds instead of a full FlashInfer nightly download and DeepEP link.
#
# pyproject.toml is copied to its own path purely to make that possible: the tree
# itself only lands at COPY below, far too late to gate anything. Both copies
# come from the same build context, so they cannot disagree.
COPY docker/qwen38/cuda_pins.sh /opt/qwen38/cuda_pins.sh
COPY python/pyproject.toml /opt/qwen38/pyproject.toml

RUN bash /opt/qwen38/cuda_pins.sh check-torch /opt/qwen38/pyproject.toml

# --- 1. FlashInfer: the pinned nightly trio ---
# Named apart from the base's ENV FLASHINFER_VERSION, which would otherwise
# shadow a same-named ARG and silently resolve to the base's 0.6.15.post1.
#
# The MNNVL CuTe DSL all-reduce backend (flashinfer#4358) is NOT expected
# from this nightly: SGLang carries the kernel in-tree at
# python/sglang/kernels/ops/communication/mnnvl_cutedsl* until a FlashInfer
# release ships it. The nightly only supplies the communication
# infrastructure (mnnvl probing, pattern enum, workspace ABC) plus the
# PR #4266 GEMM kernel, so all three packages come from the same nightly
# and no git-commit install is needed.
ARG FLASHINFER_NIGHTLY_VERSION=0.6.18.dev20260807
ARG FLASHINFER_JIT_CACHE_CUDA_TAG=cu129
ARG CUTLASS_DSL_MIN_VERSION=4.7.0

# Uninstall first: a mixed python/cubin/jit-cache installation fails at import,
# and pip would otherwise leave the base's jit-cache shadowing JIT compilation.
# Installed with dependency resolution so apache-tvm-ffi lands at whatever the
# jit-cache wheel's metadata requires -- an exact pin here could contradict it.
RUN python3 -m pip uninstall -y \
      flashinfer-python flashinfer-cubin flashinfer-jit-cache && \
    rm -rf /root/.cache/flashinfer && \
    python3 -m pip install \
      "flashinfer-python==${FLASHINFER_NIGHTLY_VERSION}" \
      "flashinfer-cubin==${FLASHINFER_NIGHTLY_VERSION}" \
      "flashinfer-jit-cache==${FLASHINFER_NIGHTLY_VERSION}+${FLASHINFER_JIT_CACHE_CUDA_TAG}" \
      --extra-index-url https://flashinfer.ai/whl/nightly/ \
      --extra-index-url "https://flashinfer.ai/whl/nightly/${FLASHINFER_JIT_CACHE_CUDA_TAG}/" && \
    python3 -m pip install "nvidia-cutlass-dsl>=${CUTLASS_DSL_MIN_VERSION}" && \
    FLASHINFER_EXPECTED="${FLASHINFER_NIGHTLY_VERSION}" \
    FLASHINFER_CUDA_TAG="${FLASHINFER_JIT_CACHE_CUDA_TAG}" \
    python3 -c 'import os; from importlib.metadata import version; e = os.environ["FLASHINFER_EXPECTED"]; tag = os.environ["FLASHINFER_CUDA_TAG"]; got = {p: version(p) for p in ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache")}; assert got["flashinfer-python"] == e, got; assert got["flashinfer-cubin"].split("+")[0] == e, got; assert got["flashinfer-jit-cache"].startswith(e + "+" + tag), got' && \
    rm -rf /root/.cache/pip

ENV FLASHINFER_VERSION=${FLASHINFER_NIGHTLY_VERSION}

LABEL ai.radixark.flashinfer.prebuilt_nightly="${FLASHINFER_NIGHTLY_VERSION}"

# This tree's BF16 Split-K GEMM loads FlashInfer PR #4266's standalone direct
# kernel by file path out of SGLANG_FLASHINFER_PR4266_SOURCE (see
# python/sglang/srt/layers/quantization/unquant.py). On SM100 that path is on by
# DEFAULT -- bf16_gemm_backend=auto resolves to cutedsl, and
# SGLANG_ENABLE_BF16_SPLITK_GEMM defaults to True -- so an unset variable makes
# the server raise at startup rather than degrade. (On Hopper the backend stays
# unoptimized and the variable is never read, but B200 runs this image too.)
# Point it at the installed FlashInfer through a stable symlink instead of a
# hardcoded dist-packages path, and fail the build now if the kernel is missing.
#
# `import flashinfer` here is load-bearing beyond resolving the path: it runs
# flashinfer/jit/env.py, so a python/cubin/jit-cache version mismatch would
# surface here. The file check then confirms the nightly actually ships the
# GEMM kernel this image depends on -- checked as a file rather than an import
# because the module pulls in CuTe DSL, which cannot load on a CPU build host.
# The MNNVL CuTe DSL all-reduce kernel needs no such check: it is part of the
# SGLang tree copied in below.
RUN FI_ROOT="$(python3 -c 'import pathlib, flashinfer; print(pathlib.Path(flashinfer.__file__).resolve().parent.parent)')" && \
    ln -sfn "${FI_ROOT}" /opt/flashinfer-src && \
    if [ ! -f /opt/flashinfer-src/flashinfer/gemm/kernels/dense_bf16_gemm_direct.py ]; then \
        echo "ERROR: flashinfer ${FLASHINFER_NIGHTLY_VERSION} does not ship flashinfer/gemm/kernels/dense_bf16_gemm_direct.py (PR #4266)." >&2; \
        echo "       Pin a FlashInfer that carries it, or serve with SGLANG_ENABLE_BF16_SPLITK_GEMM=0." >&2; \
        exit 1; \
    fi

ENV SGLANG_FLASHINFER_PR4266_SOURCE=/opt/flashinfer-src

# --- 2. Qwen38 SGLang code (replaces the base's stock sglang, editable) ---
# rm first: COPY merges into an existing directory, so files the stock release
# has and this tree does not would otherwise survive.
RUN rm -rf /sgl-workspace/sglang

COPY . /sgl-workspace/sglang

# .git is discarded, so setuptools-scm cannot derive a version and would fall
# back to 0.0.0.dev0; pass SGLANG_VERSION to label the build.
# Keep the installed extension modules, but discard Rust and pip build
# artifacts that are not used at runtime.
ARG SGLANG_VERSION=0.0.0.dev0
# Which +<tag> build of the SGLang wheels to pull. Kept separate from the
# FlashInfer jit-cache tag even though both read "cu129" today: they index two
# unrelated wheel sets, and silently reusing one for the other would make a
# FlashInfer retag quietly change which kernel ABI gets installed.
ARG SGL_WHL_CUDA_TAG=cu129
RUN cd /sgl-workspace/sglang && \
    rm -rf .git && \
    test ! -e .git && \
    SETUPTOOLS_SCM_PRETEND_VERSION="${SGLANG_VERSION}" \
      pip install -e python --no-deps && \
    # --no-deps protects the base's CUDA-tagged wheels from being replaced by
    # untagged PyPI builds, but it also drops any pin this tree raised above
    # what the base ships. Reinstall those from SGLang's index at the pinned
    # version, read out of pyproject.toml so a future bump needs no edit here.
    bash docker/qwen38/cuda_pins.sh reconcile \
      python/pyproject.toml "${SGL_WHL_CUDA_TAG}" \
      sglang-kernel sgl-deep-gemm && \
    kernels lock python && \
    ( success=0; \
      if [ "$(uname -m)" = "aarch64" ]; then \
          echo "Skipping sgl-flash-attn3 cubin download on aarch64; kernels will be JIT-compiled at runtime"; \
          success=1; \
      else \
          for i in 1 2 3; do \
              echo "Attempt $i/3: downloading sgl-kernel cubins..."; \
              if kernels download python; then success=1; break; fi; \
              [ "$i" = "3" ] || { echo "sgl-kernel cubin download failed, retrying in 30s..."; sleep 30; }; \
          done; \
      fi; \
      [ "$success" = "1" ] || \
        echo "WARNING: no matching sgl-flash-attn3 cubin variant; kernels will be JIT-compiled at runtime" ) && \
    mkdir -p /root/.cache/huggingface /root/.cache/sglang && \
    ( if [ -f python/kernels.lock ]; then mv python/kernels.lock /root/.cache/sglang/; fi ) && \
    rm -rf \
      rust/target \
      rust/sglang-grpc/target \
      rust/sglang-mm/target \
      rust/sglang-server/target \
      /root/.cargo/registry \
      /root/.cache/pip

# --- 3. Verify the image can actually import what it ships ---
# This image keeps the base's DeepEP rather than rebuilding it, so the NCCL
# re-pin the CUDA 13 recipe needs does not apply here. What does apply is the
# check: the FlashInfer install above resolves dependencies, and torch declares
# a hard `nvidia-nccl-cu12==<pin>`, so the NCCL under DeepEP can move without
# any build step failing. The CUDA 13 image shipped green while deep_ep._C had an
# unresolved ncclGetLsaDevicePointer, so assert it instead of assuming it.
#
# deep_ep goes through import-if-gpu rather than a bare import. On the dev-cu12
# base the module is provided by the packaged `sgl-deep-ep`, which checks for a
# usable CUDA device at import and raises "The NVIDIA driver does not expose a
# usable CUDA device" -- so a bare import fails on every CPU build host, which is
# all of them. The check is kept rather than dropped so it still runs when the
# image is built on a GPU host; a GPU smoke test has to cover the rest.
#
# `import sglang` proves nothing about the dependency versions: assert_pkg_version
# lives in srt/entrypoints/engine.py and only runs once a server starts. A tree
# that outgrew the base's sglang-kernel therefore builds green here, ships, and
# dies on every rank at launch -- exactly the shape of the deep_ep bug above.
# Run that same assertion at build time so the mismatch fails the build instead.
# Unlike the imports, verify reads installed metadata, so it works everywhere.
RUN cd /sgl-workspace/sglang && \
    bash docker/qwen38/cuda_pins.sh import-if-gpu deep_ep && \
    python3 -c 'import sglang; print("sglang", sglang.__version__)' && \
    bash docker/qwen38/cuda_pins.sh verify \
      python/pyproject.toml sglang-kernel sgl-deep-gemm

WORKDIR /sgl-workspace/sglang
