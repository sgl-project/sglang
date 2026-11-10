# syntax=docker/dockerfile:1.7
# Kimi-K3.1 serving image built from a sglang-k31 checkout (the build context), with the compiled
# parts that the K3.1 optimizations rely on. Without them the server still starts and answers
# correctly, only slower (silent fallback):
#   1. Rust radix TreeCore   built by pip install -e "python" (needs .git: setuptools-scm version)
#   2. FlashInfer DynBlock   TRT-LLM MoE routing modules rebuilt for 1024 experts (K3.1 has 896)
# verify_image.py fails the build if either fell back. Two more parts back opt-in paths:
#   3. FlashInfer Prims-TS   main 784bfaf5's prims_ts/ over 0.7.2, for --dsa-*-backend prims_ts
#   4. Dynamo 1.6.0          ai-dynamo-runtime rebuilt with the session-affinity patches and the
#                            K3.1 edits to dynamo.sglang, so Dynamo+SGLang serves K3.1 as is
# FlashMLA and DeepGEMM come from the sgl-flash-mla and sgl-deep-gemm wheels python/pyproject.toml pins.
#
# Files: this Dockerfile, k31.Dockerfile.dockerignore and docker/k31-image/ (helper scripts and
# the Dynamo patches). Everything else is fetched from public sources at pinned commits: the base
# image, flashinfer, ai-dynamo/dynamo and rustup.
#
# Build from the root of a sglang-k31 checkout (keep .git), on amd64 or arm64 with BuildKit; no GPU
# is needed:
#   docker buildx build -f docker/k31.Dockerfile -t <tag> --load .
# BASE_IMAGE must be a nightly whose commit is an ancestor of the checkout; update the default
# when the tree moves to a newer upstream base. The last step runs verify_image.py; it needs no
# GPU, so an image can be rechecked with:
#   docker run --rm <tag> python3 /opt/k31-image/verify_image.py
ARG BASE_IMAGE=lmsysorg/sglang:nightly-dev-20261024-5994c559@sha256:40da9cac78a558237426971c048ab0f4f1a4b13bfb1b6e7a48d4d980e4163c8e
FROM ${BASE_IMAGE}

# Source: the checkout replaces the nightly's upstream tree at the same path, which the nightly's
# editable install and verify_image.py expect. k31.Dockerfile.dockerignore (used instead of the
# repo's .dockerignore) keeps .git and drops only build outputs.
RUN rm -rf /sgl-workspace/sglang
COPY . /sgl-workspace/sglang
WORKDIR /sgl-workspace/sglang
RUN git config --global --add safe.directory /sgl-workspace/sglang \
    && git rev-parse HEAD > .k31-source-commit \
    && pip install -e "python"

# FlashInfer DynBlock: raise DynBlockKernelMaxNumExperts to 1024 in the installed headers and
# rebuild the prebuilt jit-cache modules that include it, in place (no runtime env var needed).
COPY docker/k31-image/patch_flashinfer.py docker/k31-image/build_flashinfer_dynblock.py \
     docker/k31-image/overlay_flashinfer_prims_ts.py docker/k31-image/build_dynamo.py \
     docker/k31-image/verify_image.py /opt/k31-image/
COPY docker/k31-image/dynamo/ /opt/k31-image/dynamo/
RUN python3 /opt/k31-image/patch_flashinfer.py \
    && python3 /opt/k31-image/build_flashinfer_dynblock.py

# Prims-TS and Dynamo: each script is a no-op when its marker matches what is installed.
RUN python3 /opt/k31-image/overlay_flashinfer_prims_ts.py \
    && python3 /opt/k31-image/build_dynamo.py

# Fails the build if a DynBlock module was not rebuilt, the Prims-TS overlay or the patched Dynamo
# runtime is missing, or the Rust TreeCore does not load.
# Re-run any time: python3 /opt/k31-image/verify_image.py
RUN python3 /opt/k31-image/verify_image.py \
    && echo "sglang-k31 $(cat /sgl-workspace/sglang/.k31-source-commit)"
