# syntax=docker/dockerfile:1.7
# Kimi-K3.1 serving image built from a sglang-k31 checkout (the build context), with the compiled
# parts that dev's optimizations rely on. Without them the server still starts and answers
# correctly, only slower (silent fallback):
#   1. Rust radix TreeCore   built by pip install -e "python" (needs .git: setuptools-scm version)
#   2. FlashMLA flashmla_ops #20 native 96-head sparse decode + sparse_decode_num_sm_parts,
#                            #30 prefill token packing; PyPI sglang-kernel 0.4.10 has neither
#   3. FlashInfer DynBlock   TRT-LLM MoE routing modules rebuilt for 1024 experts (K3.1 has 896)
# verify_image.py fails the build if any of the three fell back.
#
# docker/k31-image/ is the only copy of the helper scripts. The CI workflow
# (.github/workflows/build-k31-image.yml) runs this file for full builds and
# k31-incremental.Dockerfile, which reuses the same scripts, for incremental ones; keep the two
# Dockerfiles in step.
#
# Build from the root of a sglang-k31 checkout (keep .git):
#   docker buildx build -f docker/k31.Dockerfile -t <tag> --load .
# BASE_IMAGE must be a nightly whose commit is an ancestor of the checkout. CI picks the newest
# such nightly itself; manual builds use this default, so update it when dev moves to a newer
# upstream base.
ARG BASE_IMAGE=lmsysorg/sglang:nightly-dev-20261024-5994c559@sha256:40da9cac78a558237426971c048ab0f4f1a4b13bfb1b6e7a48d4d980e4163c8e
FROM ${BASE_IMAGE}
# Parallel nvcc jobs for flashmla_ops: 12 peaks at ~19 GB of RAM (amd64); the whole build peaks at
# ~23 GB on arm64.
ARG FLASHMLA_JOBS=12

# Source: the checkout replaces the nightly's upstream tree at the same path, which the nightly's
# editable install and verify_image.py expect. k31.Dockerfile.dockerignore (used instead of the
# repo's .dockerignore) keeps .git and drops only build outputs.
RUN rm -rf /sgl-workspace/sglang
COPY . /sgl-workspace/sglang
WORKDIR /sgl-workspace/sglang
RUN git config --global --add safe.directory /sgl-workspace/sglang \
    && git rev-parse HEAD > .k31-source-commit \
    && pip install -e "python"

# FlashMLA: only the flashmla_ops target of sgl-kernel, from this tree's aot sources (the build
# applies apply_flashmla_patches.cmake and fails if a patch does not apply); it replaces the
# PyPI wheel's module, the other sgl-kernel ops stay as installed.
RUN cd /sgl-workspace/sglang/python/sglang/kernels/aot \
    && CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake -S . -B /tmp/flashmla-build -G Ninja \
         -DCMAKE_BUILD_TYPE=Release \
         -DSKBUILD_SABI_COMPONENT=Development.SABIModule -DSKBUILD_SABI_VERSION=3.11 \
         -DPython_EXECUTABLE="$(command -v python3)" \
         -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
         -DSGL_KERNEL_ENABLE_SM100A=ON -DENABLE_BELOW_SM90=OFF -DENABLE_CCACHE=OFF \
    && cmake --build /tmp/flashmla-build --target flashmla_ops -j "${FLASHMLA_JOBS}" \
    && cp /tmp/flashmla-build/flashmla_ops.abi3.so \
         "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/sgl_kernel/" \
    && rm -rf /tmp/flashmla-build

# FlashInfer DynBlock: raise DynBlockKernelMaxNumExperts to 1024 in the installed headers and
# rebuild the prebuilt jit-cache modules that include it, in place (no runtime env var needed).
COPY docker/k31-image/patch_flashinfer.py docker/k31-image/build_flashinfer_dynblock.py \
     docker/k31-image/verify_image.py /opt/k31-image/
RUN python3 /opt/k31-image/patch_flashinfer.py \
    && python3 /opt/k31-image/build_flashinfer_dynblock.py

# Fails the build if flashmla_ops lacks an op of this tree, a DynBlock module was not rebuilt, or
# the Rust TreeCore does not load. Re-run any time: python3 /opt/k31-image/verify_image.py
RUN python3 /opt/k31-image/verify_image.py \
    && echo "sglang-k31 $(cat /sgl-workspace/sglang/.k31-source-commit)"
