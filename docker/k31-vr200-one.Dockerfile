# syntax=docker/dockerfile:1.7
# Kimi-K3.1 VR200 serving image (ONE image for every point of the K3.1 VR200 front): this SGLang tree + the sgl-nvidia-mega
# plugin + the compiled parts the K3.1 paths use, on the SGLang nightly of 2026-11-07 for CUDA 13.4 (arm64), pinned by digest.
# The build steps are docker/k31-image/vr200/build_one.sh; the same script built and gated the published image (a cluster
# bake in an enroot rootfs instead of a Docker layer), so this file adds only the inputs.
#
# Steps in build_one.sh (see its header): Rust TreeCore (pip install --no-deps -e python), FlashMLA flashmla_ops and the
# sgl_kernel flash_mla Python modules from this tree's aot sources, FlashInfer DynBlock (1024 experts) + the SM107 MoE module,
# FlashInfer Prims-TS (main 26bb6dc9 + flashinfer-ai/flashinfer#8186), flash-attn-4 4.0.0b36, the plugin (editable), Dynamo 1.6.0
# with the session-affinity runtime patches and the K3.1 dynamo.sglang edits, and the SM107 small-M GEMM table.
# Library versions stay the base's (torch 2.15, FlashInfer 0.7.3, sgl-deep-gemm 0.2.6, sglang-kernel 0.4.10); the script fails if a
# step moves them.
#
# Build (BuildKit, arm64 host or emulation; no GPU needed), from the root of this checkout, with the plugin checkout as a named
# build context:
#   docker buildx build --platform linux/arm64 -f docker/k31-vr200-one.Dockerfile \
#     --build-context plugin=<sgl-nvidia-mega checkout> \
#     --build-arg SGLANG_COMMIT=$(git rev-parse HEAD) --build-arg PLUGIN_COMMIT=$(git -C <plugin checkout> rev-parse HEAD) -t <tag> --load .
ARG BASE_IMAGE=lmsysorg/sglang@sha256:2163259be9cc2423e10f5f5b1827d73ea6b9d5f4b6ddd4d2226c5ceaa50b1486
FROM ${BASE_IMAGE}
ARG SGLANG_COMMIT
ARG PLUGIN_COMMIT
ARG K31_BUILD_JOBS=32

RUN rm -rf /sgl-workspace/sglang /sgl-workspace/sgl-nvidia-mega
COPY . /sgl-workspace/sglang
COPY --from=plugin . /sgl-workspace/sgl-nvidia-mega
RUN test -n "${SGLANG_COMMIT}" && test -n "${PLUGIN_COMMIT}" \
    && echo "${SGLANG_COMMIT}" > /sgl-workspace/sglang/.k31-source-commit \
    && echo "${PLUGIN_COMMIT}" > /sgl-workspace/sgl-nvidia-mega/COMMIT \
    && rm -rf /sgl-workspace/sglang/.git /sgl-workspace/sgl-nvidia-mega/.git
COPY docker/k31-image/vr200/build_one.sh /opt/k31-image/one/build_one.sh
COPY docker/k31-image/vr200/addons/ /opt/k31-image/one/addons/
RUN K31_BUILD_JOBS=${K31_BUILD_JOBS} bash /opt/k31-image/one/build_one.sh
