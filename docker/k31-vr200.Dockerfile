# syntax=docker/dockerfile:1.7
# Kimi-K3.1 VR200 serving image for the NVIDIA megakernel points: this SGLang tree + the sgl-nvidia-mega
# plugin, on the exact base image those points ran. One image serves every point; the launch words per
# point are in benchmark/kimi_k31_vr200/.
#
# How it relates to docker/k31.Dockerfile (#48414's K3.1 recipe): that recipe starts from another nightly,
# reinstalls this tree with `pip install -e python` (which would move flashinfer-python to the pyproject's
# 0.7.3 and sgl-deep-gemm to 0.2.6), builds the Rust TreeCore, FlashMLA, DeepGEMM, Prims-TS and Dynamo, and
# does not install the plugin. The VR200 points ran none of that. They ran:
#   - the SGLang nightly base image of 2026-10-27 (CUDA 13.4, SGLang main 3927a73; BASE_IMAGE below) unchanged
#     (torch 2.15.0+cu134, flashinfer 0.7.2,
#     sgl-deep-gemm 0.2.3.post2, its own FlashMLA; package set in docker/k31-image/vr200/served_packages.txt);
#   - the SGLang tree and the plugin bind-mounted over /sgl-workspace/sglang and /sgl-workspace/sgl-nvidia-mega,
#     where the base image's editable installs already point;
#   - the FlashInfer DynBlock sm107 module (docker/k31-image/vr200/build_fi_sm107.py + install_k31_parts.sh);
#   - the Python TreeCore (SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python, SGLANG_RUST_BUILD_MODE=never in the
#     launch words).
# This recipe reproduces exactly that, with copies in place of bind mounts and nothing else installed.
#
# Build (BuildKit; arm64, the base image is arm64 only; no GPU needed), from the root of this checkout, with the
# plugin checkout as a named build context:
#   docker buildx build --platform linux/arm64 -f docker/k31-vr200.Dockerfile \
#     --build-context plugin=<path to the sgl-nvidia-mega checkout> \
#     --build-arg SGLANG_COMMIT=$(git rev-parse HEAD) \
#     --build-arg PLUGIN_COMMIT=$(git -C <plugin checkout> rev-parse HEAD) -t <tag> --load .
# Recheck an image any time (CPU only): docker run --rm <tag> python3 /opt/k31-image/vr200/verify_vr200_image.py
# The base image: lmsysorg/sglang on Docker Hub, the nightly of 2026-10-27 for CUDA 13.4 (SGLang main 3927a73), pinned by its
# arm64 manifest digest.
ARG BASE_IMAGE=lmsysorg/sglang@sha256:be6da9c3bb351c36345a72cc290270d934733e0b0ff4519c393af85ab4183a8c
FROM ${BASE_IMAGE}
ARG SGLANG_COMMIT=unknown
ARG PLUGIN_COMMIT=unknown

# SGLang tree: replaces the nightly's tree at the path its editable install maps (no reinstall, so no
# dependency moves). k31-vr200.Dockerfile.dockerignore drops .git and build outputs.
RUN rm -rf /sgl-workspace/sglang
COPY . /sgl-workspace/sglang
RUN echo "sglang ${SGLANG_COMMIT}" > /sgl-workspace/sglang/.k31_tree

# Plugin: same, at the path the nightly's editable sgl_nvidia_mega install maps. Its kernels compile at run
# time (CuTe DSL / NVRTC / sgl_kernel JIT against this tree's python/sglang/kernels/jit/include).
RUN rm -rf /sgl-workspace/sgl-nvidia-mega
COPY --from=plugin . /sgl-workspace/sgl-nvidia-mega
RUN echo "${PLUGIN_COMMIT}" > /sgl-workspace/sgl-nvidia-mega/COMMIT

# FlashInfer DynBlock for K3.1's 896 experts on SM107: patched header + rebuilt fused_moe_trtllm_sm107 module.
COPY docker/k31-image/vr200/ /opt/k31-image/vr200/
RUN python3 /opt/k31-image/vr200/build_fi_sm107.py /opt/k31-image/vr200/fi_sm107 \
    && bash /opt/k31-image/vr200/install_k31_parts.sh /opt/k31-image/vr200/fi_sm107 none \
         | tee /opt/k31-image/vr200/install.log \
    && grep -q '^K31_PARTS_OK ' /opt/k31-image/vr200/install.log

# Fails the build if the package set differs from the served image's, the DynBlock parts are missing, the trees
# are not the ones imported, or a plugin sgl_kernel include has no header in this tree.
RUN python3 /opt/k31-image/vr200/verify_vr200_image.py
