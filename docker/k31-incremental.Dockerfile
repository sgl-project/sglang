# syntax=docker/dockerfile:1.7
# Incremental Kimi-K3.1 image: a previous K3.1 image (built by k31.Dockerfile or by this file)
# plus the commits since its source commit. The CI workflow (.github/workflows/build-k31-image.yml)
# uses it for pushes that need no full rebuild; a Python-only change then adds one small layer.
# It keeps the guarantees of k31.Dockerfile: the Rust TreeCore, flashmla_ops and the FlashInfer
# DynBlock modules are rebuilt when their inputs changed, and verify_image.py runs on every build.
#
# Build context (assembled by the workflow, not a checkout):
#   merged.patch     git diff --binary <FROM_SHA> <SRC_SHA>
#   changed_py.txt   changed python/**/*.py paths, syntax-checked after the patch
#   k31-image/       docker/k31-image/ of SRC_SHA
ARG PARENT_IMAGE
FROM ${PARENT_IMAGE}
ARG FROM_SHA
ARG SRC_SHA
# 1 when the patch touches python/pyproject.toml, python/setup.py or rust/.
ARG REINSTALL=0
# 1 when python/sglang/kernels/aot differs from the parent's, or after a reinstall.
ARG BUILD_FLASHMLA=0
ARG FLASHMLA_JOBS=12
# 1 when the DynBlock scripts differ from the parent's, or after a reinstall.
ARG BUILD_FI_DYNBLOCK=0

COPY merged.patch changed_py.txt /tmp/
COPY k31-image/ /opt/k31-image/

# The parent must hold exactly the tree the patch was computed against: every K3.1 image records
# its source commit in .k31-source-commit.
WORKDIR /sgl-workspace/sglang
RUN test "$(cat .k31-source-commit)" = "${FROM_SHA}" \
    && git apply --binary -p1 /tmp/merged.patch \
    && echo "${SRC_SHA}" > .k31-source-commit \
    && if [ "${REINSTALL}" = 1 ]; then pip install -e "python"; fi \
    && python3 -c 'import sys; [compile(open(p, "rb").read(), p, "exec") for p in map(str.strip, open(sys.argv[1])) if p]' /tmp/changed_py.txt \
    && rm -f /tmp/merged.patch /tmp/changed_py.txt

# FlashMLA: same build as k31.Dockerfile, only when its inputs changed.
RUN if [ "${BUILD_FLASHMLA}" = 1 ]; then \
      cd /sgl-workspace/sglang/python/sglang/kernels/aot \
      && CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake -S . -B /tmp/flashmla-build -G Ninja \
           -DCMAKE_BUILD_TYPE=Release \
           -DSKBUILD_SABI_COMPONENT=Development.SABIModule -DSKBUILD_SABI_VERSION=3.11 \
           -DPython_EXECUTABLE="$(command -v python3)" \
           -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
           -DSGL_KERNEL_ENABLE_SM100A=ON -DENABLE_BELOW_SM90=OFF -DENABLE_CCACHE=OFF \
      && cmake --build /tmp/flashmla-build --target flashmla_ops -j "${FLASHMLA_JOBS}" \
      && cp /tmp/flashmla-build/flashmla_ops.abi3.so \
           "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/sgl_kernel/" \
      && rm -rf /tmp/flashmla-build; \
    fi

# FlashInfer DynBlock: the header patch is idempotent; the module rebuild only when its inputs changed.
RUN python3 /opt/k31-image/patch_flashinfer.py \
    && if [ "${BUILD_FI_DYNBLOCK}" = 1 ]; then python3 /opt/k31-image/build_flashinfer_dynblock.py; fi

RUN python3 /opt/k31-image/verify_image.py \
    && echo "sglang-k31 $(cat /sgl-workspace/sglang/.k31-source-commit)"
