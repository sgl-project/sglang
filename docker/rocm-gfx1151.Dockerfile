# SGLang for AMD Strix Halo / Ryzen AI MAX+ (gfx1151, RDNA3.5 iGPU).
#
# This is NOT a variant of docker/rocm.Dockerfile. That file targets CDNA
# (gfx942/gfx950) and includes components which do not support gfx1151. This
# image starts from AMD's stable ROCm/PyTorch image with native gfx1151 support.
#
# Build:
#   docker build -f docker/rocm-gfx1151.Dockerfile -t sglang-rocm:gfx1151 .
#
# Run (Strix Halo has no discrete VRAM; the GPU carves out of system RAM):
#   docker run -it --rm \
#     --device=/dev/kfd --device=/dev/dri \
#     --group-add video --group-add render \
#     --security-opt seccomp=unconfined \
#     --ipc=host --shm-size 16g \
#     -p 30000:30000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     sglang-rocm:gfx1151 \
#     python3 -m sglang.launch_server --model-path <model> \
#       --attention-backend triton --host 0.0.0.0

# ROCm 7.2.4 / PyTorch 2.9.1 is AMD's stable gfx1151-supported combination.
# Pin the image digest so rebuilding cannot silently change the toolchain.
ARG BASE_IMAGE="rocm/pytorch@sha256:7fe531fa185af260352fe7fbb3fa64ad749abe72adf0600a648c4692801b125a"

# =============================================================================
# Stage 1: stable ROCm + PyTorch for gfx1151.
# Pullable and testable on its own:
#   docker build --target rocm-torch -f docker/rocm-gfx1151.Dockerfile -t rocm-torch:gfx1151 .
# =============================================================================
FROM ${BASE_IMAGE} AS rocm-torch

ARG GPU_ARCH=gfx1151

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTORCH_ROCM_ARCH=${GPU_ARCH}
# ROCDXG requires this under WSL. It is inert when /dev/dxg is absent.
ENV HSA_ENABLE_DXG_DETECTION=1

# Fail loudly if the base image or its expected development toolchain changes.
RUN python3 -c "import torch; print('torch', torch.__version__); assert torch.version.hip" \
    && test -x /opt/rocm/bin/hipcc

# =============================================================================
# Stage 2: SGLang on top of the gfx1151 ROCm stack.
# =============================================================================
FROM rocm-torch AS sglang

ARG GPU_ARCH=gfx1151
# sgl-kernel's ROCm build (python/sglang/kernels/aot/setup_rocm.py) only accepts
# gfx942/gfx950/gfx1250 and hard-exits on anything else; the patch below lifts
# that gate. Set to 0 to skip the AOT kernels entirely and run Triton-only.
ARG BUILD_SGL_KERNEL=1
ARG MAX_JOBS=12

WORKDIR /sgl-workspace

# setuptools-rust builds the sglang-mm extension during the pip install below.
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && rustc --version
ENV CARGO_BUILD_JOBS=8

COPY . /sgl-workspace/sglang

# pyproject.toml pins the CUDA stack (torch, flashinfer[cu13], cuda-python,
# ...). pyproject_other.toml carries the srt_hip extra, which is
# torch-version-agnostic -- same swap docker/rocm.Dockerfile performs.
RUN cd /sgl-workspace/sglang \
    && rm -f python/pyproject.toml \
    && mv python/pyproject_other.toml python/pyproject.toml

# One problem in setup_rocm.py for this target, plus one in include/utils.h:
# the arch gate sys.exit(1)s outside {gfx942, gfx950, gfx1250}, and WARP_SIZE
# resolves to 64 on the host pass but 32 on the device pass for a wave32 part,
# which mismatches the MoE TopK launch bounds. Current main already limits
# non-gfx942 TopK dynamic LDS to 40KB, which fits gfx1151's 64KB limit. The
# remaining two problems are fixed here rather than upstream:
# gfx1151 is not a supported SGLang target, and the sources themselves compile
# clean for it. Each edit greps for the expected text first, so a rewrite
# upstream breaks the build loudly instead of silently misconfiguring kernels.
COPY docker/patches/sgl-kernel-gfx1151.sh /tmp/sgl-kernel-gfx1151.sh

RUN cd /sgl-workspace/sglang/python/sglang/kernels/aot \
    && if [ "${BUILD_SGL_KERNEL}" = "1" ]; then \
         rm -f pyproject.toml \
         && mv pyproject_rocm.toml pyproject.toml \
         && sh /tmp/sgl-kernel-gfx1151.sh setup_rocm.py \
         && AMDGPU_TARGET=${GPU_ARCH} MAX_JOBS=${MAX_JOBS} python3 setup_rocm.py install; \
       else \
         echo "Skipping sgl-kernel build (BUILD_SGL_KERNEL=0)"; \
       fi

# Current main composes extras through self-references
# (srt_hip -> sglang[runtime_common] -> sglang[runtime_base]). pip's resolver
# recursively walks that cycle from an editable source checkout. Flatten those
# three groups before installing the package itself without dependency solving.
# Keep compressed-tensors at its last torch-2.9-compatible release.
RUN cd /sgl-workspace/sglang \
    && python3 - <<'PY'
import subprocess
import sys
import tomllib
from pathlib import Path

project = tomllib.loads(Path("python/pyproject.toml").read_text())["project"]
extras = project["optional-dependencies"]
requirements = list(project["dependencies"])
for group in ("runtime_base", "runtime_common", "srt_hip"):
    requirements.extend(
        "compressed-tensors==0.15.0"
        if requirement == "compressed-tensors"
        else requirement
        for requirement in extras[group]
        if not requirement.startswith("sglang[") and requirement != "torch"
    )
requirements = list(dict.fromkeys(requirements))
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", *requirements]
)
PY
RUN cd /sgl-workspace/sglang \
    && pip install --no-cache-dir --no-deps -e python

# aiter is not optional on ROCm despite being CDNA-oriented:
# sglang/srt/layers/quantization/__init__.py imports quark, which imports
# aiter.ops.triton at module scope, so `import sglang.srt.layers.activation`
# fails outright without it. Installed WITHOUT PREBUILD_KERNELS -- that step
# AOT-compiles the CDNA assembly kernels and is what actually fails on gfx1151.
# In JIT mode aiter builds module_aiter_core for gfx1151 on demand instead.
ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT="c16d44b93a528b2a4bfd6d8d3409116d465872a9"

RUN git clone --recursive ${AITER_REPO} /sgl-workspace/aiter \
    && cd /sgl-workspace/aiter \
    && git checkout ${AITER_COMMIT} \
    && git submodule update --init --recursive \
    && GPU_ARCHS=${GPU_ARCH} pip install --no-cache-dir --no-build-isolation \
         --config-settings editable_mode=compat -e .

# aiter's compiled attention/MoE kernels are CDNA-only; keep sglang on the
# Triton paths. ServerArgs defaults to aiter on ROCm, so callers must pass
# `--attention-backend triton` until the RDNA default is fixed upstream.
# This is load-bearing beyond attention: aiter's RMSNorm uses v_pk_mul_f32,
# a CDNA-only instruction, and its CK attention templates assume wave64.
ENV SGLANG_USE_AITER=0

WORKDIR /sgl-workspace/sglang
CMD ["/bin/bash"]
