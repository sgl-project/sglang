# SGLang for AMD Strix Halo / Ryzen AI MAX+ (gfx1151, RDNA3.5 iGPU).
#
# This is NOT a variant of docker/rocm.Dockerfile. That file targets CDNA
# (gfx942/gfx950) and unconditionally builds AITER, a custom LLVM, and TileLang,
# none of which support gfx1151. This file instead pulls the whole ROCm stack --
# runtime, libraries, and PyTorch -- as pip wheels from AMD's TheRock nightly
# index, which publishes a gfx1151-specific build.
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
#     sglang_launch --model-path <model> --host 0.0.0.0

ARG BASE_IMAGE="ubuntu:24.04"

# =============================================================================
# Stage 1: ROCm + PyTorch for gfx1151.
# Buildable and testable on its own:
#   docker build --target rocm-torch -f docker/rocm-gfx1151.Dockerfile -t rocm-torch:gfx1151 .
# =============================================================================
FROM ${BASE_IMAGE} AS rocm-torch

ARG GPU_ARCH=gfx1151
# TheRock nightly index, per-arch. Carries torch/triton/torchvision/torchaudio
# plus the `rocm` meta-package, which pulls the ROCm userspace itself as wheels
# (rocm-sdk-core + rocm-sdk-libraries-gfx1151). No /opt/rocm involved.
ARG ROCM_INDEX="https://rocm.nightlies.amd.com/v2/gfx1151/"
# Nightlies are mutable; pin one known-good build. torch 2.9.1 is also what the
# CDNA image ships, so sglang's srt_hip extra is tested against that line.
ARG ROCM_NIGHTLY="7.13.0a20260513"
ARG TORCH_VERSION="2.9.1+rocm7.13.0a20260513"
# The nightly carries two torch lines side by side (2.9.1 and 2.10.0) with a
# matching torchvision/torchaudio for each. They are not interchangeable: the
# 2.10 torchvision fails to register torchvision::nms against torch 2.9.1 and
# takes down `import sglang`. Bump all three together or none.
ARG TORCHVISION_VERSION="0.24.0+rocm7.13.0a20260513"
ARG TORCHAUDIO_VERSION="2.9.0+rocm7.13.0a20260513"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libnuma-dev \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 marks the system interpreter externally-managed (PEP 668), so
# everything lands in a venv rather than fighting apt.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN pip install --no-cache-dir -U pip setuptools wheel

# ROCM_INDEX only -- no --extra-index-url. PyPI also publishes `torch` and
# `rocm-sdk-devel`, and pip merges indexes by version, so PyPI's newer CUDA
# torch (and a rocm-sdk-devel stub) win and silently replace the ROCm stack.
# The nightly index carries torch's whole dependency closure, so it stands alone.
# torch pins rocm[libraries] and triton transitively; [devel] adds hipcc for
# the sgl-kernel build in stage 2.
RUN pip install --no-cache-dir \
        --index-url ${ROCM_INDEX} \
        "torch==${TORCH_VERSION}" \
        "rocm[devel]==${ROCM_NIGHTLY}" \
    && pip install --no-cache-dir --index-url ${ROCM_INDEX} \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}"

# Fail loudly here rather than at runtime if a CUDA torch slipped in.
RUN python3 -c "import torch; v = torch.__version__; print('torch', v); assert 'rocm' in v, v"

# TheRock installs ROCm under site-packages (_rocm_sdk_devel), not /opt/rocm.
# Symlink it to the conventional location so the paths below can be literal ENVs
# -- ENV cannot read a value computed in a RUN step, and hardcoding the
# site-packages path would bake in the Python version.
RUN ROCM_PATH="$(rocm-sdk path --root)" \
    && ln -s "${ROCM_PATH}" /opt/rocm \
    && test -x /opt/rocm/bin/hipcc \
    && test -e /opt/rocm/lib/libamdhip64.so

ENV ROCM_PATH=/opt/rocm
ENV ROCM_HOME=/opt/rocm
ENV PATH="/opt/rocm/bin:${PATH}"
# LIBRARY_PATH is for link time, LD_LIBRARY_PATH for load time. sglang's tvm-ffi
# JIT links its kernels with a bare `c++ ... -lamdhip64` and only passes
# -L/opt/venv/lib, where libamdhip64 is not; without LIBRARY_PATH the JIT build
# fails at link with "cannot find -lamdhip64" once the server starts compiling
# kernels. _rocm_sdk_core carries the runtime sonames, _rocm_sdk_devel the
# unversioned .so symlinks the linker resolves against.
ENV LIBRARY_PATH="/opt/rocm/lib"
ENV LD_LIBRARY_PATH="/opt/rocm/lib:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib"

ENV PYTORCH_ROCM_ARCH=${GPU_ARCH}
# Deliberately no HSA_OVERRIDE_GFX_VERSION: gfx1151 is natively supported by
# these wheels. Do not set it even to an empty string -- ROCr treats the
# variable as present and fails HSA init with HSA_STATUS_ERROR_OUT_OF_RESOURCES,
# which surfaces as torch.cuda.is_available() == False.

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

# Two problems in setup_rocm.py for this target, plus one in include/utils.h:
# the arch gate sys.exit(1)s outside {gfx942, gfx950, gfx1250}; the TopK LDS
# budget is keyed on "am I gfx942" so gfx1151 falls through to the 128KB
# branch despite reporting lds_size_in_kb 64; and WARP_SIZE resolves to 64 on
# the host pass but 32 on the device pass for a wave32 part, which mismatches
# the MoE TopK launch bounds. All three are fixed here rather than upstream:
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

RUN cd /sgl-workspace/sglang \
    && pip install --no-cache-dir -e "python[srt_hip]"

# aiter is not optional on ROCm despite being CDNA-oriented:
# sglang/srt/layers/quantization/__init__.py imports quark, which imports
# aiter.ops.triton at module scope, so `import sglang.srt.layers.activation`
# fails outright without it. Installed WITHOUT PREBUILD_KERNELS -- that step
# AOT-compiles the CDNA assembly kernels and is what actually fails on gfx1151.
# In JIT mode aiter builds module_aiter_core for gfx1151 on demand instead.
ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT="d9e5ef7ce08ee7045d583aed768cff41aa9210fe"

RUN git clone --recursive ${AITER_REPO} /sgl-workspace/aiter \
    && cd /sgl-workspace/aiter \
    && git checkout ${AITER_COMMIT} \
    && git submodule update --init --recursive \
    && GPU_ARCHS=${GPU_ARCH} pip install --no-cache-dir --no-build-isolation \
         --config-settings editable_mode=compat -e .

# Warm aiter's JIT cache into the image so the first request doesn't pay for it.
# Best-effort: the build host has no GPU, so this can legitimately no-op, in
# which case aiter compiles module_aiter_core on first use at runtime instead.
RUN GPU_ARCHS=${GPU_ARCH} python3 -c \
         "import aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat" \
       || echo "aiter JIT prewarm skipped; will build on first use"

# aiter's compiled attention/MoE kernels are CDNA-only; keep sglang on the
# Triton paths. There is no env var for the attention backend -- ServerArgs
# defaults it to "aiter" on ROCm -- so callers must pass
# `--attention-backend triton` explicitly. sglang_launch below does that.
# This is load-bearing beyond attention: aiter's RMSNorm uses v_pk_mul_f32,
# a CDNA-only instruction, and its CK attention templates assume wave64.
ENV SGLANG_USE_AITER=0

# ROCm tuning. HIP_FORCE_DEV_KERNARG and the AOTriton flag are free -- no state,
# no warmup. TunableOp is not: it benchmarks GEMM variants the first time it
# sees each shape, so a cold cache makes early requests SLOWER. It only pays off
# when the results file persists, which means mounting the directory:
#     -v ~/.sglang-tunableop:/root/.tunableop
# Without that mount, every container start re-tunes from scratch. Set
# PYTORCH_TUNABLEOP_ENABLED=1 to opt in.
ENV HIP_FORCE_DEV_KERNARG=1
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
# Default OFF. Measured on gfx1151: TunableOp tunes each new GEMM signature for
# ~20 s synchronously inside the first serving request, the result only
# persists on graceful exit (a SIGKILLed server loses it), and the untuned
# hipBLASLt default is within 2x of tuned on the shapes that matter here
# (3.14 ms vs 1.94 ms on a speculative draft's largest projection) while the
# tuning stall and per-call overhead cost far more -- disabling it took a
# DFLASH decode workload from ~6 to ~17 tok/s. Opt in for offline pre-tuning
# only; freeze with PYTORCH_TUNABLEOP_TUNING=0 afterwards.
ENV PYTORCH_TUNABLEOP_ENABLED=0
ENV PYTORCH_TUNABLEOP_FILENAME=/root/.tunableop/tunableop_results.csv
RUN mkdir -p /root/.tunableop

COPY docker/patches/sglang_launch.sh /usr/local/bin/sglang_launch
RUN chmod +x /usr/local/bin/sglang_launch

WORKDIR /sgl-workspace/sglang
CMD ["/bin/bash"]
