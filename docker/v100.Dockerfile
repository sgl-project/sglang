# syntax=docker/dockerfile:1.7

# sglang-sxm2 image for NVIDIA V100 (Volta, SM70). Serves GLM-5.3-Flash,
# Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash and MiniMax-H3 through the same
# scripts as a host install; see docker/v100-compose.yaml for the usage.
# Expensive native builds are separate layers, so a later validation failure
# never invalidates completed FlashInfer, sglang-kernel, or Marlin compilation.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
    CUDAHOSTCXX=/usr/bin/g++-12 \
    TORCH_CUDA_ARCH_LIST=7.0 \
    NVCC_THREADS=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.cargo/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl ffmpeg git g++-12 g++-14 libnuma-dev \
      ninja-build patch pkg-config protobuf-compiler python3.12 python3.12-dev \
      python3-pip python3-venv \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && python -m venv "$VIRTUAL_ENV"

FROM base AS builder

ARG MAX_JOBS
ENV MAX_JOBS=${MAX_JOBS}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal

WORKDIR /opt/sglang
COPY scripts/v100_safe_jobs.sh /usr/local/bin/v100-safe-jobs
RUN chmod +x /usr/local/bin/v100-safe-jobs

# Install the validated V100 dependency set without resolution. Upstream's
# pyproject pins target CUDA 13 and cannot resolve against torch 2.9.1 / cu128.
COPY requirements.txt /tmp/requirements-v100.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade pip \
    && python -m pip install --no-deps -r /tmp/requirements-v100.txt

# Patched FlashInfer SM70 source. This is editable because its JIT headers and
# Python sources are both needed at runtime.
COPY patches/flashinfer-sm70.patch \
      /opt/sglang/patches/flashinfer-sm70.patch
RUN git clone https://github.com/haohervchb/flashinfer.git \
      /opt/deps/flashinfer-sm70 \
    && git -C /opt/deps/flashinfer-sm70 checkout --detach \
      c3c40a7b90b792fc59f90f8f55c9e2de9c1b6833 \
    && git -C /opt/deps/flashinfer-sm70 apply \
      /opt/sglang/patches/flashinfer-sm70.patch
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip uninstall -y flashinfer-python flashinfer-cubin || true \
    && python -m pip install --no-deps --no-build-isolation \
      -e /opt/deps/flashinfer-sm70

# TurboMind is the one temporarily retained external SM70 component. Fetch
# only its attributed source directories; never install the 1Cat vLLM or
# attention packages and never place a full 1Cat checkout in the image.
RUN git clone --filter=blob:none --sparse --no-checkout \
      https://github.com/1CatAI/1Cat-vLLM.git \
      /opt/deps/turbomind-sm70-source \
    && git -C /opt/deps/turbomind-sm70-source sparse-checkout set \
      LICENSE csrc/core csrc/sm70_turbomind csrc/moe \
    && git -C /opt/deps/turbomind-sm70-source checkout --detach \
      6ada86ed64af6d1a7b3cb0f34df237fd86f06d48
RUN git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git \
      /opt/deps/cutlass-turbomind \
    && git -C /opt/deps/cutlass-turbomind checkout --detach \
      da5e086dab31d63815acafdac9a9c5893b1c69e2

# Build SGLang's adapter for the attributed TurboMind W8A16 block-FP8 and
# FP16 MoE source against the pinned CUTLASS revision used for host validation.
COPY scripts/build_sm70_turbomind.py /opt/sglang/scripts/build_sm70_turbomind.py
COPY python/sglang/kernels/jit/csrc/sm70_turbomind_bindings.cpp \
     /opt/sglang/python/sglang/kernels/jit/csrc/sm70_turbomind_bindings.cpp
COPY python/sglang/kernels/jit/csrc/sm70_fp16_moe_gemm.cu \
     /opt/sglang/python/sglang/kernels/jit/csrc/sm70_fp16_moe_gemm.cu
COPY python/sglang/kernels/jit/csrc/sm70_fp8_e5m2_cache.cu \
     /opt/sglang/python/sglang/kernels/jit/csrc/sm70_fp8_e5m2_cache.cu
RUN --mount=type=cache,target=/root/.cache/torch_extensions,sharing=locked \
    export MAX_JOBS="$(v100-safe-jobs)" \
    && export SGLANG_TURBOMIND_SM70_ROOT=/opt/deps/turbomind-sm70-source \
    && export SGLANG_TURBOMIND_CUTLASS_ROOT=/opt/deps/cutlass-turbomind \
    && python /opt/sglang/scripts/build_sm70_turbomind.py

# Only this source tree invalidates the sglang-kernel layer. The context ignores
# all local .so/build outputs, preventing the stale-binary bug from the host.
COPY python/sglang/kernels/aot /opt/sglang/python/sglang/kernels/aot
RUN python -m pip uninstall -y sglang-kernel || true
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    find /opt/sglang/python/sglang/kernels/aot/python/sgl_kernel \
      -type f -name 'common_ops*.so' -delete \
    && JOBS="$(v100-safe-jobs)" \
    && export MAX_JOBS="${JOBS}" \
    && export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" \
    && export CMAKE_ARGS="-DSGL_KERNEL_V100_ONLY=ON -DSGL_KERNEL_COMPILE_THREADS=1" \
    && python -m pip install --no-deps --no-build-isolation \
      /opt/sglang/python/sglang/kernels/aot

# Marlin uses the dedicated marlin_v100 repository and the exact proven local
# compatibility/tuning patches. Its smoke test is deferred to the next layer.
RUN git clone --depth 1 --branch v4.2.1 \
      https://github.com/NVIDIA/cutlass.git /opt/cutlass
COPY scripts/setup_v100_marlin.sh /opt/sglang/scripts/setup_v100_marlin.sh
COPY patches/marlin-v100-qwen-sm70-tuning.patch \
      /opt/sglang/patches/marlin-v100-qwen-sm70-tuning.patch
COPY patches/marlin-v100-qwen38-nvfp4-tuning.patch \
      /opt/sglang/patches/marlin-v100-qwen38-nvfp4-tuning.patch
RUN --mount=type=cache,target=/opt/deps/marlin-v100,sharing=locked \
    export CUTLASS_DIR=/opt/cutlass \
    && export MARLIN_V100_REPO=/opt/deps/marlin-v100 \
    && export MARLIN_V100_REF=6d72a49939701d26b15b617a4cd2423174adb2d1 \
    && export MARLIN_V100_INSTALL_DIR=/opt/v100-artifacts \
    && export MARLIN_V100_SKIP_SMOKE=1 \
    && export MARLIN_V100_SKIP_BF16_COMPAT=1 \
    && export MAX_JOBS="$(v100-safe-jobs)" \
    && bash /opt/sglang/scripts/setup_v100_marlin.sh

# Application changes do not invalidate any native compilation layer.
COPY python /opt/sglang/python
# The Rust radix tree needs torch >= 2.11 and is optional (the Python tree is
# the default backend), so build only the other extensions.
ENV SGLANG_BUILD_RUST_EXTS=grpc,multimodal,server
RUN --mount=type=bind,source=rust,target=/mnt/rust,ro \
    --mount=type=bind,source=proto,target=/mnt/proto,ro \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    cp -a /mnt/rust /opt/sglang/rust \
    && cp -a /mnt/proto /opt/sglang/proto \
    && python -m pip install --no-deps --no-build-isolation -e /opt/sglang/python \
    && python -m pip install cuda-tile==1.5.0 \
    && cp /opt/v100-artifacts/_sm70_marlin_v100_*.abi3.so \
      /opt/sglang/python/sglang/kernels/prebuilt/

# GPU-independent artifact validation is intentionally after every expensive
# compilation RUN, so BuildKit retains those layers if this check ever changes.
RUN python - <<'PY'
import importlib.util
from pathlib import Path
import subprocess

site = Path("/opt/venv/lib/python3.12/site-packages")
common_ops = list((site / "sgl_kernel" / "sm70").glob("common_ops*.so"))
assert len(common_ops) == 1, common_ops
assert common_ops[0].name == "common_ops.abi3.so", common_ops[0]
marlin_dir = Path("/opt/sglang/python/sglang/kernels/prebuilt")
marlin = [
    marlin_dir / "_sm70_marlin_v100_dense.abi3.so",
    marlin_dir / "_sm70_marlin_v100_moe.abi3.so",
]
turbomind = marlin_dir / "_sm70_turbomind_v100.so"
# The runtime stage has no rust/ workspace, so the loader uses these bundled
# extension modules and never invokes Cargo.
pkg = Path("/opt/sglang/python")
for module in (
    "sglang.srt.rust_extensions._server",
    "sglang.srt.rust_extensions._grpc",
    "sglang.srt.rust_extensions._multimodal",
):
    *parent, name = module.split(".")
    found = list(pkg.joinpath(*parent).glob(f"{name}*.so"))
    assert len(found) == 1, (module, found)
assert Path("/opt/deps/flashinfer-sm70/flashinfer/sampling.py").is_file()
assert Path(
    "/opt/sglang/python/sglang/srt/layers/attention/"
    "tilelang_fa_v100/_kernels_paged_decode.py"
).is_file()
for module in ("flash_attn_v100", "flash_attn_v100_cuda", "flash_qla_v100"):
    assert importlib.util.find_spec(module) is None, module

def validate_sm70(binary, required_strings):
    binary = Path(binary)
    assert binary.stat().st_size > 100_000, binary
    cubins = subprocess.check_output(
        ["cuobjdump", "--list-elf", str(binary)], text=True
    )
    assert ".sm_70.cubin" in cubins, (binary, cubins)
    assert all(f".sm_{arch}." not in cubins for arch in (75, 80, 86, 89, 90, 100))
    strings = subprocess.check_output(["strings", str(binary)], text=True)
    for value in required_strings:
        assert value in strings, (binary, value)

validate_sm70(common_ops[0], ["all_reduce", "causal_conv1d_fwd"])
validate_sm70(marlin[0], ["marlin_gemm"])
validate_sm70(marlin[1], ["moe_wna16_marlin_gemm"])
validate_sm70(turbomind, ["fp8_gemm", "f16_moe_gemm", "fp8_e5m2_cache_write"])
print("SM70 build artifacts validated")
PY

FROM base AS runtime

# Model-specific settings live in the serve scripts, as on a host install.
ENV HF_HOME=/root/.cache/huggingface \
    FLASHINFER_WORKSPACE_BASE=/root/sglang-v100-jit \
    TILELANG_CACHE_DIR=/root/sglang-v100-jit/tilelang \
    TRITON_CACHE_DIR=/root/sglang-v100-jit/triton \
    TORCHINDUCTOR_CACHE_DIR=/root/sglang-v100-jit/torchinductor \
    SGLANG_V100_PYTHON=/opt/venv/bin/python \
    SGLANG_V100_VENV=/opt/venv

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/deps/flashinfer-sm70 /opt/deps/flashinfer-sm70
COPY --from=builder /opt/sglang/python /opt/sglang/python
COPY scripts/smoke_v100.sh \
     scripts/serve_glm53_flash_nvfp4_v100.sh \
     scripts/serve_qwen38_flash_next_nvfp4_v100.sh \
     scripts/serve_dsv41_v100.sh \
     scripts/dsv41_flash_cold_set_ep8.json \
     scripts/serve_minimax_h3_v100.sh \
     /opt/sglang/scripts/
COPY docker/v100-entrypoint.sh /usr/local/bin/v100-entrypoint
RUN chmod +x /opt/sglang/scripts/*.sh /usr/local/bin/v100-entrypoint

WORKDIR /opt/sglang
# Language models on 11435, MiniMax-H3 on 30010.
EXPOSE 11435 30010

ENTRYPOINT ["/usr/local/bin/v100-entrypoint"]
CMD ["glm53"]
