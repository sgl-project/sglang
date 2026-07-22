"""TRT-LLM-gen fused MoE (SiTU) compiled through the sglang JIT system.

Builds the trtllm-gen fused-MoE host/runner sources from the NVIDIA x
Moonshot private FlashInfer snapshot ("SDK") with sglang's own tvm-ffi
`load_jit`, so the kernel is available inside the sglang process without
importing the fork's ``flashinfer`` python package (the box serves with
flashinfer 0.6.15; the fork snapshot is 0.6.13rc1 and the two cannot be
imported side by side).

The SDK directory (fork checkout: ``csrc/``, ``include/``, ``3rdparty/``,
``local_cubins/<pool>/``) and the SiTU cubin pool stay on-box as artifacts —
INTERNAL COLLABORATION material, never committed here — and are located via
``SGLANG_TRTLLM_GEN_MOE_SDK`` (and optionally
``SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL``). This module vendors only glue:

  * header staging: the pool ships the batched-gemm ABI headers flat; they
    are copied into a content-addressed include tree shaped like
    ``flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/`` (mirrors
    ``gen_trtllm_gen_fused_moe_sm100_module(is_private=True)``);
  * JIT build of the 12 launcher/runner/routing sources with the private
    ABI defines (``TLLM_GEN_LOCAL_CUBINS_ABI`` etc.);
  * the ctypes cubin-loader callback (the .so asks for cubins by absolute
    path + sha256; we read them from the pool);
  * a thin ``trtllm_fp4_block_scale_moe`` wrapper (FromLogits routing,
    ``do_finalize=True``); kernel tile config ("tactic") defaults to the
    runner's built-in heuristic — pass an explicit one for tuned setups.

Validated for the Kimi K3 decode/prefill MoE regime: MxFP4 weights with
bf16 (w4a16) or MxFP8 (w4a8) activations, ``ActivationType.Situ`` (SiTuGlu:
``a*tanh(g/a)*sigmoid(g) * b*tanh(u/b)``), DeepSeekV3/noaux_tc routing.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import pathlib
import shutil
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    override_jit_cuda_arch,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# ActivationType / RoutingMethodType values from the fork's tllm_enums
# (kept as plain ints here so this module has no fork import).
ACTIVATION_SITU = 9
ROUTING_DEEPSEEK_V3 = 2
_ROUTING_TOPK = 5
_ROUTING_INPUT_FROM_LOGITS = 0
# NOTE: the enum VALUES start at 0; the "Mode 1/2/3" wording in upstream
# comments is documentation numbering, not the enum value.
_ROUTING_INPUT_PACKED = 1

# Batched-gemm ABI headers shipped flat in the cubin pool; the launcher
# includes them as flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/<h>.
_BMM_EXPORT_HEADERS = [
    "BatchedGemmEnums.h",
    "BatchedGemmInterface.h",
    "BatchedGemmOptions.h",
    "Enums.h",
    "GemmGatedActOptions.h",
    "GemmOptions.h",
    "KernelParams.h",
    "KernelParamsDecl.h",
    "KernelTraits.h",
    "TmaDescriptor.h",
    "trtllm/gen/CommonUtils.h",
    "trtllm/gen/CudaArchDecl.h",
    "trtllm/gen/CudaKernelLauncher.h",
    "trtllm/gen/DtypeDecl.h",
    "trtllm/gen/MmaDecl.h",
    "trtllm/gen/SfLayoutDecl.h",
    "trtllm/gen/SparsityDecl.h",
]

_SOURCES = [
    "csrc/nv_internal/cpp/kernels/quantization.cu",
    "csrc/nv_internal/cpp/common/envUtils.cpp",
    "csrc/nv_internal/cpp/common/logger.cpp",
    "csrc/nv_internal/cpp/common/stringUtils.cpp",
    "csrc/nv_internal/cpp/common/tllmException.cpp",
    "csrc/nv_internal/cpp/common/memoryUtils.cu",
    "csrc/trtllm_fused_moe_kernel_launcher.cu",
    "csrc/trtllm_fused_moe_runner.cu",
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_deepseek.cu",
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_llama4.cu",
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu",
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu",
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu",
    "csrc/trtllm_batched_gemm_runner.cu",
]


logger = logging.getLogger(__name__)

# Formal on-box install location, searched when SGLANG_TRTLLM_GEN_MOE_SDK is
# unset. The pre-formalization ad-hoc location (/scratch/nv_work/...) is still
# honored as a fallback but is deprecated.
_SDK_DEFAULT_DIR = "/opt/sglang/trtllm_gen_moe_sdk"
_SDK_LEGACY_DIR = "/scratch/nv_work/trtllmgen_MOE"
_warned_legacy_sdk_dir = False


def _warn_legacy_sdk_dir(path: pathlib.Path) -> None:
    global _warned_legacy_sdk_dir
    if _warned_legacy_sdk_dir:
        return
    _warned_legacy_sdk_dir = True
    logger.warning(
        "trtllm-gen MoE SDK resolved from the deprecated location %s; move it "
        "to %s (or point SGLANG_TRTLLM_GEN_MOE_SDK at it). The legacy "
        "nv_work fallback will be removed.",
        path,
        _SDK_DEFAULT_DIR,
    )


def _sdk_dir_if_valid(p: str) -> Optional[pathlib.Path]:
    path = pathlib.Path(p)
    return path if (path / "csrc").is_dir() else None


def sdk_dir() -> Optional[pathlib.Path]:
    p = envs.SGLANG_TRTLLM_GEN_MOE_SDK.get()
    if p:
        path = _sdk_dir_if_valid(p)
        if path is not None and "/nv_work/" in str(path.resolve()):
            _warn_legacy_sdk_dir(path)
        return path
    path = _sdk_dir_if_valid(_SDK_DEFAULT_DIR)
    if path is not None:
        return path
    path = _sdk_dir_if_valid(_SDK_LEGACY_DIR)
    if path is not None:
        _warn_legacy_sdk_dir(path)
    return path


def cubin_pool_dir() -> Optional[pathlib.Path]:
    p = envs.SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL.get()
    if p:
        pool = pathlib.Path(p)
        return pool if pool.is_dir() else None
    sdk = sdk_dir()
    if sdk is None:
        return None
    pools = sorted((sdk / "local_cubins").glob("*/"))
    return pools[-1] if pools else None


def available() -> bool:
    pool = cubin_pool_dir()
    return (
        sdk_dir() is not None
        and pool is not None
        and (pool / "flashinferMetaInfo.h").is_file()
        and (pool / "local").is_dir()
    )


def _stage_headers(pool: pathlib.Path) -> pathlib.Path:
    """Copy the pool's ABI headers into a content-addressed include tree."""
    meta = (pool / "flashinferMetaInfo.h").read_bytes()
    tag = hashlib.sha256(meta).hexdigest()[:12]
    cache = pathlib.Path(
        os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")
    ).expanduser()
    root = cache / "trtllm_gen_moe_headers" / tag
    dest = root / "flashinfer" / "trtllm" / "batched_gemm" / "trtllmGen_bmm_export"
    stamp = root / ".staged"
    if not stamp.is_file():
        for name in _BMM_EXPORT_HEADERS:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pool / name, target)
        shutil.copyfile(pool / "flashinferMetaInfo.h", dest / "flashinferMetaInfo.h")
        stamp.touch()
    return root


def _cuda_include_dir() -> str:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not home:
        nvcc = shutil.which("nvcc")
        home = str(pathlib.Path(nvcc).parent.parent) if nvcc else "/usr/local/cuda"
    return str(pathlib.Path(home) / "include")


_CUBIN_CB_KEEPALIVE = {}


def _setup_cubin_loader(so_path: str, pool_local: pathlib.Path) -> None:
    """Register the ctypes callback the .so uses to fetch cubins by name.

    The runner requests ``<TLLM_GEN_GEMM_CUBIN_PATH>/<kernel>`` (absolute,
    because the pool path is baked in at compile time); we read the bytes
    and hand them back via FlashInferSetCurrentCubin.
    """
    if so_path in _CUBIN_CB_KEEPALIVE:
        return
    lib = ctypes.CDLL(so_path)
    cb_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)

    def _get_cubin(name: bytes, sha256: bytes) -> None:
        rel = name.decode()
        path = pathlib.Path(rel)
        if not path.is_absolute():
            path = pool_local / rel
        if path.suffix != ".cubin":
            path = path.with_name(path.name + ".cubin")
        data = path.read_bytes()
        want = sha256.decode()
        if want:
            got = hashlib.sha256(data).hexdigest()
            if got != want:
                raise RuntimeError(
                    f"cubin sha mismatch for {path}: want {want} got {got}"
                )
        lib.FlashInferSetCurrentCubin(
            ctypes.cast(ctypes.create_string_buffer(data, len(data)), ctypes.c_char_p),
            ctypes.c_int(len(data)),
        )

    cb = cb_type(_get_cubin)
    _CUBIN_CB_KEEPALIVE[so_path] = (lib, cb)
    lib.FlashInferSetCubinCallback(cb)


@cache_once
def _jit_trtllm_gen_moe_module() -> Module:
    sdk = sdk_dir()
    pool = cubin_pool_dir()
    if sdk is None or pool is None:
        raise RuntimeError(
            "trtllm-gen MoE SDK not found: set SGLANG_TRTLLM_GEN_MOE_SDK to the "
            "private FlashInfer snapshot checkout (internal collaboration "
            "artifact; ships csrc/, include/ and local_cubins/)."
        )
    staged = _stage_headers(pool)
    meta_tag = staged.name
    cubin_path = str((pool / "local").resolve())

    cache = pathlib.Path(
        os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")
    ).expanduser()
    # Flags are not part of load_jit's source hash: fold the pool identity
    # (meta hash + path) into the module marker so a pool change rebuilds.
    path_tag = hashlib.sha256(cubin_path.encode()).hexdigest()[:8]
    build_dir = cache / f"sgl_trtllm_gen_moe_{meta_tag}_{path_tag}"

    cpp_files = [str(sdk / s) for s in _SOURCES if s.endswith(".cpp")]
    cuda_files = [str(sdk / s) for s in _SOURCES if s.endswith(".cu")]
    # quantization.cu emits fp4 cvt instructions (.e2m1x2) that need the
    # arch-specific feature set: compile for sm_XXXa, not plain sm_XXX.
    # The trtllm-gen cubins themselves are prebuilt (sm100f) and loaded at
    # runtime, unaffected by this flag.
    arch = get_jit_cuda_arch()
    with override_jit_cuda_arch(arch.major, arch.minor, "a"):
        module = load_jit(
            "trtllm_gen_moe",
            meta_tag,
            path_tag,
            cpp_files=cpp_files,
            cuda_files=cuda_files,
            header_only=False,  # the launcher exports its own tvm-ffi functions
            extra_cflags=["-fvisibility=hidden"],
            extra_cuda_cflags=[
                "-DTLLM_GEN_EXPORT_INTERFACE",
                "-DTLLM_GEN_EXPORT_FLASHINFER",
                "-DTLLM_ENABLE_CUDA",
                "-DENABLE_BF16",
                "-DENABLE_FP8",
                "-DENABLE_FP4",
                "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
                "-DTLLM_GEN_LOCAL_CUBINS_ABI",
                "-DFLASHINFER_PRIVATE_MOE_FFI_NAMES",
                "-DFLASHINFER_PRIVATE_MOE_LEAN_ROUTING",
                f'-DTLLM_GEN_GEMM_CUBIN_PATH=\\"{cubin_path}\\"',
                "-Xcompiler=-fvisibility=hidden",
            ],
            extra_ldflags=["-lcuda", "-lnvrtc"],
            extra_include_paths=[
                str(staged),
                str(
                    staged
                    / "flashinfer"
                    / "trtllm"
                    / "batched_gemm"
                    / "trtllmGen_bmm_export"
                ),
                str(sdk / "include"),
                str(sdk / "csrc"),
                str(sdk / "csrc" / "nv_internal"),
                str(sdk / "csrc" / "nv_internal" / "include"),
                str(sdk / "3rdparty" / "cutlass" / "include"),
                # Host .cpp files (g++) need the CUDA headers explicitly; nvcc
                # adds them implicitly for .cu. CUDA 13's bundled CCCL is used
                # as-is (mixing the fork's pinned 3rdparty/cccl with the
                # toolkit's explodes).
                _cuda_include_dir(),
            ],
            build_directory=str(build_dir),
        )
    so_files = sorted(build_dir.glob("*.so"))
    if not so_files:
        raise RuntimeError(f"no built .so under {build_dir}")
    _setup_cubin_loader(str(so_files[-1]), pool / "local")
    return module


def trtllm_fp4_block_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = ROUTING_DEEPSEEK_V3,
    activation_type: int = ACTIVATION_SITU,
    norm_topk_prob: bool = True,
    tactic: Sequence[int] = (-1, -1),
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP4 block-scale MoE with routing from logits and finalize fused.

    ``hidden_states``: bf16 ``[T, hidden]`` (w4a16) or MxFP8-packed uint8
    with ``hidden_states_scale`` (w4a8). Weights are trtllm-gen shuffled
    MxFP4 (uint8 packed, fp8 block scales, MajorK). ``tactic`` is the
    (gemm1, gemm2) config index pair; ``(-1, -1)`` = runner heuristic.
    """
    module = _jit_trtllm_gen_moe_module()
    # The FFI launcher reads these as dense row-major; a strided slice
    # (e.g. a fused-GEMM split) would silently mis-route.
    routing_logits = routing_logits.contiguous()
    hidden_states = hidden_states.contiguous()
    num_tokens = routing_logits.shape[0]
    hidden_size = hidden_states.shape[-1]
    if hidden_states.dtype == torch.uint8:
        hidden_size *= 2
    device = hidden_states.device
    topk_ids = torch.empty(num_tokens, top_k, dtype=torch.int32, device=device)
    topk_weights = torch.empty(
        num_tokens, top_k, dtype=routing_logits.dtype, device=device
    )
    if output is None:
        output = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )
    module.trtllm_fp4_block_scale_moe_private(
        _ROUTING_INPUT_FROM_LOGITS,
        routing_logits,
        topk_ids,
        topk_weights,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        None,  # gemm1_bias
        gemm1_alpha,
        gemm1_beta,
        None,  # gemm1_clamp_limit
        gemm2_weights,
        gemm2_weights_scale,
        None,  # gemm2_bias
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        None,  # per_token_scale
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        0,  # local_expert_offset
        num_experts,  # local_num_experts
        routed_scaling_factor,
        routing_method_type,
        True,  # do_finalize
        True,  # enable_pdl
        activation_type,
        output,
        list(tactic),
        norm_topk_prob,
        None,  # routing_replay_out
    )
    return output


def trtllm_fp4_block_scale_routed_moe(
    packed_topk_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    activation_type: int = ACTIVATION_SITU,
    tactic: Sequence[int] = (-1, -1),
    output: Optional[torch.Tensor] = None,
    do_finalize: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP4 block-scale MoE with PRECOMPUTED routing (PackedPrecomputed).

    ``packed_topk_ids``: int32 ``[T, top_k]`` with ``(expert_id << 16) |
    bf16-weight-bits`` (PackTopkIds layout) — selection and weights come
    from the caller's router, the in-op routing kernels are skipped. This
    is the fast path at small T, where the in-op single-CTA routing kernel
    (~22 µs at 896 experts) costs more than an external radix router.

    ``do_finalize=False`` skips the in-op finalize (top-k weighted
    unpermute) and returns its inputs instead:
    ``(gemm2_output [padded_rows, hidden] bf16 in permuted layout,
    topk_weights [T, top_k] bf16 unpacked from packed_topk_ids,
    expanded_idx_to_permuted_idx [T*top_k] int32 with -1 = dropped slot)``.
    ``output`` is left unwritten in that mode.
    """
    module = _jit_trtllm_gen_moe_module()
    hidden_states = hidden_states.contiguous()
    num_tokens = packed_topk_ids.shape[0]
    hidden_size = hidden_states.shape[-1]
    if hidden_states.dtype == torch.uint8:
        hidden_size *= 2
    device = hidden_states.device
    # Mode 2 unpacks the weights in-kernel; this is its output buffer.
    topk_weights = torch.empty(num_tokens, top_k, dtype=torch.bfloat16, device=device)
    if output is None:
        output = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )
    result = module.trtllm_fp4_block_scale_moe_private(
        _ROUTING_INPUT_PACKED,
        None,  # routing_logits
        packed_topk_ids.contiguous(),
        topk_weights,
        None,  # routing_bias (already applied by the external router)
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        None,  # gemm1_bias
        gemm1_alpha,
        gemm1_beta,
        None,  # gemm1_clamp_limit
        gemm2_weights,
        gemm2_weights_scale,
        None,  # gemm2_bias
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        None,  # per_token_scale
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        0,  # local_expert_offset
        num_experts,  # local_num_experts
        1.0,  # routed_scaling_factor (already applied by the router)
        _ROUTING_TOPK,  # routing_method_type (unused for precomputed)
        do_finalize,
        True,  # enable_pdl
        activation_type,
        output,
        list(tactic),
        True,  # norm_topk_prob (unused for precomputed)
        None,  # routing_replay_out
    )
    if do_finalize:
        return output
    # Deferred: [gemm2_output, expert_weights (None in packed mode — the
    # weights live in the topk_weights buffer mode 2 unpacked into),
    # expanded_idx_to_permuted_idx]. Index access — iterating the tvm-ffi
    # Array yields one-shot dlpack capsules.
    return result[0], topk_weights, result[2]
