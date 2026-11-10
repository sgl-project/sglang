"""TurboMind W8A16 block-FP8 linear kernels for NVIDIA SM70."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from sglang.kernels.sm70_paths import sm70_csrc, sm70_prebuilt
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_DEFAULT_OPS_PATH = sm70_prebuilt("_sm70_turbomind_v100.so")
_OPS_LOAD_ATTEMPTED = False
_OPS_AVAILABLE = False

_PREFILL_BACKENDS = ("auto", "turbomind", "fp16")

# --- SM70 QPN8 decode backend (W8A16, M<=8, default on) ----------------
# Adapted from v100-skinny's QPN8 kernel for our block-wise [N/128][K/128]
# fp32 scale checkpoint format. Packed codes + per-K-block scales are built
# once at weight load; decode GEMMs (M<=8) route to the QPN8 kernel instead
# of the TurboMind fp8_gemm when enabled. The exact TP4 gate/up shape
# additionally fuses its paired projections with the SiLU multiply at decode.
_QPN8_SRC_PATH = sm70_csrc("sm70_fp8_qpn8_decode.cu")
_QPN8_EXT = None
_QPN8_EXACT_DENSE_SHAPES = {
    "down_proj": (5120, 4352),
    "gate_up_proj": (8704, 5120),
    "in_proj_qkvz": (4096, 5120),
    "out_proj": (5120, 1536),
    "o_proj": (5120, 1536),
}
_QPN8_SPLITK = 16
_QPN8_NACC = 1
_QPN8_FASTDEC_NACC_ARG = 3  # NACC=1 plus the extension's word-parallel decoder
_QPN8_FASTDEC_SUFFIXES = {"down_proj", "gate_up_proj", "in_proj_qkvz"}
_KORDER8 = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]

# Qwen3.8-27B-FP8 TP4 shapes admitted by 1Cat's real-weight, bitwise gates.
# The TurboMind tensors have already been converted to logical KxN here.
_SM70_FP8_PREFILL_DENSE_MIN_M = 3920
_SM70_FP8_PREFILL_DENSE_SHAPES = {
    "gate_up_proj": (5120, 8704),
    "down_proj": (4352, 5120),
    "out_proj": (1536, 5120),
    "o_proj": (1536, 5120),
}
_SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS = max(
    k * n for k, n in _SM70_FP8_PREFILL_DENSE_SHAPES.values()
)
_SM70_FP8_PREFILL_DENSE_WORKSPACE_BYTES = (
    _SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS * torch.float16.itemsize
)
# Layers deliberately retain only a CUDA address. This strong cache owns the
# allocation and keeps the 85 MiB buffer out of torch.compile/CUDA-graph inputs.
_SM70_FP8_PREFILL_DENSE_WORKSPACES: dict[tuple[int, torch.dtype], torch.Tensor] = {}
_SM70_FP8_PREFILL_DENSE_OOM_WARNED = False


def _get_sm70_fp8_prefill_backend() -> str:
    backend = envs.SGLANG_SM70_FP8_PREFILL_BACKEND.get().lower()
    if backend not in _PREFILL_BACKENDS:
        raise ValueError(
            "SGLANG_SM70_FP8_PREFILL_BACKEND must be auto, turbomind, or "
            f"fp16; got {backend!r}."
        )
    return backend


def _get_sm70_fp8_prefill_min_tokens() -> int:
    value = envs.SGLANG_SM70_FP8_PREFILL_MIN_TOKENS.get()
    if value <= 0:
        raise ValueError(
            f"SGLANG_SM70_FP8_PREFILL_MIN_TOKENS must be positive; got {value}."
        )
    return value


def _get_sm70_fp8_decode_qpn8_enabled() -> bool:
    # Default on for SM70 FP8 decode; set SGLANG_SM70_FP8_DECODE_QPN8=0 to
    # fall back to the TurboMind decode GEMM path.
    return envs.SGLANG_SM70_FP8_DECODE_QPN8.get()


def _get_sm70_fp8_qpn8_fastdec_enabled() -> bool:
    return envs.SGLANG_SM70_FP8_QPN8_FASTDEC.get()


def _get_sm70_fp8_qpn8_fused_gate_enabled() -> bool:
    return envs.SGLANG_SM70_FP8_QPN8_FUSED_GATE.get()


def _load_sm70_qpn8_ops():
    """Lazy-load the standalone SM70 QPN8 decode extension (JIT-built)."""
    global _QPN8_EXT
    if _QPN8_EXT is not None:
        return _QPN8_EXT
    if not _QPN8_SRC_PATH.is_file():
        raise RuntimeError(f"SM70 QPN8 decode source not found: {_QPN8_SRC_PATH}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("SM70 QPN8 decode requires an NVIDIA SM70 GPU.")
    from torch.utils.cpp_extension import load_inline

    build_directory = envs.SGLANG_SM70_QPN8_BUILD_DIR.get()
    os.makedirs(build_directory, exist_ok=True)
    _QPN8_EXT = load_inline(
        name="sglang_sm70_qpn8_v100",
        cpp_sources="",
        cuda_sources=_QPN8_SRC_PATH.read_text(),
        functions=None,
        is_python_module=True,
        verbose=False,
        build_directory=build_directory,
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
    )
    logger.info_once(
        "SM70 (V100): QPN8 decode extension loaded from %s.",
        _QPN8_SRC_PATH,
    )
    return _QPN8_EXT


def _qpn8_prepack(w8: torch.Tensor) -> torch.Tensor:
    """(N,K) uint8 -> [tile N/32][group K/16][lane 32][16B] contiguous codes."""
    n, k = w8.shape
    dev = w8.device
    tiles, groups = n // 32, k // 16
    lane = torch.arange(32, device=dev)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    korder = torch.tensor(_KORDER8, device=dev)
    g = torch.arange(groups, device=dev)
    kidx = g.view(groups, 1) * 16 + korder.view(1, 16)
    ncol = torch.arange(tiles, device=dev).view(tiles, 1) * 32 + col.view(1, 32)
    out = w8[
        ncol.view(tiles, 1, 32, 1).expand(tiles, groups, 32, 16),
        kidx.view(1, groups, 1, 16).expand(tiles, groups, 32, 16),
    ]
    return out.contiguous().view(-1)


def _qpn8_gscales(scale_inv: torch.Tensor, k: int) -> torch.Tensor:
    """[N/128][K/128] fp32 -> half[K/128][N/32], decoder's *256 folded."""
    nb, _ = scale_inv.shape
    gs = scale_inv.t().reshape(k // 128, nb) * 256.0
    gs = gs.repeat_interleave(4, dim=1).to(torch.float16).contiguous()
    return gs


def _is_sm70_fp8_qpn8_exact_dense_layer(layer: torch.nn.Module) -> bool:
    if getattr(layer, "tp_size", 1) != 4:
        return False
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    expected = _QPN8_EXACT_DENSE_SHAPES.get(suffix)
    return expected is not None and tuple(layer.weight.shape) == expected


def _maybe_prepare_sm70_fp8_qpn8(layer: torch.nn.Module) -> None:
    if not _get_sm70_fp8_decode_qpn8_enabled():
        return
    if not _is_sm70_fp8_qpn8_exact_dense_layer(layer):
        return
    weight = layer.weight.data
    scales = layer.weight_scale_inv.data.to(torch.float32).contiguous()
    n, k = weight.shape
    if n % 128 or k % 128:
        return
    _load_sm70_qpn8_ops()
    raw = weight.view(torch.uint8).contiguous()  # already [N,K]
    layer.sm70_fp8_qpn8_codes = _qpn8_prepack(raw)
    layer.sm70_fp8_qpn8_gscales = _qpn8_gscales(scales, k)
    layer.sm70_fp8_qpn8_shape = (k, n)
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    layer.sm70_fp8_qpn8_gated = suffix == "gate_up_proj"
    layer.sm70_fp8_qpn8_fused_gate = (
        layer.sm70_fp8_qpn8_gated and _get_sm70_fp8_qpn8_fused_gate_enabled()
    )
    layer.sm70_fp8_qpn8_nacc_arg = (
        _QPN8_FASTDEC_NACC_ARG if suffix in _QPN8_FASTDEC_SUFFIXES else _QPN8_NACC
    )
    if not _get_sm70_fp8_qpn8_fastdec_enabled():
        layer.sm70_fp8_qpn8_nacc_arg = _QPN8_NACC
    layer.sm70_fp8_qpn8_ready = True
    logger.info_once(
        "SM70 (V100): QPN8 decode backend prepared for %s K=%d N=%d.",
        getattr(layer, "prefix", "?"),
        k,
        n,
    )


def _is_sm70_fp8_prefill_exact_dense_layer(layer: torch.nn.Module) -> bool:
    if getattr(layer, "tp_size", 1) != 4:
        return False
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    expected = _SM70_FP8_PREFILL_DENSE_SHAPES.get(suffix)
    return expected is not None and tuple(layer.weight.shape) == expected


def _get_sm70_fp8_prefill_exact_dense_workspace(
    weight: torch.Tensor,
) -> Optional[torch.Tensor]:
    global _SM70_FP8_PREFILL_DENSE_OOM_WARNED
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, torch.float16)
    workspace = _SM70_FP8_PREFILL_DENSE_WORKSPACES.get(key)
    if workspace is not None:
        return workspace
    try:
        workspace = torch.empty(
            _SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS,
            dtype=torch.float16,
            device=weight.device,
        )
    except torch.OutOfMemoryError:
        if not _SM70_FP8_PREFILL_DENSE_OOM_WARNED:
            logger.warning(
                "Insufficient memory for the bounded SM70 FP8 prefill "
                "workspace; falling back to TurboMind W8A16."
            )
            _SM70_FP8_PREFILL_DENSE_OOM_WARNED = True
        return None
    _SM70_FP8_PREFILL_DENSE_WORKSPACES[key] = workspace
    return workspace


def _use_sm70_fp8_prefill_dispatch(layer: torch.nn.Module) -> bool:
    return getattr(
        layer, "sm70_fp8_prefill_exact_dense_workspace_ptr", 0
    ) != 0 and hasattr(torch.ops.sglang_sm70_turbomind, "fp8_prefill_dispatch")


def _load_sm70_turbomind_fp8_ops() -> bool:
    global _OPS_AVAILABLE, _OPS_LOAD_ATTEMPTED
    if _OPS_LOAD_ATTEMPTED:
        return _OPS_AVAILABLE
    _OPS_LOAD_ATTEMPTED = True

    ops_path = Path(envs.SGLANG_SM70_TURBOMIND_OPS_PATH.get() or _DEFAULT_OPS_PATH)
    try:
        torch.ops.load_library(str(ops_path))
        _OPS_AVAILABLE = hasattr(
            torch.ops.sglang_sm70_turbomind, "fp8_prepare"
        ) and hasattr(torch.ops.sglang_sm70_turbomind, "fp8_gemm")
        if not _OPS_AVAILABLE:
            raise RuntimeError("the required fp8_prepare/fp8_gemm operators are absent")
    except Exception as exc:
        logger.warning("SM70 TurboMind FP8 operators unavailable: %s", exc)
        _OPS_AVAILABLE = False
    return _OPS_AVAILABLE


def can_use_sm70_turbomind_fp8(
    weight_block_size: Optional[list[int]],
) -> bool:
    """Return whether the local SM70 W8A16 kernel covers this quantization."""
    backend = envs.SGLANG_SM70_FP8_BACKEND.get().lower()
    if backend not in ("auto", "turbomind", "marlin"):
        raise ValueError(
            "SGLANG_SM70_FP8_BACKEND must be auto, turbomind, or marlin; "
            f"got {backend!r}."
        )
    if backend == "marlin":
        return False
    if (
        weight_block_size != [128, 128]
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (7, 0)
    ):
        if backend == "turbomind":
            raise RuntimeError(
                "The SM70 TurboMind FP8 backend requires an NVIDIA SM70 GPU "
                "and block-wise FP8 weights with block_size=[128, 128]."
            )
        return False

    available = _load_sm70_turbomind_fp8_ops()
    if backend == "turbomind" and not available:
        raise RuntimeError(
            "SGLANG_SM70_FP8_BACKEND=turbomind was requested, but the "
            "TurboMind FP8 extension could not be loaded."
        )
    return available


def prepare_sm70_turbomind_fp8_linear(layer: torch.nn.Module) -> None:
    if layer.orig_dtype != torch.float16:
        raise RuntimeError(
            "SM70 TurboMind FP8 requires FP16 activations, "
            f"but the layer dtype is {layer.orig_dtype}."
        )
    if getattr(layer, "weight_block_size", None) != [128, 128]:
        raise RuntimeError("SM70 TurboMind FP8 requires weight_block_size=[128, 128].")

    # QPN8 decode codes must be captured from the raw checkpoint weight before
    # the TurboMind repack replaces it.
    _maybe_prepare_sm70_fp8_qpn8(layer)

    weight = layer.weight.data
    scales = layer.weight_scale_inv.data.to(torch.float32).contiguous()
    logical_widths = getattr(layer, "logical_widths", None)
    is_gated_silu = (
        getattr(layer, "prefix", "").rsplit(".", 1)[-1] == "gate_up_proj"
        and isinstance(logical_widths, list)
        and len(logical_widths) == 2
        and logical_widths[0] == logical_widths[1]
    )
    tm_weight, tm_scales, meta = torch.ops.sglang_sm70_turbomind.fp8_prepare(
        weight,
        scales,
        128,
        is_gated_silu,
    )
    layer.weight = Parameter(tm_weight, requires_grad=False)
    layer.weight_scale_inv = Parameter(tm_scales, requires_grad=False)
    layer.sm70_fp8_k_ld = int(meta[0].item())
    layer.sm70_fp8_q_ld = int(meta[1].item())
    layer.sm70_fp8_turbomind = True
    layer.sm70_fp8_gated_silu = is_gated_silu
    prefill_backend = _get_sm70_fp8_prefill_backend()
    exact_dense_enabled = envs.SGLANG_SM70_FP8_PREFILL_EXACT_DENSE.get()
    if (
        prefill_backend != "turbomind"
        and exact_dense_enabled
        and hasattr(torch.ops.sglang_sm70_turbomind, "fp8_prefill_dispatch")
        and _is_sm70_fp8_prefill_exact_dense_layer(layer)
    ):
        workspace = _get_sm70_fp8_prefill_exact_dense_workspace(layer.weight)
        if workspace is not None:
            layer.sm70_fp8_prefill_exact_dense_workspace_ptr = workspace.data_ptr()
            layer.sm70_fp8_prefill_min_tokens = _get_sm70_fp8_prefill_min_tokens()
    logger.info_once("SM70 (V100): using TurboMind W8A16 block-FP8 dense GEMM.")
    if is_gated_silu:
        logger.info_once(
            "SM70 (V100): using the TurboMind fused gate/up SiLU epilogue."
        )
    if _use_sm70_fp8_prefill_dispatch(layer):
        logger.info_once(
            "SM70 (V100): using the graph-safe shared 85 MiB exact-dense "
            "FP8 prefill workspace for admitted TP4 projections at M >= %d.",
            layer.sm70_fp8_prefill_min_tokens,
        )


def apply_sm70_turbomind_fp8_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1])
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    k, n = (
        layer.sm70_fp8_qpn8_shape
        if getattr(layer, "sm70_fp8_qpn8_ready", False)
        else (None, layer.output_size_per_partition)
    )
    m = x_2d.shape[0]
    if (
        getattr(layer, "sm70_fp8_qpn8_ready", False)
        and x_2d.dtype == torch.float16
        and 0 < m <= 8
        and x_2d.shape[-1] == k
    ):
        out_2d = _load_sm70_qpn8_ops().qpn8_linear(
            x_2d,
            layer.sm70_fp8_qpn8_codes,
            layer.sm70_fp8_qpn8_gscales,
            n,
            k,
            _QPN8_SPLITK,
            layer.sm70_fp8_qpn8_nacc_arg,
        )
    else:
        out_2d = torch.empty(
            (m, layer.output_size_per_partition),
            dtype=x.dtype,
            device=x.device,
        )
        if _use_sm70_fp8_prefill_dispatch(layer) and x_2d.dtype == torch.float16:
            torch.ops.sglang_sm70_turbomind.fp8_prefill_dispatch(
                out_2d,
                layer.sm70_fp8_prefill_exact_dense_workspace_ptr,
                x_2d,
                layer.weight,
                layer.weight_scale_inv,
                128,
                layer.sm70_fp8_k_ld,
                layer.sm70_fp8_q_ld,
                False,
                layer.sm70_fp8_prefill_min_tokens,
            )
        else:
            torch.ops.sglang_sm70_turbomind.fp8_gemm(
                out_2d,
                x_2d,
                layer.weight,
                layer.weight_scale_inv,
                128,
                layer.sm70_fp8_k_ld,
                layer.sm70_fp8_q_ld,
                False,
            )
    if getattr(layer, "sm70_fp8_gated_silu", False):
        # The packed weight is interleaved for the fused epilogue. Preserve the
        # ordinary linear contract when a caller does not request fusion.
        out_features = layer.output_size_per_partition // 2
        out_2d = (
            out_2d.reshape(x_2d.shape[0], out_features, 2)
            .transpose(1, 2)
            .reshape(x_2d.shape[0], layer.output_size_per_partition)
        )
    if bias is not None:
        out_2d.add_(bias)
    return out_2d.reshape(*x.shape[:-1], layer.output_size_per_partition)


def apply_sm70_turbomind_fp8_fused_silu_and_mul(
    layer: torch.nn.Module,
    x: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Apply a packed gate/up projection and SiLU multiplication in one GEMM."""
    if not getattr(layer, "sm70_fp8_gated_silu", False):
        return None

    x_2d = x.reshape(-1, x.shape[-1])
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    out_features = layer.output_size_per_partition // 2
    k, n = (
        layer.sm70_fp8_qpn8_shape
        if getattr(layer, "sm70_fp8_qpn8_gated", False)
        else (None, None)
    )
    m = x_2d.shape[0]
    if (
        getattr(layer, "sm70_fp8_qpn8_gated", False)
        and x_2d.dtype == torch.float16
        and 0 < m <= 8
        and x_2d.shape[-1] == k
    ):
        if getattr(layer, "sm70_fp8_qpn8_fused_gate", False):
            fused = _load_sm70_qpn8_ops().qpn8_gated_silu(
                x_2d,
                layer.sm70_fp8_qpn8_codes,
                layer.sm70_fp8_qpn8_gscales,
                out_features,
                k,
            )
            return fused.reshape(*x.shape[:-1], out_features)
        gate_up = _load_sm70_qpn8_ops().qpn8_linear(
            x_2d,
            layer.sm70_fp8_qpn8_codes,
            layer.sm70_fp8_qpn8_gscales,
            n,
            k,
            _QPN8_SPLITK,
            layer.sm70_fp8_qpn8_nacc_arg,
        )
        gate = gate_up[:, :out_features]
        up = gate_up[:, out_features:]
        fused = torch.nn.functional.silu(gate) * up
        return fused.reshape(*x.shape[:-1], out_features)
    out_2d = torch.empty(
        (x_2d.shape[0], out_features),
        dtype=x.dtype,
        device=x.device,
    )
    if _use_sm70_fp8_prefill_dispatch(layer) and x_2d.dtype == torch.float16:
        torch.ops.sglang_sm70_turbomind.fp8_prefill_dispatch(
            out_2d,
            layer.sm70_fp8_prefill_exact_dense_workspace_ptr,
            x_2d,
            layer.weight,
            layer.weight_scale_inv,
            128,
            layer.sm70_fp8_k_ld,
            layer.sm70_fp8_q_ld,
            True,
            layer.sm70_fp8_prefill_min_tokens,
        )
    else:
        torch.ops.sglang_sm70_turbomind.fp8_gemm(
            out_2d,
            x_2d,
            layer.weight,
            layer.weight_scale_inv,
            128,
            layer.sm70_fp8_k_ld,
            layer.sm70_fp8_q_ld,
            True,
        )
    return out_2d.reshape(*x.shape[:-1], out_features)
