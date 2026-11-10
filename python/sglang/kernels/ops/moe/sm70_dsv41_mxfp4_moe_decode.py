"""SM70 MXFP4 MoE decode GEMV (DeepSeek-V4.1-Flash, M<=4).

Consumes marlin_v100 packed MXFP4 + logical UE8M0. Not NVFP4 decode.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import torch

logger = logging.getLogger(__name__)
_GEMV_LIVE_LOGGED = False
_GEMV_CAPTURE_LOGGED = False

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.marlin_utils import (
    DSV41_FLASH_HIDDEN_SIZE,
    DSV41_FLASH_MOE_INTERMEDIATE_SIZE,
    DSV41_FLASH_MXFP4_GROUP_SIZE,
    DSV41_FLASH_NUM_EXPERTS_PER_TOK,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_K_HIDDEN = DSV41_FLASH_HIDDEN_SIZE
_K_INTERMEDIATE = DSV41_FLASH_MOE_INTERMEDIATE_SIZE
_K_GATE_UP = 2 * _K_INTERMEDIATE
_K_TOPK = DSV41_FLASH_NUM_EXPERTS_PER_TOK
_K_GROUP = DSV41_FLASH_MXFP4_GROUP_SIZE
_K_MAX_M = 4
_K_SPLIT_K_GATE = 8
_K_SPLIT_K_DOWN = 8
_K_MAX_ROUTES = _K_MAX_M * _K_TOPK

_WS: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


@cache_once
def _module() -> Module:
    cap = torch.cuda.get_device_capability()
    if cap[0] != 7:
        raise RuntimeError(
            f"sm70_dsv41 mxfp4_moe_decode requires SM70 (Volta); got SM{cap[0]}{cap[1]}"
        )
    return load_jit(
        "sm70_dsv41_mxfp4_moe_decode",
        cuda_files=["sm70_dsv41_mxfp4_moe_decode.cuh"],
        cuda_wrappers=[("mxfp4_moe_decode", "sm70_dsv41::mxfp4_moe_decode")],
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
        extra_cuda_cflags=["-O3"],
    )


def sm70_dsv41_mxfp4_moe_decode_enabled() -> bool:
    return bool(envs.SGLANG_DSV41_MOE_GEMV.get())


def sm70_dsv41_mxfp4_moe_decode_available() -> bool:
    if not sm70_dsv41_mxfp4_moe_decode_enabled() or not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability()[0] == 7
    except (AssertionError, RuntimeError):
        return False


def sm70_dsv41_mxfp4_moe_decode_eligible(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    weight_bits: int = 4,
    w13_qzeros: Optional[torch.Tensor] = None,
    w2_qzeros: Optional[torch.Tensor] = None,
    w13_global_scale: Optional[torch.Tensor] = None,
    w2_global_scale: Optional[torch.Tensor] = None,
    w13_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    is_gated: bool = True,
    activation: str = "silu",
    gate_up_input_scale: float = 1.0,
    wide_output_scale: float = 1.0,
    gemm1_alpha: Optional[float] = None,
    gemm1_clamp_limit: Optional[float] = None,
    swiglu_limit: Optional[float] = None,
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
) -> bool:
    """True iff the DSV4.1 Flash decode GEMV can replace Marlin for this call."""
    if not sm70_dsv41_mxfp4_moe_decode_available():
        return False
    if hidden_states.dtype != torch.float16 or hidden_states.ndim != 2:
        return False
    m = int(hidden_states.shape[0])
    if m < 1 or m > _K_MAX_M or int(hidden_states.shape[1]) != _K_HIDDEN:
        return False
    if tuple(topk_ids.shape) != (m, _K_TOPK):
        return False
    if w13.ndim != 3 or w2.ndim != 3:
        return False
    e = int(w13.shape[0])
    if e < 1 or int(w2.shape[0]) != e:
        return False
    if tuple(w13.shape[1:]) != (_K_HIDDEN // 16, _K_GATE_UP * 2):
        return False
    if tuple(w2.shape[1:]) != (_K_INTERMEDIATE // 16, _K_HIDDEN * 2):
        return False
    if tuple(w13_scales.shape) != (e, _K_HIDDEN // _K_GROUP, _K_GATE_UP):
        return False
    if tuple(w2_scales.shape) != (e, _K_INTERMEDIATE // _K_GROUP, _K_HIDDEN):
        return False
    if w13_scales.dtype != torch.float8_e8m0fnu or w2_scales.dtype != torch.float8_e8m0fnu:
        return False
    if weight_bits != 4:
        return False
    if w13_qzeros is not None or w2_qzeros is not None:
        return False
    if w13_global_scale is not None or w2_global_scale is not None:
        return False
    if w13_bias is not None or w2_bias is not None:
        return False
    if not is_gated or activation != "silu":
        return False
    if apply_router_weight_on_input or no_combine:
        return False
    if gate_up_input_scale != 1.0 or wide_output_scale != 1.0:
        return False
    # GPT-OSS SwiGLU (alpha + clamp) is a different epilogue. DSV4.1-Flash
    # only sets swiglu_limit=10; that used to make this always False, so
    # decode graphs captured Marlin instead of the GEMV.
    if gemm1_alpha is not None or gemm1_clamp_limit is not None:
        return False
    if swiglu_limit is not None:
        try:
            lim = float(swiglu_limit)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(lim) or lim < 0.0:
            return False
    return True


def _workspace(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = device.index if device.index is not None else torch.cuda.current_device()
    ws = _WS.get(key)
    if ws is None:
        gate = torch.empty(
            (_K_SPLIT_K_GATE, _K_MAX_ROUTES, _K_GATE_UP),
            dtype=torch.float32,
            device=device,
        )
        activated = torch.empty(
            (_K_MAX_ROUTES, _K_INTERMEDIATE),
            dtype=torch.float16,
            device=device,
        )
        down = torch.empty(
            (_K_SPLIT_K_DOWN, _K_MAX_ROUTES, _K_HIDDEN),
            dtype=torch.float32,
            device=device,
        )
        ws = (gate, activated, down)
        _WS[key] = ws
    return ws


def sm70_dsv41_mxfp4_moe_decode(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor:
    """Run the decode GEMV. Caller must have checked ``eligible``."""
    global _GEMV_LIVE_LOGGED, _GEMV_CAPTURE_LOGGED
    capturing = False
    if hidden_states.is_cuda:
        try:
            capturing = bool(torch.cuda.is_current_stream_capturing())
        except RuntimeError:
            capturing = False
    if capturing and not _GEMV_CAPTURE_LOGGED:
        logger.info(
            "DSV4.1 decode MXFP4 GEMV live M=%s E=%s swiglu_limit=%s capturing=%s",
            int(hidden_states.shape[0]),
            int(w13.shape[0]),
            swiglu_limit,
            capturing,
        )
        _GEMV_LIVE_LOGGED = True
        _GEMV_CAPTURE_LOGGED = True
    elif not _GEMV_LIVE_LOGGED:
        logger.info(
            "DSV4.1 decode MXFP4 GEMV live M=%s E=%s swiglu_limit=%s capturing=%s",
            int(hidden_states.shape[0]),
            int(w13.shape[0]),
            swiglu_limit,
            capturing,
        )
        _GEMV_LIVE_LOGGED = True
    hidden_states = (
        hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    )
    topk_ids = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()
    s13 = w13_scales.view(torch.uint8)
    s2 = w2_scales.view(torch.uint8)
    if not s13.is_contiguous():
        s13 = s13.contiguous()
    if not s2.is_contiguous():
        s2 = s2.contiguous()
    gate, activated, down = _workspace(hidden_states.device)
    output = torch.empty_like(hidden_states)
    limit = 0.0 if swiglu_limit is None else float(swiglu_limit)
    _module().mxfp4_moe_decode(
        hidden_states,
        w13 if w13.is_contiguous() else w13.contiguous(),
        w2 if w2.is_contiguous() else w2.contiguous(),
        s13,
        s2,
        topk_ids,
        topk_weights,
        gate,
        activated,
        down,
        output,
        limit,
    )
    return output
