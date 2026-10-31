"""Fused K3 MoE-front prep: radix routing + trtllm id pack + mxfp8 quant.

One launch replaces the three tiny kernels between the K3 fused-front GEMM and
the trtllm-gen routed-MoE op (route_radix -> triton
(id<<16|bf16(w)) pack -> per_token_group_quant, ~7.5us busy + 2 extra launches
per MoE layer with the SMs near idle). CTAs [0, M) run the route_radix body
with the pack folded into its epilogue; CTAs [M, 2M) run the
per_token_group_quant math, one CTA per token row. Both halves reuse the
standalone kernels' device code, so ids/weights, packed ids, and quantized
activations are bit-identical to the unfused chain.

Specialized like route_radix itself: 896 experts, top-16, bf16/fp32 scores,
and a 3584-wide bf16 activation row quantized to fp8 with row-major packed
UE8M0 group-32 scales (the trtllm-gen SiTU MoE input format). Wired into
serving through sglang.srt.layers.moe.route_quant_handoff.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.moe import moe_route_radix
from sglang.srt.utils.common import get_device_sm

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HIDDEN = 3584
_MAX_TOKENS = 64  # route_radix covers up to 64 tokens per launch
_GROUP_SIZE = 32
_NUM_GROUPS = _HIDDEN // _GROUP_SIZE


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "moe_route_quant_fused",
        *args,
        cuda_files=["moe/route_quant_fused.cuh"],
        cuda_wrappers=[("run", f"RouteQuantFusedKernel<{args}>::run")],
        # No fast-math: the routing half must stay bit-identical to
        # route_radix (see its module comment); the quant half's math is
        # fast-math-independent (explicit intrinsics + bit manipulation).
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_metadata_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "moe_route_quant_metadata",
        *args,
        cuda_files=["moe/route_quant_metadata.cuh"],
        cuda_wrappers=[
            ("run", f"RouteQuantMetadataKernel<{args}, Mxfp8RowQuant>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_nvfp4_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "moe_route_quant_nvfp4",
        *args,
        cuda_files=["moe/route_quant_metadata.cuh"],
        cuda_wrappers=[
            ("run", f"RouteQuantMetadataKernel<{args}, Nvfp4RowQuant>::run")
        ],
        extra_cuda_cflags=["-O3", "-DENABLE_BF16", "-DENABLE_FP4"],
        extra_dependencies=["flashinfer", "flashinfer_nv_internal"],
    )


@cache_once
def metadata_available() -> bool:
    try:
        _jit_metadata_module()
        return True
    except Exception as error:
        logging.getLogger(__name__).warning(
            "Failed to load fused routing metadata; using ordinary routing: %s", error
        )
        return False


def allocate_metadata(device, num_tokens=1, weight_dtype=torch.bfloat16):
    # The native body reads only live permuted slots, so skip the memset.
    def empty(n):
        return torch.empty(n, device=device, dtype=torch.int32)

    routes = num_tokens * 16
    return [
        empty(1),
        empty(routes),
        empty(routes * 8 + 1),
        torch.empty((num_tokens, 16), device=device, dtype=weight_dtype),
        empty(1792),
        empty(896),
        empty(routes),
        empty(routes),
        empty(1),
    ]


_arrivals: dict[torch.device, torch.Tensor] = {}


def _arrival_counter(device) -> torch.Tensor:
    device = torch.device(device)
    if device not in _arrivals:
        _arrivals[device] = torch.zeros(1, device=device, dtype=torch.int32)
    return _arrivals[device]


@cache_once
def available() -> bool:
    try:
        _jit_module()
        return True
    except Exception as e:  # pragma: no cover - toolchain dependent
        logging.getLogger(__name__).warning(
            f"Failed to load the JIT fused route+quant kernel: {e}"
        )
        return False


def covered(
    scores: torch.Tensor, bias: torch.Tensor, topk: int, x: torch.Tensor
) -> bool:
    """route_radix coverage plus the quant half: [M<=64, 3584] bf16 rows with
    32B-aligned starts (base and stride).

    Activations disagreeing with the scores on token count is a caller bug, not
    an unsupported layout (the only staging site splits both from one GEMM
    output), so it is left to the launcher, which reports it as "scores and
    activations must have the same token count".
    """
    return (
        moe_route_radix.covered(scores, bias, topk)
        and x.dim() == 2
        and 0 < x.shape[0] <= _MAX_TOKENS
        and x.shape[1] == _HIDDEN
        and x.dtype in (torch.bfloat16, torch.float32)
        and x.stride(1) == 1
        and x.data_ptr() % 32 == 0
        and (x.stride(0) * x.element_size()) % 32 == 0
    )


def route_quant_fused(
    scores: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
    metadata=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(weights [M, topk] fp32, ids [M, topk] int32, packed [M, topk]
    int32, x_q [M, 3584] fp8_e4m3, x_s [M, 28] int32 row-major packed UE8M0)``.
    Caller must have checked covered(). Winners come out in expert-id-ascending
    order (the standalone production dispatch's sorted=False)."""
    M = scores.shape[0]
    device = scores.device
    out_w = torch.empty((M, topk), dtype=torch.float32, device=device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=device)
    out_packed = torch.empty((M, topk), dtype=torch.int32, device=device)
    out_q = torch.empty((M, _HIDDEN), dtype=torch.float8_e4m3fn, device=device)
    out_s = torch.empty((M, _NUM_GROUPS // 4), dtype=torch.int32, device=device)
    module = _jit_module() if metadata is None else _jit_metadata_module()
    metadata_args = (
        []
        if metadata is None
        else [metadata[i] for i in (0, 1, 2, 3, 6, 7, 8)]
        + [_arrival_counter(device), out_s]  # out_s: unused global-scale slot
    )
    module.run(
        scores,
        bias,
        out_w,
        out_i,
        out_packed,
        x,
        out_q,
        out_s,
        *metadata_args,
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
        bool(apply_scale),
    )
    return out_w, out_i, out_packed, out_q, out_s


def nvfp4_covered(scores, bias, topk, x, global_scale) -> bool:
    """covered() plus the SM107 NVFP4 quantizer: a one-element FP32 global
    scale and FlashInfer's default FP4 fast math."""
    return (
        covered(scores, bias, topk, x)
        and x.is_cuda
        and get_device_sm() == 107
        and scores.device == bias.device == global_scale.device == x.device
        and global_scale.numel() == 1
        and global_scale.dtype == torch.float32
        and global_scale.is_contiguous()
        and not any(
            os.getenv(name, "").strip().lower() not in {"", "0", "false", "off", "no"}
            for name in (
                "FLASHINFER_NVFP4_4OVER6",
                "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
                "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH",
            )
        )
    )


def route_quant_nvfp4(
    scores,
    bias,
    x,
    global_scale,
    topk,
    renormalize,
    routed_scaling_factor,
    apply_scale,
    metadata,
):
    """Returns ``(weights [M, topk] fp32, ids [M, topk] int32, x_q [M, 1792]
    packed FP4, x_s [M, 224] linear fp8_e4m3)`` and fills ``metadata``
    (allocate_metadata with FP32 weights) when given. Caller must have checked
    nvfp4_covered(); metadata is only produced for M<=8."""
    M = x.shape[0]
    weights = torch.empty((M, topk), device=x.device, dtype=torch.float32)
    ids = torch.empty((M, topk), device=x.device, dtype=torch.int32)
    x_q = torch.empty((M, _HIDDEN // 2), device=x.device, dtype=torch.uint8)
    x_s = torch.empty((M, _HIDDEN // 16), device=x.device, dtype=torch.uint8)
    _jit_nvfp4_module().run(
        scores,
        bias,
        weights,
        ids,
        torch.empty_like(ids),
        x,
        x_q,
        x_s,
        *(
            [metadata[i] for i in (0, 1, 2, 3, 6, 7, 8)]
            if metadata is not None
            else [ids] * 7
        ),
        _arrival_counter(x.device),
        global_scale.reshape(1),
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
        bool(apply_scale),
    )
    return weights, ids, x_q, x_s.view(torch.float8_e4m3fn)
