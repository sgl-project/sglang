"""Shape-specialized Inkling MoE gate top-k + renorm JIT kernels.

Three families, all specialized for the Inkling gate layout (logits
``[tokens, 258]`` fp32 = 256 routed + 2 shared experts, top-6 selection by
``sigmoid(logit) + bias``, logsigmoid renorm over selected ++ shared):

- ``inkling_gate_topk_renorm``      -- v1 warp-per-row gate (int64 indices).
- ``inkling_gate_topk_renorm_v2``   -- v2 gate: wide vector loads, int32
  indices, optional PDL, in-register raw-logit carry (no re-gather).
- ``inkling_gate_gemv`` / ``inkling_gate_gemv_fused`` -- expert-per-block GEMV
  of the gate linear (x [tokens, 6144] bf16 @ W [264, 6144] bf16 -> fp32
  logits), standalone or with the gate epilogue fused into the same launch
  (last finishing block runs it; ticket+workspace are cached per device).

NOTE: the fused/gemv wrappers cache CUDA buffers and JIT-compile on first use;
run them eagerly once before CUDA-graph capture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_LOGITS_PAD = 264  # fp32 logits row pitch shared with the padded gate GEMM
_HIDDEN = 6144
_TOPK = 6
_N_SHARED = 2
_FUSED_MAX_TOKENS = 64


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "inkling_gate_topk_renorm",
        "fast_math",
        cuda_files=["moe/inkling_gate_topk_renorm.cuh"],
        cuda_wrappers=[
            ("inkling_gate_topk_renorm", "inkling_gate_topk_renorm"),
            ("inkling_gate_topk_renorm_packed", "inkling_gate_topk_renorm_packed"),
            ("inkling_gate_topk_renorm_v2", "inkling_gate_topk_renorm_v2"),
            (
                "inkling_gate_topk_renorm_v2_packed",
                "inkling_gate_topk_renorm_v2_packed",
            ),
            ("inkling_gate_gemv", "inkling_gate_gemv"),
            ("inkling_gate_gemv_fused", "inkling_gate_gemv_fused"),
            ("inkling_gate_gemv_fused_packed", "inkling_gate_gemv_fused_packed"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def _launch_inkling_gate_topk_renorm(
    logits: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    routed_w: torch.Tensor,
    shared_w: torch.Tensor,
    indices: torch.Tensor,
    route_scale: float,
) -> None:
    module = _jit_module()
    module.inkling_gate_topk_renorm(
        logits, bias, global_scale, routed_w, shared_w, indices, float(route_scale)
    )


def _check_gate_inputs(
    logits: torch.Tensor, bias: torch.Tensor, global_scale: torch.Tensor
) -> None:
    assert logits.is_cuda and logits.dtype == torch.float32 and logits.dim() == 2
    assert logits.shape[1] == 258 and logits.stride(1) == 1
    assert bias.is_cuda and bias.dtype == torch.float32 and bias.shape == (256,)
    assert global_scale.is_cuda and global_scale.dtype == torch.float32
    assert global_scale.numel() == 1


def inkling_gate_topk_renorm(
    logits: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    route_scale: float,
    *,
    return_packed: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]
):
    """Select top-6 routed experts from 256 and renorm with 2 shared experts.

    This is specialized for the Inkling fused gate layout:
    ``logits`` is ``[tokens, 258]`` fp32, where columns ``0:256`` are routed
    experts and columns ``256:258`` are shared experts. The top-k selection key is
    ``sigmoid(logits[:, :256]) + bias``; renorm is over sigmoid(raw logits) for
    the selected routed experts plus both shared experts.

    ``return_packed=True`` emits the FlashInfer routed-MoE pack instead of the
    routed_w + indices pair: ``packed[t,6]`` int32 = ``(expert_id << 16) | bf16
    weight bits``. Returns ``(packed, shared_w)``.
    """
    _check_gate_inputs(logits, bias, global_scale)

    tokens = logits.shape[0]
    shared_w = torch.empty(
        (tokens, _N_SHARED), dtype=torch.float32, device=logits.device
    )
    if return_packed:
        packed = torch.empty((tokens, _TOPK), dtype=torch.int32, device=logits.device)
        if tokens == 0:
            return packed, shared_w
        _jit_module().inkling_gate_topk_renorm_packed(
            logits,
            bias.contiguous(),
            global_scale.contiguous(),
            packed,
            shared_w,
            float(route_scale),
        )
        return packed, shared_w

    routed_w = torch.empty((tokens, _TOPK), dtype=torch.float32, device=logits.device)
    indices = torch.empty((tokens, _TOPK), dtype=torch.int64, device=logits.device)
    if tokens == 0:
        return routed_w, shared_w, indices

    _launch_inkling_gate_topk_renorm(
        logits,
        bias.contiguous(),
        global_scale.contiguous(),
        routed_w,
        shared_w,
        indices,
        route_scale,
    )
    return routed_w, shared_w, indices


def inkling_gate_topk_renorm_v2(
    logits: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    route_scale: float,
    *,
    return_packed: bool = False,
    enable_pdl: bool = False,
    warps_per_block: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """v2 gate kernel; same math as v1 but int32 indices and optional PDL.

    Returns ``(routed_w, indices, shared_w, packed)`` where the unused half is
    ``None`` depending on ``return_packed`` -- mirroring the triton
    ``sigmoid_gate_topk_renorm`` contract. ``warps_per_block`` in
    ``{0 (auto), 1, 2, 4, 8}`` selects the launch shape.

    Requires 32B-aligned logits rows: the production ``[tokens, 264]``-padded
    GEMM output sliced to ``[:, :258]`` qualifies.
    """
    _check_gate_inputs(logits, bias, global_scale)
    assert logits.stride(0) % 8 == 0, f"rows must be 32B-aligned: {logits.stride()=}"

    tokens = logits.shape[0]
    shared_w = torch.empty(
        (tokens, _N_SHARED), dtype=torch.float32, device=logits.device
    )
    if return_packed:
        packed = torch.empty((tokens, _TOPK), dtype=torch.int32, device=logits.device)
        if tokens > 0:
            _jit_module().inkling_gate_topk_renorm_v2_packed(
                logits,
                bias.contiguous(),
                global_scale.contiguous(),
                packed,
                shared_w,
                float(route_scale),
                bool(enable_pdl),
                int(warps_per_block),
            )
        return None, None, shared_w, packed

    routed_w = torch.empty((tokens, _TOPK), dtype=torch.float32, device=logits.device)
    indices = torch.empty((tokens, _TOPK), dtype=torch.int32, device=logits.device)
    if tokens > 0:
        _jit_module().inkling_gate_topk_renorm_v2(
            logits,
            bias.contiguous(),
            global_scale.contiguous(),
            routed_w,
            shared_w,
            indices,
            float(route_scale),
            bool(enable_pdl),
            int(warps_per_block),
        )
    return routed_w, indices, shared_w, None


def _check_gemv_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2
    assert x.shape[1] == _HIDDEN and x.stride(1) == 1 and x.stride(0) == _HIDDEN
    assert weight.is_cuda and weight.dtype == torch.bfloat16 and weight.dim() == 2
    assert weight.shape[0] >= 258 and weight.shape[1] == _HIDDEN
    assert weight.stride(1) == 1 and weight.stride(0) == _HIDDEN


def inkling_gate_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    enable_pdl: bool = False,
    experts_per_block: int = 0,
) -> torch.Tensor:
    """Gate linear as an expert-per-block GEMV: returns fp32 logits [tokens, 258].

    Drop-in for ``inkling_fused_gate_linear_with_fp32_out`` (the returned view
    shares the same padded [tokens, 264] layout). Meant for small token counts
    where the PDL split pair (this + v2 gate) beats cublas + gate.
    """
    _check_gemv_inputs(x, weight)
    tokens = x.shape[0]
    logits = torch.empty((tokens, _LOGITS_PAD), dtype=torch.float32, device=x.device)
    if tokens > 0:
        _jit_module().inkling_gate_gemv(
            x, weight, logits, bool(enable_pdl), int(experts_per_block)
        )
    return logits[:, :258]


# Per-device (workspace [64, 264] fp32, ticket int32[1]) reused by every fused
# call. The kernel resets the ticket to zero on completion, so the buffers are
# CUDA-graph replay-safe; allocate them eagerly (warmup) before graph capture.
_fused_scratch: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_fused_scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = device.index if device.index is not None else torch.cuda.current_device()
    scratch = _fused_scratch.get(key)
    if scratch is None:
        # Allocating inside CUDA graph capture would place the persistent
        # buffers in the capture pool, where other graphs' replays can reuse
        # (clobber) them. Call ensure_gate_gemv_fused_scratch() eagerly first
        # (InklingGate.__init__ does).
        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "fused gate scratch must be allocated before CUDA graph capture"
        workspace = torch.empty(
            (_FUSED_MAX_TOKENS, _LOGITS_PAD), dtype=torch.float32, device=device
        )
        ticket = torch.zeros((1,), dtype=torch.int32, device=device)
        scratch = (workspace, ticket)
        _fused_scratch[key] = scratch
    return scratch


def ensure_gate_gemv_fused_scratch(device: torch.device) -> None:
    """Eagerly allocate the fused-gate workspace/ticket (call at model init,
    before any CUDA graph capture)."""
    _get_fused_scratch(device)


def inkling_gate_gemv_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    route_scale: float,
    *,
    return_packed: bool = False,
    enable_pdl: bool = False,
    experts_per_block: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """Fully fused Inkling gate: GEMV + sigmoid+bias top-6 + renorm, one launch.

    ``x`` is ``[tokens, 6144]`` bf16 (tokens <= 64), ``weight`` the padded
    ``[264, 6144]`` bf16 gate weight. Output contract matches
    ``sigmoid_gate_topk_renorm``: ``(routed_w, indices, shared_w, packed)``.
    """
    _check_gemv_inputs(x, weight)
    tokens = x.shape[0]
    assert tokens <= _FUSED_MAX_TOKENS, f"fused gate supports <= 64 tokens: {tokens=}"
    assert bias.is_cuda and bias.dtype == torch.float32 and bias.shape == (256,)
    assert global_scale.is_cuda and global_scale.dtype == torch.float32

    workspace, ticket = _get_fused_scratch(x.device)
    shared_w = torch.empty((tokens, _N_SHARED), dtype=torch.float32, device=x.device)
    if return_packed:
        packed = torch.empty((tokens, _TOPK), dtype=torch.int32, device=x.device)
        if tokens > 0:
            _jit_module().inkling_gate_gemv_fused_packed(
                x,
                weight,
                bias.contiguous(),
                global_scale.contiguous(),
                workspace,
                ticket,
                packed,
                shared_w,
                float(route_scale),
                bool(enable_pdl),
                int(experts_per_block),
            )
        return None, None, shared_w, packed

    routed_w = torch.empty((tokens, _TOPK), dtype=torch.float32, device=x.device)
    indices = torch.empty((tokens, _TOPK), dtype=torch.int32, device=x.device)
    if tokens > 0:
        _jit_module().inkling_gate_gemv_fused(
            x,
            weight,
            bias.contiguous(),
            global_scale.contiguous(),
            workspace,
            ticket,
            routed_w,
            shared_w,
            indices,
            float(route_scale),
            bool(enable_pdl),
            int(experts_per_block),
        )
    return routed_w, indices, shared_w, None
