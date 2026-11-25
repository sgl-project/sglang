"""Fused MoE routing for DeepSeek-V4.1-Flash on gfx950, behind SGLANG_OPT_HIP_FUSED_MOE_ROUTING=1.

For decode / target-verify MoE calls of 257-2048 rows (DSpark verify steps), AITER's fused routing kernel
(aiter.ops.flydsl.moe_route_sort_quant, two launches) replaces aiter topk_gating, the shared-expert append,
fused_moe's sort and its stage-1 MXFP8 quant + scale sort. Wired as a consume-once handoff:

    DeepseekV2MoE.forward_normal   stage(hidden_states) before self.topk, clear() after the experts call
    select_experts (sqrtsoftplus)  try_route(...) runs the kernel and returns the top-7 ids / weights (shared slot
                                   included), or None to fall back to the unfused chain
    aiter fused_moe                its decode sort+quant hook takes the kernel's sorted ids and MXFP8 input when the
                                   call's activations and topk_ids are the ones the kernel saw; else AITER's own path

Every step falls back: another model or shape, a miss in the hook, or the switch off leave today's chain untouched.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_ENABLED = envs.SGLANG_OPT_HIP_FUSED_MOE_ROUTING.get()
_MIN_ROWS = 257
_MAX_ROWS = 2048
_LOG_FIRST = 8  # log lines per event kind and process: proof the path ran, without flooding the server log
_state = {
    "x": None,
    "out": None,
    "tried": False,
    "decode": False,
    "router": None,
    "bias": {},
}


def _log_first(kind, msg, *args):
    _state[kind] = _state.get(kind, 0) + 1
    if _state[kind] <= _LOG_FIRST:
        logger.info(
            "hip_fused_moe_routing: " + msg + " (%s #%d)", *args, kind, _state[kind]
        )


def enabled() -> bool:
    return _ENABLED


def set_decode(flag: bool) -> None:
    """The MoE layer's batch is a decode or target-verify step (only those are routed, not prefill)."""
    _state["decode"] = flag


def stage(x: torch.Tensor) -> None:
    _state["x"] = x
    _state["out"] = None
    _state["tried"] = False


def note_skip(m: int, why) -> None:
    """A staged MoE call of m rows that the kernel will not route, with the failed conditions."""
    _state["tried"] = True
    if _MIN_ROWS <= m <= _MAX_ROWS:
        _log_first("skipped: " + ", ".join(why), "%d rows not routed", m)


def clear() -> None:
    x = _state["x"]
    if x is not None and not _state["tried"] and _MIN_ROWS <= x.shape[0] <= _MAX_ROWS:
        _log_first(
            "unreached",
            "select_experts did not reach the fused routing branch (%d rows)",
            x.shape[0],
        )
    if _state["out"] is not None:
        _log_first(
            "unused",
            "routing result not taken by fused_moe (%d rows); AITER sorted again",
            _state["out"]["topk_ids"].shape[0],
        )
    _state["x"] = None
    _state["out"] = None


def prepare(device) -> None:
    """Build the kernel's buffers and compile both sort-block variants eagerly (model init, not inside a graph
    capture): RouteA zeroes its workspace once, and FlyDSL compiles on a kernel's first launch."""
    if not _ENABLED or _state["router"] is not None:
        return
    from aiter.ops.flydsl.moe_route_sort_quant import RouteA

    r = RouteA(device)
    bias = torch.zeros(384, dtype=torch.float32, device=device)
    for m in (300, 1100):  # sort block 32 and 64
        r(
            logits=torch.zeros((m, 384), dtype=torch.float32, device=device),
            bias_f32=bias,
            x=torch.zeros((m, 5120), dtype=torch.bfloat16, device=device),
        )
    torch.cuda.synchronize(device)
    _state["router"] = r
    _install_aiter_hook()
    logger.info(
        "hip_fused_moe_routing: on for %d-%d rows; kernel compiled for sort blocks 32 and 64",
        _MIN_ROWS,
        _MAX_ROWS,
    )


def route_or_skip(*, skip_reasons, **route_kwargs):
    """try_route(**route_kwargs), unless one of the caller's (name, failed) conditions holds."""
    why = [name for name, bad in skip_reasons if bad]
    if why:
        note_skip(m=route_kwargs["hidden_states"].shape[0], why=why)
        return None
    return try_route(**route_kwargs)


def try_route(
    *,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    shared_w: float,
    num_token_non_padded: Optional[torch.Tensor],
):
    """(topk_weights [M, 7] f32, topk_ids [M, 7] i32) from the fused kernel, or None (caller runs the chain)."""
    m = hidden_states.shape[0]
    _state["tried"] = True
    if not (_ENABLED and _state["router"] is not None and _MIN_ROWS <= m <= _MAX_ROWS):
        return None
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )

    why = [
        name
        for name, bad in (
            ("not decode / verify", not _state["decode"]),
            ("breakable prefill graph", is_in_breakable_cuda_graph()),
            ("not staged", _state["x"] is not hidden_states),
            ("renormalize off", not renormalize),
            ("no correction bias", correction_bias is None),
            (
                "logits dtype",
                router_logits.dtype not in (torch.float32, torch.bfloat16),
            ),
            ("experts", router_logits.shape[1] != 384),
            ("x dtype", hidden_states.dtype != torch.bfloat16),
            ("hidden size", hidden_states.shape[1] != 5120),
            (
                "not contiguous",
                not (hidden_states.is_contiguous() and router_logits.is_contiguous()),
            ),
        )
        if bad
    ]
    if why:
        note_skip(m=m, why=why)
        return None
    # above 64 rows AITER's router GEMM returns bf16 logits; the kernel reads fp32 (bf16 -> fp32 is exact)
    if router_logits.dtype != torch.float32:
        router_logits = router_logits.float()
    capturing = torch.cuda.is_current_stream_capturing()
    # A tensor made inside a CUDA-graph capture is only filled when that graph replays, so the fp32 bias is cached
    # from eager calls only; inside a capture it is converted in the graph itself.
    key = (correction_bias.data_ptr(), correction_bias.dtype)
    bias = _state["bias"].get(key)
    if bias is None:
        bias = correction_bias.float().contiguous()
        if capturing:
            _log_first(
                "bias in capture",
                "fp32 bias converted inside the graph (layer bias %#x)",
                key[0],
            )
        else:
            _state["bias"][key] = bias
    r = _state["router"](
        logits=router_logits,
        bias_f32=bias,
        x=hidden_states,
        rsf=1.0 if routed_scaling_factor is None else routed_scaling_factor,
        shared_w=shared_w,
        num_token_non_padded=num_token_non_padded,
    )
    _state["out"] = r
    _log_first(
        "routed, graph capture" if capturing else "routed, eager",
        "fused routing of %d rows",
        m,
    )
    return r["topk_w"], r["topk_ids"]


def _install_aiter_hook():
    import aiter.fused_moe as fm

    orig_enabled, orig_sort_quant = (
        fm._is_fused_decode_sort_quant_enabled,
        fm._fused_decode_sort_quant,
    )

    def _hit(hidden_states, topk_ids):
        r = _state["out"]
        return (
            r is not None
            and _state["x"] is not None
            and hidden_states.data_ptr() == _state["x"].data_ptr()
            and topk_ids.data_ptr() == r["topk_ids"].data_ptr()
        )

    def enabled_hook(hidden_states, topk_ids, topk_weights, **kw):
        if (
            _hit(hidden_states, topk_ids)
            and kw.get("block_m") == _state["out"]["block_m"]
            and kw.get("expert_mask") is None
            and kw.get("num_local_tokens") is None
        ):
            return True
        return orig_enabled(hidden_states, topk_ids, topk_weights, **kw)

    def sort_quant_hook(
        hidden_states,
        topk_ids,
        topk_weights,
        *,
        model_dim,
        global_experts,
        block_m,
        moebuf_dtype,
        accumulate,
        compact_scale=False,
        num_local_tokens=None,
    ):
        if not _hit(hidden_states, topk_ids):
            return orig_sort_quant(
                hidden_states,
                topk_ids,
                topk_weights,
                model_dim=model_dim,
                global_experts=global_experts,
                block_m=block_m,
                moebuf_dtype=moebuf_dtype,
                accumulate=accumulate,
                compact_scale=compact_scale,
                num_local_tokens=num_local_tokens,
            )
        r = _state["out"]
        _state["out"] = None  # consume once
        _log_first(
            "fed",
            "fused_moe took the kernel's sorted ids and MXFP8 input (%d rows, block_m %d)",
            hidden_states.shape[0],
            block_m,
        )
        if (
            block_m != r["block_m"]
            or global_experts != 385
            or compact_scale
            or num_local_tokens is not None
        ):
            raise RuntimeError(
                f"hip_fused_moe_routing: fused_moe wants block_m {block_m}, kernel sorted for {r['block_m']}"
            )
        buf = (
            torch.zeros(
                (hidden_states.shape[0], model_dim),
                dtype=moebuf_dtype,
                device=hidden_states.device,
            )
            if accumulate
            else torch.empty((0, 0), dtype=moebuf_dtype, device=hidden_states.device)
        )
        return (
            r["sorted_ids"],
            r["sorted_w"],
            r["sorted_eids"],
            r["num_valid"],
            buf,
            r["a1"],
            r["a1_scale"],
        )

    fm._is_fused_decode_sort_quant_enabled = enabled_hook
    fm._fused_decode_sort_quant = sort_quant_hook
