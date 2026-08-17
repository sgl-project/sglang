"""Production shared-rank finalizers for the BF16 MoE row domains.

The materialized fallback stays on :meth:`MoeBaseProvider.finalize`.  This
module supplies the promoted Step-6 candidate:

``shared_rank_reduce``
    Valid only when down-B is shared across all routed experts of an adapter.
    It folds router weights in rank space, then one from-scratch finalizer
    combines weighted base rows with the shared B tail and applies routed
    scaling to the complete sum.

Both kernels are row-domain-agnostic and serve the masked AND contiguous
providers verbatim: the rank reduction is pure pair-domain (it reads the
canonical pair-major down-A bridge and never touches a physical row), and
the tail reads base down rows exclusively through ``src2dst`` over a flat
row view — the same lever that makes ``post_reorder`` and the down-B
scatter portable.  Only the tail's host validation distinguishes the
masked ``[E_local, m_max, hidden]`` slab from the contiguous compact
``[m_pad_ceiling, hidden]`` buffer.

The family is deterministic: reductions use a fixed order and every
destination cell has exactly one writer.

The fused B tail accumulates in FP32 and is combined with the BF16 base rows
before the requested output cast. This is the mathematical
``scale * sum(weight * (base + A @ B))`` contract; it intentionally omits the
serial reference's intermediate BF16 delta-materialization rounding.
Correctness is therefore judged against the independent numerical contract
rather than bitwise equality with the materialized staging path.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.routing import RouteView

MASKED_FINALIZE_TRITON = "triton"

SHARED_RANK_DEFAULT_CONFIG: dict[str, dict[str, int]] = {
    "reduce": {
        "BLOCK_SIZE_T": 32,
        "num_warps": 4,
        "num_stages": 2,
    },
    "tail": {
        "BLOCK_SIZE_H": 128,
        "BLOCK_SIZE_K": 32,
        "num_warps": 4,
        "num_stages": 3,
    },
}

_OUTPUT_DTYPES = (torch.bfloat16, torch.float32)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _validate_output_boundary(
    *,
    down_masked: torch.Tensor,
    src2dst: torch.Tensor,
    routing: RouteView,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    num_local_experts: int,
) -> tuple[int, int, int]:
    num_tokens, top_k = routing.topk_ids.shape
    pairs = num_tokens * top_k
    # The tail kernel addresses base rows only as src2dst[pair] over the flat
    # row view, so both physical row domains are valid inputs: the masked
    # [E_local, m_max, hidden] slab (rows e * m_max + slot) and the
    # contiguous compact [rows, hidden] buffer (rows seg_offsets[e] + slot).
    # The compact row count is a device-side quantity (seg_offsets[-1]), so
    # only the masked slab admits a host-side row-domain check.
    if down_masked.ndim == 3:
        if down_masked.shape[0] != num_local_experts or down_masked.shape[2] < 1:
            raise ValueError("down_masked must be [num_local_experts, m_max, hidden]")
    elif down_masked.ndim != 2 or down_masked.shape[1] < 1:
        raise ValueError(
            "base down rows must be [num_local_experts, m_max, hidden] or "
            "flat [rows, hidden]"
        )
    hidden = down_masked.shape[-1]
    if src2dst.dtype != torch.int32 or src2dst.numel() != pairs:
        raise ValueError(f"src2dst must be int32 with {pairs} entries")
    if topk_weights.shape != (num_tokens, top_k):
        raise ValueError(
            f"topk_weights must be {(num_tokens, top_k)}, got "
            f"{tuple(topk_weights.shape)}"
        )
    if topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must remain FP32 until the exactly-once finalize")
    if output.shape != (num_tokens, hidden):
        raise ValueError(
            f"output must be {(num_tokens, hidden)}, got {tuple(output.shape)}"
        )
    if output.dtype not in _OUTPUT_DTYPES:
        raise TypeError(f"output dtype must be one of {_OUTPUT_DTYPES}")
    if down_masked.dtype != torch.bfloat16:
        raise TypeError("the shared-rank BF16 finalizer requires BF16 base down rows")
    tensors = (
        down_masked,
        src2dst,
        routing.topk_ids,
        routing.token_slots,
        topk_weights,
        output,
    )
    if len({item.device for item in tensors}) != 1:
        raise ValueError("masked-finalize tensors must share one device")
    return num_tokens, top_k, hidden


def _validate_config(
    config: Mapping[str, int],
    *,
    block_name: str,
    needs_k: bool,
) -> tuple[int, int, int, int]:
    required = {block_name, "num_warps", "num_stages"}
    if needs_k:
        required.add("BLOCK_SIZE_K")
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"finalizer config is missing {missing}")
    block = int(config[block_name])
    block_k = int(config.get("BLOCK_SIZE_K", 16))
    warps = int(config["num_warps"])
    stages = int(config["num_stages"])
    if min(block, block_k, warps, stages) < 1:
        raise ValueError("finalizer launch parameters must be positive")
    if needs_k and block_k < 16:
        raise ValueError("BLOCK_SIZE_K must be at least 16")
    if not _is_power_of_two(block) or not _is_power_of_two(block_k):
        raise ValueError("Triton finalizer block sizes must be powers of two")
    return block, block_k, warps, stages


def _validate_shared_route(routing: RouteView) -> None:
    if routing.shared_outer_local_expert_count is None:
        raise ValueError(
            "shared_rank_reduce requires a shared-outer route; applying one "
            "B after rank reduction is invalid for per-expert weights"
        )
    if routing.lora_experts_per_adapter != 1:
        raise ValueError("shared rank reduction requires one LoRA expert per adapter")


@triton.jit
def _shared_rank_reduce_kernel(
    bridge_ptr,
    token_rank_ptr,
    weights_ptr,
    topk_ids_ptr,
    token_slots_ptr,
    num_tokens,
    stride_xm,
    stride_xk,
    stride_tm,
    stride_tk,
    stride_wm,
    stride_wk,
    rank: tl.constexpr,
    top_k: tl.constexpr,
    max_loras: tl.constexpr,
    local_expert_count: tl.constexpr,
    block_t: tl.constexpr,
    block_r: tl.constexpr,
):
    # Provenance: finalize_candidates.py::_shared_rank_reduce_kernel /
    # invoke_shared_rank_reduce. The fixed-k FP32 reduction and BLOCK_SIZE_T
    # semantics are retained. Routed scaling is deliberately deferred to the
    # B tail so every finalizer implements scale * sum(weight * pair).
    tokens = tl.program_id(0) * block_t + tl.arange(0, block_t)
    token_mask = tokens < num_tokens
    tokens64 = tokens.to(tl.int64)
    rank_offsets = tl.arange(0, block_r).to(tl.int64)
    rank_mask = rank_offsets < rank
    adapter = tl.load(token_slots_ptr + tokens, mask=token_mask, other=-1)
    adapter_valid = (adapter >= 0) & (adapter < max_loras)
    acc = tl.zeros((block_t, block_r), tl.float32)
    for k in range(top_k):
        pairs = tokens * top_k + k
        expert = tl.load(topk_ids_ptr + pairs, mask=token_mask, other=-1)
        valid = (
            token_mask & adapter_valid & (expert >= 0) & (expert < local_expert_count)
        )
        weight = tl.load(
            weights_ptr + tokens64 * stride_wm + k * stride_wk,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            bridge_ptr
            + pairs.to(tl.int64)[:, None] * stride_xm
            + rank_offsets[None, :] * stride_xk,
            mask=valid[:, None] & rank_mask[None, :],
            other=0.0,
        )
        acc += weight[:, None] * x.to(tl.float32)
    tl.store(
        token_rank_ptr
        + tokens64[:, None] * stride_tm
        + rank_offsets[None, :] * stride_tk,
        acc.to(token_rank_ptr.dtype.element_ty),
        mask=token_mask[:, None] & rank_mask[None, :],
    )


@triton.jit
def _shared_from_scratch_finalize_kernel(
    down_ptr,
    src2dst_ptr,
    token_rank_ptr,
    b_ptr,
    output_ptr,
    weights_ptr,
    topk_ids_ptr,
    token_slots_ptr,
    stride_dm,
    stride_dh,
    stride_tm,
    stride_tk,
    stride_bg,
    stride_bh,
    stride_bk,
    stride_om,
    stride_oh,
    stride_wm,
    stride_wk,
    num_tokens,
    routed_scaling,
    num_local_experts: tl.constexpr,
    hidden: tl.constexpr,
    rank: tl.constexpr,
    top_k: tl.constexpr,
    max_loras: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
):
    """Finalize unscaled weighted base/rank sums, then apply routed scaling.

    Scaling after the complete base+LoRA sum matches the stock DeepGEMM
    coefficient order while avoiding the post-reorder + full-H
    read-modify-write sequence.
    """
    token = tl.program_id(0)
    pid_h = tl.program_id(1)
    token64 = token.to(tl.int64)
    h_offsets = pid_h.to(tl.int64) * block_h + tl.arange(0, block_h).to(tl.int64)
    h_mask = h_offsets < hidden
    base_acc = tl.zeros((block_h,), tl.float32)
    for k in range(top_k):
        pair = token * top_k + k
        expert = tl.load(topk_ids_ptr + pair, mask=token < num_tokens, other=-1)
        valid = (expert >= 0) & (expert < num_local_experts)
        dst = tl.load(src2dst_ptr + pair, mask=valid, other=0).to(tl.int64)
        base = tl.load(
            down_ptr + dst * stride_dm + h_offsets * stride_dh,
            mask=valid & h_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weights_ptr + token64 * stride_wm + k * stride_wk,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        base_acc += weight * base

    adapter = tl.load(token_slots_ptr + token, mask=token < num_tokens, other=-1)
    adapter_valid = (adapter >= 0) & (adapter < max_loras)
    safe_adapter = tl.maximum(adapter, 0).to(tl.int64)
    delta = tl.zeros((block_h,), tl.float32)
    for k_begin in range(0, rank, block_k):
        rank_offsets = k_begin + tl.arange(0, block_k).to(tl.int64)
        rank_mask = rank_offsets < rank
        x = tl.load(
            token_rank_ptr + token64 * stride_tm + rank_offsets * stride_tk,
            mask=adapter_valid & rank_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + safe_adapter * stride_bg
            + h_offsets[:, None] * stride_bh
            + rank_offsets[None, :] * stride_bk,
            mask=adapter_valid & h_mask[:, None] & rank_mask[None, :],
            other=0.0,
        )
        delta += tl.sum(b.to(tl.float32) * x[None, :].to(tl.float32), axis=1)

    tl.store(
        output_ptr + token64 * stride_om + h_offsets * stride_oh,
        (routed_scaling * (base_acc + tl.where(adapter_valid, delta, 0.0))).to(
            output_ptr.dtype.element_ty
        ),
        mask=(token < num_tokens) & h_mask,
    )


def invoke_shared_rank_reduce(
    *,
    bridge: torch.Tensor,
    routing: RouteView,
    topk_weights: torch.Tensor,
    routed_scaling_factor: float | None,
    token_rank: torch.Tensor,
    config: Mapping[str, int],
) -> None:
    """Reduce unscaled pair-rank rows while base W2 runs independently."""
    # The provider ABI carries the forward's scaling factor through both
    # scheduled halves so alternate implementations can bind the same inputs.
    # This implementation deliberately applies it in the tail, after the
    # fixed-order router-weight reduction.
    del routed_scaling_factor
    _validate_shared_route(routing)
    num_tokens, top_k = routing.topk_ids.shape
    if bridge.ndim != 2 or bridge.shape[0] != num_tokens * top_k:
        raise ValueError(f"bridge must have {num_tokens * top_k} pair rows")
    rank = bridge.shape[1]
    if rank < 1:
        raise ValueError("shared bridge rank must be positive")
    if token_rank.shape != (num_tokens, rank):
        raise ValueError(
            f"token_rank must be {(num_tokens, rank)}, got {tuple(token_rank.shape)}"
        )
    if token_rank.dtype != bridge.dtype or bridge.dtype != torch.bfloat16:
        raise TypeError("shared rank reduction requires BF16 bridge/token_rank")
    if topk_weights.shape != routing.topk_ids.shape:
        raise ValueError("topk_weights must match route [T,K]")
    if topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must remain FP32 through rank reduction")
    if (
        len(
            {
                bridge.device,
                token_rank.device,
                topk_weights.device,
                routing.topk_ids.device,
            }
        )
        != 1
    ):
        raise ValueError("shared rank-reduce tensors must share one device")
    block_t, _, reduce_warps, reduce_stages = _validate_config(
        config, block_name="BLOCK_SIZE_T", needs_k=False
    )
    if num_tokens == 0:
        return
    block_r = max(16, triton.next_power_of_2(rank))
    if block_r > 256:
        raise ValueError(
            "shared rank-reduce production candidate is capped at rank 256"
        )
    _shared_rank_reduce_kernel[(triton.cdiv(num_tokens, block_t),)](
        bridge,
        token_rank,
        topk_weights,
        routing.topk_ids,
        routing.token_slots,
        num_tokens,
        bridge.stride(0),
        bridge.stride(1),
        token_rank.stride(0),
        token_rank.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        rank=rank,
        top_k=top_k,
        max_loras=routing.max_loras,
        local_expert_count=routing.shared_outer_local_expert_count,
        block_t=block_t,
        block_r=block_r,
        num_warps=reduce_warps,
        num_stages=reduce_stages,
    )


def invoke_shared_from_scratch_finalize(
    *,
    down_masked: torch.Tensor,
    src2dst: torch.Tensor,
    token_rank: torch.Tensor,
    b_down: torch.Tensor,
    routing: RouteView,
    topk_weights: torch.Tensor,
    routed_scaling_factor: float | None,
    output: torch.Tensor,
    num_local_experts: int,
    config: Mapping[str, int],
) -> None:
    """Write weighted base plus shared-B delta in one deterministic launch."""
    num_tokens, _, hidden = _validate_output_boundary(
        down_masked=down_masked,
        src2dst=src2dst,
        routing=routing,
        topk_weights=topk_weights,
        output=output,
        num_local_experts=num_local_experts,
    )
    _validate_shared_route(routing)
    if token_rank.ndim != 2 or token_rank.shape[0] != num_tokens:
        raise ValueError(f"token_rank must have {num_tokens} rows")
    rank = token_rank.shape[1]
    if b_down.shape != (routing.max_loras, hidden, rank):
        raise ValueError(
            f"shared b_down must be {(routing.max_loras, hidden, rank)}, got "
            f"{tuple(b_down.shape)}"
        )
    if token_rank.dtype != b_down.dtype or b_down.dtype != torch.bfloat16:
        raise TypeError("shared from-scratch finalize requires BF16 rank/B tensors")
    block_h, block_k, warps, stages = _validate_config(
        config, block_name="BLOCK_SIZE_H", needs_k=True
    )
    if num_tokens == 0:
        return
    _shared_from_scratch_finalize_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
        down_masked.view(-1, hidden),
        src2dst,
        token_rank,
        b_down,
        output,
        topk_weights,
        routing.topk_ids,
        routing.token_slots,
        down_masked.stride(-2),
        down_masked.stride(-1),
        token_rank.stride(0),
        token_rank.stride(1),
        b_down.stride(0),
        b_down.stride(1),
        b_down.stride(2),
        output.stride(0),
        output.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        num_tokens,
        1.0 if routed_scaling_factor is None else routed_scaling_factor,
        num_local_experts=num_local_experts,
        hidden=hidden,
        rank=rank,
        top_k=routing.topk_ids.shape[1],
        max_loras=routing.max_loras,
        block_h=block_h,
        block_k=block_k,
        num_warps=warps,
        num_stages=stages,
    )
