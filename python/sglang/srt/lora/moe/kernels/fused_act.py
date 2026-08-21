"""One kernel that computes the gate/up LoRA-B product and the activation over
the masked MoE rows.

The base GEMM writes ``[E, m_max, slices * width]`` rows. LoRA uses the
original pair order. The kernel joins the two orders through ``src2dst``.

The activation always goes to the masked rows that the base down GEMM reads.
The pair-major copy is optional. It exists for a down-A family that cannot read
the masked rows through the pair-to-row map in the provider ABI. An invalid
routed pair gets exact zero in the pair output, and no masked row at all.

NUMERICS: the kernel adds LoRA-B to the BF16 base row in FP32, then applies the
activation. It computes ``activation(base + A @ B)``. The serial reference
rounds the delta to BF16 first. The kernel then rounds the activation itself
to BF16. That dtype is the provider ABI for down-A and the base down GEMM.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.activation import ActivationFn
from sglang.srt.lora.moe.kernels.activation_delta import (
    apply_activation,
)
from sglang.srt.lora.moe.route_view import RouteView

B_ACT_FAMILIES = ("b_activation",)
MASKED_ACT_TRITON = "triton"

FUSED_B_ACT_DEFAULT_CONFIG: dict[str, int] = {
    "BLOCK_SIZE_W": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 3,
}


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _require_config(config: Mapping[str, int]) -> tuple[int, int, int, int, int]:
    required = {
        "BLOCK_SIZE_W",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"fused-act config is missing {missing}")
    block_w = int(config["BLOCK_SIZE_W"])
    block_k = int(config["BLOCK_SIZE_K"])
    group_m = int(config["GROUP_SIZE_M"])
    num_warps = int(config["num_warps"])
    num_stages = int(config["num_stages"])
    if min(block_w, block_k) < 16:
        raise ValueError("Triton dot tiles BLOCK_SIZE_W/K must be at least 16")
    if not _is_power_of_two(block_w) or not _is_power_of_two(block_k):
        raise ValueError("Triton BLOCK_SIZE_W/K must be powers of two")
    if min(group_m, num_warps, num_stages) < 1:
        raise ValueError("GROUP_SIZE_M, num_warps, and num_stages must be positive")
    return block_w, block_k, group_m, num_warps, num_stages


def _validate_common(
    *,
    activation: str,
    base_gateup: torch.Tensor,
    act_masked: torch.Tensor,
    act_pairs: torch.Tensor | None,
    src2dst: torch.Tensor,
    routing: RouteView,
    num_local_experts: int,
) -> tuple[int, int, int]:
    ActivationFn.parse(activation)
    pairs = routing.topk_ids.numel()
    if src2dst.dtype != torch.int32 or src2dst.numel() != pairs:
        raise ValueError(f"src2dst must be int32 with {pairs} entries")
    if (
        act_masked.ndim != 3
        or act_masked.shape[0] != num_local_experts
        or act_masked.shape[2] < 1
    ):
        raise ValueError("act_masked must be [num_local_experts, m_max, intermediate]")
    width = act_masked.shape[2]
    slices = base_gateup.shape[-1] // width
    if slices not in (1, 2) or slices * width != base_gateup.shape[-1]:
        raise ValueError(
            f"base gate/up width {base_gateup.shape[-1]} is not 1x or 2x "
            f"activation width {width}"
        )
    if base_gateup.shape != (
        num_local_experts,
        act_masked.shape[1],
        slices * width,
    ):
        raise ValueError(
            "base_gateup must share masked [E,m] rows and carry "
            f"{slices * width} columns, got {tuple(base_gateup.shape)}"
        )
    if act_pairs is not None:
        if act_pairs.shape != (*routing.topk_ids.shape, width):
            raise ValueError(
                f"act_pairs must be {(*routing.topk_ids.shape, width)}, got "
                f"{tuple(act_pairs.shape)}"
            )
        if act_masked.dtype != act_pairs.dtype:
            raise TypeError(
                f"dual activation stores need one dtype, got "
                f"{act_masked.dtype} and {act_pairs.dtype}"
            )
    if base_gateup.dtype != torch.bfloat16 or act_masked.dtype != torch.bfloat16:
        raise TypeError("masked BF16 middle requires BF16 base and activation rows")
    tensors = (
        base_gateup,
        act_masked,
        src2dst,
        routing.topk_ids,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
    )
    if act_pairs is not None:
        tensors += (act_pairs,)
    devices = {item.device for item in tensors}
    if len(devices) != 1:
        raise ValueError(f"masked-act tensors span devices {devices}")
    return slices, pairs, width


def _validate_b_inputs(
    bridge: torch.Tensor,
    b_gate_up: torch.Tensor,
    routing: RouteView,
    *,
    pairs: int,
    slices: int,
    width: int,
    bridge_top_k: int,
) -> int:
    if bridge_top_k not in (1, routing.topk_ids.shape[1]):
        raise ValueError(
            "bridge_top_k must be 1 (pair-major) or the route top_k "
            f"{routing.topk_ids.shape[1]} (token-major)"
        )
    expected_rows = pairs if bridge_top_k == 1 else routing.topk_ids.shape[0]
    if b_gate_up.ndim != 3 or b_gate_up.shape[:2] != (
        routing.num_virtual_experts,
        slices * width,
    ):
        raise ValueError("b_gate_up must be [num_virtual_experts, slices*width, rank]")
    rank = b_gate_up.shape[2]
    if bridge.shape != (expected_rows, slices * rank):
        raise ValueError(
            f"bridge must be {(expected_rows, slices * rank)}, got "
            f"{tuple(bridge.shape)}"
        )
    if bridge.dtype != b_gate_up.dtype:
        raise TypeError("bridge and b_gate_up must have the same dot dtype")
    if bridge.dtype != torch.bfloat16:
        raise TypeError("masked BF16 gate/up LoRA factors must be BF16")
    return rank


@triton.jit
def _base_columns(
    offsets,
    width: tl.constexpr,
    num_slices: tl.constexpr,
    gate_first: tl.constexpr,
    interleaved: tl.constexpr,
):
    if num_slices == 1:
        return offsets, offsets
    if interleaved:
        gate_offsets = 2 * offsets
        up_offsets = 2 * offsets + 1
    else:
        gate_offsets = offsets
        up_offsets = width + offsets
    if not gate_first:
        gate_offsets, up_offsets = up_offsets, gate_offsets
    return gate_offsets, up_offsets


@triton.jit
def _delta_slice(
    bridge_ptr,
    weight_group_ptr,
    bridge_rows,
    pair_mask,
    w_offsets,
    w_mask,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    slice_id: tl.constexpr,
    rank: tl.constexpr,
    width: tl.constexpr,
    block_m: tl.constexpr,
    block_w: tl.constexpr,
    block_k: tl.constexpr,
):
    acc = tl.zeros((block_m, block_w), tl.float32)
    for k_begin in range(0, rank, block_k):
        k_offsets = k_begin + tl.arange(0, block_k).to(tl.int64)
        k_mask = k_offsets < rank
        lhs = tl.load(
            bridge_ptr
            + bridge_rows[:, None] * stride_xm
            + (slice_id * rank + k_offsets)[None, :] * stride_xk,
            mask=pair_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        rhs = tl.load(
            weight_group_ptr
            + (slice_id * width + w_offsets)[None, :] * stride_wn
            + k_offsets[:, None] * stride_wk,
            mask=k_mask[:, None] & w_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(lhs, rhs, out_dtype=tl.float32)
    return acc


@triton.jit
def _b_act_kernel(
    bridge_ptr,
    b_ptr,
    base_ptr,
    act_masked_ptr,
    act_pairs_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    sorted_pairs_ptr,
    block_veids_ptr,
    pairs_post_padded_ptr,
    num_pairs,
    stride_xm,
    stride_xk,
    stride_bg,
    stride_bn,
    stride_bk,
    stride_pm,
    stride_pn,
    stride_am,
    stride_an,
    stride_qm,
    stride_qn,
    num_local_experts: tl.constexpr,
    top_k: tl.constexpr,
    width: tl.constexpr,
    rank: tl.constexpr,
    num_slices: tl.constexpr,
    activation_type: tl.constexpr,
    gate_first: tl.constexpr,
    interleaved: tl.constexpr,
    bridge_token_major: tl.constexpr,
    num_m_blocks: tl.constexpr,
    block_m: tl.constexpr,
    block_w: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    store_pair_act: tl.constexpr,
    consume_base_pdl: tl.constexpr,
):
    pid = tl.program_id(0)
    # ``tl.cdiv`` returns a tensor-like value. The multiplication below needs a
    # constexpr, so this line uses Python arithmetic.
    num_w_tiles: tl.constexpr = (width + block_w - 1) // block_w
    programs_per_group = group_m * num_w_tiles
    group_id = pid // programs_per_group
    first_m = group_id * group_m
    group_size = min(num_m_blocks - first_m, group_m)
    pid_m = first_m + (pid % programs_per_group) % group_size
    pid_w = (pid % programs_per_group) // group_size
    if pid_m * block_m >= tl.load(pairs_post_padded_ptr):
        return

    slots = pid_m * block_m + tl.arange(0, block_m).to(tl.int64)
    pair_ids = tl.load(sorted_pairs_ptr + slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    expert = tl.load(topk_ids_ptr + pair_ids, mask=pair_mask, other=-1)
    base_valid = pair_mask & (expert >= 0) & (expert < num_local_experts)
    dst_rows = tl.load(src2dst_ptr + pair_ids, mask=base_valid, other=0).to(tl.int64)
    veid = tl.load(block_veids_ptr + pid_m).to(tl.int64)

    w_offsets = pid_w * block_w + tl.arange(0, block_w).to(tl.int64)
    w_mask = w_offsets < width
    gate_cols, up_cols = _base_columns(
        w_offsets,
        width,
        num_slices,
        gate_first,
        interleaved,
    )

    delta_gate = tl.zeros((block_m, block_w), tl.float32)
    delta_up = tl.zeros((block_m, block_w), tl.float32)
    if veid != -1:
        bridge_rows = pair_ids // top_k if bridge_token_major else pair_ids
        group_ptr = b_ptr + veid * stride_bg
        delta_gate += _delta_slice(
            bridge_ptr,
            group_ptr,
            bridge_rows,
            pair_mask,
            w_offsets,
            w_mask,
            stride_xm,
            stride_xk,
            stride_bn,
            stride_bk,
            slice_id=0,
            rank=rank,
            width=width,
            block_m=block_m,
            block_w=block_w,
            block_k=block_k,
        )
        if num_slices == 2:
            delta_up += _delta_slice(
                bridge_ptr,
                group_ptr,
                bridge_rows,
                pair_mask,
                w_offsets,
                w_mask,
                stride_xm,
                stride_xk,
                stride_bn,
                stride_bk,
                slice_id=1,
                rank=rank,
                width=width,
                block_m=block_m,
                block_w=block_w,
                block_k=block_k,
            )

    if consume_base_pdl:
        # The LoRA-B tile above does not depend on the base gate/up GEMM. It
        # runs while that GEMM is still on the GPU. The kernel waits here, just
        # before it loads the base output.
        tl.extra.cuda.gdc_wait()
    base_gate = tl.load(
        base_ptr + dst_rows[:, None] * stride_pm + gate_cols[None, :] * stride_pn,
        mask=base_valid[:, None] & w_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    act = apply_activation(base_gate + delta_gate, activation_type)
    if num_slices == 2:
        base_up = tl.load(
            base_ptr + dst_rows[:, None] * stride_pm + up_cols[None, :] * stride_pn,
            mask=base_valid[:, None] & w_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        act = act * (base_up + delta_up)
    value = act.to(act_masked_ptr.dtype.element_ty)
    tl.store(
        act_masked_ptr + dst_rows[:, None] * stride_am + w_offsets[None, :] * stride_an,
        value,
        mask=base_valid[:, None] & w_mask[None, :],
    )
    if store_pair_act:
        # Every pair row gets a store. An invalid pair gets exact zero.
        tl.store(
            act_pairs_ptr
            + pair_ids[:, None] * stride_qm
            + w_offsets[None, :] * stride_qn,
            tl.where(base_valid[:, None], act, 0.0).to(act_pairs_ptr.dtype.element_ty),
            mask=pair_mask[:, None] & w_mask[None, :],
        )


def fused_b_act_masked(
    family: str,
    *,
    activation: str,
    base_gateup: torch.Tensor,
    act_masked: torch.Tensor,
    act_pairs: torch.Tensor | None,
    src2dst: torch.Tensor,
    routing: RouteView,
    num_local_experts: int,
    gate_first: bool,
    interleaved: bool,
    config: Mapping[str, int],
    bridge_gateup: torch.Tensor | None = None,
    b_gate_up: torch.Tensor | None = None,
    bridge_top_k: int = 1,
    consume_base_pdl: bool = False,
) -> None:
    slices, pairs, width = _validate_common(
        activation=activation,
        base_gateup=base_gateup,
        act_masked=act_masked,
        act_pairs=act_pairs,
        src2dst=src2dst,
        routing=routing,
        num_local_experts=num_local_experts,
    )
    if bridge_gateup is None or b_gate_up is None:
        raise ValueError(f"family {family!r} requires bridge_gateup and b_gate_up")

    gate_rank = _validate_b_inputs(
        bridge_gateup,
        b_gate_up,
        routing,
        pairs=pairs,
        slices=slices,
        width=width,
        bridge_top_k=bridge_top_k,
    )

    if pairs == 0:
        return
    block_w, block_k, group_m, num_warps, num_stages = _require_config(config)
    if routing.block_size < 16 or not _is_power_of_two(routing.block_size):
        raise ValueError(
            "aligned fused-act route block size must be a power of two >= 16"
        )
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    pair_target = act_pairs if act_pairs is not None else act_masked
    num_w_tiles = triton.cdiv(width, block_w)
    _b_act_kernel[(num_m_blocks * num_w_tiles,)](
        bridge_gateup,
        b_gate_up,
        base_gateup.view(-1, slices * width),
        act_masked.view(-1, width),
        pair_target.view(-1, width),
        src2dst,
        routing.topk_ids,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        pairs,
        bridge_gateup.stride(0),
        bridge_gateup.stride(1),
        b_gate_up.stride(0),
        b_gate_up.stride(1),
        b_gate_up.stride(2),
        base_gateup.stride(-2),
        base_gateup.stride(-1),
        act_masked.stride(-2),
        act_masked.stride(-1),
        pair_target.stride(-2),
        pair_target.stride(-1),
        num_local_experts=num_local_experts,
        top_k=routing.topk_ids.shape[1],
        width=width,
        rank=gate_rank,
        num_slices=slices,
        activation_type=activation,
        gate_first=gate_first,
        interleaved=interleaved,
        bridge_token_major=bridge_top_k != 1,
        num_m_blocks=num_m_blocks,
        block_m=routing.block_size,
        block_w=block_w,
        block_k=block_k,
        group_m=group_m,
        store_pair_act=act_pairs is not None,
        consume_base_pdl=consume_base_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **({"launch_pdl": True} if consume_base_pdl else {}),
    )


def fused_b_act_contiguous(
    family: str,
    *,
    activation: str,
    base_gateup: torch.Tensor,  # [m_pad_ceiling, slices * inter] bf16
    act_compact: torch.Tensor,  # [m_pad_ceiling, inter] bf16
    act_pairs: torch.Tensor | None,  # [num_tokens, top_k, inter] or None
    src2dst: torch.Tensor,  # [num_tokens * top_k] int32 COMPACT rows
    routing: RouteView,
    num_local_experts: int,
    gate_first: bool,
    interleaved: bool,
    config: Mapping[str, int],
    bridge_gateup: torch.Tensor | None = None,
    b_gate_up: torch.Tensor | None = None,
    bridge_top_k: int = 1,
    consume_base_pdl: bool = False,
) -> None:
    """Run the fused LoRA-B GEMM and the activation over the compact rows.

    The kernel, the grid and the per-pair arithmetic match
    :func:`fused_act.fused_b_act_masked`. Only the shape check
    differs, because the compact domain is one flat 2-D buffer. The kernel
    writes a zero into ``act_pairs`` once for each invalid pair.

    The kernel skips the segment padding rows, so they keep stale values. The
    down GEMM still reads them, but it treats every row separately. No consumer
    reads the output of a padding row.
    """
    if family not in B_ACT_FAMILIES:
        raise ValueError(f"family={family!r} is not one of {B_ACT_FAMILIES}")
    ActivationFn.parse(activation)
    pairs = routing.topk_ids.numel()
    if src2dst.dtype != torch.int32 or src2dst.numel() != pairs:
        raise ValueError(f"src2dst must be int32 with {pairs} entries")
    if act_compact.ndim != 2 or act_compact.shape[1] < 1:
        raise ValueError("act_compact must be compact [m_pad_ceiling, intermediate]")
    width = act_compact.shape[1]
    # The slice count comes from the weight shape, not from the activation.
    slices = base_gateup.shape[-1] // width
    if slices not in (1, 2) or slices * width != base_gateup.shape[-1]:
        raise ValueError(
            f"base gate/up width {base_gateup.shape[-1]} is not 1x or 2x "
            f"activation width {width}"
        )
    if base_gateup.shape != (act_compact.shape[0], slices * width):
        raise ValueError(
            "base_gateup must share the compact rows and carry "
            f"{slices * width} columns, got {tuple(base_gateup.shape)}"
        )
    if act_pairs is not None:
        if act_pairs.shape != (*routing.topk_ids.shape, width):
            raise ValueError(
                f"act_pairs must be {(*routing.topk_ids.shape, width)}, got "
                f"{tuple(act_pairs.shape)}"
            )
        if act_compact.dtype != act_pairs.dtype:
            raise TypeError(
                f"dual activation stores need one dtype, got "
                f"{act_compact.dtype} and {act_pairs.dtype}"
            )
    if base_gateup.dtype != torch.bfloat16 or act_compact.dtype != torch.bfloat16:
        raise TypeError("contiguous BF16 middle requires BF16 base and activation rows")
    tensors = (
        base_gateup,
        act_compact,
        src2dst,
        routing.topk_ids,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
    )
    if act_pairs is not None:
        tensors += (act_pairs,)
    devices = {item.device for item in tensors}
    if len(devices) != 1:
        raise ValueError(f"contiguous-middle tensors span devices {devices}")
    if bridge_gateup is None or b_gate_up is None:
        raise ValueError(f"family {family!r} requires bridge_gateup and b_gate_up")
    gate_rank = _validate_b_inputs(
        bridge_gateup,
        b_gate_up,
        routing,
        pairs=pairs,
        slices=slices,
        width=width,
        bridge_top_k=bridge_top_k,
    )

    if pairs == 0:
        return
    block_w, block_k, group_m, num_warps, num_stages = _require_config(config)
    if routing.block_size < 16 or not _is_power_of_two(routing.block_size):
        raise ValueError(
            "aligned fused-act route block size must be a power of two >= 16"
        )
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    pair_target = act_pairs.view(-1, width) if act_pairs is not None else act_compact
    num_w_tiles = triton.cdiv(width, block_w)
    _b_act_kernel[(num_m_blocks * num_w_tiles,)](
        bridge_gateup,
        b_gate_up,
        base_gateup,
        act_compact,
        pair_target,
        src2dst,
        routing.topk_ids,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        pairs,
        bridge_gateup.stride(0),
        bridge_gateup.stride(1),
        b_gate_up.stride(0),
        b_gate_up.stride(1),
        b_gate_up.stride(2),
        base_gateup.stride(0),
        base_gateup.stride(1),
        act_compact.stride(0),
        act_compact.stride(1),
        pair_target.stride(0),
        pair_target.stride(1),
        num_local_experts=num_local_experts,
        top_k=routing.topk_ids.shape[1],
        width=width,
        rank=gate_rank,
        num_slices=slices,
        activation_type=activation,
        gate_first=gate_first,
        interleaved=interleaved,
        bridge_token_major=bridge_top_k != 1,
        num_m_blocks=num_m_blocks,
        block_m=routing.block_size,
        block_w=block_w,
        block_k=block_k,
        group_m=group_m,
        store_pair_act=act_pairs is not None,
        consume_base_pdl=consume_base_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **({"launch_pdl": True} if consume_base_pdl else {}),
    )
