"""LoRA-A projections over aligned pairs, raw pairs, or shared token rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.kernels.routing import (
    grouped_tile_coords,
    virtual_expert_ids_inline,
)
from sglang.srt.lora.moe.route_view import RouteView

if TYPE_CHECKING:
    from sglang.srt.lora.moe.execution_plan import LoraASpec


@triton.jit
def _grouped_lora_a_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    pair_to_row_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_pairs_post_padded_ptr,
    num_input_rows,
    num_pairs,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    TOP_K: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PAIR_INPUT: tl.constexpr,
    USE_PAIR_TO_ROW: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    PRODUCE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = grouped_tile_coords(pid, num_pid_n, NUM_M_BLOCKS, GROUP_SIZE_M)
    if pid_m * BLOCK_SIZE_M >= num_pairs_post_padded:
        return

    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    virtual_expert_id = tl.load(block_virtual_expert_ids_ptr + pid_m).to(tl.int64)
    if virtual_expert_id == -1:
        return

    if USE_PAIR_TO_ROW:
        input_rows = tl.load(
            pair_to_row_ptr + pair_ids,
            mask=pair_mask,
            other=-1,
        ).to(tl.int64)
    elif PAIR_INPUT:
        input_rows = pair_ids
    else:
        input_rows = pair_ids // TOP_K
    input_mask = pair_mask & (input_rows >= 0) & (input_rows < num_input_rows)

    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = n_offsets < N
    if PRODUCE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_begin in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < K
        lhs = tl.load(
            input_ptr
            + input_rows[:, None] * stride_im
            + k_offsets[None, :] * stride_ik,
            mask=input_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        rhs = tl.load(
            weight_ptr
            + virtual_expert_id * stride_we
            + n_offsets[None, :] * stride_wn
            + k_offsets[:, None] * stride_wk,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(lhs, rhs, out_dtype=tl.float32)

    tl.store(
        output_ptr + pair_ids[:, None] * stride_om + n_offsets[None, :] * stride_on,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=pair_mask[:, None] & n_mask[None, :],
    )


def grouped_lora_a(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routing: RouteView,
    *,
    config: Mapping[str, int],
    pair_input: bool = False,
    pair_to_row: torch.Tensor | None = None,
    produce_pdl: bool = False,
) -> None:
    num_pairs = routing.topk_ids.numel()
    if num_pairs == 0:
        return

    block_size_n = int(config["BLOCK_SIZE_N"])
    block_size_k = int(config["BLOCK_SIZE_K"])
    group_size_m = int(config["GROUP_SIZE_M"])
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    num_n_blocks = triton.cdiv(weight.shape[1], block_size_n)
    _grouped_lora_a_kernel[(num_m_blocks * num_n_blocks,)](
        input,
        weight,
        output,
        output if pair_to_row is None else pair_to_row,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        input.shape[0],
        num_pairs,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        TOP_K=routing.topk_ids.shape[1],
        N=weight.shape[1],
        K=weight.shape[2],
        PAIR_INPUT=pair_input,
        USE_PAIR_TO_ROW=pair_to_row is not None,
        NUM_M_BLOCKS=num_m_blocks,
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        PRODUCE_PDL=produce_pdl,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


@triton.jit
def _per_pair_lora_a_kernel(
    input_ptr,
    weight_ptr,
    topk_ids_ptr,
    token_lora_mapping_ptr,
    output_ptr,
    num_pairs,
    routed_expert_id_bound,
    stride_im,
    stride_ik,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    N: tl.constexpr,
    K: tl.constexpr,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    PAIR_INPUT: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pair_id = tl.program_id(0)
    pid_n = tl.program_id(1)
    key = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_lora_mapping_ptr,
        pair_id,
        pair_id < num_pairs,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        SHARED_OUTER=SHARED_OUTER,
    )
    valid = key != -1
    group = tl.maximum(key, 0).to(tl.int64)
    pair64 = pair_id.to(tl.int64)
    input_row = pair64 if PAIR_INPUT else pair64 // TOP_K

    n_offsets = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(
        tl.int64
    )
    n_mask = n_offsets < N
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k_begin in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < K
        lhs = tl.load(
            input_ptr + input_row * stride_im + k_offsets * stride_ik,
            mask=valid & k_mask,
            other=0.0,
        )
        rhs = tl.load(
            weight_ptr
            + group * stride_wg
            + n_offsets[:, None] * stride_wn
            + k_offsets[None, :] * stride_wk,
            mask=valid & n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        accumulator += tl.sum(rhs.to(tl.float32) * lhs[None, :].to(tl.float32), axis=1)

    # B zeros sentinel destinations without reading these bridge rows.
    tl.store(
        output_ptr + pair64 * stride_om + n_offsets * stride_on,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=valid & n_mask,
    )


def per_pair_lora_a(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routing: RouteView,
    *,
    config: Mapping[str, int],
    pair_input: bool = False,
) -> None:
    num_pairs = routing.topk_ids.numel()
    if num_pairs == 0:
        return

    block_size_n = int(config["BLOCK_SIZE_N"])
    _per_pair_lora_a_kernel[(num_pairs, triton.cdiv(weight.shape[1], block_size_n))](
        input,
        weight,
        routing.topk_ids,
        routing.token_lora_mapping,
        output,
        num_pairs,
        routing.num_local_experts,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        N=weight.shape[1],
        K=weight.shape[2],
        LORA_EXPERTS_PER_ADAPTER=routing.lora_experts_per_adapter,
        MAX_LORAS=routing.max_loras,
        TOP_K=routing.topk_ids.shape[1],
        SHARED_OUTER=routing.is_shared_outer,
        PAIR_INPUT=pair_input,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=int(config["BLOCK_SIZE_K"]),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def run_lora_a(
    spec: LoraASpec,
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routing: RouteView,
    config: Mapping[str, int],
    pair_to_row: torch.Tensor | None = None,
    produce_pdl: bool = False,
) -> torch.Tensor:
    family = spec.family.value
    pair_input = spec.site.value == "down"
    match family:
        case "token_grouped":
            grouped_lora_a(
                input,
                weight,
                output,
                routing,
                config=config,
                produce_pdl=produce_pdl,
            )
        case "grouped":
            grouped_lora_a(
                input,
                weight,
                output,
                routing,
                config=config,
                pair_input=pair_input,
                pair_to_row=pair_to_row,
                produce_pdl=produce_pdl,
            )
        case "per_pair":
            if produce_pdl:
                raise ValueError(
                    f"{family} A has no qualified programmatic-dependent-launch "
                    "producer"
                )
            per_pair_lora_a(
                input,
                weight,
                output,
                routing,
                config=config,
                pair_input=pair_input,
            )
        case _:
            raise NotImplementedError(f"no production LoRA-A executor for {family!r}")
    return output
