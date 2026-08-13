"""Deterministic BF16 math core for routed MoE LoRA.

The core consumes canonical token/expert-pair routing and contains no execution
policy.  Callers choose tiles and provider column order explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.srt.lora.sgl_lora.routing import RouteView


@triton.jit
def _grouped_lora_a_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_map_ptr,
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
    USE_INPUT_ROW_MAP: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    programs_per_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // programs_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(NUM_M_BLOCKS - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % programs_per_group) % group_size_m)
    pid_n = (pid % programs_per_group) // group_size_m
    if pid_m * BLOCK_SIZE_M >= num_pairs_post_padded:
        return

    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    virtual_expert_id = tl.load(block_virtual_expert_ids_ptr + pid_m).to(tl.int64)
    if virtual_expert_id == -1:
        return

    if USE_INPUT_ROW_MAP:
        input_rows = tl.load(
            input_row_map_ptr + pair_ids,
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
    input_row_map: torch.Tensor | None = None,
) -> None:
    """Write one single-K grouped LoRA-A result in canonical pair order.

    ``input`` is token-major by default.  ``pair_input`` selects canonical
    pair-major input.  A supplied ``input_row_map[pair]`` instead selects a
    provider-private row and may contain ``-1``; such rows are overwritten by
    zero in ``output``. Rows whose virtual expert ID is ``-1`` are undefined;
    the paired B primitive never observes them and overwrites its destination.

    ``config`` is chosen by the caller. This primitive contains no serving
    selector or provisional rank/token threshold.
    """
    num_pairs = routing.topk_ids.numel()
    if num_pairs == 0:
        return

    block_size_n = int(config["BLOCK_SIZE_N"])
    block_size_k = int(config["BLOCK_SIZE_K"])
    group_size_m = int(config["GROUP_SIZE_M"])
    input_row_map_ptr = output if input_row_map is None else input_row_map
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    num_n_blocks = triton.cdiv(weight.shape[1], block_size_n)
    _grouped_lora_a_kernel[(num_m_blocks * num_n_blocks,)](
        input,
        weight,
        output,
        input_row_map_ptr,
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
        USE_INPUT_ROW_MAP=input_row_map is not None,
        NUM_M_BLOCKS=num_m_blocks,
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def stock_grouped_lora_b(
    intermediate: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    destination_offsets: Sequence[int],
    config: Mapping[str, int],
    intermediate_top_k: int = 1,
) -> None:
    """Materialize one slice or matching gate/up slices with the stock GEMM.

    Factors are canonical ``[gate, up]``.  ``destination_offsets=(0, I)``
    writes a ``[gate, up]`` provider, while ``(I, 0)`` writes ``[up, gate]``
    directly.  Sentinel route blocks overwrite every targeted destination cell
    with zero, making buffer reuse safe under one CUDA graph. The caller owns
    static shape/layout/config validation.

    ``intermediate_top_k`` selects the intermediate's row domain: the stock
    kernel reads row ``pair_id // top_k``, so 1 (default) consumes the
    canonical pair-major ``[P, slices*R]`` bridge and ``top_k`` consumes a
    TOKEN-major ``[T, slices*R]`` bridge — the shared-outer token-dedup form
    (plan section 41.1), where every pair of one token reads the single
    deduplicated A row and no broadcast copy exists.
    """
    num_slices = len(destination_offsets)
    num_pairs = routing.topk_ids.numel()
    if num_pairs == 0:
        return
    # Fail-closed row-domain contract (third S3 review): exactly the two
    # shapes that exist. A pair-major bridge passed with top_k=K would
    # otherwise read row pair//K and SUCCEED with silently wrong results.
    num_tokens = routing.topk_ids.shape[0]
    if intermediate_top_k == 1:
        if intermediate.shape[0] != num_pairs:
            raise ValueError(
                f"pair-major intermediate must have {num_pairs} rows, got "
                f"{intermediate.shape[0]}"
            )
    elif intermediate_top_k == routing.topk_ids.shape[1]:
        if intermediate.shape[0] != num_tokens:
            raise ValueError(
                f"token-major intermediate must have {num_tokens} rows, got "
                f"{intermediate.shape[0]}"
            )
    else:
        raise ValueError(
            f"intermediate_top_k must be 1 (pair-major) or the route's "
            f"top_k {routing.topk_ids.shape[1]} (token-major), got "
            f"{intermediate_top_k}"
        )

    slice_width = weight.shape[1] // num_slices
    rank = weight.shape[2]
    # The stock kernel does not read either tensor when routed weighting is
    # disabled. Reusing the contiguous route tensor avoids a forward allocation.
    # The stock kernel takes a weights tensor it must not read
    # (mul_routed_weight=False); any [T, K] tensor satisfies the signature.
    unused_topk_weights = routing.topk_ids
    for slice_id, destination_offset in enumerate(destination_offsets):
        invoke_fused_moe_kernel(
            intermediate[:, slice_id * rank : (slice_id + 1) * rank],
            weight[:, slice_id * slice_width : (slice_id + 1) * slice_width, :],
            None,
            destination[
                :, int(destination_offset) : int(destination_offset) + slice_width
            ],
            None,
            None,
            None,
            unused_topk_weights,
            routing.topk_ids,
            routing.sorted_pair_ids,
            routing.block_virtual_expert_ids,
            routing.num_pairs_post_padded,
            False,
            intermediate_top_k,
            config,
            tl.bfloat16,
            False,
            False,
            False,
            False,
            False,
            filter_expert=True,
        )
