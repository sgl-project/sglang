# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""LoRA hooks for MoE runners.

LoRA deltas are injected at two points in the MoE pipeline:
1. After gate_up projection, BEFORE activation
2. After down projection, BEFORE final reduction

This module provides hook closures that any MoE backend can call at those points,
without needing a per-backend LoRA runner subclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.utils import is_cuda, is_hip, is_xpu, next_power_of_2

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_hip = is_hip()
_is_xpu = is_xpu()

if _is_cuda or _is_hip or _is_xpu:
    from sglang.jit_kernel.moe_lora_align import moe_lora_align_block_size


def _get_moe_lora_block_config(max_lora_rank: int) -> dict:
    """Compute rank-aware block sizes for MoE LoRA kernels.

    Shrink: output dim is the rank -> cap BLOCK_SIZE_N to avoid waste.
    Expand: input dim is the rank -> cap BLOCK_SIZE_K similarly.
    """
    if max_lora_rank <= 0:
        rank_pow2 = 64
    else:
        rank_pow2 = next_power_of_2(max_lora_rank)

    shrink_n = min(64, rank_pow2)
    expand_k = max(16, min(64, rank_pow2))

    return {
        "shrink_block_size_n": shrink_n,
        "expand_block_size_k": expand_k,
    }


_SPARSITY_FACTOR = 8


def _naive_moe_lora_align_block_size(
    topk_ids: torch.Tensor,
    seg_indptr: torch.Tensor,
    req_to_lora: torch.Tensor,
    num_experts: int,
    block_size_m: int,
    max_loras: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    adapter_enabled: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct LoRA token-expert alignment on CPU for small batches.

    When the number of tokens is very small, the overhead of launching the
    CUDA-based moe_lora_align_block_size kernel exceeds the actual
    computation. This function builds the same data structures using simple
    Python loops on CPU and transfers the result to GPU in one shot.
    """
    M, top_k = topk_ids.shape
    num_valid_tokens = M * top_k

    sorted_token_ids = torch.full(
        (max_loras * max_num_tokens_padded,),
        num_valid_tokens,
        dtype=torch.int32,
    )
    expert_ids_out = torch.full((max_loras * max_num_m_blocks,), -1, dtype=torch.int32)
    num_tokens_post_padded = torch.zeros(max_loras, dtype=torch.int32)

    seg_indptr_list = seg_indptr.cpu().tolist()
    req_to_lora_list = req_to_lora.cpu().tolist()
    topk_ids_list = topk_ids.cpu().tolist()
    adapter_enabled_list = adapter_enabled.cpu().tolist()

    for lora_id in range(max_loras):
        if not adapter_enabled_list[lora_id]:
            continue

        pairs: list[tuple[int, int]] = []
        for seg_idx in range(len(seg_indptr_list) - 1):
            if req_to_lora_list[seg_idx] != lora_id:
                continue
            start = seg_indptr_list[seg_idx]
            end = seg_indptr_list[seg_idx + 1]
            for m in range(start, end):
                for k in range(top_k):
                    pairs.append((topk_ids_list[m][k], m * top_k + k))

        if not pairs:
            continue

        pairs.sort()

        base_t = lora_id * max_num_tokens_padded
        base_e = lora_id * max_num_m_blocks
        pos = 0
        block_idx = 0
        i = 0
        while i < len(pairs):
            cur_expert = pairs[i][0]
            group_start = pos
            while i < len(pairs) and pairs[i][0] == cur_expert:
                sorted_token_ids[base_t + pos] = pairs[i][1]
                pos += 1
                i += 1
            group_len = pos - group_start
            padded_len = ((group_len + block_size_m - 1) // block_size_m) * block_size_m
            num_blocks = padded_len // block_size_m
            for b in range(num_blocks):
                expert_ids_out[base_e + block_idx + b] = cur_expert
            block_idx += num_blocks
            pos = group_start + padded_len

        num_tokens_post_padded[lora_id] = pos

    return (
        sorted_token_ids.to(device),
        expert_ids_out.to(device),
        num_tokens_post_padded.to(device),
    )


def _get_moe_lora_block_config(max_lora_rank: int) -> dict:
    """Compute rank-aware block sizes for MoE LoRA kernels.

    Shrink: output dim is the rank -> cap BLOCK_SIZE_N to avoid waste.
    Expand: input dim is the rank -> cap BLOCK_SIZE_K similarly.
    """
    if max_lora_rank <= 0:
        rank_pow2 = 64
    else:
        rank_pow2 = next_power_of_2(max_lora_rank)

    shrink_n = min(64, rank_pow2)
    expand_k = max(16, min(64, rank_pow2))

    return {
        "shrink_block_size_n": shrink_n,
        "expand_block_size_k": expand_k,
    }


_SPARSITY_FACTOR = 8


def _naive_moe_lora_align_block_size(
    topk_ids: torch.Tensor,
    seg_indptr: torch.Tensor,
    req_to_lora: torch.Tensor,
    num_experts: int,
    block_size_m: int,
    max_loras: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    adapter_enabled: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct LoRA token-expert alignment on CPU for small batches.

    When the number of tokens is very small, the overhead of launching the
    CUDA-based moe_lora_align_block_size kernel exceeds the actual
    computation. This function builds the same data structures using simple
    Python loops on CPU and transfers the result to GPU in one shot.
    """
    M, top_k = topk_ids.shape
    num_valid_tokens = M * top_k

    sorted_token_ids = torch.full(
        (max_loras * max_num_tokens_padded,),
        num_valid_tokens,
        dtype=torch.int32,
    )
    expert_ids_out = torch.full((max_loras * max_num_m_blocks,), -1, dtype=torch.int32)
    num_tokens_post_padded = torch.zeros(max_loras, dtype=torch.int32)

    seg_indptr_list = seg_indptr.cpu().tolist()
    req_to_lora_list = req_to_lora.cpu().tolist()
    topk_ids_list = topk_ids.cpu().tolist()
    adapter_enabled_list = adapter_enabled.cpu().tolist()

    for lora_id in range(max_loras):
        if not adapter_enabled_list[lora_id]:
            continue

        pairs: list[tuple[int, int]] = []
        for seg_idx in range(len(seg_indptr_list) - 1):
            if req_to_lora_list[seg_idx] != lora_id:
                continue
            start = seg_indptr_list[seg_idx]
            end = seg_indptr_list[seg_idx + 1]
            for m in range(start, end):
                for k in range(top_k):
                    pairs.append((topk_ids_list[m][k], m * top_k + k))

        if not pairs:
            continue

        pairs.sort()

        base_t = lora_id * max_num_tokens_padded
        base_e = lora_id * max_num_m_blocks
        pos = 0
        block_idx = 0
        i = 0
        while i < len(pairs):
            cur_expert = pairs[i][0]
            group_start = pos
            while i < len(pairs) and pairs[i][0] == cur_expert:
                sorted_token_ids[base_t + pos] = pairs[i][1]
                pos += 1
                i += 1
            group_len = pos - group_start
            padded_len = ((group_len + block_size_m - 1) // block_size_m) * block_size_m
            num_blocks = padded_len // block_size_m
            for b in range(num_blocks):
                expert_ids_out[base_e + block_idx + b] = cur_expert
            block_idx += num_blocks
            pos = group_start + padded_len

        num_tokens_post_padded[lora_id] = pos

    return (
        sorted_token_ids.to(device),
        expert_ids_out.to(device),
        num_tokens_post_padded.to(device),
    )


@dataclass
class LoRAInfo:
    """LoRA weights and dispatch info for MoE computation."""

    # LoRA weights: [num_loras, num_experts_or_1, dim1, dim2]
    # When experts_shared_outer_loras=True:
    #   gate_up_lora_a: [num_loras, 1, max_rank, hidden_dim] (shared)
    #   down_lora_b: [num_loras, 1, hidden_dim, max_rank] (shared)
    gate_up_lora_a_weights: (
        torch.Tensor
    )  # [num_loras, num_experts_or_1, max_rank, hidden_dim]
    gate_up_lora_b_weights: (
        torch.Tensor
    )  # [num_loras, num_experts, gate_up_dim, max_rank]
    down_lora_a_weights: (
        torch.Tensor
    )  # [num_loras, num_experts, max_rank, intermediate_dim]
    down_lora_b_weights: (
        torch.Tensor
    )  # [num_loras, num_experts_or_1, hidden_dim, max_rank]

    # Indice pointers of each segment in shape (num_segments + 1, )
    seg_indptr: torch.Tensor

    # The index of lora adapter used by each segment, in shape (num_segments,)
    req_to_lora: torch.Tensor

    # LoRA config per adapter
    lora_ranks: torch.Tensor  # [num_loras]
    adapter_enabled: torch.Tensor  # [num_loras] - which adapters are enabled
    max_lora_rank: int  # Maximum LoRA rank across all adapters

    num_experts: int
    experts_shared_outer_loras: bool = False
    cg_buffers: dict | None = None
    cg_buffers: dict | None = None

    fully_sharded: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    hidden_size: int = 0
    lora_use_virtual_experts: bool = False


@dataclass
class LoRAHooks:
    """Hook callbacks for injecting LoRA deltas into the MoE pipeline."""

    after_gate_up: (
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None] | None
    ) = None
    after_down: (
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None] | None
    ) = None


def _compute_token_lora_mapping(
    hidden_states: torch.Tensor,
    lora_info: LoRAInfo,
) -> torch.Tensor:
    """Map each token to its LoRA adapter index (-1 for no LoRA)."""
    token_positions = torch.arange(
        hidden_states.shape[0], device=hidden_states.device, dtype=torch.int32
    )
    req_indices = torch.searchsorted(
        lora_info.seg_indptr[1:].to(torch.int32),
        token_positions,
        right=True,
    )
    return lora_info.req_to_lora.to(torch.int32)[req_indices]


def _compute_lora_alignment(
    topk_ids: torch.Tensor,
    lora_info: LoRAInfo,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute LoRA alignment tensors for the non-virtual-expert (classic) path.

    Returns: (sorted_token_ids_reshaped, expert_ids_reshaped, num_tokens_post_padded_lora, lora_ids)
    """
    cg = lora_info.cg_buffers if get_is_capture_mode() else None
    shrink_config = {"BLOCK_SIZE_M": 64}
    M = topk_ids.shape[0]
    block_size_m = shrink_config["BLOCK_SIZE_M"]
    max_loras = len(lora_info.lora_ranks)

    max_num_tokens_padded = topk_ids.numel() + lora_info.num_experts * (
        block_size_m - 1
    )
    max_num_tokens_padded = (
        (max_num_tokens_padded + block_size_m - 1) // block_size_m
    ) * block_size_m
    max_num_m_blocks = (max_num_tokens_padded + block_size_m - 1) // block_size_m

    device = topk_ids.device

    use_naive = (
        cg is None
        and M * topk_ids.shape[1] * _SPARSITY_FACTOR
        <= lora_info.num_experts * max_loras
    )

    if use_naive:
        sorted_token_ids_lora, expert_ids_lora, num_tokens_post_padded_lora = (
            _naive_moe_lora_align_block_size(
                topk_ids,
                lora_info.seg_indptr,
                lora_info.req_to_lora,
                int(lora_info.num_experts),
                int(block_size_m),
                int(max_loras),
                int(max_num_tokens_padded),
                int(max_num_m_blocks),
                lora_info.adapter_enabled,
                device,
            )
        )
        lora_ids = torch.arange(max_loras, dtype=torch.int32, device=device)
    else:
        if cg is not None:
            sorted_token_ids_lora = cg["sorted_token_ids_lora"][
                : max_loras * max_num_tokens_padded
            ]
            expert_ids_lora = cg["expert_ids_lora"][: max_loras * max_num_m_blocks]
            num_tokens_post_padded_lora = cg["num_tokens_post_padded_lora"][:max_loras]
        else:
            sorted_token_ids_lora = torch.empty(
                (max_loras * max_num_tokens_padded,),
                dtype=torch.int32,
                device=device,
            )
            expert_ids_lora = torch.empty(
                (max_loras * max_num_m_blocks,),
                dtype=torch.int32,
                device=device,
            )
            num_tokens_post_padded_lora = torch.empty(
                (max_loras,), dtype=torch.int32, device=device
            )

        if cg is not None and "lora_ids" in cg:
            lora_ids = cg["lora_ids"][:max_loras]
        else:
            lora_ids = torch.arange(max_loras, dtype=torch.int32, device=device)

        moe_lora_align_block_size(
            topk_ids,
            lora_info.seg_indptr,
            lora_info.req_to_lora,
            int(lora_info.num_experts),
            int(block_size_m),
            int(max_loras),
            int(max_num_tokens_padded),
            int(max_num_m_blocks),
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            lora_info.adapter_enabled,
            lora_ids,
            cumsum_buffer=cg.get("cumsum_buffer") if cg is not None else None,
            token_mask=cg.get("token_mask") if cg is not None else None,
        )

    return (
        sorted_token_ids_lora.view(max_loras, -1),
        expert_ids_lora.view(max_loras, -1),
        num_tokens_post_padded_lora,
        lora_ids,
    )


def _add_lora_gate_up_delta(
    hidden_states: torch.Tensor,
    intermediate_cache: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    lora_info: LoRAInfo,
    token_lora_mapping: torch.Tensor | None,
    sorted_token_ids_reshaped: torch.Tensor | None,
    expert_ids_reshaped: torch.Tensor | None,
    num_tokens_post_padded_lora: torch.Tensor | None,
    lora_ids: torch.Tensor | None,
    routing_cache: dict | None = None,
) -> None:
    """Add LoRA gate_up delta to intermediate_cache in-place."""
    from sglang.srt.lora.triton_ops import (
        fused_moe_lora,
        merged_experts_fused_moe_lora_add,
    )

    if get_is_capture_mode():
        # During CUDA graph capture, always enter the LoRA path so that
        # the LoRA kernels are recorded in the graph.  adapter_enabled is
        # all-zeros during capture, so the Triton kernel early-exits per
        # program (zero overhead).  During replay the tensor is updated
        # in-place with the real adapter mask before graph.replay().
        has_active_lora = True
    else:
        num_loras = len(lora_info.lora_ranks)
        has_active_lora = (
            (
                lora_info.adapter_enabled[:num_loras]
                * (lora_info.lora_ranks > 0).to(lora_info.adapter_enabled.dtype)
            )
            .any()
            .item()
        )
    if not has_active_lora or lora_info is None or lora_info.max_lora_rank == 0:
        return

    M, top_k, gate_up_dim = intermediate_cache.shape
    r = lora_info.max_lora_rank
    gate_up_a = lora_info.gate_up_lora_a_weights
    gate_up_b = lora_info.gate_up_lora_b_weights
    inter_size = gate_up_b.shape[2] // 2
    M, top_k, gate_up_dim = intermediate_cache.shape
    r = lora_info.max_lora_rank
    gate_up_a = lora_info.gate_up_lora_a_weights
    gate_up_b = lora_info.gate_up_lora_b_weights
    inter_size = gate_up_b.shape[2] // 2

    if lora_info.experts_shared_outer_loras and not lora_info.lora_use_virtual_experts:
        gate_up_a = gate_up_a.expand(-1, lora_info.num_experts, -1, -1)
    inter_size = gate_up_b.shape[2] // 2
    lora_a_stacked = [gate_up_a[:, :, :r, :], gate_up_a[:, :, r : 2 * r, :]]
    lora_b_stacked = [gate_up_b[:, :, :inter_size, :], gate_up_b[:, :, inter_size:, :]]

    if lora_info.lora_use_virtual_experts:
        merged_experts_fused_moe_lora_add(
            output=intermediate_cache,
            hidden_states=hidden_states,
            lora_a=gate_up_a,
            lora_b=gate_up_b,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=False,
            routing_cache=routing_cache,
        )
    else:
        blk = _get_moe_lora_block_config(r)
        fused_moe_lora(
            output=intermediate_cache,
            qcurr_hidden_states=hidden_states,
            lora_a_stacked=lora_a_stacked,
            lora_b_stacked=lora_b_stacked,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids_reshaped,
            expert_ids=expert_ids_reshaped,
            num_tokens_post_padded=num_tokens_post_padded_lora,
            max_lora_rank=r,
            top_k_num=top_k,
            lora_ids=lora_ids,
            adapter_enabled=lora_info.adapter_enabled,
            shrink_block_size_m=64,
            shrink_block_size_n=blk["shrink_block_size_n"],
            shrink_block_size_k=64,
            shrink_group_size_m=8,
            shrink_num_warps=4,
            shrink_num_stages=2,
            shrink_split_k=1,
            expand_block_size_m=64,
            expand_block_size_n=64,
            expand_block_size_k=blk["expand_block_size_k"],
            expand_group_size_m=8,
            expand_num_warps=4,
            expand_num_stages=2,
            expand_split_k=1,
            fully_sharded=lora_info.fully_sharded,
        )


def _add_lora_down_delta(
    intermediate_input: torch.Tensor,
    intermediate_cache: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    lora_info: LoRAInfo,
    token_lora_mapping: torch.Tensor | None,
    sorted_token_ids_reshaped: torch.Tensor | None,
    expert_ids_reshaped: torch.Tensor | None,
    num_tokens_post_padded_lora: torch.Tensor | None,
    lora_ids: torch.Tensor | None,
    routing_cache: dict | None = None,
) -> None:
    """Add LoRA down delta to intermediate_cache in-place."""
    from sglang.srt.lora.triton_ops import (
        fused_moe_lora,
        merged_experts_fused_moe_lora_add,
    )

    if lora_info.max_lora_rank == 0:
        return

    M, top_k, hidden_dim = intermediate_cache.shape

    down_lora_a = lora_info.down_lora_a_weights
    down_lora_b = lora_info.down_lora_b_weights
    if lora_info.experts_shared_outer_loras and not lora_info.lora_use_virtual_experts:
        down_lora_b = down_lora_b.expand(-1, lora_info.num_experts, -1, -1)

    if lora_info.fully_sharded and lora_info.tp_size > 1:
        shard_size = lora_info.hidden_size // lora_info.tp_size
        offset = shard_size * lora_info.tp_rank
    else:
        offset = 0
    if lora_info.fully_sharded and lora_info.tp_size > 1:
        shard_size = lora_info.hidden_size // lora_info.tp_size
        offset = shard_size * lora_info.tp_rank
    else:
        offset = 0

    if lora_info.lora_use_virtual_experts:
        merged_experts_fused_moe_lora_add(
            output=intermediate_cache,
            hidden_states=intermediate_input,
            lora_a=down_lora_a,
            lora_b=down_lora_b,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=routing_cache,
        )
    else:
        blk = _get_moe_lora_block_config(lora_info.max_lora_rank)
        fused_moe_lora(
            output=intermediate_cache,
            qcurr_hidden_states=intermediate_input,
            lora_a_stacked=[down_lora_a],
            lora_b_stacked=[down_lora_b],
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids_reshaped,
            expert_ids=expert_ids_reshaped,
            num_tokens_post_padded=num_tokens_post_padded_lora,
            max_lora_rank=lora_info.max_lora_rank,
            top_k_num=top_k,
            lora_ids=lora_ids,
            adapter_enabled=lora_info.adapter_enabled,
            shrink_block_size_m=64,
            shrink_block_size_n=blk["shrink_block_size_n"],
            shrink_block_size_k=64,
            shrink_group_size_m=8,
            shrink_num_warps=4,
            shrink_num_stages=2,
            shrink_split_k=1,
            expand_block_size_m=64,
            expand_block_size_n=64,
            expand_block_size_k=blk["expand_block_size_k"],
            expand_group_size_m=8,
            expand_num_warps=4,
            expand_num_stages=2,
            expand_split_k=1,
            mul_routed_weight=True,
            fully_sharded=lora_info.fully_sharded,
            offset=offset,
        )


def build_lora_hooks(
    hidden_states: torch.Tensor,
    lora_info: LoRAInfo,
    topk_ids: torch.Tensor,
) -> LoRAHooks:
    """Build LoRA hook closures for injection into any MoE runner.

    Computes token_lora_mapping and alignment tensors once, then returns
    closures that capture them for the two injection points.
    """
    if lora_info is None or lora_info.max_lora_rank == 0:
        return LoRAHooks()

    # Compute alignment / mapping (once, shared by both hooks)
    token_lora_mapping: torch.Tensor | None = None
    sorted_token_ids_reshaped: torch.Tensor | None = None
    expert_ids_reshaped: torch.Tensor | None = None
    num_tokens_post_padded_lora: torch.Tensor | None = None
    lora_ids: torch.Tensor | None = None

    if lora_info.lora_use_virtual_experts:
        token_lora_mapping = _compute_token_lora_mapping(hidden_states, lora_info)
    else:
        (
            sorted_token_ids_reshaped,
            expert_ids_reshaped,
            num_tokens_post_padded_lora,
            lora_ids,
        ) = _compute_lora_alignment(topk_ids, lora_info)

    # Shared routing cache: gate_up and down reuse routing for same (num_experts, shared_outer, block_size)
    routing_cache: dict = {}

    def after_gate_up(
        hidden_states: torch.Tensor,
        intermediate_cache1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        _add_lora_gate_up_delta(
            hidden_states,
            intermediate_cache1,
            topk_weights,
            topk_ids,
            lora_info,
            token_lora_mapping,
            sorted_token_ids_reshaped,
            expert_ids_reshaped,
            num_tokens_post_padded_lora,
            lora_ids,
            routing_cache=routing_cache,
        )

    def after_down(
        intermediate_input: torch.Tensor,
        intermediate_cache3: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        _add_lora_down_delta(
            intermediate_input,
            intermediate_cache3,
            topk_weights,
            topk_ids,
            lora_info,
            token_lora_mapping,
            sorted_token_ids_reshaped,
            expert_ids_reshaped,
            num_tokens_post_padded_lora,
            lora_ids,
            routing_cache=routing_cache,
        )

    return LoRAHooks(after_gate_up=after_gate_up, after_down=after_down)
