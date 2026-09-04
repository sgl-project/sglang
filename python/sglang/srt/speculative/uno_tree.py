# SPDX-License-Identifier: Apache-2.0
# Adapted from nano-vllm-uno's fixed-budget draft-tree builder for SGLang.

"""GPU-native UNO proposal-tree construction for SGLang spec-v2.

UNO owns only proposal ranking and fixed-budget best-first selection.  The
result is expressed directly in EAGLE's candidate-lineage ABI so the existing
EAGLE implementation can build masks, positions, traversal links, verify the
tree, sample the accepted path, and compact KV state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from flashinfer import top_k as _flashinfer_top_k
from torch import Tensor


@dataclass(frozen=True)
class UnoTreeProposal:
    """EAGLE-native representation of one fixed-width UNO tree batch.

    For non-root node ``i``, ``top_scores_index[:, i - 1]`` is the implicit
    candidate edge ``parent_node * candidate_top_k + candidate_rank``.
    ``parent_list`` maps an EAGLE candidate row back to the edge that created
    that row's node.  Its rows use EAGLE's native
    ``candidate_top_k * (max_depth - 1) + 1`` stride, including unused
    padding.  The tensors can therefore be passed directly to
    ``build_tree_kernel_efficient`` without constructing direct parent arrays.

    Tensors may be backed by the caller's reusable workspace and remain valid
    only until that workspace is reused.
    """

    root_tokens: Tensor
    draft_tokens: Tensor
    parent_list: Tensor
    top_scores_index: Tensor
    candidate_top_k: int
    max_depth: int

    @property
    def num_verify_tokens(self) -> int:
        return int(self.draft_tokens.shape[1]) + 1


@triton.jit
def _candidate_lse_partials_kernel(
    logits,
    partial_max,
    partial_sum,
    temperature_values,
    inverse_temperature,
    BATCH_STRIDE: tl.constexpr,
    DEPTH_STRIDE: tl.constexpr,
    VOCAB_STRIDE: tl.constexpr,
    NUM_DEPTHS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
    TEMPERATURE_BATCH_STRIDE: tl.constexpr,
    TEMPERATURE_IS_TENSOR: tl.constexpr,
):
    batch = tl.program_id(0)
    depth = tl.program_id(1)
    block = tl.program_id(2)
    row = batch * NUM_DEPTHS + depth
    offsets = block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    values = tl.load(
        logits + batch * BATCH_STRIDE + depth * DEPTH_STRIDE + offsets * VOCAB_STRIDE,
        mask=offsets < VOCAB_SIZE,
        other=-float("inf"),
    ).to(tl.float32)
    if TEMPERATURE_IS_TENSOR:
        row_temperature = tl.load(
            temperature_values + batch * TEMPERATURE_BATCH_STRIDE
        ).to(tl.float32)
        row_inverse_temperature = tl.where(
            row_temperature > 0.0,
            1.0 / row_temperature,
            1.0,
        )
    else:
        row_inverse_temperature = inverse_temperature
    values *= row_inverse_temperature
    maximum = tl.max(values, axis=0)
    total = tl.sum(tl.exp(values - maximum), axis=0)
    output = row * NUM_BLOCKS + block
    tl.store(partial_max + output, maximum)
    tl.store(partial_sum + output, total)


@triton.jit
def _candidate_lse_finalize_kernel(
    top_values,
    partial_max,
    partial_sum,
    top_log_probs,
    temperature_values,
    inverse_temperature,
    K: tl.constexpr,
    NUM_DEPTHS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
    TEMPERATURE_BATCH_STRIDE: tl.constexpr,
    TEMPERATURE_IS_TENSOR: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // NUM_DEPTHS
    blocks = tl.arange(0, BLOCK_PARTIALS)
    block_max = tl.load(
        partial_max + row * NUM_BLOCKS + blocks,
        mask=blocks < NUM_BLOCKS,
        other=-float("inf"),
    )
    maximum = tl.max(block_max, axis=0)
    block_sum = tl.load(
        partial_sum + row * NUM_BLOCKS + blocks,
        mask=blocks < NUM_BLOCKS,
        other=0.0,
    )
    total = tl.sum(block_sum * tl.exp(block_max - maximum), axis=0)
    normalizer = maximum + tl.log(total)

    ranks = tl.arange(0, BLOCK_K)
    candidates = tl.load(
        top_values + row * K + ranks,
        mask=ranks < K,
        other=-float("inf"),
    ).to(tl.float32)
    if TEMPERATURE_IS_TENSOR:
        row_temperature = tl.load(
            temperature_values + batch * TEMPERATURE_BATCH_STRIDE
        ).to(tl.float32)
        row_inverse_temperature = tl.where(
            row_temperature > 0.0,
            1.0 / row_temperature,
            1.0,
        )
    else:
        row_inverse_temperature = inverse_temperature

    log_probs = candidates * row_inverse_temperature - normalizer
    # Proposal mass affects efficiency, not correctness.  A malformed row
    # must not leave the best-first frontier without a deterministic winner.
    log_probs = tl.where(log_probs == log_probs, log_probs, -float("inf"))
    log_probs = tl.where(log_probs > 0.0, 0.0, log_probs)
    tl.store(
        top_log_probs + row * K + ranks,
        log_probs,
        mask=ranks < K,
    )


def _temperature_rows(
    temperature: float | Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, int] | None:
    if not isinstance(temperature, Tensor):
        return None
    if temperature.device != device:
        raise ValueError("temperature and logits must share a device")
    if not temperature.is_floating_point():
        raise TypeError("temperature tensor must use a floating dtype")
    if temperature.ndim == 0:
        return temperature.reshape(1), 0
    if temperature.shape not in ((batch_size,), (batch_size, 1)):
        raise ValueError(
            "temperature tensor must be scalar, [B], or [B, 1]; got "
            f"{tuple(temperature.shape)}"
        )
    values = temperature.reshape(batch_size)
    return values, values.stride(0)


def _build_candidate_log_probs(
    logits: Tensor,
    top_values: Tensor,
    top_log_probs: Tensor,
    partial_max: Tensor,
    partial_sum: Tensor,
    temperature: float | Tensor,
) -> None:
    """Normalize selected logits over the full vocabulary in FP32."""

    batch_size, num_depths, vocab_size = logits.shape
    num_rows = batch_size * num_depths
    candidate_top_k = int(top_values.size(-1))
    block_vocab = 8192
    num_blocks = triton.cdiv(vocab_size, block_vocab)
    temperature_rows = _temperature_rows(
        temperature,
        batch_size=batch_size,
        device=logits.device,
    )
    if temperature_rows is None:
        temperature_values = logits
        temperature_batch_stride = 0
        temperature_is_tensor = False
        scalar_temperature = float(temperature)
        inverse_temperature = (
            1.0 / scalar_temperature if scalar_temperature > 0.0 else 1.0
        )
    else:
        temperature_values, temperature_batch_stride = temperature_rows
        temperature_is_tensor = True
        inverse_temperature = 1.0

    _candidate_lse_partials_kernel[(batch_size, num_depths, num_blocks)](
        logits,
        partial_max,
        partial_sum,
        temperature_values,
        inverse_temperature,
        BATCH_STRIDE=logits.stride(0),
        DEPTH_STRIDE=logits.stride(1),
        VOCAB_STRIDE=logits.stride(-1),
        NUM_DEPTHS=num_depths,
        VOCAB_SIZE=vocab_size,
        NUM_BLOCKS=num_blocks,
        BLOCK_VOCAB=block_vocab,
        TEMPERATURE_BATCH_STRIDE=temperature_batch_stride,
        TEMPERATURE_IS_TENSOR=temperature_is_tensor,
        num_warps=4,
        num_stages=1,
    )
    _candidate_lse_finalize_kernel[(num_rows,)](
        top_values,
        partial_max,
        partial_sum,
        top_log_probs,
        temperature_values,
        inverse_temperature,
        K=candidate_top_k,
        NUM_DEPTHS=num_depths,
        NUM_BLOCKS=num_blocks,
        BLOCK_K=triton.next_power_of_2(candidate_top_k),
        BLOCK_PARTIALS=triton.next_power_of_2(num_blocks),
        TEMPERATURE_BATCH_STRIDE=temperature_batch_stride,
        TEMPERATURE_IS_TENSOR=temperature_is_tensor,
        num_warps=1,
        num_stages=1,
    )


@triton.jit
def _build_tree_kernel(
    top_token_ids,
    top_log_probs,
    draft_tokens,
    selected_edges,
    parent_list,
    search_depths,
    search_log_masses,
    TOKEN_BATCH_STRIDE: tl.constexpr,
    TOKEN_DEPTH_STRIDE: tl.constexpr,
    TOKEN_RANK_STRIDE: tl.constexpr,
    PROB_BATCH_STRIDE: tl.constexpr,
    PROB_DEPTH_STRIDE: tl.constexpr,
    PROB_RANK_STRIDE: tl.constexpr,
    PARENT_BATCH_STRIDE: tl.constexpr,
    NUM_DEPTHS: tl.constexpr,
    K: tl.constexpr,
    Q: tl.constexpr,
    BLOCK_CANDIDATES: tl.constexpr,
):
    batch = tl.program_id(0)
    search_offset = batch * Q
    edge_output_offset = batch * (Q - 1)
    parent_output_offset = batch * PARENT_BATCH_STRIDE
    candidate_slots = tl.arange(0, BLOCK_CANDIDATES)
    candidate_parents = candidate_slots // K
    candidate_ranks = candidate_slots % K
    candidate_in_bounds = candidate_slots < Q * K
    used = tl.zeros((BLOCK_CANDIDATES,), dtype=tl.int1)

    # The root token itself is returned by reference.  Only its private search
    # state is stored here; EAGLE prepends it as the verify-tree root later.
    tl.store(
        parent_list + parent_output_offset + candidate_slots,
        -1,
        mask=candidate_slots < PARENT_BATCH_STRIDE,
    )
    tl.store(search_depths + search_offset, 0)
    tl.store(search_log_masses + search_offset, 0.0)
    tl.debug_barrier()

    for node_index in range(1, Q):
        parent_valid = candidate_in_bounds & (candidate_parents < node_index)
        safe_parent = tl.where(parent_valid, candidate_parents, 0)
        parent_depth = tl.load(
            search_depths + search_offset + safe_parent,
            mask=parent_valid,
            other=NUM_DEPTHS,
        )
        parent_mass = tl.load(
            search_log_masses + search_offset + safe_parent,
            mask=parent_valid,
            other=-float("inf"),
        ).to(tl.float32)
        valid = parent_valid & (parent_depth < NUM_DEPTHS) & ~used
        safe_depth = tl.where(valid, parent_depth, 0)
        token = tl.load(
            top_token_ids
            + batch * TOKEN_BATCH_STRIDE
            + safe_depth * TOKEN_DEPTH_STRIDE
            + candidate_ranks * TOKEN_RANK_STRIDE,
            mask=valid,
            other=0,
        )
        log_prob = tl.load(
            top_log_probs
            + batch * PROB_BATCH_STRIDE
            + safe_depth * PROB_DEPTH_STRIDE
            + candidate_ranks * PROB_RANK_STRIDE,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        mass = parent_mass + log_prob
        child_depth = parent_depth + 1

        # Deterministic best-first order: mass, shallower depth, lower rank,
        # lower token ID, then lower parent node.
        best_mass = tl.max(tl.where(valid, mass, -float("inf")), axis=0)
        winner = valid & (mass == best_mass)
        best_depth = tl.min(tl.where(winner, child_depth, 1 << 30), axis=0)
        winner &= child_depth == best_depth
        best_rank = tl.min(tl.where(winner, candidate_ranks, 1 << 30), axis=0)
        winner &= candidate_ranks == best_rank
        best_token = tl.min(tl.where(winner, token, 1 << 30), axis=0)
        winner &= token == best_token
        best_parent = tl.min(tl.where(winner, candidate_parents, 1 << 30), axis=0)
        winner &= candidate_parents == best_parent
        best_slot = tl.min(tl.where(winner, candidate_slots, 1 << 30), axis=0)

        selected_parent = best_slot // K
        selected_rank = best_slot % K
        selected_depth = tl.load(search_depths + search_offset + selected_parent)
        selected_mass = tl.load(search_log_masses + search_offset + selected_parent).to(
            tl.float32
        ) + tl.load(
            top_log_probs
            + batch * PROB_BATCH_STRIDE
            + selected_depth * PROB_DEPTH_STRIDE
            + selected_rank * PROB_RANK_STRIDE
        ).to(tl.float32)
        selected_token = tl.load(
            top_token_ids
            + batch * TOKEN_BATCH_STRIDE
            + selected_depth * TOKEN_DEPTH_STRIDE
            + selected_rank * TOKEN_RANK_STRIDE
        )

        output_index = edge_output_offset + node_index - 1
        tl.store(draft_tokens + output_index, selected_token)
        # This is already EAGLE's implicit selected-edge encoding.
        tl.store(selected_edges + output_index, best_slot)
        if node_index < Q - 1:
            # EAGLE shifts each selected edge by one candidate row.  Fusing
            # this write avoids separate fill/copy launches on every step.
            tl.store(
                parent_list + parent_output_offset + node_index,
                best_slot,
            )
        tl.store(
            search_depths + search_offset + node_index,
            selected_depth + 1,
        )
        tl.store(
            search_log_masses + search_offset + node_index,
            selected_mass,
        )
        used |= candidate_slots == best_slot
        tl.debug_barrier()


def _candidate_tree_capacity(
    num_depths: int,
    candidate_top_k: int,
    stop_at: int,
) -> int:
    capacity = 1
    width = 1
    for _ in range(num_depths):
        width *= candidate_top_k
        capacity += width
        if capacity >= stop_at:
            break
    return capacity


@torch.inference_mode()
def build_uno_tree_proposal(
    root_tokens: Tensor,
    draft_logits: Tensor,
    *,
    max_nodes: int,
    candidate_top_k: int,
    temperature: float | Tensor,
    workspace: dict[str, Tensor] | None = None,
) -> UnoTreeProposal:
    """Build fixed-``Q`` UNO trees directly in EAGLE's proposal ABI.

    ``root_tokens`` is ``[B]`` and ``draft_logits`` is ``[B, F-1, V]``.
    Candidate log probabilities are normalized over the full vocabulary.  No
    CPU reference/fallback, direct parent array, attention mask, traversal
    structure, acceptance walk, or KV operation is implemented here.
    """

    if root_tokens.ndim != 1:
        raise ValueError(
            f"root_tokens must have shape [B], got {tuple(root_tokens.shape)}"
        )
    if draft_logits.ndim != 3 or draft_logits.size(0) != root_tokens.size(0):
        raise ValueError(
            "draft_logits must have shape [B, depth, vocab] with the same B "
            f"as root_tokens; got {tuple(draft_logits.shape)}"
        )
    if not root_tokens.is_cuda or not draft_logits.is_cuda:
        raise ValueError("UNO tree construction requires CUDA tensors")
    if root_tokens.device != draft_logits.device:
        raise ValueError("tree roots and draft logits must share a device")
    if root_tokens.dtype not in (torch.int32, torch.int64):
        raise TypeError("tree roots must use an integer dtype")
    if not draft_logits.is_floating_point():
        raise TypeError("draft logits must use a floating dtype")
    if root_tokens.numel() == 0:
        raise ValueError("UNO tree construction requires a non-empty batch")
    if max_nodes < 1:
        raise ValueError("max_nodes must include at least the root")
    if candidate_top_k < 1:
        raise ValueError("candidate_top_k must be >= 1")

    batch_size = int(root_tokens.size(0))
    num_depths = int(draft_logits.size(1))
    vocab_size = int(draft_logits.size(2))
    candidate_top_k = int(candidate_top_k)
    draft_width = num_depths + 1
    parent_width = candidate_top_k * max(num_depths - 1, 0) + 1
    if max_nodes < draft_width:
        raise ValueError(
            f"max_nodes Q must be >= draft width F; got Q={max_nodes}, F={draft_width}"
        )
    if candidate_top_k > vocab_size:
        raise ValueError(
            f"candidate_top_k ({candidate_top_k}) exceeds vocabulary size "
            f"({vocab_size})"
        )
    if max_nodes > 128 or max_nodes * candidate_top_k > 2048:
        raise ValueError(
            "the initial single-program UNO builder requires Q <= 128 and "
            f"Q*K <= 2048; got Q={max_nodes}, K={candidate_top_k}"
        )
    capacity = _candidate_tree_capacity(
        num_depths,
        candidate_top_k,
        max_nodes,
    )
    if capacity < max_nodes:
        raise ValueError(
            f"candidate set can produce only {capacity} tree nodes, but "
            f"fixed tree verification requires {max_nodes}"
        )
    if max_nodes - 1 > parent_width:
        raise ValueError(
            "EAGLE's parent-list ABI cannot represent this UNO tree: "
            f"Q-1={max_nodes - 1} exceeds K*(depth-1)+1={parent_width}"
        )
    _temperature_rows(
        temperature,
        batch_size=batch_size,
        device=draft_logits.device,
    )

    def buffer(
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> Tensor:
        if workspace is None:
            return torch.empty(
                shape,
                dtype=dtype,
                device=draft_logits.device,
            )
        value = workspace.get(name)
        if (
            value is None
            or value.shape != shape
            or value.dtype != dtype
            or value.device != draft_logits.device
        ):
            value = torch.empty(
                shape,
                dtype=dtype,
                device=draft_logits.device,
            )
            workspace[name] = value
        return value

    edge_shape = (batch_size, max_nodes - 1)
    draft_tokens = buffer("draft_tokens", edge_shape, torch.long)
    selected_edges = buffer("top_scores_index", edge_shape, torch.long)
    parent_list = buffer(
        "parent_list",
        (batch_size, parent_width),
        torch.long,
    )

    if max_nodes == 1:
        parent_list.fill_(-1)
        return UnoTreeProposal(
            root_tokens=root_tokens,
            draft_tokens=draft_tokens,
            parent_list=parent_list,
            top_scores_index=selected_edges,
            candidate_top_k=candidate_top_k,
            max_depth=num_depths,
        )

    candidate_shape = (batch_size, num_depths, candidate_top_k)
    flat_logits = draft_logits.contiguous().view(batch_size * num_depths, vocab_size)
    flat_top_values, flat_top_token_ids = _flashinfer_top_k(
        flat_logits,
        candidate_top_k,
        sorted=True,
        deterministic=False,
    )
    top_values = flat_top_values.view(candidate_shape)
    top_token_ids = buffer("top_token_ids", candidate_shape, torch.long)
    top_token_ids.copy_(flat_top_token_ids.view(candidate_shape))
    top_log_probs = buffer("top_log_probs", candidate_shape, torch.float32)
    num_partial_blocks = (vocab_size + 8191) // 8192
    partial_shape = (batch_size * num_depths, num_partial_blocks)
    _build_candidate_log_probs(
        draft_logits,
        top_values,
        top_log_probs,
        buffer("partial_lse_max", partial_shape, torch.float32),
        buffer("partial_lse_sum", partial_shape, torch.float32),
        temperature,
    )

    search_shape = (batch_size, max_nodes)
    search_depths = buffer("search_depths", search_shape, torch.int32)
    search_log_masses = buffer("search_log_masses", search_shape, torch.float32)
    _build_tree_kernel[(batch_size,)](
        top_token_ids,
        top_log_probs,
        draft_tokens,
        selected_edges,
        parent_list,
        search_depths,
        search_log_masses,
        TOKEN_BATCH_STRIDE=top_token_ids.stride(0),
        TOKEN_DEPTH_STRIDE=top_token_ids.stride(1),
        TOKEN_RANK_STRIDE=top_token_ids.stride(2),
        PROB_BATCH_STRIDE=top_log_probs.stride(0),
        PROB_DEPTH_STRIDE=top_log_probs.stride(1),
        PROB_RANK_STRIDE=top_log_probs.stride(2),
        PARENT_BATCH_STRIDE=parent_list.stride(0),
        NUM_DEPTHS=num_depths,
        K=candidate_top_k,
        Q=max_nodes,
        BLOCK_CANDIDATES=triton.next_power_of_2(max_nodes * candidate_top_k),
        num_warps=8,
        num_stages=1,
    )

    return UnoTreeProposal(
        root_tokens=root_tokens,
        draft_tokens=draft_tokens,
        parent_list=parent_list,
        top_scores_index=selected_edges,
        candidate_top_k=candidate_top_k,
        max_depth=num_depths,
    )
