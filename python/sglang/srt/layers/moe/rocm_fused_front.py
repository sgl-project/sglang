"""Glue for the one-launch ROCm MoE front (:func:`rocm_router_gate_sort`). aiter's sorting arguments
appear only inside ``aiter.fused_moe``'s ``moe_sorting`` call, so the override records them per router
on first sight and the next gate launch of that router sorts in the same launch; the override hands
those outputs back only after checking they match its arguments, else it sorts the ids again."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.kernels.ops.moe.rocm_router_gate import rocm_router_gate
from sglang.kernels.ops.moe.rocm_router_gate_sort import (
    ROCM_GATE_SORT_MAX_TOKENS,
    rocm_router_gate_sort,
)

SortOutputs = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


@dataclass(frozen=True)
class SortConfig:
    """What ``moe_sorting`` was asked for, beyond the ids themselves."""

    local_expert_ids: torch.Tensor  # cached per expert mask, compared by storage
    num_experts: int
    model_dim: int
    moe_buf_dtype: torch.dtype
    block_size: int
    zero_moe_buf: bool

    def same_as(self, other: SortConfig) -> bool:
        return (
            self.local_expert_ids.data_ptr() == other.local_expert_ids.data_ptr()
            and self.local_expert_ids.numel() == other.local_expert_ids.numel()
            and self.num_experts == other.num_experts
            and self.model_dim == other.model_dim
            and self.moe_buf_dtype == other.moe_buf_dtype
            and self.block_size == other.block_size
            and self.zero_moe_buf == other.zero_moe_buf
        )


@dataclass
class _PendingSort:
    key: tuple
    config: Optional[SortConfig]  # None: the gate ran alone
    num_token_non_padded: Optional[torch.Tensor]
    ids: torch.Tensor  # holds the storage, so the key cannot be reused while pending
    outputs: Optional[SortOutputs]


# process-wide by design: aiter's moe_sorting call has no per-layer hook; forward runs on one thread, no lock
# router key -> sorting arguments seen for it (None: the sorting is not fusable for this key)
_sort_configs: dict[tuple, Optional[SortConfig]] = {}
# ids storage -> what the gate launch of that batch produced
_pending_sorts: dict[int, _PendingSort] = {}
# on overflow the pending sorts are dropped and the override re-sorts the ids (correct, one launch slower)
_PENDING_LIMIT = 256


def _router_key(
    correction_bias: Optional[torch.Tensor],
    num_tokens: int,
    topk: int,
    num_experts: int,
) -> tuple:
    # the bias is a per-layer parameter: it identifies the router, and with it the layer's mask
    bias = 0 if correction_bias is None else correction_bias.data_ptr()
    return (bias, num_tokens, topk, num_experts)


def _same_count(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.data_ptr() == b.data_ptr()


def gate_partials(
    gating_output: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    topk: int,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    partials: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The ROCm decode gate on split-K partials: aiter's ``topk_gating`` weights and ids,
    with aiter's sorting folded into the launch when this router's sorting arguments are
    known and the batch is small enough."""
    num_tokens, num_experts = gating_output.shape
    key = _router_key(correction_bias, num_tokens, topk, num_experts)
    config = _sort_configs.get(key) if num_tokens <= ROCM_GATE_SORT_MAX_TOKENS else None
    if config is None:
        weights, ids = rocm_router_gate(
            gating_output,
            correction_bias,
            topk,
            renormalize,
            routed_scaling_factor,
            partials=partials,
        )
        outputs = None
    else:
        gate_and_sort = rocm_router_gate_sort(
            gating_output,
            correction_bias,
            topk,
            renormalize,
            routed_scaling_factor,
            partials,
            config.local_expert_ids,
            config.num_experts,
            config.model_dim,
            config.moe_buf_dtype,
            config.block_size,
            config.zero_moe_buf,
            num_token_non_padded=num_token_non_padded,
        )
        weights, ids = gate_and_sort[0], gate_and_sort[1]
        outputs = gate_and_sort[2:]
    if num_tokens <= ROCM_GATE_SORT_MAX_TOKENS:
        if len(_pending_sorts) >= _PENDING_LIMIT:
            _pending_sorts.clear()
        _pending_sorts[ids.data_ptr()] = _PendingSort(
            key, config, num_token_non_padded, ids, outputs
        )
    return weights, ids


def _pop_pending(topk_ids: torch.Tensor) -> Optional[_PendingSort]:
    pending = _pending_sorts.pop(topk_ids.data_ptr(), None)
    if pending is None:
        return None
    if (
        pending.ids.shape != topk_ids.shape
        or pending.ids.dtype != topk_ids.dtype
        or not topk_ids.is_contiguous()
    ):
        return None
    return pending


def take_pending_sort(
    topk_ids: torch.Tensor,
    config: SortConfig,
    num_token_non_padded: Optional[torch.Tensor],
) -> Optional[SortOutputs]:
    """The sorting outputs the gate launch of ``topk_ids`` already produced for exactly
    ``config`` and ``num_token_non_padded``, or None (and the arguments are recorded for the
    router's next gate launch)."""
    pending = _pop_pending(topk_ids)
    if pending is None:
        return None
    if (
        pending.outputs is not None
        and pending.config is not None
        and pending.config.same_as(config)
        and _same_count(pending.num_token_non_padded, num_token_non_padded)
    ):
        return pending.outputs
    _sort_configs[pending.key] = config
    return None


def disable_pending_sort(topk_ids: torch.Tensor) -> None:
    """``moe_sorting`` was called in a form the fused launch cannot serve: stop fusing the
    sorting for this router."""
    pending = _pop_pending(topk_ids)
    if pending is not None:
        _sort_configs[pending.key] = None
