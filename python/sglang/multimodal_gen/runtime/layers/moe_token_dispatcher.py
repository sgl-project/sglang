# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import msgspec
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    sequence_parallel_is_initialized,
)

NON_LOCAL_EXPERT_ID = -1


class FixedCapacityMoeDispatchedTokens(msgspec.Struct, frozen=True):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    local_expert_ids: torch.Tensor
    num_source_tokens: int


def _validate_dispatch_inputs(
    *,
    global_expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    if global_expert_ids.shape != topk_weights.shape:
        raise ValueError(
            "global_expert_ids and topk_weights must have the same shape, got "
            f"{tuple(global_expert_ids.shape)} and {tuple(topk_weights.shape)}"
        )
    if global_expert_ids.ndim != 2:
        raise ValueError("global_expert_ids and topk_weights must be two-dimensional")


class FixedCapacityMoeTokenDispatcher:
    """Equal-split all-to-all dispatcher for ranks with equal token counts."""

    def __init__(
        self,
        *,
        process_group: ProcessGroup,
        ep_size: int,
        num_local_experts: int,
    ) -> None:
        self._process_group = process_group
        self._ep_size = ep_size
        self._num_local_experts = num_local_experts

    def _all_to_all(self, tensor: torch.Tensor) -> torch.Tensor:
        # The reused SP group guarantees the equal row counts required here.
        output = torch.empty_like(tensor)
        dist.all_to_all_single(output, tensor.contiguous(), group=self._process_group)
        return output

    def dispatch(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        global_expert_ids: torch.Tensor,
    ) -> FixedCapacityMoeDispatchedTokens:
        _validate_dispatch_inputs(
            global_expert_ids=global_expert_ids,
            topk_weights=topk_weights,
        )
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[0] != global_expert_ids.shape[0]
        ):
            raise ValueError(
                "hidden_states must be two-dimensional and match the routed token count"
            )

        num_tokens = hidden_states.shape[0]
        destination_ranks = torch.arange(
            self._ep_size,
            device=global_expert_ids.device,
        ).view(self._ep_size, 1, 1)
        owner_ranks = torch.div(
            global_expert_ids,
            self._num_local_experts,
            rounding_mode="floor",
        )
        local_routes = owner_ranks.unsqueeze(0) == destination_ranks
        local_ids = (
            global_expert_ids.unsqueeze(0) - destination_ranks * self._num_local_experts
        )
        local_ids = torch.where(
            local_routes,
            local_ids,
            torch.full_like(local_ids, NON_LOCAL_EXPERT_ID),
        )

        # Non-local routes keep their weights; -1 ids make the reducer skip them.
        return FixedCapacityMoeDispatchedTokens(
            hidden_states=self._all_to_all(hidden_states.repeat(self._ep_size, 1)),
            topk_weights=self._all_to_all(topk_weights.repeat(self._ep_size, 1)),
            local_expert_ids=self._all_to_all(local_ids.flatten(0, 1).to(torch.int32)),
            num_source_tokens=num_tokens,
        )

    def combine(
        self,
        *,
        expert_output: torch.Tensor,
        dispatched: FixedCapacityMoeDispatchedTokens,
    ) -> torch.Tensor:
        returned = self._all_to_all(expert_output)
        return returned.view(
            self._ep_size,
            dispatched.num_source_tokens,
            expert_output.shape[-1],
        ).sum(dim=0)


def create_moe_token_dispatcher(
    *,
    ep_size: int,
    num_local_experts: int,
) -> FixedCapacityMoeTokenDispatcher | None:
    if ep_size <= 1:
        return None
    if not sequence_parallel_is_initialized():
        raise RuntimeError(
            "expert parallelism reuses the SP group, but it is not initialized"
        )
    sp_group = get_sp_group()
    if sp_group.world_size != ep_size:
        raise RuntimeError(
            "expert parallelism reuses the SP group, so ep_size must equal the SP "
            f"world size; got ep_size={ep_size}, sp_world_size={sp_group.world_size}"
        )
    return FixedCapacityMoeTokenDispatcher(
        process_group=sp_group.device_group,
        ep_size=ep_size,
        num_local_experts=num_local_experts,
    )
