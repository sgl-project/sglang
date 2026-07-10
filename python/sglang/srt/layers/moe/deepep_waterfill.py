# Copyright 2023-2026 SGLang Team
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
"""DeepEP Waterfill: shared expert as 9th routed expert, dispatched to least-loaded rank."""

from typing import Optional, Tuple

import torch
from torch import Tensor

from sglang.kernels.ops.moe.deepep_waterfill_kernels import (
    LOCAL_SHARED_MARKER,
    WaterfillDispatchPlan,
    _count_routed_per_rank_kernel,
    _empty_expanded,
    materialize_waterfill_dispatch_fused,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import StandardTopKOutput


@torch.compile(dynamic=True)
def expand_topk_with_shared_expert(
    topk_ids: Tensor,
    topk_weights: Tensor,
    num_routed_experts: int,
    world_size: int,
    source_rank: int,
    shared_weight: float,
) -> Tuple[Tensor, Tensor]:
    """Expand topk [N, 8] → [N, 9] with ID remap; shared expert always local."""
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    device = topk_ids.device
    old_epr = num_routed_experts // world_size
    new_epr = old_epr + 1
    has_valid = (topk_ids >= 0).any(dim=1)
    valid_mask = topk_ids >= 0
    old_ranks = torch.where(valid_mask, topk_ids // old_epr, torch.zeros_like(topk_ids))
    expanded_topk_ids = torch.empty(
        num_tokens, topk + 1, dtype=topk_ids.dtype, device=device
    )
    expanded_topk_ids[:, :topk] = torch.where(
        valid_mask, topk_ids + old_ranks, topk_ids
    )

    shared_id = source_rank * new_epr + old_epr
    expanded_topk_ids[:, topk] = torch.where(has_valid, shared_id, LOCAL_SHARED_MARKER)
    expanded_topk_weights = torch.empty(
        num_tokens, topk + 1, dtype=topk_weights.dtype, device=device
    )
    expanded_topk_weights[:, :topk] = torch.where(valid_mask, topk_weights, 0.0)
    expanded_topk_weights[:, topk] = torch.where(has_valid, shared_weight, 0.0).to(
        topk_weights.dtype
    )
    return expanded_topk_ids, expanded_topk_weights


class DeepEPWaterfillBalancer:
    """Waterfill load balancer: shared expert fused as real routed expert (topk 8→9)."""

    MIN_BATCH_FOR_BALANCE = 64

    def __init__(
        self,
        num_routed_experts: int,
        world_size: int,
        rank: int,
        layer_id: int,
        routed_scaling_factor: float = 1.0,
    ):
        self.num_routed_experts = num_routed_experts
        self.world_size = world_size
        self.rank = rank
        self.layer_id = layer_id
        self.old_experts_per_rank = num_routed_experts // world_size
        self.shared_weight = (
            1.0 / routed_scaling_factor if routed_scaling_factor != 0 else 1.0
        )
        self._counts_buf: Optional[Tensor] = None
        self.use_static_waterfill = not envs.SGLANG_DISABLE_STATIC_WATERFILL.get()

    def count_local_routed(self, topk_ids: Tensor) -> Tensor:
        """Count routed tokens per rank via Triton kernel (uses original expert IDs)."""
        if self._counts_buf is None:
            self._counts_buf = torch.zeros(
                self.world_size, dtype=torch.int64, device=topk_ids.device
            )
        buf = self._counts_buf
        buf.zero_()
        num_tokens = topk_ids.shape[0]
        if num_tokens == 0:
            return buf
        topk = topk_ids.shape[1]
        BLOCK_SIZE = 256
        grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _count_routed_per_rank_kernel[grid](
            topk_ids,
            buf,
            num_tokens,
            topk,
            self.old_experts_per_rank,
            self.world_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return buf

    def _is_low_batch(self, num_tokens: int) -> bool:
        """Return whether waterfill should skip balancing for small batches."""
        return num_tokens < self.MIN_BATCH_FOR_BALANCE

    def _can_skip_dispatch_plan_for_low_batch(self, num_tokens: int) -> bool:
        """Return whether static mode can skip dispatch-plan setup entirely."""
        return self.use_static_waterfill and self._is_low_batch(num_tokens)

    def _build_static_dispatch_plan(
        self, routed_counts: Tensor
    ) -> WaterfillDispatchPlan:
        """Build static-mode Waterfill inputs from current local routed counts."""
        return WaterfillDispatchPlan(
            rank_load=routed_counts,
            allow_all_ranks=True,
            target_total=0,
        )

    def _build_dynamic_dispatch_plan(
        self,
        routed_counts: Tensor,
        local_tokens_per_rank: Optional[Tensor],
        topk: int,
    ) -> WaterfillDispatchPlan:
        """Build dynamic waterfill inputs from globally reduced routed counts."""
        # Dynamic Waterfill balances against effective rank load: globally
        # reduced routed counts plus each rank's active token count.
        rank_load = (
            routed_counts + local_tokens_per_rank
            if local_tokens_per_rank is not None
            else routed_counts
        )
        total_routed_t = routed_counts.sum()
        total_tokens_global_t = total_routed_t // topk
        total_effective_t = rank_load.sum()
        max_effective_t = rank_load.max()
        target_total = int(
            (total_effective_t + total_tokens_global_t + self.world_size - 1)
            // self.world_size
        )
        allow_all_ranks = bool(max_effective_t <= target_total)
        return WaterfillDispatchPlan(
            rank_load=rank_load,
            allow_all_ranks=allow_all_ranks,
            target_total=target_total,
        )

    @staticmethod
    def _all_reduce_dynamic_rank_load(
        local_routed_counts: Tensor, num_tokens: int
    ) -> Tuple[Tensor, Tensor]:
        """Aggregate dynamic load with SGLang EP communication."""
        from sglang.srt.distributed import get_moe_ep_group
        from sglang.srt.distributed.communication_op import (
            moe_expert_parallel_all_reduce,
        )

        group = get_moe_ep_group()
        world = group.world_size
        buf = torch.zeros(
            world * 2, dtype=torch.int64, device=local_routed_counts.device
        )
        buf[:world] = local_routed_counts
        rank = group.rank_in_group
        buf[world + rank : world + rank + 1].fill_(num_tokens)
        buf = moe_expert_parallel_all_reduce(buf)
        return buf[:world], buf[world:]

    def _build_dispatch_plan(
        self, topk_ids: Tensor, num_tokens: int
    ) -> Optional[WaterfillDispatchPlan]:
        """Prepare dispatch state for the waterfill selection boundary."""
        local_routed_counts = self.count_local_routed(topk_ids)
        if self.use_static_waterfill:
            return self._build_static_dispatch_plan(local_routed_counts)

        global_routed_counts, local_tokens_per_rank = (
            DeepEPWaterfillBalancer._all_reduce_dynamic_rank_load(
                local_routed_counts, num_tokens
            )
        )
        if self._is_low_batch(num_tokens):
            return None
        return self._build_dynamic_dispatch_plan(
            global_routed_counts,
            local_tokens_per_rank=local_tokens_per_rank,
            topk=topk_ids.shape[1],
        )

    def _materialize_dispatch(
        self,
        topk_ids: Tensor,
        topk_weights: Tensor,
        dispatch_plan: WaterfillDispatchPlan,
    ) -> Tuple[Tensor, Tensor]:
        """Expand TopK using local expansion or fused Waterfill."""
        num_tokens = topk_ids.shape[0]
        if num_tokens == 0:
            return _empty_expanded(topk_ids, topk_weights)

        if self._is_low_batch(num_tokens):
            return expand_topk_with_shared_expert(
                topk_ids,
                topk_weights,
                self.num_routed_experts,
                self.world_size,
                self.rank,
                self.shared_weight,
            )

        return materialize_waterfill_dispatch_fused(
            topk_ids,
            topk_weights,
            dispatch_plan.rank_load,
            self.num_routed_experts,
            self.world_size,
            self.rank,
            self.shared_weight,
            allow_all_ranks=dispatch_plan.allow_all_ranks,
            target_total=dispatch_plan.target_total,
        )

    @staticmethod
    def _with_expanded_topk(
        topk_output: StandardTopKOutput,
        expanded_ids: Tensor,
        expanded_weights: Tensor,
    ) -> StandardTopKOutput:
        """Wrap expanded tensors back into SGLang's StandardTopKOutput."""
        return StandardTopKOutput(
            topk_weights=expanded_weights,
            topk_ids=expanded_ids,
            router_logits=topk_output.router_logits,
        )

    def _expand_local_shared(
        self, topk_output: StandardTopKOutput
    ) -> StandardTopKOutput:
        expanded_ids, expanded_weights = expand_topk_with_shared_expert(
            topk_output.topk_ids,
            topk_output.topk_weights,
            self.num_routed_experts,
            self.world_size,
            self.rank,
            self.shared_weight,
        )
        return self._with_expanded_topk(topk_output, expanded_ids, expanded_weights)

    def expand_topk(
        self, topk_output: StandardTopKOutput, num_tokens: int
    ) -> StandardTopKOutput:
        """Expand topk [N, 8] -> [N, 9] with waterfill-assigned shared expert."""
        if self._can_skip_dispatch_plan_for_low_batch(num_tokens):
            # Static mode can use local expansion without communication for small
            # decode-sized batches. Dynamic mode still all-reduces before local
            # expansion so all ranks participate consistently.
            return self._expand_local_shared(topk_output)

        dispatch_plan = self._build_dispatch_plan(topk_output.topk_ids, num_tokens)
        if dispatch_plan is None:
            if num_tokens == 0:
                expanded_ids, expanded_weights = _empty_expanded(
                    topk_output.topk_ids, topk_output.topk_weights
                )
                return self._with_expanded_topk(
                    topk_output, expanded_ids, expanded_weights
                )
            else:
                return self._expand_local_shared(topk_output)
        expanded_ids, expanded_weights = self._materialize_dispatch(
            topk_output.topk_ids,
            topk_output.topk_weights,
            dispatch_plan,
        )
        return self._with_expanded_topk(topk_output, expanded_ids, expanded_weights)
