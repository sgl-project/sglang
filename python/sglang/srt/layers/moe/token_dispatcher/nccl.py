from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInputFormat,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopKOutput,
    TopKOutputChecker,
)


@dataclass
class NcclRouteHandle:
    """State required to reverse one variable-split dispatch."""

    send_counts: torch.Tensor
    recv_counts: torch.Tensor
    send_token_idx: torch.Tensor
    send_route_idx: torch.Tensor
    num_input_tokens: int

    @property
    def send_splits(self) -> list[int]:
        return self.send_counts.cpu().tolist()

    @property
    def recv_splits(self) -> list[int]:
        return self.recv_counts.cpu().tolist()


class NcclDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    topk_output: StandardTopKOutput
    route_handle: NcclRouteHandle

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.NCCL


class NcclCombineInput(NamedTuple):
    hidden_states: torch.Tensor
    route_handle: NcclRouteHandle

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.NCCL


class NcclDispatcher(BaseDispatcher):
    """Correctness-first NCCL dispatcher for CUDA architectures such as SM80.

    The eager implementation emits one record per token/expert pair. It uses
    PyTorch's variable-split ``all_to_all_single`` and intentionally does not
    claim CUDA-graph support or a fused metadata transport yet.
    """

    def __init__(self, group: dist.ProcessGroup, moe_runner_config: MoeRunnerConfig):
        super().__init__()
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.num_experts = moe_runner_config.num_experts
        self.num_local_experts = moe_runner_config.num_local_experts
        self.num_fused_shared_experts = moe_runner_config.num_fused_shared_experts or 0

        if self.num_experts is None or self.num_local_experts is None:
            raise ValueError("NCCL dispatcher requires global and local expert counts")
        if self.num_fused_shared_experts:
            raise NotImplementedError(
                "NCCL dispatcher does not yet support fused shared experts"
            )
        if self.num_local_experts * self.world_size != self.num_experts:
            raise ValueError(
                "NCCL dispatcher currently requires an equal contiguous expert "
                "partition: num_local_experts * world_size == num_experts"
            )

    def _exchange_counts(self, send_counts: torch.Tensor) -> torch.Tensor:
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        return recv_counts

    def _exchange(
        self,
        tensor: torch.Tensor,
        send_splits: list[int],
        recv_splits: list[int],
    ) -> torch.Tensor:
        output = torch.empty(
            (sum(recv_splits), *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_to_all_single(
            output,
            tensor,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.group,
        )
        return output

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> NcclDispatchOutput:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "NCCL MoE dispatcher is correctness-first and eager-only; "
                "launch with --disable-cuda-graph"
            )
        if hidden_states.dtype != torch.bfloat16:
            raise NotImplementedError(
                "NCCL MoE dispatcher currently supports BF16 activations only"
            )
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise TypeError("NCCL MoE dispatcher requires standard top-k output")

        topk_ids = topk_output.topk_ids.to(torch.int64)
        topk_weights = topk_output.topk_weights.to(torch.float32)
        num_tokens, top_k = topk_ids.shape
        token_idx = torch.arange(
            num_tokens, dtype=torch.int64, device=hidden_states.device
        ).repeat_interleave(top_k)
        # Retain the original flattened (token, top-k slot) row identity. The
        # reverse all-to-all returns rows in destination-chunk order, which
        # differs across physical expert maps; combine uses this key to restore
        # the model's canonical top-k accumulation order.
        route_idx = torch.arange(
            num_tokens * top_k, dtype=torch.int64, device=hidden_states.device
        )
        expert_idx = topk_ids.reshape(-1)
        router_weight = topk_weights.reshape(-1)

        valid = (expert_idx >= 0) & (expert_idx < self.num_experts)
        token_idx = token_idx[valid]
        route_idx = route_idx[valid]
        expert_idx = expert_idx[valid]
        router_weight = router_weight[valid]
        destination = torch.div(
            expert_idx, self.num_local_experts, rounding_mode="floor"
        )

        order = torch.argsort(destination, stable=True)
        destination = destination[order]
        send_counts = torch.bincount(
            destination, minlength=self.world_size
        ).to(torch.int64)
        recv_counts = self._exchange_counts(send_counts)
        send_splits = send_counts.cpu().tolist()
        recv_splits = recv_counts.cpu().tolist()

        send_token_idx = token_idx[order].contiguous()
        send_route_idx = route_idx[order].contiguous()
        send_expert_idx = expert_idx[order].contiguous()
        send_weight = router_weight[order].contiguous()
        send_hidden_states = hidden_states[send_token_idx].contiguous()

        recv_hidden_states = self._exchange(
            send_hidden_states, send_splits, recv_splits
        )
        recv_expert_idx = self._exchange(send_expert_idx, send_splits, recv_splits)
        recv_weight = self._exchange(send_weight, send_splits, recv_splits)

        local_expert_idx = torch.remainder(
            recv_expert_idx, self.num_local_experts
        ).to(torch.int32).unsqueeze(1)
        local_weight = recv_weight.unsqueeze(1)
        local_topk_output = StandardTopKOutput(
            topk_weights=local_weight,
            topk_ids=local_expert_idx,
            router_logits=torch.empty(
                (recv_hidden_states.shape[0], 0),
                dtype=torch.float32,
                device=hidden_states.device,
            ),
        )
        handle = NcclRouteHandle(
            send_counts=send_counts,
            recv_counts=recv_counts,
            send_token_idx=send_token_idx,
            send_route_idx=send_route_idx,
            num_input_tokens=num_tokens,
        )
        return NcclDispatchOutput(
            hidden_states=recv_hidden_states,
            topk_output=local_topk_output,
            route_handle=handle,
        )

    def combine(self, combine_input: NcclCombineInput) -> torch.Tensor:
        hidden_states, handle = combine_input
        recv_splits = handle.recv_splits
        send_splits = handle.send_splits
        expected_records = sum(recv_splits)
        if hidden_states.shape[0] != expected_records:
            raise RuntimeError(
                "NCCL MoE runner changed the dispatched record count: "
                f"output={hidden_states.shape[0]}, "
                f"expected={expected_records}"
            )

        returned_hidden_states = self._exchange(
            hidden_states,
            recv_splits,
            send_splits,
        )
        # Reverse all-to-all preserves row order inside every peer chunk. The
        # returned rows therefore line up with the source-side send order, so
        # token indices never need to cross the network in either direction.
        #
        # Do not use CUDA scatter_add_ here: every token has up to top-k rows,
        # so duplicate indices would be accumulated by unordered atomics. That
        # made greedy generations placement-dependent and non-reproducible.
        # Sorting once and adding one unique token row at a time keeps the
        # accumulation order fixed while preserving the BF16/FP32 dtype. The
        # route key restores the original flattened top-k slot order, so the
        # result is independent of which physical map determined chunk order.
        return self._deterministic_combine(
            returned_hidden_states,
            handle.send_token_idx,
            handle.num_input_tokens,
            handle.send_route_idx,
        )

    @staticmethod
    def _deterministic_combine(
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor,
        num_input_tokens: int,
        route_order: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sum dispatched rows per token without duplicate-index atomics."""
        if hidden_states.shape[0] != token_indices.numel():
            raise RuntimeError(
                "NCCL combine token-index count does not match returned rows: "
                f"rows={hidden_states.shape[0]}, indices={token_indices.numel()}"
            )
        if route_order is None:
            route_order = torch.arange(
                hidden_states.shape[0], dtype=torch.int64, device=hidden_states.device
            )
        if route_order.numel() != token_indices.numel():
            raise RuntimeError(
                "NCCL combine route-order count does not match returned rows: "
                f"rows={hidden_states.shape[0]}, route_order={route_order.numel()}"
            )

        combined = torch.zeros(
            (num_input_tokens, hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if hidden_states.shape[0] == 0:
            return combined

        # First restore the original flattened top-k order, then stably group
        # by token. Stable sorting on the second key preserves slot order within
        # each token regardless of destination/expert chunk order.
        route_order_idx = torch.argsort(route_order, stable=True)
        route_ordered_tokens = token_indices.index_select(0, route_order_idx)
        token_order = torch.argsort(route_ordered_tokens, stable=True)
        order = route_order_idx.index_select(0, token_order)
        sorted_token_indices = token_indices.index_select(0, order)
        sorted_hidden_states = hidden_states.index_select(0, order)
        counts = torch.bincount(
            sorted_token_indices,
            minlength=num_input_tokens,
        )
        starts = torch.cumsum(counts, dim=0) - counts

        # The maximum count is top-k for valid rows (and less for padded rows),
        # so this loop is bounded by the model's small routing fan-out.
        for slot in range(int(counts.max().item())):
            active_tokens = torch.nonzero(counts > slot, as_tuple=False).flatten()
            sorted_rows = starts[active_tokens] + slot
            updates = combined.index_select(0, active_tokens) + sorted_hidden_states.index_select(
                0, sorted_rows
            )
            combined.index_copy_(0, active_tokens, updates)
        return combined
