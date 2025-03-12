try:
    from deep_ep import Buffer

    use_deepep = True
except ImportError:
    use_deepep = False

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.model_executor.forward_batch_info import ForwardMode

_buffer = None
_low_latency_buffer = None


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """
    Copy from DeepEP example usage in model inference prefilling.
    https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-model-training-or-inference-prefilling
    """

    global _buffer

    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def get_low_latency_buffer(
    group: dist.ProcessGroup,
    num_max_dispatch_tokens_per_rank: int,
    hidden: int,
    num_experts: int,
):
    """
    Copy from DeepEP example usage in model inference decoding.
    https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
    """

    global _low_latency_buffer
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank, hidden, group.size(), num_experts
    )

    if (
        _low_latency_buffer is None
        or _low_latency_buffer.group != group
        or not _low_latency_buffer.low_latency_mode
        or _low_latency_buffer.num_rdma_bytes < num_rdma_bytes
    ):
        assert num_experts % group.size() == 0
        _low_latency_buffer = Buffer(
            group,
            0,
            num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_experts // group.size(),
        )
    return _low_latency_buffer


def permute(
    tokens,
    routing_map,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """
    Copy from Megatron-Core moe for token permutation
    https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py#L221
    """

    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[1]
    if drop_and_pad and not (num_out_tokens is None):
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
            :, :capacity
        ].contiguous()
        sorted_indices = sorted_indices.view(-1)
    else:
        routing_map = routing_map.bool().T.contiguous()
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device)
            .unsqueeze(0)
            .expand(num_experts, -1)
        )
        sorted_indices = token_indices.masked_select(routing_map)
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """
    Copy from Megatron-Core moe for token unpermutation
    https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py#L282
    """

    _, hidden = restore_shape

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size(1)
            num_permuted_tokens = sorted_indices.size(0)
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = probs.size(0)

            probs_T_1D = probs.T.contiguous().view(-1)

            indices_dim0 = torch.arange(
                num_experts, device=routing_map.device
            ).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)

            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            permuted_probs = probs.T.contiguous().masked_select(
                routing_map.T.contiguous()
            )
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    output_tokens = torch.zeros(
        restore_shape, device=permuted_tokens.device, dtype=permuted_tokens.dtype
    )
    output_tokens.scatter_add_(
        0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens
    )

    return output_tokens


class DeepEPManager:
    """
    Copy from Megatron-Core token_dispatcher using DeepEP backend
    https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py#L696
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        capacity_factor: float = None,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
    ):
        self.group = group
        self.router_topk = router_topk
        self.capacity_factor = capacity_factor
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.recv_expert_count = None
        self.tokens_per_expert = None
        self.params_dtype = params_dtype
        self.params_bytes = 2
        # Metadata
        self.token_indices = None
        self.token_probs = None
        # Handle used for combine operation
        self.handle = None

        if not use_deepep:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )
        self.buffer = get_buffer(self.group, self.hidden_size * self.params_bytes)
        # self.low_latency_buffer = get_low_latency_buffer(
        #     self.group, 256, self.hidden_size * self.params_bytes, self.num_experts
        # )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        forward_mode: ForwardMode,
        previous_event=None,
        num_max_dispatch_tokens_per_rank: int = 256,
    ):
        # Todo: for decode batch, use low_latency_dispatch
        if True:  # not forward_mode.is_decode():
            (
                hidden_states,
                topk_idx,
                topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                event,
            ) = self.dispatch_forward(
                hidden_states, topk_idx, topk_weights, num_experts, previous_event
            )
            self.tokens_per_expert = torch.tensor(
                num_recv_tokens_per_expert_list,
                device=hidden_states.device,
                dtype=torch.int64,
            )
        else:
            hidden_states, recv_expert_count, handle, event, hook = (
                self.low_latency_dispatch(
                    hidden_states,
                    topk_idx,
                    num_max_dispatch_tokens_per_rank,
                    num_experts,
                )
            )
            self.recv_expert_count = recv_expert_count
        self.handle = handle
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights
        return hidden_states, topk_idx, topk_weights

    def dispatch_forward(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        previous_event=None,
    ):
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.buffer.get_dispatch_layout(
            topk_idx,
            num_experts,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = self.buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        self.tokens_per_expert = torch.tensor(
            num_recv_tokens_per_expert_list, device=x.device, dtype=torch.int64
        )

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        )

    def combine_forward(self, x: torch.Tensor, handle: Tuple, previous_event=None):
        combined_x, _, event = self.buffer.combine(
            x,
            handle,
            async_finish=False,
            previous_event=previous_event,
            allocate_on_comm_stream=False,
        )
        return combined_x, event

    def low_latency_dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
    ):
        recv_hidden_states, recv_expert_count, handle, event, hook = (
            self.low_latency_buffer.low_latency_dispatch(
                hidden_states,
                topk_idx,
                num_max_dispatch_tokens_per_rank,
                num_experts,
                async_finish=False,
                return_recv_hook=True,
            )
        )
        return recv_hidden_states, recv_expert_count, handle, event, hook

    def low_latency_combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: Tuple,
    ):
        combined_hidden_states, event_overlap, hook = (
            self.low_latency_buffer.low_latency_combine(
                hidden_states,
                topk_idx,
                topk_weights,
                handle,
                async_finish=False,
                return_recv_hook=True,
            )
        )
        return combined_hidden_states, event_overlap, hook

    def _indices_to_multihot(self, indices, probs):
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_local_experts),
            dtype=torch.long,
            device=indices.device,
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_local_experts),
            dtype=torch.float,
            device=indices.device,
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(
            mask.sum(dim=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs

    def get_dispached_metadata(self) -> torch.Tensor:
        return self.topk_ids, self.topk_weights

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert

    def combine(
        self, hidden_states: torch.Tensor, forward_mode: ForwardMode
    ) -> torch.Tensor:
        # Todo: for decode batch, use low_latency_combine
        if True:  # not forward_mode.is_decode():
            hidden_states, event = self.combine_forward(hidden_states, self.handle)
        else:
            hidden_states, event, hook = self.low_latency_combine(
                hidden_states, self.topk_idx, self.topk_weights, self.handle
            )
        self.handle = None
        return hidden_states

    def get_permuted_hidden_states_by_experts(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        self.dispatched_routing_map, self.topk_weights = self._indices_to_multihot(
            self.topk_idx, self.topk_weights
        )
        self.hidden_shape_before_permute = hidden_states.shape
        hidden_states, self.reversed_mapping_for_combine = permute(
            hidden_states,
            self.dispatched_routing_map,
            num_out_tokens=self.tokens_per_expert.sum(),
            fused=self.permute_fusion,
        )
        return hidden_states

    def get_restored_hidden_states_by_experts(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        assert (
            self.topk_weights.dtype == torch.float32
        ), "DeepEP only supports float32 probs"
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            probs=self.topk_weights,
            fused=self.permute_fusion,
        )
        return hidden_states.to(input_dtype)


class DeepEPTokenDispatcher:
    """
    Copy from Megatron-Core token_dispatcher MoEFlexTokenDispatcher
    https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py#L851
    """

    def __init__(
        self,
        num_local_experts: int,
        tok_k: int,
        num_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
    ):
        self.num_local_experts = num_local_experts

        self._comm_manager = DeepEPManager(
            group=get_tp_group().device_group,
            router_topk=tok_k,
            permute_fusion=True,
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
        )

    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        forward_mode: ForwardMode,
        num_max_dispatch_tokens_per_rank: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.hidden_shape = hidden_states.shape
        topk_ids = topk_ids.to(torch.int64)
        hidden_states, topk_ids, topk_weights = self._comm_manager.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            num_experts,
            forward_mode=forward_mode,
            previous_event=None,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
        )
        if hidden_states.shape[0] > 0:
            global_input_tokens = hidden_states
            global_input_tokens = (
                self._comm_manager.get_permuted_hidden_states_by_experts(hidden_states)
            )
        else:
            global_input_tokens = hidden_states
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()

        return global_input_tokens, topk_ids, topk_weights, tokens_per_expert

    def token_unpermutation(
        self, hidden_states: torch.Tensor, forward_mode: ForwardMode
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if hidden_states.shape[0] > 0:
            hidden_states = self._comm_manager.get_restored_hidden_states_by_experts(
                hidden_states
            )
        hidden_states = self._comm_manager.combine(hidden_states, forward_mode)
        return hidden_states.view(self.hidden_shape)
