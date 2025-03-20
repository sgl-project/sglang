try:
    from deep_ep import Buffer

    use_deepep = True
except ImportError:
    use_deepep = False

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.ep_moe.kernels import (
    compute_src2dst_triton_kernel,
    deepep_permute_triton_kernel,
    deepep_post_reorder_triton_kernel,
    deepep_run_moe_deep_preprocess,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

_buffer_normal = None
_buffer_low_latency = None


def get_buffer_normal(group: dist.ProcessGroup, hidden_bytes: int):
    """
    Copy from DeepEP example usage in model inference prefilling.
    https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-model-training-or-inference-prefilling
    """

    global _buffer_normal

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
        _buffer_normal is None
        or _buffer_normal.group != group
        or _buffer_normal.num_nvl_bytes < num_nvl_bytes
        or _buffer_normal.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer_normal = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer_normal


def get_buffer_low_latency(
    group: dist.ProcessGroup,
    num_max_dispatch_tokens_per_rank: int,
    hidden: int,
    num_experts: int,
):
    """
    Copy from DeepEP example usage in model inference decoding.
    https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
    """

    global _buffer_low_latency
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank, hidden, group.size(), num_experts
    )

    if (
        _buffer_low_latency is None
        or _buffer_low_latency.group != group
        or not _buffer_low_latency.low_latency_mode
        or _buffer_low_latency.num_rdma_bytes < num_rdma_bytes
    ):
        assert num_experts % group.size() == 0
        _buffer_low_latency = Buffer(
            group,
            0,
            num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_experts // group.size(),
        )
    return _buffer_low_latency


def permute(
    tokens,
    routing_map,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """
    Copy from Megatron-Core moe for token permutation
    https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py
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
    https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py
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


class DeepEPDispatcher:
    """
    Copy from Megatron-Core token_dispatcher MoEFlexTokenDispatcher
    https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py
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
        self.params_dtype = params_dtype
        self.params_bytes = 2
        # Metadata
        self.token_indices = None
        self.token_probs = None
        # Handle used for combine operation
        self.handle = None

        # `num_max_dispatch_tokens_per_rank` (the actual batch size in the decoding engine) should be less than 256
        # https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
        self.num_max_dispatch_tokens_per_rank = 128

        if not use_deepep:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )
        self.buffer_normal = get_buffer_normal(
            self.group, self.hidden_size * self.params_bytes
        )
        self.buffer_low_latency = None
        # Todo: enable low latency dispatch
        """
        self.buffer_low_latency = get_buffer_low_latency(
            self.group,
            self.num_max_dispatch_tokens_per_rank,
            self.hidden_size * self.params_bytes,
            self.num_experts,
        )
        """

    def deepep_permute(
        self,
        topk_ids,
        hidden_states,
        num_experts,
        top_k,
        use_fp8_w8a8,
        use_block_quant,
        fp8_dtype,
    ):
        reorder_topk_ids, src2dst, seg_indptr = deepep_run_moe_deep_preprocess(
            topk_ids, num_experts
        )
        num_total_tokens = reorder_topk_ids.numel()
        gateup_input = torch.empty(
            (int(num_total_tokens), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=(
                fp8_dtype
                if (use_fp8_w8a8 and not use_block_quant)
                else hidden_states.dtype
            ),
        )
        # PreReorder
        deepep_permute_triton_kernel[(hidden_states.shape[0],)](
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            None,
            top_k,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
        )
        self.src2dst = src2dst
        return reorder_topk_ids, seg_indptr, gateup_input

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        forward_mode: ForwardMode,
        previous_event=None,
        num_max_dispatch_tokens_per_rank: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.hidden_shape = hidden_states.shape
        topk_idx = topk_idx.to(torch.int64)
        # Todo: enable low latency dispatch
        if True:  # not forward_mode.is_decode():
            (
                hidden_states,
                topk_idx,
                topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                event,
            ) = self.dispatch_normal(
                hidden_states, topk_idx, topk_weights, num_experts, previous_event
            )
            self.tokens_per_expert = torch.tensor(
                num_recv_tokens_per_expert_list,
                device=hidden_states.device,
                dtype=torch.int64,
            )
        else:
            hidden_states, recv_expert_count, handle, event, hook = (
                self.dispatch_low_latency(
                    hidden_states,
                    topk_idx,
                    num_max_dispatch_tokens_per_rank,
                    num_experts,
                )
            )
            self.recv_expert_count = recv_expert_count
        tokens_per_expert = self.get_number_of_tokens_per_expert()
        self.handle = handle
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights
        if hidden_states.shape[0] > 0:
            hidden_states = self.get_permuted_hidden_states_by_experts(hidden_states)
        return hidden_states, topk_idx, topk_weights, tokens_per_expert

    def dispatch_normal(
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
        ) = self.buffer_normal.get_dispatch_layout(
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
        ) = self.buffer_normal.dispatch(
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

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        )

    def dispatch_low_latency(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
    ):
        """
        # For H20, there will be an CUDA error: DeepEP/csrc/kernels/internode_ll.cu:337 'too many blocks in cooperative launch'
        # Please please make sure to change DeepEP code in internode_ll.cu dispatch / combine first and then reinstall!
        # More details refer: https://github.com/deepseek-ai/DeepEP/issues/15#issuecomment-2709715782
        +
        diff --git a/csrc/kernels/internode_ll.cu b/csrc/kernels/internode_ll.cu
        index f60e933..cddaabf 100644
        --- a/csrc/kernels/internode_ll.cu
        +++ b/csrc/kernels/internode_ll.cu
        @@ -307,14 +307,14 @@ void dispatch(void* packed_recv_x, float* packed_recv_x_scales,
                    int num_topk, int num_experts, int rank, int num_ranks,
                    void* workspace, cudaStream_t stream, int phases) {
            constexpr int kNumMaxTopK = 9;
        -    constexpr int kNumWarpsPerGroup = 10;
        -    constexpr int kNumWarpGroups = 3;
        +    constexpr int kNumWarpsPerGroup = 8;
        +    constexpr int kNumWarpGroups = 4;
            EP_STATIC_ASSERT(kNumMaxTopK + 1 <= kNumWarpGroups * kNumWarpsPerGroup, "Too many top-k selections");
        +
            const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
            const auto num_sms = cell_div(num_experts, kNumWarpGroups);
            EP_HOST_ASSERT(num_topk <= kNumMaxTopK);
        -    EP_HOST_ASSERT(cell_div(static_cast<int>(hidden * 2 / sizeof(int4)), 32 * (num_warps - 1)) <= 2);
        +    // EP_HOST_ASSERT(cell_div(static_cast<int>(hidden * 2 / sizeof(int4)), 32 * (num_warps - 1)) <= 2);
        +
            // Workspace checks
            auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
        @@ -505,8 +505,8 @@ void combine(void* combined_x,
                    int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
                    int num_topk, int num_experts, int rank, int num_ranks,
                    void* workspace, cudaStream_t stream, int phases) {
        -    constexpr int kNumWarpsPerGroup = 10;
        -    constexpr int kNumWarpGroups = 3;
        +    constexpr int kNumWarpsPerGroup = 8;
        +    constexpr int kNumWarpGroups = 4;
            constexpr int kNumMaxTopk = 9;
        +
            const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
        """

        recv_hidden_states, recv_expert_count, handle, event, hook = (
            self.buffer_low_latency.low_latency_dispatch(
                hidden_states,
                topk_idx,
                num_max_dispatch_tokens_per_rank,
                num_experts,
                async_finish=False,
                return_recv_hook=False,  # True for double-batch overlapping, need call hook()
            )
        )
        # hook()
        return recv_hidden_states, recv_expert_count, handle, event, hook

    def combine(
        self, hidden_states: torch.Tensor, forward_mode: ForwardMode
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Todo: enable low latency combine
        if True:  # not forward_mode.is_decode():
            if hidden_states.shape[0] > 0:
                hidden_states = self.get_restored_hidden_states_by_experts(
                    hidden_states
                )
            hidden_states, event = self.combine_normal(hidden_states, self.handle)
        else:
            hidden_states, event, hook = self.combine_low_latency(
                hidden_states, self.topk_idx, self.topk_weights, self.handle
            )
        self.handle = None
        return hidden_states.view(self.hidden_shape)

    def combine_normal(self, x: torch.Tensor, handle: Tuple, previous_event=None):
        combined_x, _, event = self.buffer_normal.combine(
            x,
            handle,
            async_finish=False,
            previous_event=previous_event,
            allocate_on_comm_stream=False,
        )
        return combined_x, event

    def combine_low_latency(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: Tuple,
    ):
        combined_hidden_states, event_overlap, hook = (
            self.buffer_low_latency.low_latency_combine(
                hidden_states,
                topk_idx,
                topk_weights,
                handle,
                async_finish=False,
                return_recv_hook=False,  # True for double-batch overlapping, need call hook()
            )
        )
        # hook()
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
        return self.topk_idx, self.topk_weights

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert

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
