try:
    from deep_ep import Buffer

    use_deepep = True
except ImportError:
    use_deepep = False

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.ep_moe.kernels import (
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
        async_finish: bool = False,
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
        self.async_finish = async_finish

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
        hidden_states,
        fp8_dtype=None,
        use_fp8_w8a8=False,
        use_block_quant=False,
    ):
        reorder_topk_ids, src2dst, seg_indptr = deepep_run_moe_deep_preprocess(
            self.topk_idx, self.num_experts
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
            self.topk_idx,
            None,
            self.router_topk,
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
        num_max_dispatch_tokens_per_rank: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            ) = self.dispatch_normal(hidden_states, topk_idx, topk_weights, num_experts)
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

        if self.async_finish:
            event.current_stream_wait()

        self.handle = handle
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights
        if hidden_states.shape[0] > 0:
            reorder_topk_ids, seg_indptr, hidden_states = self.deepep_permute(
                hidden_states, fp8_dtype=hidden_states.dtype
            )
        else:
            reorder_topk_ids = torch.empty(
                (0,), device=hidden_states.device, dtype=torch.int64
            )
            seg_indptr = torch.zeros(
                (num_experts + 1,), device=hidden_states.device, dtype=torch.int64
            )
        return hidden_states, reorder_topk_ids, seg_indptr

    def dispatch_normal(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
    ):
        previous_event = Buffer.capture() if self.async_finish else None

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
            async_finish=self.async_finish,
            allocate_on_comm_stream=previous_event is not None,
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
            async_finish=self.async_finish,
            allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
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
                async_finish=self.async_finish,
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
                num_tokens = self.src2dst.shape[0] // self.router_topk
                output = torch.empty(
                    (num_tokens, hidden_states.shape[1]),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                deepep_post_reorder_triton_kernel[(num_tokens,)](
                    hidden_states,
                    output,
                    self.src2dst,
                    self.topk_idx,
                    self.topk_weights,
                    self.router_topk,
                    hidden_states.shape[1],
                    BLOCK_SIZE=512,
                )
            else:
                output = torch.zeros(
                    (0, hidden_states.shape[1]),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            hidden_states, event = self.combine_normal(output, self.handle)
        else:
            hidden_states, event, hook = self.combine_low_latency(
                hidden_states, self.topk_idx, self.topk_weights, self.handle
            )

        if self.async_finish:
            event.current_stream_wait()

        self.handle = None
        return hidden_states

    def combine_normal(self, x: torch.Tensor, handle: Tuple):
        previous_event = Buffer.capture() if self.async_finish else None

        combined_x, _, event = self.buffer_normal.combine(
            x,
            handle,
            async_finish=self.async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
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
                async_finish=self.async_finish,
                return_recv_hook=False,  # True for double-batch overlapping, need call hook()
            )
        )
        # hook()
        return combined_hidden_states, event_overlap, hook
