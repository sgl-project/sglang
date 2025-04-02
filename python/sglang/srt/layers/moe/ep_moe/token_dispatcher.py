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
            num_rdma_bytes=num_rdma_bytes,
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
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: str = "auto",
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        if not use_deepep:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.params_bytes = 2

        self.deepep_mode = deepep_mode
        self.handle = None

        if self.deepep_mode in ["normal", "auto"]:  # for normal / auto mode
            self.buffer_normal = get_buffer_normal(
                self.group, self.hidden_size * self.params_bytes
            )
            self.async_finish = async_finish
            self.src2dst = None
        if self.deepep_mode in ["low_latency", "auto"]:  # for low_latency / auto mode
            """
            num_max_dispatch_tokens_per_rank: the actual batch size in the decoding engine should be less than 256
            https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
            """
            # TODO(ch-wan): allow users to set this value
            self.num_max_dispatch_tokens_per_rank = 128
            self.buffer_low_latency = get_buffer_low_latency(
                self.group,
                self.num_max_dispatch_tokens_per_rank,
                self.hidden_size,
                self.num_experts,
            )
            self.return_recv_hook = return_recv_hook

    def deepep_permute(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        fp8_dtype: Optional[torch.dtype] = None,
        use_fp8_w8a8: bool = False,
        use_block_quant: bool = False,
    ):
        reorder_topk_ids, self.src2dst, seg_indptr = deepep_run_moe_deep_preprocess(
            topk_idx, self.num_experts
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
            self.src2dst,
            topk_idx,
            None,
            self.router_topk,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
        )
        return reorder_topk_ids, seg_indptr, gateup_input

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        num_max_dispatch_tokens_per_rank: int = 128,
        forward_mode: ForwardMode = None,
    ) -> Tuple:
        topk_idx = topk_idx.to(torch.int64)
        reorder_topk_ids = torch.empty(
            (0,), device=hidden_states.device, dtype=torch.int64
        )
        seg_indptr = torch.zeros(
            (num_experts + 1,), device=hidden_states.device, dtype=torch.int64
        )
        masked_m = torch.empty(
            (self.num_local_experts,), device=hidden_states.device, dtype=torch.int64
        )
        expected_m = 0

        if self.deepep_mode == "normal" or (
            self.deepep_mode == "auto" and not forward_mode.is_decode()
        ):
            (
                hidden_states,
                topk_idx,
                topk_weights,
                event,
            ) = self.dispatch_normal(hidden_states, topk_idx, topk_weights, num_experts)
            event.current_stream_wait() if self.async_finish else ()
            if hidden_states.shape[0] > 0:
                reorder_topk_ids, seg_indptr, hidden_states = self.deepep_permute(
                    hidden_states, topk_idx, fp8_dtype=hidden_states.dtype
                )
        elif self.deepep_mode == "low_latency" or (
            self.deepep_mode == "auto" and forward_mode.is_decode()
        ):
            expected_m = (
                hidden_states.shape[0]
                * self.buffer_low_latency.group_size
                * topk_idx.shape[1]
                + num_experts
            ) // num_experts
            hidden_states, masked_m, event, hook = self.dispatch_low_latency(
                hidden_states,
                topk_idx,
                num_max_dispatch_tokens_per_rank,
                num_experts,
                use_fp8=True,
            )
            hook() if self.return_recv_hook else event.current_stream_wait()
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

        return (
            hidden_states,
            topk_idx,
            topk_weights,
            reorder_topk_ids,
            seg_indptr,
            masked_m,
            expected_m,
        )

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

        # FIXME: `handle` should be transmitted with tokens from dispatch to combine.
        # However, doing this would incur an unknown synchronization error, but keeping
        # `handle` as a member variable works.
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            _,  # num_recv_tokens_per_expert_list
            self.handle,
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
            event,
        )

    def dispatch_low_latency(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        use_fp8: bool = False,
    ):
        """
        # For H20, there will be an CUDA error: DeepEP/csrc/kernels/internode_ll.cu:337 'too many blocks in cooperative launch'.
        # Please make sure to change DeepEP code in internode_ll.cu dispatch / combine as below first and then reinstall.
        # More details refer: https://github.com/deepseek-ai/DeepEP/issues/15#issuecomment-2709715782

        diff --git a/csrc/kernels/internode_ll.cu b/csrc/kernels/internode_ll.cu
        index 76ae2e2..8ecd08f 100644
        --- a/csrc/kernels/internode_ll.cu
        +++ b/csrc/kernels/internode_ll.cu
        @@ -310,8 +310,8 @@ void dispatch(void* packed_recv_x, float* packed_recv_x_scales,
                    int num_topk, int num_experts, int rank, int num_ranks, bool use_fp8,
                    void* workspace, cudaStream_t stream, int phases) {
            constexpr int kNumMaxTopK = 9;
        -    constexpr int kNumWarpsPerGroup = 10;
        -    constexpr int kNumWarpGroups = 3;
        +    constexpr int kNumWarpsPerGroup = 8;
        +    constexpr int kNumWarpGroups = 4;
            EP_STATIC_ASSERT(kNumMaxTopK + 1 <= kNumWarpGroups * kNumWarpsPerGroup, "Too many top-k selections");

            const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
        @@ -501,8 +501,8 @@ void combine(void* combined_x,
                    int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
                    int num_topk, int num_experts, int rank, int num_ranks,
                    void* workspace, cudaStream_t stream, int phases) {
        -    constexpr int kNumWarpsPerGroup = 10;
        -    constexpr int kNumWarpGroups = 3;
        +    constexpr int kNumWarpsPerGroup = 8;
        +    constexpr int kNumWarpGroups = 4;
            constexpr int kNumMaxTopk = 9;

            const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
        """

        packed_recv_hidden, packed_recv_count, self.handle, event, hook = (
            self.buffer_low_latency.low_latency_dispatch(
                hidden_states,
                topk_idx,
                num_max_dispatch_tokens_per_rank,
                num_experts,
                use_fp8=use_fp8,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
            )
        )
        return packed_recv_hidden, packed_recv_count, event, hook

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_mode: ForwardMode,
    ) -> torch.Tensor:
        if self.deepep_mode == "normal" or (
            self.deepep_mode == "auto" and not forward_mode.is_decode()
        ):
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
                    topk_idx,
                    topk_weights,
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
            hidden_states, event = self.combine_normal(
                output,
            )
            event.current_stream_wait() if self.async_finish else ()
        elif self.deepep_mode == "low_latency" or (
            self.deepep_mode == "auto" and forward_mode.is_decode()
        ):
            hidden_states, event, hook = self.combine_low_latency(
                hidden_states,
                topk_idx,
                topk_weights,
            )
            hook() if self.return_recv_hook else event.current_stream_wait()
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

        return hidden_states

    def combine_normal(self, x: torch.Tensor):
        previous_event = Buffer.capture() if self.async_finish else None

        combined_x, _, event = self.buffer_normal.combine(
            x,
            self.handle,
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
    ):
        combined_hidden_states, event, hook = (
            self.buffer_low_latency.low_latency_combine(
                hidden_states,
                topk_idx,
                topk_weights,
                self.handle,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
            )
        )
        return combined_hidden_states, event, hook
