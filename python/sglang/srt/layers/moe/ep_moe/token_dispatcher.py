from sglang.srt.utils import DeepEPMode

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

_buffer = None

def _get_deepep_buffer(
    group: dist.ProcessGroup,
    hidden: int,
    low_latency_mode: bool = False,
    num_max_dispatch_tokens_per_rank: int = None,
    num_experts: int = None,
):
    global _buffer
    
    # if _buffer is not None and _buffer.low_latency_mode == low_latency_mode:
    #     return _buffer

    num_nvl_bytes, num_rdma_bytes = 0, 0
    if not low_latency_mode:
        hidden_bytes = hidden * 2
        for config in (
            Buffer.get_dispatch_config(group.size()),
            Buffer.get_combine_config(group.size()),
        ):
            num_nvl_bytes = max(
                config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                num_nvl_bytes,
            )
            num_rdma_bytes = max(
                config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                num_rdma_bytes,
            )
        if not (
            _buffer is None
            or _buffer.group != group
            or _buffer.num_nvl_bytes < num_nvl_bytes
            or _buffer.num_rdma_bytes < num_rdma_bytes
        ):
            return _buffer
    else:
        assert num_max_dispatch_tokens_per_rank is not None
        assert num_experts is not None and num_experts % group.size() == 0
        num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, group.size(), num_experts
        )
        if not (
            _buffer is None
            or _buffer.group != group
            or not _buffer.low_latency_mode
            or _buffer.num_rdma_bytes < num_rdma_bytes
        ):
            return _buffer

    # Todo: simply set num_nvl_bytes as 1e9 to W/A intranode dispatch assertion error: num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes, need better solution
    _buffer = Buffer(
        group,
        int(1e9),
        num_rdma_bytes,
        low_latency_mode=low_latency_mode,
        num_qps_per_rank=(num_experts // group.size() if low_latency_mode else 1),
    )

    return _buffer


class _DeepEPDispatcherImplBase:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
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

        self.handle = None

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
    ):
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError
    
    def _get_buffer(self) -> Buffer:
        raise NotImplementedError


class _DeepEPDispatcherImplNormal(_DeepEPDispatcherImplBase):
    def __init__(self, async_finish: bool, **kwargs):
        super().__init__(**kwargs)

        # self.buffer_normal = _get_buffer_normal(
        #     self.group, self.hidden_size * self.params_bytes
        # )
        self.async_finish = async_finish
        self.src2dst = None

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
    ):
        topk_idx = topk_idx.to(torch.int64)
        previous_event = Buffer.capture() if self.async_finish else None
        return hidden_states, topk_idx, topk_weights, previous_event

    def dispatch_b(
        self, hidden_states, topk_idx, topk_weights, previous_event
    ):
        (
            hidden_states,
            topk_idx,
            topk_weights,
            event,
        ) = self._dispatch_core(
            hidden_states, topk_idx, topk_weights, previous_event
        )
        event.current_stream_wait() if self.async_finish else ()
        if hidden_states.shape[0] > 0:
            reorder_topk_ids, seg_indptr, hidden_states = self._deepep_permute(
                hidden_states, topk_idx, fp8_dtype=hidden_states.dtype
            )
        else:
            reorder_topk_ids = torch.empty(
                (0,), device=hidden_states.device, dtype=torch.int64
            )
            seg_indptr = torch.zeros(
                (self.num_experts + 1,), device=hidden_states.device, dtype=torch.int64
            )

        masked_m = expected_m = None

        return (
            hidden_states,
            topk_idx,
            topk_weights,
            reorder_topk_ids,
            seg_indptr,
            masked_m,
            expected_m,
        )

    def _dispatch_core(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event,
    ):
        buffer = self._get_buffer()
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            topk_idx,
            self.num_experts,
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
        ) = buffer.dispatch(
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

    def _deepep_permute(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        fp8_dtype: Optional[torch.dtype] = None,
        use_fp8_w8a8: bool = False,
        use_block_quant: bool = False,
    ):
        """
        Copy from Megatron-Core token_dispatcher MoEFlexTokenDispatcher
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py
        """

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

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
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
        previous_event = Buffer.capture() if self.async_finish else None
        return output, previous_event

    def combine_b(self, output, previous_event):
        hidden_states, event = self._combine_core(output, previous_event)
        event.current_stream_wait() if self.async_finish else ()
        self.handle = None
        self.src2dst = None
        return hidden_states

    def _combine_core(self, x: torch.Tensor, previous_event):
        buffer = self._get_buffer(
            self.group, self.hidden_size  # * self.params_bytes
        )
        combined_x, _, event = buffer.combine(
            x,
            self.handle,
            async_finish=self.async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
        )
        return combined_x, event
    
    def _get_buffer(self):
        return _get_deepep_buffer(
            self.group, self.hidden_size  #  * self.params_bytes
        )


class _DeepEPDispatcherImplLowLatency(_DeepEPDispatcherImplBase):
    def __init__(self, return_recv_hook: bool, **kwargs):
        super().__init__(**kwargs)

        """
        num_max_dispatch_tokens_per_rank: the actual batch size in the decoding engine should be less than 256
        https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
        """
        # TODO(ch-wan): allow users to set this value
        self.num_max_dispatch_tokens_per_rank = 128
        # self.buffer_low_latency = _get_buffer(
        #     self.group,
        #     self.num_max_dispatch_tokens_per_rank,
        #     self.hidden_size,
        #     self.num_experts,
        # )
        self.return_recv_hook = return_recv_hook

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
    ):
        buffer = self._get_buffer()
        topk_idx = topk_idx.to(torch.int64)
        expected_m = (
            hidden_states.shape[0]
            * buffer.group_size
            * topk_idx.shape[1]
            + self.num_experts
        ) // self.num_experts
        hidden_states, masked_m, event, hook = self._dispatch_core(
            hidden_states,
            topk_idx,
            num_max_dispatch_tokens_per_rank,
            use_fp8=True,
        )
        return (
            hidden_states,
            topk_idx,
            topk_weights,
            masked_m,
            expected_m,
            event,
            hook,
        )

    def dispatch_b(
        self,
        hidden_states,
        topk_idx,
        topk_weights,
        masked_m,
        expected_m,
        event,
        hook,
    ):
        hook() if self.return_recv_hook else event.current_stream_wait()

        reorder_topk_ids = seg_indptr = None

        return (
            hidden_states,
            topk_idx,
            topk_weights,
            reorder_topk_ids,
            seg_indptr,
            masked_m,
            expected_m,
        )

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
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
        buffer = self._get_buffer()
        packed_recv_hidden, packed_recv_count, self.handle, event, hook = (
            buffer.low_latency_dispatch(
                hidden_states,
                topk_idx,
                num_max_dispatch_tokens_per_rank,
                self.num_experts,
                use_fp8=use_fp8,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
            )
        )
        return packed_recv_hidden, packed_recv_count, event, hook

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        hidden_states, event, hook = self._combine_core(
            hidden_states,
            topk_idx,
            topk_weights,
        )
        return hidden_states, event, hook

    def combine_b(self, hidden_states, event, hook):
        hook() if self.return_recv_hook else event.current_stream_wait()
        return hidden_states

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        buffer = self._get_buffer()
        combined_hidden_states, event, hook = (
            buffer.low_latency_combine(
                hidden_states,
                topk_idx,
                topk_weights,
                self.handle,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
            )
        )
        self.handle = None
        return combined_hidden_states, event, hook

    def _get_buffer(self):
        return _get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.num_max_dispatch_tokens_per_rank,
            self.hidden_size,
            self.num_experts,
        )


class DeepEPDispatcher:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.auto,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        self.deepep_mode = deepep_mode

        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
        )

        if self.deepep_mode.enable_normal():
            self._normal_dispatcher = _DeepEPDispatcherImplNormal(
                async_finish=async_finish,
                **common_kwargs,
            )
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _DeepEPDispatcherImplLowLatency(
                return_recv_hook=return_recv_hook,
                **common_kwargs,
            )

    def dispatch(self, *args, **kwargs) -> Tuple:
        self.dispatch_a(*args, **kwargs)
        return self.dispatch_b()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int = 128,
        forward_mode: ForwardMode = None,
    ):
        inner_state = self._get_impl(forward_mode).dispatch_a(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
        )
        self._dispatch_intermediate_state = forward_mode, inner_state

    def dispatch_b(self):
        forward_mode, inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl(forward_mode).dispatch_b(*inner_state)

    def combine(self, *args, **kwargs) -> Tuple:
        self.combine_a(*args, **kwargs)
        return self.combine_b()

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        inner_state = self._get_impl(forward_mode).combine_a(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = forward_mode, inner_state

    def combine_b(self):
        forward_mode, inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl(forward_mode).combine_b(*inner_state)

    def _get_impl(self, forward_mode: ForwardMode) -> "_DeepEPDispatcherImplBase":
        resolved_deepep_mode = self.deepep_mode.resolve(forward_mode)
        if resolved_deepep_mode == DeepEPMode.normal:
            return self._normal_dispatcher
        elif resolved_deepep_mode == DeepEPMode.low_latency:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")
