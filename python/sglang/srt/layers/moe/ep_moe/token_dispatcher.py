from sglang.srt.distributed import get_tensor_model_parallel_rank

try:
    from deep_ep import Buffer

    use_deepep = True
except ImportError:
    use_deepep = False

from typing import Tuple

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
        self.enable_async = async_finish

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

    # TODO wait for low_latency code, so currently this file is just hacky refactor
    # TODO low_latency_dispatch/low_latency_combine's async_finish should be false, return_recv_hook should be self.enable_async
    def dispatch(self, *args, **kwargs):
        self.dispatch_a(*args, **kwargs)
        return self.dispatch_b()

    def combine(self, *args, **kwargs):
        self.combine_a(*args, **kwargs)
        return self.combine_b()

    # TODO actual name should be prepare-execute and issue-receive instead of a-b, can rename
    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        forward_mode: ForwardMode,
        num_max_dispatch_tokens_per_rank: int = 128,
    ):
        topk_idx = topk_idx.to(torch.int64)
        previous_event = Buffer.capture() if self.enable_async else None
        self.dispatch_intermediate_state = hidden_states, topk_idx, topk_weights, num_experts

    def dispatch_b(self):
        hidden_states, topk_idx, topk_weights, num_experts, previous_event = self.dispatch_intermediate_state

        (
            hidden_states,
            topk_idx,
            topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = self.dispatch_normal(hidden_states, topk_idx, topk_weights, num_experts, previous_event)

        if self.enable_async:
            event.current_stream_wait()

        # TODO move from above
        self.tokens_per_expert = torch.tensor(
            num_recv_tokens_per_expert_list,
            device=hidden_states.device,
            dtype=torch.int64,
        )

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
        previous_event,
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
            async_finish=self.enable_async,
            allocate_on_comm_stream=(previous_event is not None) and self.enable_async,
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
            async_finish=self.enable_async,
            allocate_on_comm_stream=(previous_event is not None) and self.enable_async,
        )

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        )

    def combine_a(
        self, hidden_states: torch.Tensor, forward_mode: ForwardMode
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

        self.combine_intermediate_state = event, hidden_states

    def combine_b(self):
        event, hidden_states = self.combine_intermediate_state

        if self.enable_async:
            event.current_stream_wait()

        self.handle = None
        return hidden_states

    def combine_normal(self, x: torch.Tensor, handle: Tuple):
        previous_event = Buffer.capture() if self.enable_async else None

        combined_x, _, event = self.buffer_normal.combine(
            x,
            handle,
            async_finish=self.enable_async,
            previous_event=previous_event,
            allocate_on_comm_stream=(previous_event is not None) and self.enable_async,
        )
        return combined_x, event
