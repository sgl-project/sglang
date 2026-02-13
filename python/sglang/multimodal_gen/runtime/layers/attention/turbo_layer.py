# copy and modify from https://github.com/thu-ml/TurboDiffusion/blob/main/turbodiffusion/rcm/utils/a2a_cp.py and https://github.com/thu-ml/TurboDiffusion/blob/main/turbodiffusion/SLA/core.py

from typing import Any, Callable, List, Tuple, Type, Union

import torch
import torch.distributed as dist
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn import Module

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionImpl,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn import (
    SageSparseLinearAttentionBackend,
    SparseLinearAttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.managers.forward_context import (
    ForwardContext,
    get_forward_context,
)
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import get_compute_dtype

logger = init_logger(__name__)


def post_all2all(local_seq_2_local_head, seq_world_size):
    def post_func(input):
        # b, s, n, h
        if local_seq_2_local_head:
            output = rearrange(input, "w bs seq h d -> bs (w seq) h d")
        else:
            output = rearrange(input, "w bs s h d -> bs s (w h) d", w=seq_world_size)

        return output

    return post_func


def single_all_to_all(input, local_seq_2_local_head, group, async_op=False):
    seq_world_size = dist.get_world_size(group)

    # b, s, n, h
    if local_seq_2_local_head:
        bs, local_seq_len, num_total_head, head_dim = input.shape
        assert (
            num_total_head % seq_world_size == 0
        ), f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
        input_t = rearrange(
            input,
            "bs seq_len (w h) d -> w bs seq_len h d",
            w=seq_world_size,
            h=num_total_head // seq_world_size,
        ).contiguous()
        post_all2all_fun = post_all2all(local_seq_2_local_head, seq_world_size)
    else:
        bs, global_seq_len, num_local_head, head_dim = input.shape
        input_t = rearrange(
            input,
            "bs (w s) h d -> w bs s h d",
            w=seq_world_size,
            s=global_seq_len // seq_world_size,
        ).contiguous()
        post_all2all_fun = post_all2all(local_seq_2_local_head, seq_world_size)

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group, async_op=async_op)

    res = post_all2all_fun(output)
    return res


def async_a2a_communicate(
    a2a_inputs: Union[torch.Tensor, List[torch.Tensor]],
    cp_size: int,
    cp_group: ProcessGroup,
    cp_stream: torch.cuda.Stream,
    local_seq_2_local_head: bool,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    A2A communication for context parallelism. best used in communicate qkv
    Modified from Nvidia Transformer Engine.
    """
    a2a_inputs = [a2a_inputs] if not isinstance(a2a_inputs, list) else a2a_inputs
    a2a_outputs, a2a_reqs = [None] * len(a2a_inputs), [None] * len(a2a_inputs)
    a2a_post_fns = [None] * len(a2a_inputs)
    if local_seq_2_local_head:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
                    a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True
                )
                a2a_post_fns[i - 1] = post_all2all(local_seq_2_local_head, cp_size)
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    a2a_outputs[i - 2] = a2a_post_fns[i - 2](a2a_outputs[i - 2])
            if i < len(a2a_inputs):
                a2a_inputs[i] = rearrange(
                    a2a_inputs[i], "bs seq_len (w h) d -> w bs seq_len h d", w=cp_size
                ).contiguous()
    else:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
                    a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True
                )
                a2a_post_fns[i - 1] = post_all2all(local_seq_2_local_head, cp_size)
            if i < len(a2a_inputs):
                a2a_inputs[i] = rearrange(
                    a2a_inputs[i], "bs (w s) h d -> w bs s h d", w=cp_size
                ).contiguous()
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    a2a_outputs[i - 2] = a2a_post_fns[i - 2](a2a_outputs[i - 2])
    torch.cuda.current_stream().wait_stream(cp_stream)
    return a2a_outputs[0] if len(a2a_inputs) == 1 else a2a_outputs


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, group: dist.ProcessGroup, input: Tensor, local_seq_2_local_head: bool
    ) -> Tensor:
        ctx.group = group
        res = single_all_to_all(input, local_seq_2_local_head, group, False)
        ctx.local_seq_2_local_head = local_seq_2_local_head
        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None]:
        return (
            None,
            _SeqAllToAll.apply(ctx.group, *grad_output, not ctx.local_seq_2_local_head),
            None,
        )


class _SeqAllToAllQKV(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cp_size: int,
        cp_stream: torch.cuda.Stream,
        local_seq_2_local_head: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_stream = cp_stream
        ctx.local_seq_2_local_head = local_seq_2_local_head
        q, k, v = async_a2a_communicate(
            [q, k, v], cp_size, group, cp_stream, local_seq_2_local_head
        )
        return q, k, v

    @staticmethod
    def backward(
        ctx: Any, *grad_output: Tensor
    ) -> Tuple[None, Tensor, Tensor, Tensor, None, None, None]:
        q_grad, k_grad, v_grad = _SeqAllToAllQKV.apply(
            ctx.group,
            *grad_output,
            ctx.cp_size,
            ctx.cp_stream,
            not ctx.local_seq_2_local_head,
        )
        return (None, q_grad, k_grad, v_grad, None, None, None)


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
    """

    def __init__(self, local_attention: Union[Module, Callable]) -> None:
        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.pg = None
        self.stream = None

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, ctx_attn_metadata
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer

        Returns:
            * output (Tensor): context output
        """
        if self.pg is None:
            return self.local_attn(query, key, value, ctx_attn_metadata)
        pg_size = dist.get_world_size(self.pg)
        if pg_size < 2:
            return self.local_attn(query, key, value, ctx_attn_metadata)

        query_layer, key_layer, value_layer = _SeqAllToAllQKV.apply(
            self.pg, query, key, value, pg_size, self.stream, True
        )
        context_layer = self.local_attn(
            query_layer, key_layer, value_layer, ctx_attn_metadata
        )

        output = _SeqAllToAll.apply(self.pg, context_layer, False)
        return output

    def set_context_parallel_group(self, group, stream):
        self.pg = group
        self.stream = stream


class MinimalA2AAttnOp(DistributedAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        attention_type: str,
        topk: float,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        dtype = get_compute_dtype()
        attn_backend = get_attn_backend(
            head_size, dtype, supported_attention_backends=supported_attention_backends
        )
        # Maintained for compatibility purposes; can be removed when CI allows setting Attention_backend or when TurboWan supports FA.
        if attn_backend not in (
            SparseLinearAttentionBackend,
            SageSparseLinearAttentionBackend,
        ):
            logger.warning(
                "TurboWan now only supports `sla_attn` or `sage_sla_attn` and has been automatically set to attention_type. Please set --attention-backend to `sla_attn` or `sage_sla_attn`."
            )
            if attention_type == "sagesla":
                attn_backend = SageSparseLinearAttentionBackend
            else:
                attn_backend = SparseLinearAttentionBackend
        impl_cls: Type["AttentionImpl"] = attn_backend.get_impl_cls()
        local_attn = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            topk_ratio=topk,
        )
        super(MinimalA2AAttnOp, self).__init__(local_attn)

    def set_context_parallel_group(self, process_group, ranks, stream):
        del ranks
        super().set_context_parallel_group(process_group, stream)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs
    ) -> Tensor:
        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata
        results = super().forward(query, key, value, ctx_attn_metadata)
        return rearrange(results, "b ... h l -> b ... (h l)")
