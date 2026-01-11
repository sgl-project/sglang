from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch
import torch.distributed as dist

from sglang.srt.environ import envs

from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput, StandardTopKOutput
from sglang.srt.distributed import (
    get_moe_tensor_parallel_world_size,
    get_tp_group,
)
from sglang.srt.utils import round_up

try:
    import pplx_kernels as pplx
    use_pplx = True
except ImportError:
    use_pplx = False


class PPLXDispatchOutput(NamedTuple):
    """PPLX dispatch output.
    Contains expert-grouped hidden states after all-to-all dispatch.
    """

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    expert_num_tokens: torch.Tensor
    topk_output: StandardTopKOutput
    
    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.PPLX


assert isinstance(PPLXDispatchOutput, DispatchOutput)


class PPLXCombineInput(NamedTuple):
    """PPLX Combine Input.
    Contains expert outputs after MoE computation, ready for a2a combine.
    """

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.PPLX


assert isinstance(PPLXCombineInput, CombineInput)

class PPLXDispatcher(BaseDispatcher):
    def __init__(
        self,
        num_experts: int,
        num_local_experts: int,
        experts_per_token: int,
        hidden_dim: int,
        params_dtype: torch.dtype,
    ):
        super().__init__()

        if not use_pplx:
            raise ImportError(
                "PPLX is not installed. Please install PPLX package from "
                "https://github.com/perplexityai/pplx-kernels."
            )

        self.max_num_tokens = envs.SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        self.internode = envs.SGLANG_PPLX_INTERNODE.get()
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_per_token = experts_per_token # topk
        self.rank = get_tp_group().rank
        self.world_size = get_tp_group().world_size # vllm: dp_size is tp_size, bugs in pplx-kernels
        self.dp_size = get_moe_tensor_parallel_world_size()
        self.hidden_dim = hidden_dim

        print(f"self.dp_size = {self.dp_size}")
        print(f"self.world_size = {self.world_size}")

        assert params_dtype.itemsize == 2, (
            "!!!current only support bfloat16/fp16!!!"
        )
        # currently only for fp16
        self.hidden_dim_bytes = round_up(
            hidden_dim * params_dtype.itemsize,
            16
        ) # vllm: All pplx byte sizes must be 16-bit aligned.

        self.hidden_dim_scale_bytes = 0

        if self.internode:
            self.ata = pplx.AllToAll.internode(
                max_num_tokens=self.max_num_tokens,
                num_experts=self.num_experts,
                experts_per_token=self.experts_per_token,
                rank=self.rank,
                world_size=self.world_size,
                dp_size=self.dp_size,
                hidden_dim=self.hidden_dim,
                hidden_dim_bytes=self.hidden_dim_bytes,
                hidden_dim_scale_bytes=self.hidden_dim_bytes
            )
        else:
            self.ata = pplx.AllToAll.intranode(
                max_num_tokens=self.max_num_tokens,
                num_experts=self.num_experts,
                experts_per_token=self.experts_per_token,
                rank=self.rank,
                world_size=self.world_size,
                dp_size=self.dp_size,
                hidden_dim=self.hidden_dim,
                hidden_dim_bytes=self.hidden_dim_bytes,
                hidden_dim_scale_bytes=self.hidden_dim_scale_bytes,
                group_name=get_tp_group().cpu_group.group_name
            )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: StandardTopKOutput,
        hidden_states_scale: Optional[torch.Tensor] = None,
    ) -> PPLXDispatchOutput:
        num_tokens = hidden_states.size(0)
        hidden_dim = hidden_states.size(-1)
        topk_ids = topk_output.topk_ids
        
        assert topk_ids.size(0) == num_tokens

        device = hidden_states.device

        expert_num_tokens = torch.empty(
            self.num_local_experts,
            dtype=torch.int32,
            device=device,
        )

        num_dispatchers = self.world_size // self.dp_size

        expert_x = torch.empty(
            (
                self.num_local_experts,
                self.max_num_tokens * num_dispatchers,
                hidden_dim,
            ),
            dtype=hidden_states.dtype,
            device=device,
        )

        expert_x_scale: torch.Tensor | None = None
        # vllm: This argument is optional, defaults to indices.size(0)
        # There's not much point setting this unless it is != indices.size(0)
        bound_m: Optional[torch.Tensor] = None
        topk_ids_u32 = topk_output.topk_ids.view(dtype=torch.uint32)

        self.ata.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x = hidden_states,
            dp_x_scale=hidden_states_scale,
            indices=topk_ids_u32,
            bound_m=bound_m,
            do_send=True,
            do_recv=True,
        )

        return PPLXDispatchOutput(
            hidden_states=expert_x,
            hidden_states_scale=hidden_states_scale,
            expert_num_tokens=expert_num_tokens,
            topk_output=topk_output,
        )
    
    def combine(
        self,
        combine_input: PPLXCombineInput,
    ) -> torch.Tensor:
        hidden_states = combine_input.hidden_states
        topk_ids = combine_input.topk_ids
        topk_weights = combine_input.topk_weights
        num_tokens = topk_ids.shape[0]

        device = hidden_states.device
        dtype = hidden_states.dtype

        # vllm: This argument is optional, defaults to indices.size(0)
        # There's not much point setting this unless it is != indices.size(0)
        bound_m: Optional[torch.Tensor] = None
        topk_ids_u32 = topk_ids.view(dtype=torch.uint32)

        out_tokens = torch.empty(
            (num_tokens, self.hidden_dim),
            dtype=dtype,
            device=device,
        )

        self.ata.combine(
            out_tokens=out_tokens,
            indices=topk_ids_u32,
            weights=topk_weights,
            expert_y=hidden_states,
            bound_m=bound_m,
        )

        return out_tokens
