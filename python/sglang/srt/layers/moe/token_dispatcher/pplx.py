from __future__ import annotations

import os
from typing import TYPE_CHECKING, NamedTuple

import torch
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher, CombineInput, DispatchOutput, DispatchOutputFormat, CombineInputFormat
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.distributed import get_world_group, get_world_size
from sglang.srt.layers.dp_attention import get_attention_tp_size, get_attention_tp_group, get_attention_dp_size
from sglang.srt.server_args import get_global_server_args
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

try:
    from rose.kernels.efa_all_to_all import EfaAllToAll
    from rose.distributed.torch_group import TorchParallelGroup as RoseParallelGroup
    use_rose = True
except ImportError:
    use_rose = False

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig


GLOBAL_GROUP = None
NUM_NODES = None


class PplxDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    num_recv_tokens_per_expert: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    
    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.PPLX


class PplxCombineInput(NamedTuple):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.PPLX

assert isinstance(PplxDispatchOutput, DispatchOutput)
assert isinstance(PplxCombineInput, CombineInput)


class PplxDispatcher(BaseDispatcher):
    def __init__(self, moe_runner_config: MoeRunnerConfig):
        
        assert use_rose, "Rose is not installed"
        
        global GLOBAL_GROUP
        global NUM_NODES
        
        if GLOBAL_GROUP is None:
            assert NUM_NODES is None
            device = get_world_group().device
            local_rank = get_world_group().local_rank
            global_rank = get_world_group().rank
            ranks = get_world_group().ranks
            hostname = os.environ.get("HOSTNAME", "unknown")
            host_list = [None] * get_world_size()
            torch.distributed.all_gather_object(host_list, hostname)
            unique_nodes = sorted(set(host_list))
            node_rank = unique_nodes.index(hostname)
            
            NUM_NODES = len(unique_nodes)
            GLOBAL_GROUP = RoseParallelGroup(
                device=device,
                node_rank=node_rank,
                local_rank=local_rank,
                global_rank=global_rank,
                ranks=ranks,
            )
        
        global_group = GLOBAL_GROUP
        world_size = get_world_size()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_dp_size  = get_attention_dp_size()
        self.num_local_experts = moe_runner_config.num_local_experts
        tp_group = global_group.slice_by_count(world_size // self.attn_tp_size)
        node_group = global_group.slice_by_count(NUM_NODES)

        self.max_num_tokens = get_global_server_args().chunked_prefill_size // self.attn_tp_size
        self.dtype, self.scale_dtype = torch.float8_e4m3fn, torch.float32  # TODO: support other dtypes
        self.num_experts = moe_runner_config.num_experts

        with torch.device('cpu'):
            self.dispatcher = EfaAllToAll(
                max_num_tokens=self.max_num_tokens,
                num_experts=moe_runner_config.num_experts,
                expert_padding=128,  # deep_gemm expert alignment
                hidden_dim=moe_runner_config.hidden_size,
                hidden_dim_scale=moe_runner_config.hidden_size // 128,
                max_private_tokens=None,
                in_dtype=self.dtype,
                out_dtype=torch.bfloat16,
                scale_dtype=self.scale_dtype,
                num_experts_per_token=moe_runner_config.top_k,
                nets_per_gpu=1,  # TODO: support multiple nets per GPU
                device=get_attention_tp_group().device,
                dp_group=tp_group,
                node_group=node_group,
                global_group=global_group,
            )
        

    def dispatch(self, hidden_states: torch.Tensor, topk_output: TopKOutput) -> PplxDispatchOutput:
        
        from sglang.srt.layers.moe.topk import TopKOutputChecker
        assert TopKOutputChecker.format_is_standard(topk_output)
        
        max_recv_tokens = hidden_states.shape[0] * self.attn_dp_size * self.num_local_experts

        out_expert_num_tokens = torch.empty(
            (self.num_local_experts,),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        out_expert_x = torch.empty(
            (max_recv_tokens, hidden_states.shape[1]),
            dtype=self.dtype,
            device=hidden_states.device,
        )

        if self.dtype == torch.float8_e4m3fn:
            hidden_states_fp8, hidden_states_scale = sglang_per_token_group_quant_fp8(
                hidden_states,
                128,
            )
            assert hidden_states_scale.dtype == self.scale_dtype
            out_expert_x_scale = torch.empty(
                (max_recv_tokens, hidden_states_scale.shape[1]),
                dtype=hidden_states_scale.dtype,
                device=hidden_states.device,
            )
        else:
            raise NotImplementedError()

        if hidden_states_fp8.shape[0] > 0:
            print(f"hidden_states_fp8.shape: {hidden_states_fp8.shape}, hidden_states_scale.shape: {hidden_states_scale.shape}, topk_output.topk_ids.shape: {topk_output.topk_ids.shape}, topk_output.topk_weights.shape: {topk_output.topk_weights.shape}, out_expert_x.shape: {out_expert_x.shape}, out_expert_x_scale.shape: {out_expert_x_scale.shape}, out_expert_num_tokens.shape: {out_expert_num_tokens.shape}, topk_output.topk_ids.min(): {topk_output.topk_ids.min()}, topk_output.topk_ids.max(): {topk_output.topk_ids.max()}")
            assert topk_output.topk_ids.min() >= 0 and topk_output.topk_ids.max() < self.num_experts
        
        topk_ids = topk_output.topk_ids.to(torch.uint32)
        topk_weights = topk_output.topk_weights

        self.dispatcher.dispatch(
            out_expert_num_tokens=out_expert_num_tokens,
            out_expert_x=out_expert_x,
            out_expert_x_scale=out_expert_x_scale,
            dp_x=hidden_states_fp8,
            dp_x_scale=hidden_states_scale,
            indices=topk_ids,
            weights=topk_weights,
            bound_m=None,
        )
        
        # print(out_expert_x, out_expert_x_scale, flush=True)

        return PplxDispatchOutput(
            hidden_states=out_expert_x,
            hidden_states_scale=out_expert_x_scale,
            num_recv_tokens_per_expert=out_expert_num_tokens,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    def combine(self, combine_input: PplxCombineInput) -> torch.Tensor:
        out_tokens = torch.empty(
            (combine_input.topk_ids.shape[0], combine_input.hidden_states.shape[1]),
            dtype=combine_input.hidden_states.dtype,
            device=combine_input.hidden_states.device,
        )

        # print(out_tokens.shape, combine_input.topk_ids.shape, combine_input.topk_weights.shape, combine_input.hidden_states.shape)

        self.dispatcher.combine(
            out_tokens=out_tokens,
            indices=combine_input.topk_ids,
            weights=combine_input.topk_weights,
            expert_y=combine_input.hidden_states,
            bound_m=None,
        )
        
        # print(out_tokens)
        
        return out_tokens
