from __future__ import annotations

import logging
from typing import NamedTuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode

logger = logging.getLogger(__name__)


class FuseEPDispatchOutput(NamedTuple):
    """DeepEP low latency dispatch output."""

    hidden_state: torch.Tensor

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_LL


class FuseEPCombineInput(NamedTuple):
    """DeepEP low latency combine input."""

    hidden_state: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_LL


class NpuFuseEPDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.LOW_LATENCY,
    ):
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode

        self.params_bytes = 2
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs
    ) -> DispatchOutput:
        hidden_states, _ = self._get_buffer().fused_deep_moe(
            hidden_states,
            topk_idx=topk_output.topk_ids,
            topk_weights=topk_output.topk_weights,
            gmm1_permuted_weight=kwargs["gmm1_permuted_weight"],
            gmm1_permuted_weight_scale=kwargs["gmm1_permuted_weight_scale"],
            gmm2_weight=kwargs["gmm2_weight"],
            gmm2_weight_scale=kwargs["gmm2_weight_scale"],
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            num_experts=self.num_experts,
        )
        return FuseEPDispatchOutput(hidden_states)

    def combine(self, combine_input: CombineInput, **kwargs) -> torch.Tensor:
        pass

    def _get_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )
