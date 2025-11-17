from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.utils import get_int_env_var

if TYPE_CHECKING:
    from sglang.srt.single_batch_overlap import CombineOverlapArgs

from enum import Enum, auto

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class MooncakeDispatchOutput(NamedTuple):
    """Mooncake EP dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_LL


assert isinstance(MooncakeDispatchOutput, DispatchOutput)


class MooncakeCombineInput(NamedTuple):
    """Mooncake EP combine input."""

    pass

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_LL


assert isinstance(MooncakeCombineInput, CombineInput)


class EPBuffer:
    _buffer = None
    _hidden_size: Optional[int] = None
    _num_max_dispatch_tokens_per_rank: Optional[int] = None
    _num_experts: Optional[int] = None

    @classmethod
    def get_ep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        param_bytes: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = -1,
        num_experts: int = -1,
    ):
        if cls._buffer is not None:
            return cls._buffer

        # Lazy import Buffer to avoid creating CUDA context at module import time
        from mooncake.mooncake_ep_buffer import Buffer

        cls._hidden_size = hidden_size
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        cls._num_experts = num_experts

        num_ep_buffer_bytes = 0
        if deepep_mode.enable_normal():
            raise NotImplementedError(
                "Normal mode is not supported for Mooncake EP yet."
            )
        if deepep_mode.enable_low_latency():
            assert num_max_dispatch_tokens_per_rank != -1
            assert num_experts != -1 and num_experts % group.size() == 0
            num_ep_buffer_bytes = Buffer.get_ep_buffer_size_hint(
                num_max_dispatch_tokens_per_rank,
                hidden_size,
                group.size(),
                num_experts,
            )

        cls._buffer = Buffer(group, num_ep_buffer_bytes)
        return cls._buffer


class _MooncakeEPDispatcherImpl:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        return_recv_hook: bool,
        deepep_mode: DeepEPMode,
    ):
        try:
            from mooncake.mooncake_ep_buffer import Buffer  # noqa: F401
        except ImportError:
            raise ImportError(
                "Mooncake EP is not installed. Please install Mooncake package at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                "with EP support to run SGLang with Mooncake EP."
            )
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.return_recv_hook = return_recv_hook
        self.deepep_mode = deepep_mode

        self.params_bytes = 2
        self.num_max_dispatch_tokens_per_rank = get_int_env_var(
            "SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 128
        )
        # Mooncake EP dispatch uses FINISHED_SUM_TAG=1024
        # and the logic requires num-tokens-sent-from-one-rank-to-another-rank less than it
        assert self.num_max_dispatch_tokens_per_rank <= 1024

        self.first_execution = True
        self.timeout_us = 10000000

        self.active_ranks = ElasticEPStateManager.instance().active_ranks

        self.handle = None

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        topk_ids, topk_weights = topk_output.topk_ids, topk_output.topk_weights
        buffer = self._get_buffer()
        topk_ids = topk_ids.to(torch.int64)
        expected_m = (
            hidden_states.shape[0] * buffer.group_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts
        hidden_states, masked_m, event, hook = self._dispatch_core(
            hidden_states,
            topk_ids,
            use_fp8=True,
        )
        return (
            hidden_states,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
            event,
            hook,
        )

    def dispatch_b(
        self,
        hidden_states,
        topk_ids,
        topk_weights,
        masked_m,
        expected_m,
        event,
        hook,
    ):
        hook() if self.return_recv_hook else event.current_stream_wait()

        get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(
            masked_m
        )

        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        else:
            hidden_states_scale = None

        return MooncakeDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
        )

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        use_fp8: bool = False,
    ):
        buffer = self._get_buffer()
        packed_recv_hidden, packed_recv_count, self.handle, event, hook = (
            buffer.dispatch(
                hidden_states,
                topk_ids,
                self.active_ranks,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                -1 if self.first_execution else self.timeout_us,
                use_fp8=use_fp8,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
            )
        )
        return packed_recv_hidden, packed_recv_count, event, hook

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[CombineOverlapArgs] = None,
    ):
        hidden_states, event, hook = self._combine_core(
            hidden_states,
            topk_ids,
            topk_weights,
        )
        return hidden_states, event, hook, overlap_args

    def combine_b(self, hidden_states, event, hook):
        hook() if self.return_recv_hook else event.current_stream_wait()
        return hidden_states

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        buffer = self._get_buffer()
        combined_hidden_states, event, hook = buffer.combine(
            hidden_states,
            topk_ids,
            topk_weights,
            self.active_ranks,
            -1 if self.first_execution else self.timeout_us,
            self.handle,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        )
        self.first_execution = False
        self.handle = None
        return combined_hidden_states, event, hook

    def _get_buffer(self):
        return EPBuffer.get_ep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )


@dataclass
class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class MooncakeEPDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        self.deepep_mode = deepep_mode

        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _MooncakeEPDispatcherImpl(
                group=group,
                router_topk=router_topk,
                permute_fusion=permute_fusion,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                hidden_size=hidden_size,
                params_dtype=params_dtype,
                return_recv_hook=return_recv_hook,
                deepep_mode=deepep_mode,
            )
        if self.deepep_mode.enable_normal():
            raise NotImplementedError

        self._stage = _Stage.INITIAL

    def dispatch(self, *args, **kwargs) -> DispatchOutput:
        self.dispatch_a(*args, **kwargs)
        ret = self.dispatch_b()
        return ret

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)

    def combine(
        self,
        combine_input: CombineInput,
        overlap_args: Optional[CombineOverlapArgs] = None,
    ) -> Tuple:
        self.combine_a(combine_input, overlap_args)
        ret = self.combine_b()
        return ret

    def combine_a(
        self,
        combine_input: CombineInput,
        overlap_args: Optional[CombineOverlapArgs] = None,
    ):
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            overlap_args=overlap_args,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)

    def _get_impl(self) -> _MooncakeEPDispatcherImpl:
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            raise NotImplementedError
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage

    def set_quant_config(self, quant_config: dict):
        pass
