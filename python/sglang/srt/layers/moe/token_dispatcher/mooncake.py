from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed.utils import get_global_tcp_store
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
from sglang.srt.runtime_context import (
    get_parallel,
    get_resources,
    should_use_elastic_mooncake_ep,
)
from sglang.srt.utils import get_int_env_var

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
    """Managing facade for the process-wide Mooncake EP buffer; the state
    itself lives on ``ctx.resources``."""

    @classmethod
    def _state(cls):
        from types import SimpleNamespace

        buffers = get_resources().buffers
        state = buffers.get("mooncake_ep_state")
        if state is None:
            state = SimpleNamespace(
                buffer=None,
                hidden_size=None,
                num_max_dispatch_tokens_per_rank=None,
                num_experts=None,
                num_local_experts=None,
                connected_ep_size=None,
                scale_to=None,
                dispatch_ep_size=None,
            )
            buffers["mooncake_ep_state"] = state
        return state

    @classmethod
    def get_existing_buffer(cls):
        """The already-created buffer (elastic-EP membership refresh)."""
        return cls._state().buffer

    @staticmethod
    def is_elastic_ep_enabled() -> bool:
        parallel = get_parallel()
        return should_use_elastic_mooncake_ep(
            max_ep_size=parallel.max_ep_size,
            elastic_ep_initial_size=parallel.elastic_ep_initial_size,
            tp_size=parallel.tp_size,
        )

    @classmethod
    def on_scale(cls, from_ep_size: int, to_ep_size: int) -> None:
        """Record the scale target; connections are opened lazily on use."""
        state = cls._state()
        state.scale_to = to_ep_size
        state.dispatch_ep_size = to_ep_size
        logger.debug(
            "[Elastic EP][mooncake] scheduling rank connections: "
            "old_ep_size=%d new_ep_size=%d",
            from_ep_size,
            to_ep_size,
        )

    @classmethod
    def _connect_new_ranks(cls, state, target_ep_size: int) -> None:
        if state.buffer is None:
            return
        connected_ep_size = state.connected_ep_size or 0
        if target_ep_size > connected_ep_size:
            current_store = get_global_tcp_store()
            if current_store is not None:
                state.buffer.set_tcp_store_group(current_store)
            state.buffer.connect_ranks(list(range(connected_ep_size, target_ep_size)))
            state.connected_ep_size = target_ep_size

    @classmethod
    def get_dispatch_ep_size(cls) -> int:
        state = cls._state()
        if state.dispatch_ep_size is not None:
            return state.dispatch_ep_size
        return ElasticEPStateManager.get_data_plane_ep_size()

    @classmethod
    def get_ep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        param_bytes: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = -1,
        num_experts: int = -1,
        num_local_experts: int = -1,
    ):
        state = cls._state()
        elastic_ep = cls.is_elastic_ep_enabled()
        if state.buffer is not None:
            if elastic_ep and state.scale_to is not None:
                cls._connect_new_ranks(state, state.scale_to)
            return state.buffer

        # Lazy import Buffer to avoid creating CUDA context at module import time
        from mooncake.mooncake_ep_buffer import Buffer

        state.hidden_size = hidden_size
        state.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        state.num_experts = num_experts
        state.num_local_experts = num_local_experts

        if deepep_mode.enable_normal():
            raise NotImplementedError(
                "Normal mode is not supported for Mooncake EP yet."
            )
        if deepep_mode.enable_low_latency():
            assert num_max_dispatch_tokens_per_rank != -1
            assert num_experts != -1
            if not elastic_ep:
                assert num_experts % group.size() == 0

        if elastic_ep:
            if not deepep_mode.enable_low_latency():
                raise NotImplementedError(
                    "Elastic Mooncake EP requires low-latency dispatch."
                )
            assert num_local_experts > 0
            rank = dist.get_rank(group)
            world_size = dist.get_world_size(group)
            rank_offset = ElasticEPStateManager.get_ep_join_rank_offset()
            global_rank = rank + rank_offset
            capacity = get_parallel().max_ep_size or world_size
            initial_ep_size = rank_offset + world_size
            if global_rank >= capacity or initial_ep_size > capacity:
                raise ValueError(
                    "Elastic Mooncake EP rank range exceeds --max-ep-size."
                )
            num_rdma_bytes = Buffer.get_rdma_size_hint(
                num_max_dispatch_tokens_per_rank,
                hidden_size,
                capacity,
                capacity * num_local_experts,
            )
            tcp_store = get_global_tcp_store()
            if tcp_store is None:
                raise RuntimeError(
                    "Mooncake EP requires the global TCPStore initialized during "
                    "distributed setup."
                )

            state.buffer = Buffer.create_elastic(
                rank=global_rank,
                tcp_store_group=tcp_store,
            )
            state.buffer.update_memory_buffers(
                num_ranks=capacity,
                num_experts_per_rank=num_local_experts,
                num_rdma_bytes=num_rdma_bytes,
            )
            target_ep_size = max(initial_ep_size, state.scale_to or 0)
            state.scale_to = target_ep_size
            state.dispatch_ep_size = target_ep_size
            state.connected_ep_size = 0
            cls._connect_new_ranks(state, target_ep_size)
            return state.buffer

        num_ep_buffer_bytes = 0
        if deepep_mode.enable_low_latency():
            num_ep_buffer_bytes = Buffer.get_ep_buffer_size_hint(
                num_max_dispatch_tokens_per_rank,
                hidden_size,
                group.size(),
                num_experts,
            )

        state.buffer = Buffer(group, num_ep_buffer_bytes)
        return state.buffer


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
        self._elastic_ep = EPBuffer.is_elastic_ep_enabled()
        self._elastic_state = (
            ElasticEPStateManager.instance() if self._elastic_ep else None
        )
        self._mask_buffer = (
            torch.empty_like(self._elastic_state.active_ranks)
            if self._elastic_state is not None
            else None
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

        self.handle = None

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        topk_ids, topk_weights = topk_output.topk_ids, topk_output.topk_weights
        buffer = self._get_buffer()
        topk_ids = topk_ids.to(torch.int64)
        if self._elastic_ep:
            rank_bound = EPBuffer.get_dispatch_ep_size()
            dispatch_num_experts = rank_bound * self.num_local_experts
            expected_m = (
                hidden_states.shape[0] * rank_bound * topk_ids.shape[1]
                + dispatch_num_experts
            ) // dispatch_num_experts
        else:
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
        if self._elastic_ep:
            dispatch_num_experts = (
                EPBuffer.get_dispatch_ep_size() * self.num_local_experts
            )
            result = buffer.dispatch(
                hidden_states,
                topk_ids,
                num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
                num_experts=dispatch_num_experts,
                use_fp8=use_fp8,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
            )
        else:
            active_ranks = self._get_active_ranks()
            result = buffer.dispatch(
                hidden_states,
                topk_ids,
                active_ranks,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                -1 if self.first_execution else self.timeout_us,
                use_fp8=use_fp8,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
            )
        packed_recv_hidden, packed_recv_count, self.handle, event, hook = result
        return packed_recv_hidden, packed_recv_count, event, hook

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        hidden_states, event, hook = self._combine_core(
            hidden_states,
            topk_ids,
            topk_weights,
        )
        return hidden_states, event, hook

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
        if self._elastic_ep:
            combined_hidden_states, event, hook = buffer.combine(
                hidden_states,
                topk_ids,
                topk_weights,
                self.handle,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
            )
            if self._elastic_state is not None:
                buffer.query_mask_buffer(self._mask_buffer)
                rank_bound = EPBuffer.get_dispatch_ep_size()
                self._elastic_state.active_ranks[:rank_bound].copy_(
                    1 - self._mask_buffer[:rank_bound]
                )
        else:
            active_ranks = self._get_active_ranks()
            combined_hidden_states, event, hook = buffer.combine(
                hidden_states,
                topk_ids,
                topk_weights,
                active_ranks,
                -1 if self.first_execution else self.timeout_us,
                self.handle,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
            )
        self.first_execution = False
        self.handle = None
        return combined_hidden_states, event, hook

    def _get_active_ranks(self) -> torch.Tensor:
        elastic_state = ElasticEPStateManager.instance()
        if elastic_state is not None:
            return elastic_state.active_ranks
        return torch.ones(self.group.size(), dtype=torch.int32, device="cuda")

    def _get_buffer(self):
        return EPBuffer.get_ep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            self.num_local_experts,
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
        super().__init__()

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

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
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
    ) -> torch.Tensor:
        self.combine_a(combine_input)
        ret = self.combine_b()
        return ret

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
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
