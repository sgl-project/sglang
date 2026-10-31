from __future__ import annotations

import logging
import time
from enum import Enum, auto
from functools import cache

import torch
import torch.distributed as dist

from sglang.srt.distributed.utils import get_global_tcp_store
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    beat_heartbeat,
    nixl_wired_barrier_via_store,
    read_heartbeats,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_resources,
)
from sglang.srt.utils.common import is_device_stream_capturing

logger = logging.getLogger(__name__)

_FAULT_POLL_INTERVAL_S = 1.0  # Wall clock: captured graph replays run no python.
_FAULT_POLL_SETTLE_S = 30.0  # A width must hold this long before its marks are read.
_FAULT_HEARTBEAT_STALE_S = 10.0  # A peer tick unchanged this long counts as stopped.

NixlEPDispatchOutput = DeepEPLLDispatchOutput
NixlEPCombineInput = DeepEPLLCombineInput


@cache
def _load_nixl_ep() -> tuple[type, torch.dtype]:
    try:
        from nixl_ep import Buffer
    except ImportError as exc:
        raise ImportError(
            "NixlEP is not installed. Please install NixlEP package from "
            "https://github.com/ai-dynamo/nixl."
        ) from exc

    try:
        from nixl_ep import topk_idx_t
    except ImportError:
        topk_idx_t = torch.int64

    assert isinstance(topk_idx_t, torch.dtype)
    return Buffer, topk_idx_t


class NixlEPBuffer:
    """Managing facade for the process-wide NIXL EP buffer; the state itself
    lives on ``ctx.resources``."""

    @classmethod
    def _state(cls):
        from types import SimpleNamespace

        buffers = get_resources().buffers
        state = buffers.get("nixl_ep_state")
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
                reuse_graphs=False,  # Read from --elastic-ep-reuse-cuda-graph at build.
                # Peer-fault state: one per process, since there is one transport mask.
                mask_buffer=None,
                mask_width=None,
                mask_settled_at=0.0,
                mask_cleared=False,
                last_fault_poll=0.0,
                heartbeats={},
                held_marks=set(),
            )
            buffers["nixl_ep_state"] = state
        return state

    @classmethod
    def poll_rank_faults(cls) -> None:
        """Fold the transport's peer mask into the elastic active-rank mask. Host side
        after a forward only: the query leaves stream work and the read is a sync."""
        state = cls._state()
        buffer = state.buffer
        if buffer is None:
            return
        inst = ElasticEPStateManager.instance()
        if inst is None or inst.active_ranks is None:
            return
        # Reading the mask is a device sync, which a graph capture cannot take.
        if is_device_stream_capturing(inst.active_ranks.device):
            return
        # Until a resize commits the mask belongs to the scale path, and the query can
        # leave work on a stream a half-arrived or half-departed peer never joins.
        if ElasticEPStateManager.is_scale_pending():
            return
        n = ElasticEPStateManager.get_data_plane_ep_size()
        if not n or state.connected_ep_size is None:
            return
        width = (state.connected_ep_size, n)
        now = time.monotonic()
        if state.mask_width != width:
            state.mask_width = width
            state.mask_settled_at = now
            state.mask_cleared = False
            state.held_marks.clear()
            # Nobody beats while a scale is pending, so old ticks would all read stale.
            state.heartbeats.clear()
            return
        if state.mask_buffer is None:
            state.mask_buffer = torch.zeros(
                inst.active_ranks.numel(), dtype=torch.int32, device="cuda"
            )
        if not state.mask_cleared:
            # Marks from wiring up stay latched, so erase them once the width settles;
            # a peer that really is gone is marked again by its next combine.
            if now - state.mask_settled_at < _FAULT_POLL_SETTLE_S:
                return
            cls._clear_marks(buffer, n, inst.active_ranks_cpu)
            state.mask_cleared = True
            state.last_fault_poll = now
            return
        if now - state.last_fault_poll < _FAULT_POLL_INTERVAL_S:
            return
        if now - state.last_fault_poll > _FAULT_HEARTBEAT_STALE_S:
            # This rank paused, so every peer tick reads old too: start the window over.
            state.heartbeats.clear()
        state.last_fault_poll = now
        buffer.query_mask_buffer(state.mask_buffer)
        marks = state.mask_buffer[:n].tolist()

        # Beat before reading, and read only the marked peers' ticks.
        beat_heartbeat()
        suspects = [rank for rank in range(n) if marks[rank]]
        for rank, tick in read_heartbeats(suspects).items():
            seen = state.heartbeats.get(rank)
            if seen is None or seen[0] != tick:
                state.heartbeats[rank] = (tick, now)

        # A rank that finds most of the width marked is describing its own transport
        # (e.g. a cold joiner), so retire only a minority. The local rank is never
        # marked, so a width of two retires nobody.
        marked = sum(marks)
        if marked * 2 >= n:
            logger.warning(
                "[Elastic EP][nixl] %d of %d ranks masked by the transport; too many "
                "to be peers failing, so retiring none of them",
                marked,
                n,
            )
            return

        for rank in suspects:
            # Both sides of a stall mark each other, so a mark alone would partition
            # the cohort. Retire only once the peer's scheduler has stopped beating.
            tick, since = state.heartbeats.get(rank, (0, now))
            if not tick or now - since < _FAULT_HEARTBEAT_STALE_S:
                if rank not in state.held_marks:
                    state.held_marks.add(rank)
                    logger.warning(
                        "[Elastic EP][nixl] rank %d masked by the transport but still "
                        "running; leaving it in the active mask",
                        rank,
                    )
                continue
            # Retire on the first mark that outlives the peer: the transport has stopped
            # exchanging with it, so each forward until the rebalance lacks its experts.
            logger.warning(
                "[Elastic EP][nixl] rank %d masked by the transport; retiring it "
                "from the active mask",
                rank,
            )
            # Clear only, never set: re-admitting a rank is the scale path's decision,
            # not a fault detector's, and dp_attention builds its collectives on this.
            inst.active_ranks[rank].zero_()

    @staticmethod
    def _clear_marks(buffer, n: int, live) -> None:
        """Erase the marks at a settled width; retired ranks stay masked (a departed
        peer back in the a2a hangs it)."""
        for rank in range(n):
            if live is not None and not int(live[rank]):
                continue
            try:
                buffer.update_mask_buffer(rank, False)
            except Exception:
                # The local rank and unconnected ranks refuse an unmask.
                logger.debug(
                    "[Elastic EP][nixl] could not clear the mark on rank %d",
                    rank,
                    exc_info=True,
                )

    @classmethod
    def on_scale(cls, from_ep_size: int, to_ep_size: int) -> None:
        """Schedule connections for newly admitted ranks."""
        state = cls._state()
        state.scale_to = to_ep_size
        # Under graph reuse the width stays at capacity; see _mask_unwired.
        if not state.reuse_graphs:
            state.dispatch_ep_size = to_ep_size
        logger.debug(
            "[Elastic EP][nixl] scheduling rank connections: old_ep_size=%d "
            "new_ep_size=%d",
            from_ep_size,
            to_ep_size,
        )

    @classmethod
    def on_retire(cls, retiree_ranks: list) -> None:
        """Survivor NIXL disconnect (drops peer QPs; contiguous tail assumed)."""
        state = cls._state()
        # Lazy connections: the vendor asserts on disconnecting a peer never connected.
        connected = state.connected_ep_size or 0
        tail = min(retiree_ranks)
        if stale := [r for r in retiree_ranks if r < connected]:
            cls._disconnect_ranks(state, stale)
        state.connected_ep_size = min(connected, tail)
        state.scale_to = tail
        if state.reuse_graphs:
            cls._mask_unwired(state, retiree_ranks)
        else:
            state.dispatch_ep_size = tail

    @classmethod
    def _mask_unwired(cls, state, ranks: list) -> None:
        """Mask peerless slots so a capacity-wide dispatch does not wait on them."""
        if not ranks:
            return
        for rank in ranks:
            try:
                state.buffer.update_mask_buffer(rank, True)
            except Exception:
                # A disconnect may already have masked it; the readback decides.
                logger.debug(
                    "[Elastic EP][nixl] could not mask rank %d", rank, exc_info=True
                )
        # The readback syncs, which a capture cannot take.
        if is_device_stream_capturing(torch.device("cuda")):
            return
        width = state.buffer.group_size
        mask = torch.zeros(width, dtype=torch.int32, device="cuda")
        state.buffer.query_mask_buffer(mask)
        unmasked = [r for r in ranks if r < width and not int(mask[r])]
        if unmasked:
            logger.warning(
                "[Elastic EP][nixl] ranks %s are still unmasked; a capacity-wide "
                "dispatch will wait on them",
                unmasked,
            )

    @classmethod
    def _connect_ranks(cls, state, ranks: list, *, tag: str) -> None:
        current_store = get_global_tcp_store()
        if current_store is not None:
            state.buffer.set_tcp_store_group(current_store)

        state.buffer.connect_ranks(ranks)
        logger.debug(
            "[Elastic EP][nixl] connect (%s) ranks=%s group_size=%s",
            tag,
            ranks,
            state.buffer.group_size,
        )

    @classmethod
    def _disconnect_ranks(cls, state, ranks: list) -> None:
        current_store = get_global_tcp_store()
        if current_store is not None:
            state.buffer.set_tcp_store_group(current_store)
        state.buffer.disconnect_ranks(ranks)

    @classmethod
    def _update_connections(cls, state, scale_to: int) -> None:
        if scale_to < state.connected_ep_size:
            # No wired barrier: a narrowing is ranks leaving, and they never arrive.
            cls._disconnect_ranks(state, list(range(scale_to, state.connected_ep_size)))
            state.connected_ep_size = scale_to
            return
        new_ranks = list(range(state.connected_ep_size, scale_to))
        cls._connect_ranks(state, new_ranks, tag="update")
        state.connected_ep_size = scale_to
        nixl_wired_barrier_via_store(scale_to)

    @classmethod
    def get_nixl_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = -1,
        num_experts: int = -1,
        num_local_experts: int = -1,
    ):
        state = cls._state()
        if state.buffer is not None:
            if (
                state.scale_to is not None
                and state.connected_ep_size is not None
                and state.scale_to != state.connected_ep_size
            ):
                cls._update_connections(state, state.scale_to)
            return state.buffer

        Buffer, _ = _load_nixl_ep()
        state.hidden_size = hidden_size
        state.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        state.num_experts = num_experts
        state.num_local_experts = num_local_experts

        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        # Joiner-local ranks are offset into the expanded global rank space.
        offset = ElasticEPStateManager.get_ep_join_rank_offset()
        global_rank = rank + offset

        max_ep_size = get_parallel().max_ep_size or world_size
        nixl_max_ranks = max_ep_size

        num_rdma_bytes = 0
        if deepep_mode.enable_normal():
            raise NotImplementedError("Normal mode is not supported for Nixl EP yet.")
        if deepep_mode.enable_low_latency():
            assert num_max_dispatch_tokens_per_rank != -1
            assert num_experts > 0 and num_local_experts > 0
            max_num_global_experts = nixl_max_ranks * num_local_experts
            num_rdma_bytes = Buffer.get_rdma_size_hint(
                num_max_dispatch_tokens_per_rank,
                hidden_size,
                nixl_max_ranks,
                max_num_global_experts,
            )

        tcp_store = get_global_tcp_store()
        if tcp_store is None:
            raise RuntimeError(
                "Global TCPStore is not initialized. "
                "Make sure init_distributed_environment was called before using NIXL EP."
            )

        logger.info(
            f"Using NIXL EP (world_size={world_size}, max_ep_size={max_ep_size}, "
            f"rank={rank}, global_rank={global_rank}, offset={offset}, "
            f"num_experts={state.num_experts}, "
            f"num_experts_per_rank={state.num_local_experts}) "
        )

        state.buffer = Buffer(
            rank=global_rank,
            tcp_store_group=tcp_store,
        )

        state.buffer.update_memory_buffers(
            num_ranks=nixl_max_ranks,
            num_experts_per_rank=state.num_local_experts,
            num_rdma_bytes=num_rdma_bytes,
        )
        initial_ep_size = offset + world_size
        scale_to = max(initial_ep_size, state.scale_to or 0)
        cls._connect_ranks(state, list(range(scale_to)), tag="initial")
        state.connected_ep_size = scale_to
        state.scale_to = scale_to
        state.reuse_graphs = get_exec().moe.elastic_ep_reuse_cuda_graph
        if state.reuse_graphs:
            state.dispatch_ep_size = nixl_max_ranks
            cls._mask_unwired(state, list(range(scale_to, nixl_max_ranks)))
        else:
            state.dispatch_ep_size = scale_to
        nixl_wired_barrier_via_store(scale_to)
        return state.buffer

    @classmethod
    def clean_buffer(cls):
        state = cls._state()
        state.buffer.clean_buffer(
            state.num_max_dispatch_tokens_per_rank,
            state.hidden_size,
            state.num_experts,
        )


class _NixlEPDispatcherImplBase:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
    ):
        _, self.topk_indices_dtype = _load_nixl_ep()

        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode

        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        # NixlEP internode_ll dispatch uses FINISHED_SUM_TAG=1024
        # and the logic requires num-tokens-sent-from-one-rank-to-another-rank less than it
        assert self.num_max_dispatch_tokens_per_rank <= 1024
        elastic_state = ElasticEPStateManager.instance()
        self.active_ranks = (
            elastic_state.active_ranks if elastic_state is not None else None
        )
        self._active_world_size = dist.get_world_size(group)

        self.handle = None
        self.quant_config = None
        self.overlap_args = None
        self.meta_overlap_args = None

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config

    def set_overlap_args(self, combine_overlap_args, meta_overlap_args) -> None:
        self.overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError

    def _get_buffer(self):
        raise NotImplementedError


class _NixlEPDispatcherImpl(_NixlEPDispatcherImplBase):
    def __init__(self, return_recv_hook: bool, **kwargs):
        super().__init__(**kwargs)

        """
        num_max_dispatch_tokens_per_rank: the actual batch size in the decoding engine should be less than 256
        https://github.com/ai-dynamo/nixl
        """
        self.return_recv_hook = return_recv_hook
        self.device_module = torch.get_device_module()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        buffer = self._get_buffer()
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(self.topk_indices_dtype)
        state = NixlEPBuffer._state()
        dispatch_ep_size = state.dispatch_ep_size
        num_local_experts = state.num_local_experts
        assert dispatch_ep_size is not None and num_local_experts is not None
        num_dispatch_experts = num_local_experts * dispatch_ep_size
        expected_m = (
            hidden_states.shape[0] * dispatch_ep_size * topk_ids.shape[1]
            + num_dispatch_experts
        ) // num_dispatch_experts

        hidden_states, masked_m, event, hook = self._dispatch_core(
            hidden_states,
            topk_ids,
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

        nixl_output = NixlEPDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
        )
        return nixl_output

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
    ):
        use_fp8 = not envs.SGLANG_NIXL_EP_BF16_DISPATCH.get()

        buffer = self._get_buffer()
        state = NixlEPBuffer._state()
        dispatch_ep_size = state.dispatch_ep_size
        num_local_experts = state.num_local_experts
        assert dispatch_ep_size is not None and num_local_experts is not None
        nixl_num_experts = num_local_experts * dispatch_ep_size
        packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
            buffer.dispatch(
                hidden_states,
                topk_idx,
                self.num_max_dispatch_tokens_per_rank,
                nixl_num_experts,
                use_fp8=use_fp8,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
                round_scale=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
                use_ue8m0=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
            )
        )
        return packed_recv_hidden, self.packed_recv_count, event, hook

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

        combined_hidden_states, event, hook = buffer.combine(
            x=hidden_states,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            handle=self.handle,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        )
        self.packed_recv_count = self.handle = None
        return combined_hidden_states, event, hook

    def _get_buffer(self):
        return NixlEPBuffer.get_nixl_buffer(
            self.group,
            self.hidden_size,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            self.num_local_experts,
        )


class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class NixlEPDispatcher(BaseDispatcher):
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
            deepep_mode=deepep_mode,
        )

        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _NixlEPDispatcherImpl(
                return_recv_hook=return_recv_hook,
                **common_kwargs,
            )
        if self.deepep_mode.enable_normal():
            raise NotImplementedError("Normal mode is not supported for Nixl EP yet.")

        self._stage = _Stage.INITIAL

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states=hidden_states, topk_output=topk_output)
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

    def _get_impl(self) -> _NixlEPDispatcherImplBase:
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            raise NotImplementedError("Normal mode is not supported for Nixl EP yet.")
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def set_quant_config(self, quant_config: dict):
        super().set_quant_config(quant_config)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_quant_config(quant_config)

    def set_overlap_args(self, combine_overlap_args, meta_overlap_args):
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage
