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
    get_parallel,
    get_resources,
)
from sglang.srt.utils.common import is_device_stream_capturing

logger = logging.getLogger(__name__)

# Seconds between peer-state polls at a steady width. Wall clock rather than a count
# of combines: a count only advances where python runs, so a captured decode graph
# replays without ever reaching it, and ranks that carry less of the routing or that
# joined later reach a given count later than their peers. The cohort then disagrees
# about who is alive, which is how a vote ends up one short of its own quorum.
_FAULT_POLL_INTERVAL_S = 1.0

# How long a width has to hold before the mask is worth reading at all. Covers server
# start and the tail of a scale, where peers are still connecting and a mark says more
# about who finished wiring up first than about who is alive.
_FAULT_POLL_SETTLE_S = 30.0

# How long a peer's liveness tick has to stand still before the peer counts as stopped.
# Ten poll intervals, so a rank that merely missed a few ticks under load is not taken
# for a dead one.
_FAULT_HEARTBEAT_STALE_S = 10.0

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
                # Peer-fault state, held once per process rather than once per
                # dispatcher: there is one transport mask, and 26 MoE layers each
                # polling it on their own counter is 26 different opinions about
                # when a peer died.
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
        """Fold the transport's peer mask into the elastic active-rank mask.

        Host side and after a forward, never from inside one. The query leaves work on
        the stream and reading the result is a sync, and the combine is the one place
        where neither is safe.

        A mark is a guess about liveness -- the transport sets it on a single combine
        receive timeout, so a peer that stalled one forward looks like a peer that
        died -- but a mark on a minority is acted on immediately all the same. The
        mark itself is what costs: once it is set the transport has stopped exchanging
        with that peer, so every forward until the rebalance is a forward missing its
        experts. Retiring promptly is what lets the rebalance route around it, and the
        cost of guessing wrong is a rebalance, which the cohort survives.
        """
        state = cls._state()
        buffer = state.buffer
        if buffer is None:
            return
        inst = ElasticEPStateManager.instance()
        if inst is None or inst.active_ranks is None:
            return
        # Not while a graph is being captured. Reading the mask is a device sync, and a
        # sync inside capture either tears the capture down or bakes this poll into a
        # graph that will replay it on every decode. The capture window is also exactly
        # where peers are least in step, so it is where the query is most likely to see
        # a mark it should not act on.
        if is_device_stream_capturing(inst.active_ranks.device):
            return
        # Until a resize commits the mask belongs to the scale path, and the query can
        # leave work on a stream that a half-arrived or half-departed peer will never
        # join. A real fault is deferred by one resize rather than missed.
        if ElasticEPStateManager.is_scale_pending():
            return
        n = ElasticEPStateManager.get_data_plane_ep_size()
        connected = state.connected_ep_size
        if not n or connected is None:
            return

        width = (connected, n)
        now = time.monotonic()
        if state.mask_width != width:
            state.mask_width = width
            state.mask_settled_at = now
            state.mask_cleared = False
            state.held_marks.clear()
            # Ticks recorded at the old width mean nothing at the new one. Nobody beats
            # while a scale is pending, so every peer's tick stood still through it, and
            # keeping those timestamps would read the pause as a cohort of deaths on the
            # first poll that follows.
            state.heartbeats.clear()
            return
        if state.mask_buffer is None:
            state.mask_buffer = torch.zeros(
                inst.active_ranks.numel(), dtype=torch.int32, device="cuda"
            )
        if not state.mask_cleared:
            # Let a new width settle, then erase whatever is marked at it before any of
            # this counts. Connections are still being wired for a moment after a width
            # lands, and a mark there says more about who finished wiring first than
            # about who is alive -- but a mark is a latch the transport never lifts, so
            # waiting alone does not help. The marks a cold start leaves behind read
            # exactly like a death for the rest of the run.
            #
            # Erasing is self correcting where remembering is not. Recording the marks
            # as a floor for later polls to beat forgives, for good, a peer that really
            # did die during the scale: nobody retires it, the router keeps sending to
            # it, and the answers are quietly wrong at 10% gsm8k. Erased, a peer that is
            # gone is marked again by its very next combine and retired a second later,
            # while one that was merely slow is not.
            if now - state.mask_settled_at < _FAULT_POLL_SETTLE_S:
                return
            cls._clear_marks(buffer, n, inst.active_ranks_cpu)
            state.mask_cleared = True
            state.last_fault_poll = now
            return
        if now - state.last_fault_poll < _FAULT_POLL_INTERVAL_S:
            return
        if now - state.last_fault_poll > _FAULT_HEARTBEAT_STALE_S:
            # This rank stopped polling for a while, so it stopped beating too, and so
            # did every peer that paused with it. Their ticks are old for the same
            # reason ours is. Start the window over rather than read the pause as death.
            state.heartbeats.clear()
        state.last_fault_poll = now
        buffer.query_mask_buffer(state.mask_buffer)
        marks = state.mask_buffer[:n].tolist()

        # Beat first, then read, so a rank is never the reason its own tick looks old.
        # Only the marked are read: a healthy width is then one store write per rank
        # per second and no reads at all, rather than a width's worth of them.
        beat_heartbeat()
        suspects = [rank for rank in range(n) if marks[rank]]
        for rank, tick in read_heartbeats(suspects).items():
            seen = state.heartbeats.get(rank)
            if seen is None or seen[0] != tick:
                state.heartbeats[rank] = (tick, now)

        # A rank that finds most of the width marked is describing itself, not the
        # width. A freshly joined rank whose own transport came up cold marks every
        # peer it failed to reach, and acting on that retires the entire serving
        # cohort from the newcomer's mask while the cohort retires nobody: a 4 -> 8
        # regrow had both joiner ranks drop all four survivors, after which any
        # request routed to a joiner slot blocked in a collective the two sides no
        # longer agreed on. So retire only a minority, the condition under which what
        # is left is still a cohort. The local rank can never be marked, which makes a
        # width of two unable to retire anyone, and that is the right answer there:
        # one of the two is wrong and the mask cannot say which.
        marked = sum(marks)
        if marked * 2 >= n:
            logger.warning(
                "[Elastic EP][nixl] %d of %d ranks masked by the transport; too many "
                "to be peers failing, so retiring none of them",
                marked,
                n,
            )
            return

        for rank in range(n):
            if not marks[rank]:
                continue
            # A mark says the transport gave up on a peer. It does not say the peer is
            # gone, and the difference decides whether retiring is a repair or a
            # self-inflicted partition: both sides of a stall mark each other, and both
            # acting on it leaves each serving a cohort the other has written off. An
            # 8 -> 6 whose top two slots had just rejoined ended exactly there, the
            # survivors dropping the two live joiner ranks off a burst of 32 timeouts
            # each while those two ranks dropped all four survivors, after which a
            # request to either side blocked in a collective built over a different
            # set of ranks. The tick settles it, because a peer that is merely cold
            # keeps running and a peer that is gone does not. A wedged peer is caught
            # too: this poll is its scheduler's, so a scheduler that stops stops
            # beating.
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
            # Act on the first mark that outlives the peer. Giving a suspect another
            # round sounds kinder and is not: a masked peer is one the transport has
            # stopped exchanging with, so every forward until it is retired is a
            # forward missing that peer's experts. Unmasking to see whether it recovers
            # was measured at 44% gsm8k against a 50% floor. Retiring promptly is what
            # lets the rebalance route around it.
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
        """Erase the marks standing at a settled width, skipping retired ranks.

        A rank the cohort has already retired stays masked: re-admitting one is the
        scale path's decision, and putting a departed peer back into the a2a hangs it.
        """
        for rank in range(n):
            if live is not None and not int(live[rank]):
                continue
            try:
                buffer.update_mask_buffer(rank, False)
            except Exception:
                # The local rank cannot be masked, and a rank that is not connected
                # cannot be unmasked. Both refuse here rather than leave a mark.
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
        # Only retirees this rank connected to: connections are lazy, so a
        # grow-shrink with no dispatch between never made them and the vendor
        # asserts on an unknown peer. The next dispatch closes the gap.
        connected = state.connected_ep_size or 0
        tail = min(retiree_ranks)
        stale = [r for r in retiree_ranks if r < connected]
        if stale:
            cls._disconnect_ranks(state, stale)
        state.connected_ep_size = min(connected, tail)
        state.scale_to = tail
        state.dispatch_ep_size = tail

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
        widened = scale_to > state.connected_ep_size
        if widened:
            cls._connect_ranks(
                state, list(range(state.connected_ep_size, scale_to)), tag="update"
            )
        elif scale_to < state.connected_ep_size:
            cls._disconnect_ranks(state, list(range(scale_to, state.connected_ep_size)))
        state.connected_ep_size = scale_to
        if widened:
            # On the way up only: a narrowing is ranks leaving, and they never arrive.
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
