"""Generation-owned LL resources for serialized full decode CUDA Graphs.

Each in-flight subbatch owns a separate group and handle. Compatible layers
and buckets reuse the resources of their subbatch serially. This is an
ownership policy, not an API handle-count limit.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from threading import RLock, get_ident
from types import SimpleNamespace

import torch

from sglang.srt.runtime_context import get_resources


class NcclEpGraphResources:
    def __init__(self, capacity: int, *, shutdown):
        if not 0 < capacity <= 1024:
            raise ValueError("NCCL EP Graph capacity must be in [1, 1024]")
        self.capacity = capacity
        self.shutdown = shutdown
        self._lanes = {}
        self.capturing = False
        self._lock = RLock()
        self._sessions = []
        self._last_stream = None
        self._fence = None
        self._session_thread = None

    def require_session(self, kind):
        if (
            not self._sessions
            or self._session_thread != get_ident()
            or self._sessions[-1][0] != kind
        ):
            raise RuntimeError(f"NCCL EP Graph requires an active {kind}_session")

    def _order_stream(self, stream):
        previous = self._last_stream
        if previous is not None and previous != stream:
            # Record now, including consumers submitted after the last replay
            # returned. An event recorded at replay's end would miss those reads.
            self._fence = previous.record_event()
            stream.wait_event(self._fence)
        self._last_stream = stream

    @contextmanager
    def submission_session(self, kind):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent NCCL EP Graph submissions are unsupported")
        entered = False
        try:
            if self._sessions and (self._sessions[-1][0], kind) not in {
                ("replay", "capture"),
                ("replay", "cleanup"),
                ("capture", "cleanup"),
                ("replay", "eager"),
                ("replay", "transaction"),
                ("eager", "transaction"),
            }:
                raise RuntimeError(
                    "Overlapping NCCL EP Graph submissions are unsupported"
                )
            stream = torch.cuda.current_stream()
            self._order_stream(stream)
            self._sessions.append((kind, stream))
            self._session_thread = get_ident()
            entered = True
            yield
        finally:
            if entered:
                self._sessions.pop()
                if self._sessions:
                    # Recapture uses a child stream inside execute's replay
                    # session. Order it back before the parent's input writes.
                    self._order_stream(self._sessions[-1][1])
                elif (
                    not any(lane.state is not None for lane in self._lanes.values())
                    and get_nccl_ep_graph_resources() is self
                ):
                    get_resources().buffers.pop("nccl_ep_graph_resources")
                if not self._sessions:
                    self._session_thread = None
            self._lock.release()

    @contextmanager
    def capture_session(self):
        previous = get_nccl_ep_graph_resources()
        if previous is not None and previous is not self:
            raise RuntimeError("Another NCCL EP Graph generation is still live")
        with self.submission_session("capture"):
            get_resources().buffers["nccl_ep_graph_resources"] = self
            self.capturing = True
            try:
                yield
            finally:
                self.capturing = False

    def lane(self, index):
        if index not in (0, 1):
            raise ValueError("NCCL EP supports two TBO subbatches")
        if index not in self._lanes:
            self._lanes[index] = _NcclEpGraphLane(self)
        return self._lanes[index]

    @property
    def state(self):
        return self.lane(0).state

    @property
    def handle(self):
        return self.lane(0).handle

    @property
    def borrower(self):
        return next(
            (
                lane.borrower
                for lane in self._lanes.values()
                if lane.borrower is not None
            ),
            None,
        )

    def prepare(self, dispatcher, x, ids, weights):
        self.require_session("capture")
        index = dispatcher.instance_id or 0
        # Create both groups collectively, in identical order on every rank,
        # before either subbatch can submit communication during warmup.
        if dispatcher.instance_id is not None:
            for lane_index in (0, 1):
                lane = self.lane(lane_index)
                if lane.state is None:
                    lane.prepare(dispatcher, x, ids, weights)
                    lane.release(dispatcher)
        return self.lane(index).prepare(dispatcher, x, ids, weights)

    def release(self, dispatcher):
        self.lane(dispatcher.instance_id or 0).release(dispatcher)

    def close(self):
        # Preflight every lane before releasing any generation-owned state.
        if self.borrower is not None:
            raise RuntimeError("NCCL EP Graph has an incomplete transaction")
        for lane in self._lanes.values():
            lane.close()
        if not self._sessions and get_nccl_ep_graph_resources() is self:
            get_resources().buffers.pop("nccl_ep_graph_resources")


class _NcclEpGraphLane:
    def __init__(self, owner):
        self.owner = owner
        self.capacity = owner.capacity
        self.state = self.handle = self.borrower = None
        self.signature = self.rows = None

    def prepare(self, dispatcher, x, ids, weights):
        from .nccl_ep import NcclEpBuffer, _load_nccl_ep

        self.owner.require_session("capture")
        if not self.owner.capturing or self.borrower is not None:
            raise RuntimeError("NCCL EP Graph requires complete serial transactions")
        t = x.shape[0]
        if not 0 < t <= self.capacity:
            raise ValueError("NCCL EP Graph bucket exceeds its frozen capacity")
        if x.ndim != 2 or x.shape[1] != dispatcher.hidden_size:
            raise ValueError(
                "NCCL EP Graph input hidden size does not match its MoE layer"
            )
        if (
            x.dtype != torch.bfloat16
            or dispatcher.params_dtype != torch.bfloat16
            or x.device != dispatcher.ep_group.device
        ):
            raise ValueError("NCCL EP Graph requires BF16 inputs on its EP device")
        if ids.shape != (t, dispatcher.router_topk) or weights.shape != ids.shape:
            raise ValueError("NCCL EP Graph routing/weight shapes do not match inputs")
        signature = (
            dispatcher.ep_group.pynccl_comm.comm.value,
            x.device,
            dispatcher.world_size,
            dispatcher.num_experts,
            dispatcher.num_local_experts,
            dispatcher.hidden_size,
            dispatcher.router_topk,
            dispatcher.params_dtype,
        )
        if self.signature is not None and signature != self.signature:
            raise ValueError(
                "Incompatible MoE layer for the shared NCCL EP Graph group"
            )
        nccl_core, ep = _load_nccl_ep()
        stream = torch.cuda.current_stream().cuda_stream
        if self.state is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Initialize NCCL EP Graph resources during warmup")
            self.signature = signature
            state = self.state = SimpleNamespace(
                group=None,
                num_experts=dispatcher.num_experts,
                num_local_experts=dispatcher.num_local_experts,
                hidden_size=dispatcher.hidden_size,
                max_dispatch_tokens_per_rank=self.capacity,
                max_recv_tokens_per_rank=dispatcher.world_size * self.capacity,
            )
            NcclEpBuffer._alloc_scratch(state, x.device)
            state.send_tokens = torch.empty(
                (self.capacity, dispatcher.hidden_size), dtype=x.dtype, device=x.device
            )
            state.topk_ids = torch.full(
                (self.capacity, dispatcher.router_topk),
                -1,
                dtype=torch.int64,
                device=x.device,
            )
            state.topk_weights = torch.empty_like(state.topk_ids, dtype=torch.float32)
            state.wrapped_comm = nccl_core.Communicator(ptr=signature[0])
            state.group = ep.Group.create(
                state.wrapped_comm,
                ep.GroupConfig(
                    algorithm=ep.Algorithm.LOW_LATENCY,
                    num_experts=state.num_experts,
                    num_topk=dispatcher.router_topk,
                    max_dispatch_tokens_per_rank=self.capacity,
                    max_recv_tokens_per_rank=0,
                    max_token_bytes=dispatcher.hidden_size * 2,
                    rdma_buffer_size=0,  # AUTO: initialize the maximum layout once.
                    max_num_sms=NcclEpBuffer._resolve_max_num_sms(state.num_experts),
                ),
            )
            self.handle = state.group.create_handle(
                layout=ep.Layout.EXPERT_MAJOR,
                topk_idx=ep.Tensor(state.topk_ids),
                config=ep.HandleConfig(),
                stream=stream,
            )
            self.rows = self.capacity
        state = self.state
        send = state.send_tokens[:t]
        routing = state.topk_ids[:t]
        factors = state.topk_weights[:t]
        send.copy_(x)
        routing.copy_(ids)
        factors.copy_(weights)
        if self.rows != t:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "NCCL EP bucket descriptors must be set during warmup"
                )
            # Same allocation/base address; only the bucket's descriptor changes.
            self.handle.update(ep.Tensor(routing), stream=stream)
            self.rows = t
        self.borrower = dispatcher
        return state, self.handle, send, routing, factors

    def release(self, dispatcher):
        if self.borrower is not dispatcher:
            raise RuntimeError("NCCL EP Graph transaction ownership mismatch")
        self.borrower = None

    def close(self):
        # The backend must wait and reset every executable before this call.
        if self.borrower is not None:
            raise RuntimeError("NCCL EP Graph has an incomplete transaction")
        if self.handle is not None:
            self.handle.destroy()
            self.handle = None
        if self.state is not None and self.state.group is not None:
            self.state.group.destroy()
        self.state = self.signature = self.rows = None


def get_nccl_ep_graph_resources():
    return get_resources().buffers.get("nccl_ep_graph_resources")


def nccl_ep_eager_session():
    owner = get_nccl_ep_graph_resources()
    return owner.submission_session("eager") if owner is not None else nullcontext()


def require_nccl_ep_eager_session():
    owner = get_nccl_ep_graph_resources()
    if owner is not None:
        owner.require_session("eager")


def destroy_nccl_ep_resources():
    """Close graph executables and EP groups before releasing coordinators."""
    owner = get_nccl_ep_graph_resources()
    if owner is not None:
        owner.shutdown()
    if any(
        (state := get_resources().buffers.get(key)) is not None
        and state.group is not None
        for key in ("nccl_ep_state", "nccl_ep_state_1")
    ):
        from .nccl_ep import NcclEpBuffer

        torch.cuda.synchronize()
        NcclEpBuffer.destroy()
