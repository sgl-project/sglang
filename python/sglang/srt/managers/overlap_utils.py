"""Relayer: scheduler-owned cross-iter relay services.

A "relay" is any value whose producer phase is forward N's completion
(model kernel output, or CPU branch in ``process_batch_result`` that
observes forward output) and whose consumer phase lives in iter N+1
(``filter_batch`` / ``merge_batch`` / ``prepare_for_decode`` /
``cache_finished_req`` / next ``forward_batch_generation``). Examples:

- ``output_ids`` (non-spec) — model output -> next iter input_ids
- ``new_seq_lens`` (spec V2 verify) -> next iter schedule prep
- ``kv_committed_len`` increment -> next iter cache release
- ``finished`` status -> next iter ``filter_batch`` keep mask
- Sampling penalizer accumulator state -> next iter sampling

Pre-Relayer, each of these relays had its own ad-hoc mechanism:
``FutureMap`` for token ids, in-place ``seq_lens.add_(1)`` for seq_lens,
``req.kv_committed_len += accept_lens[i] - 1`` for KV commit,
``verify_done.synchronize()`` CPU sync for visibility, ``batch_record_buf``
ring for lifetime, ``copy_for_forward`` for sampling penalizer. Reviewers
could not derive race-freedom from reading the code; each ad-hoc mechanism
covered one subset of the four axes (GPU memory race / GPU lifetime / CPU
TOCTOU / Lockstep) and the overall invariant lived only in design docs.

The ``Relayer`` is the single named home for every relay. Each relay
funnels through one of five typed sub-channels (see classes below) and
exposes a uniform ``alloc -> store -> resolve`` API, plus cross-stream
sync (``GpuScalarChannel.store`` records a cuda event; ``resolve_*``
event-waits on the consumer stream). The scheduler becomes the single
explicit producer of "what next iter's forward needs", and consumer-side
visibility is established by channel semantics rather than ad-hoc CPU
syncs scattered across the codebase.

Five sub-channels:

- ``gpu_scalar``: per-req GPU scalar / small tensor stacked into a
  circular buffer indexed by ``FutureIndices``. Today carries
  ``token_ids`` (non-spec) and the spec V2 draft input fields
  (``topk_p`` / ``topk_index`` / ``bonus_tokens`` / ``new_seq_lens`` /
  ``hidden_states``); future migrations route ``seq_lens`` / ``seq_lens_cpu``
  / ``orig_seq_lens`` here.
- ``gpu_tensor``: per-req GPU tensor relay for payloads too irregular
  for the stacked layout (placeholder for now; e.g. variable-length
  per-req tensors).
- ``cpu_value``: per-req Python value relay (``int`` / ``bool``).
  Holds ``kv_committed_delta`` and ``finished`` status today.
- ``cpu_action``: deferred CPU action queue for side-effect relays
  (page release / pool free) — producer enqueues, scheduler drains
  at the post-barrier point.
- ``state_obj``: cross-iter state objects (e.g. SamplingBatchInfo with
  its penalizer orchestrator).

Cross-stream sync: ``GpuScalarChannel.store`` records a CUDA event on
the producer stream after each write; ``resolve_by_*`` calls
``event.wait(consumer_stream)`` so the read is correctly ordered against
the producer's write across streams without an explicit CPU sync. This
replaces the role of legacy mechanisms like
``EagleDraftInput.verify_done.synchronize()``.

Iter-pin ring: 2-iter Python ref retention for non-channel tensor
lifetime (e.g. a verify-phase ForwardBatch whose internal tensors are
read on the forward stream after schedule_stream returns to mid-iter
work). ``Scheduler.batch_record_buf`` is a thin alias over
``Relayer._iter_pin_ring`` for back-compat with direct indexers.

This module is intentionally short and reads top-to-bottom; the
class hierarchy reflects the relay axis (GPU vs CPU, value vs action vs
state) directly without further dispatch indirection.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_is_cuda = is_cuda()
_is_hip = is_hip()


def _resolve_future_token_ids_native(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


if _is_cuda or _is_hip:
    from sglang.jit_kernel.resolve_future_token_ids import (
        resolve_future_token_ids_cuda,
    )

    _resolve_future_token_ids = resolve_future_token_ids_cuda
else:
    _resolve_future_token_ids = _resolve_future_token_ids_native


@dataclass
class FutureIndices:
    """Slot handle returned by ``Relayer.alloc_future_indices`` and by each
    sub-channel's ``alloc``.

    - ``indices``: 1-D ``int64`` tensor of slot numbers; used by the
      gpu_scalar channel for advanced indexing (``buffer[indices]``).
      Lives on the channel's device for GPU channels and on CPU for the
      cpu_value channel.
    - ``interval``: ``slice(start, end)`` view of the same slot range.
      Used by interval-based store/resolve when the consumer wants a
      stacked view rather than a gather (e.g. spec V2 draft input).

    Both views describe the *same* slot range; consumers pick whichever
    fits their kernel layout. Slot numbers are 1-based (slot 0 is the
    "no value" sentinel used by the negative-encoding ``input_ids`` trick
    for non-spec token_id relay).
    """

    indices: torch.Tensor
    interval: Optional[slice] = None


class _SlotAllocator:
    """Circular slot allocator. Each channel uses one to produce FutureIndices
    on alloc; the slot range wraps modulo ``future_limit`` while the underlying
    buffer is sized to ``future_buffer_len`` (= future_limit + headroom) so
    in-flight slots have room to coexist with newly allocated ones.
    """

    def __init__(self, future_limit: int, future_buffer_len: int, device: torch.device):
        self.future_ct = 0
        self.future_limit = future_limit
        self.future_buffer_len = future_buffer_len
        self.device = device

    def alloc(self, bs: int) -> FutureIndices:
        cur = self.future_ct
        self.future_ct = (cur + bs) % self.future_limit
        start = cur + 1
        end = cur + 1 + bs
        indices = torch.arange(start, end, dtype=torch.int64, device=self.device)
        return FutureIndices(indices=indices, interval=slice(start, end))

    def is_empty(self, intv: slice) -> bool:
        start, stop, step = intv.indices(self.future_buffer_len)
        return (start >= stop) if step > 0 else (start <= stop)


class GpuScalarChannel:
    """Per-req GPU scalar / small tensor relay.

    Holds a set of named circular GPU buffers indexed by a shared slot
    allocator. Each buffer is lazily allocated on first store from the sample
    tensor's shape/dtype, so callers do not need to declare buffer shapes up
    front. This is the channel that today's ``output_ids`` (non-spec) and
    spec-V2 draft input (topk_p / topk_index / bonus_tokens / new_seq_lens /
    hidden_states) live on.

    API: ``alloc(bs) -> FutureIndices``, ``store(fi, name, value)``,
    ``resolve_by_interval(fi, name) -> tensor``,
    ``resolve_by_indices(indices, name) -> tensor``. Future migrations route
    new relays (spec V2 accept_lens, num_correct_drafts) onto this channel by
    adding new buffer names.
    """

    def __init__(self, allocator: _SlotAllocator):
        self._allocator = allocator
        self._buffers: Dict[str, torch.Tensor] = {}
        # Cross-stream sync: producers (forward_stream) record an event after
        # each ``store``; consumers (schedule_stream) ``event.wait(stream)``
        # in ``resolve_*`` so the read is correctly ordered against the
        # producer's write across streams. We track the most recent producer
        # event globally rather than per-slot; resolves over-wait at most by
        # one extra producer write, which on the same producer stream has
        # already happened-before our event-of-interest, so the wait is still
        # correct (and the cost is one event-wait per resolve).
        self._last_producer_event: Optional[torch.cuda.Event] = None

    @property
    def device(self) -> torch.device:
        return self._allocator.device

    @property
    def future_buffer_len(self) -> int:
        return self._allocator.future_buffer_len

    def alloc(self, bs: int) -> FutureIndices:
        return self._allocator.alloc(bs)

    def ensure_buffer(
        self,
        name: str,
        per_slot_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if name not in self._buffers:
            self._buffers[name] = torch.empty(
                (self._allocator.future_buffer_len, *per_slot_shape),
                dtype=dtype,
                device=self._allocator.device,
            )
        return self._buffers[name]

    def _new_event(self) -> torch.cuda.Event:
        return torch.get_device_module(self._allocator.device).Event()

    def store(
        self,
        future_indices: FutureIndices,
        name: str,
        value: torch.Tensor,
        producer_stream: Optional[torch.cuda.Stream] = None,
    ):
        intv = future_indices.interval
        if self._allocator.is_empty(intv):
            # idle indices in dp attention do not need store info
            return
        # value is shape (bs, *per_slot_shape); per_slot_shape derived from value[0]
        if name not in self._buffers:
            self.ensure_buffer(name, value.shape[1:], value.dtype)
        self._buffers[name][intv] = value
        # Record producer-side completion event so cross-stream resolves can
        # wait without an explicit CPU sync.
        event = self._new_event()
        if producer_stream is None:
            event.record()
        else:
            event.record(producer_stream)
        self._last_producer_event = event

    def _wait_producer_on(self, consumer_stream: Optional[torch.cuda.Stream]):
        if self._last_producer_event is None:
            return
        if consumer_stream is None:
            consumer_stream = torch.get_device_module(
                self._allocator.device
            ).current_stream()
        self._last_producer_event.wait(consumer_stream)

    def resolve_by_interval(
        self,
        future_indices: FutureIndices,
        name: str,
        consumer_stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        self._wait_producer_on(consumer_stream)
        return self._buffers[name][future_indices.interval]

    def resolve_by_indices(
        self,
        indices: torch.Tensor,
        name: str,
        consumer_stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        self._wait_producer_on(consumer_stream)
        return self._buffers[name][indices]

    def buffer(self, name: str) -> torch.Tensor:
        return self._buffers[name]

    def has_buffer(self, name: str) -> bool:
        return name in self._buffers


class GpuTensorChannel:
    """Per-req GPU tensor relay for payloads too large/awkward for the
    scalar-style circular buffer (e.g. variable-length per-req tensors that
    cannot share a stacked layout).

    API mirrors GpuScalarChannel but storage uses a dict of
    ``slot_index -> tensor`` rather than a stacked buffer, so each slot owns
    its tensor's lifetime directly. No callsites yet.
    """

    def __init__(self, allocator: _SlotAllocator):
        self._allocator = allocator
        self._slots: Dict[int, Dict[str, torch.Tensor]] = {}

    def alloc(self, bs: int) -> FutureIndices:
        return self._allocator.alloc(bs)

    def store(self, future_indices: FutureIndices, name: str, value: torch.Tensor):
        intv = future_indices.interval
        if self._allocator.is_empty(intv):
            return
        start, stop, _ = intv.indices(self._allocator.future_buffer_len)
        for i, slot in enumerate(range(start, stop)):
            self._slots.setdefault(slot, {})[name] = value[i]

    def resolve(self, future_indices: FutureIndices, name: str) -> list:
        intv = future_indices.interval
        start, stop, _ = intv.indices(self._allocator.future_buffer_len)
        return [self._slots[s][name] for s in range(start, stop)]

    def free_slot(self, slot_index: int):
        self._slots.pop(slot_index, None)


class CpuValueChannel:
    """Per-req CPU value relay (Python int / bool / small object).

    Backing store: ``slot_index -> Dict[name, value]``. Producers write after
    they know the value (e.g. ``process_batch_result`` decides ``finished`` /
    ``kv_committed_len`` increment); consumers in the next iter read by slot
    index.
    """

    def __init__(self, future_buffer_len: int):
        self.future_buffer_len = future_buffer_len
        self.future_ct = 0
        self._slots: Dict[int, Dict[str, Any]] = {}

    def alloc(self, bs: int) -> FutureIndices:
        cur = self.future_ct
        self.future_ct = (cur + bs) % self.future_buffer_len
        start = cur + 1
        end = cur + 1 + bs
        # CPU channel: indices is a plain Python list-backed tensor only for
        # parity with the GPU channels; consumers use ``.interval`` for slot
        # lookup. We keep an int64 tensor on CPU so it's cheap and the
        # FutureIndices dataclass is reusable.
        indices = torch.arange(start, end, dtype=torch.int64)
        return FutureIndices(indices=indices, interval=slice(start, end))

    def store(self, future_indices: FutureIndices, name: str, values: list):
        intv = future_indices.interval
        start, stop, _ = intv.indices(self.future_buffer_len)
        for i, slot in enumerate(range(start, stop)):
            self._slots.setdefault(slot, {})[name] = values[i]

    def resolve(self, future_indices: FutureIndices, name: str) -> list:
        intv = future_indices.interval
        start, stop, _ = intv.indices(self.future_buffer_len)
        return [self._slots.get(s, {}).get(name) for s in range(start, stop)]

    def free_slot(self, slot_index: int):
        self._slots.pop(slot_index, None)


class CpuActionChannel:
    """Deferred CPU action queue.

    Producers (e.g. ``process_batch_result`` deciding ``release_pages`` for a
    finished req) enqueue callables; the scheduler drains the queue at the
    appropriate post-barrier point in the next iter, before any read of the
    freed resource. This is the "action defer" axis of relay: the value being
    relayed is a side-effect (free / release / commit) rather than a payload.
    """

    def __init__(self):
        self._queue: deque = deque()

    def enqueue(self, action: Callable[[], None]):
        self._queue.append(action)

    def drain(self):
        while self._queue:
            action = self._queue.popleft()
            action()

    def __len__(self) -> int:
        return len(self._queue)


class StateObjChannel:
    """Cross-iter state object holder for objects that carry penalizer /
    sampling state across iters (e.g. ``SamplingBatchInfo.penalizer_orchestrator``
    accumulator state) and cannot be expressed as scalar values.

    Backing store: ``name -> object``. Producers update in place during
    ``process_batch_result``; consumers read the current value at iter start.
    """

    def __init__(self):
        self._refs: Dict[str, Any] = {}

    def put(self, name: str, obj: Any):
        self._refs[name] = obj

    def get(self, name: str) -> Optional[Any]:
        return self._refs.get(name)

    def pop(self, name: str) -> Optional[Any]:
        return self._refs.pop(name, None)


class Relayer:
    """Scheduler-owned service for cross-iter relay state.

    A relay is any value where iter N+1's producer phase is bounded below by
    forward N's completion event. Examples in sglang today:

    - ``output_ids`` (non-spec) — model forward output -> next iter input_ids
    - ``new_seq_lens`` (spec V2 verify) -> next iter schedule prep
    - ``kv_committed_len`` -> next iter cache_finished_req
    - ``release_pages`` decision -> next iter alloc

    All relays funnel through a uniform ``alloc -> store -> resolve`` API,
    with the channel type chosen by data category (GPU scalar / GPU tensor /
    CPU value / CPU action / state object). The scheduler becomes the single
    explicit producer of "what next iter's forward needs", replacing the
    implicit in-place mutation of shared GPU tensors used previously.

    Today only ``gpu_scalar`` is actively populated (it absorbs the previous
    FutureMap behavior verbatim). The remaining channels are wired in so
    follow-up migrations of CPU-side relays can plug into the same API.
    """

    def __init__(
        self,
        max_running_requests: int,
        chunked_prefill_size: int,
        context_len: int,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
    ):
        # FIXME: the calculation of future_limit and future_buffer_len maybe too conservative
        # Circular buffer layout (wraps in this order):
        # Running decode batch -> Prefill chunk 1 -> ... -> Prefill chunk N
        # A running decode batch's result will be resolved after all prefill chunks are done.
        # reserve `max_num_chunks` extra future slots on top of `max_running_requests * 3`.
        max_num_chunks = (
            (context_len + chunked_prefill_size - 1) // chunked_prefill_size
            if chunked_prefill_size
            else 0
        )
        future_limit = max_running_requests * (3 + max_num_chunks)
        # Adding 2 * max_running_requests to future_limit ensures the buffer is sufficiently large.
        future_buffer_len = future_limit + 2 * max_running_requests

        self.device = device
        self.spec_algo = spec_algo

        # All GPU channels share one allocator so callsites that pass
        # FutureIndices between channels (e.g. spec V2 store_to_map writes
        # several buffers from one alloc) keep their slot ranges aligned.
        self._gpu_allocator = _SlotAllocator(future_limit, future_buffer_len, device)

        self.gpu_scalar = GpuScalarChannel(self._gpu_allocator)
        self.gpu_tensor = GpuTensorChannel(self._gpu_allocator)
        self.cpu_value = CpuValueChannel(future_buffer_len)
        self.cpu_action = CpuActionChannel()
        self.state_obj = StateObjChannel()

        if self.spec_algo.is_none():
            # Pre-init the token_ids buffer (existing behavior).
            self.gpu_scalar.ensure_buffer("token_ids", torch.Size(()), torch.int64)

    # Back-compat surface ---------------------------------------------------
    # Existing callsites in scheduler / decode_schedule_batch_mixin /
    # disaggregation see the same API; internals now route through channels.

    @property
    def future_ct(self) -> int:
        return self._gpu_allocator.future_ct

    @property
    def future_limit(self) -> int:
        return self._gpu_allocator.future_limit

    @property
    def future_buffer_len(self) -> int:
        return self._gpu_allocator.future_buffer_len

    def alloc_future_indices(self, bs: int) -> FutureIndices:
        """Reserve ``bs`` consecutive slot indices on the ``gpu_scalar``
        channel's circular buffer and return the handle.

        The handle is the same key used by ``store_to_map`` (forward stream
        write of token ids / draft input) and ``resolve_future`` (next-iter
        schedule stream read). Slot numbering wraps modulo ``future_limit``
        while the underlying buffer is ``future_limit + 2 * max_running_requests``
        long, so an in-flight slot has at least one full iter of headroom
        before its slot index is reused.
        """
        return self._gpu_allocator.alloc(bs)

    def is_empty_slice(self, s: slice) -> bool:
        """Return ``True`` when ``s`` is an empty / degenerate interval
        produced by an idle DP-attention rank's zero-sized slot range.
        Stores against such intervals are no-ops; consumers must not
        resolve them.
        """
        return self._gpu_allocator.is_empty(s)

    def resolve_future(self, batch: ScheduleBatch):
        if self.spec_algo.is_none():
            _resolve_future_token_ids(
                batch.input_ids, self.gpu_scalar.buffer("token_ids")
            )
        else:
            draft_input: EagleDraftInput = batch.spec_info
            if draft_input is None:
                # FIXME(lsyin): No future exists, only for prefill batch, not compatible with mixed mode
                return
            indices = draft_input.future_indices.indices
            # The indices tensor was allocated on the default stream but is
            # used here on the forward stream. Meanwhile, the old spec_info
            # holding this tensor will lose all Python references (replaced at
            # batch.spec_info), so the caching allocator (torch GC) could
            # reclaim the memory before the GPU finishes reading it.
            indices.record_stream(torch.get_device_module(self.device).current_stream())
            draft_input.topk_p = self.gpu_scalar.resolve_by_indices(indices, "topk_p")
            draft_input.topk_index = self.gpu_scalar.resolve_by_indices(
                indices, "topk_index"
            )
            draft_input.bonus_tokens = self.gpu_scalar.resolve_by_indices(
                indices, "bonus_tokens"
            )
            draft_input.new_seq_lens = self.gpu_scalar.resolve_by_indices(
                indices, "new_seq_lens"
            )
            if spec_need_hidden_states():
                draft_input.hidden_states = self.gpu_scalar.resolve_by_indices(
                    indices, "hidden_states"
                )

    def store_to_map(
        self, future_indices: FutureIndices, batch_result: GenerationBatchResult
    ):
        """Forward-stream write: non-spec -> ``token_ids`` buffer; spec V2 ->
        delegates to ``store_to_map_for_new_batch`` for draft input fields.
        Records a CUDA event for cross-stream resolve sync.
        """
        if self.spec_algo.is_none():
            self.gpu_scalar.store(
                future_indices, "token_ids", batch_result.next_token_ids
            )
        else:
            self.store_to_map_for_new_batch(
                future_indices, batch_result.next_draft_input
            )

    def store_to_map_for_new_batch(
        self, future_indices: FutureIndices, draft_input: EagleDraftInput
    ):
        intv = future_indices.interval
        if self._gpu_allocator.is_empty(intv):
            return
        self.gpu_scalar.store(future_indices, "topk_p", draft_input.topk_p)
        self.gpu_scalar.store(future_indices, "topk_index", draft_input.topk_index)
        self.gpu_scalar.store(future_indices, "bonus_tokens", draft_input.bonus_tokens)
        self.gpu_scalar.store(future_indices, "new_seq_lens", draft_input.new_seq_lens)
        if spec_need_hidden_states():
            self.gpu_scalar.store(
                future_indices, "hidden_states", draft_input.hidden_states
            )

    # Explicit named relays for spec V2 GPU scalars. These are thin wrappers
    # over ``gpu_scalar.store / resolve_by_indices`` to give producer / consumer
    # sites a self-documenting call (``store_new_seq_lens(fi, t)`` reads better
    # than ``gpu_scalar.store(fi, "new_seq_lens", t)``). Today's ``new_seq_lens``
    # already flows through ``store_to_map_for_new_batch`` above; these methods
    # are the public face for future migrations of ``accept_lens`` and
    # ``num_correct_drafts`` (currently still CPU-relayed via .tolist()).

    def store_new_seq_lens(self, future_indices: FutureIndices, value: torch.Tensor):
        self.gpu_scalar.store(future_indices, "new_seq_lens", value)

    def resolve_new_seq_lens(self, indices: torch.Tensor) -> torch.Tensor:
        return self.gpu_scalar.resolve_by_indices(indices, "new_seq_lens")

    def store_accept_lens(self, future_indices: FutureIndices, value: torch.Tensor):
        self.gpu_scalar.store(future_indices, "accept_lens", value)

    def resolve_accept_lens(self, indices: torch.Tensor) -> torch.Tensor:
        return self.gpu_scalar.resolve_by_indices(indices, "accept_lens")

    def store_num_correct_drafts(
        self, future_indices: FutureIndices, value: torch.Tensor
    ):
        self.gpu_scalar.store(future_indices, "num_correct_drafts", value)

    def resolve_num_correct_drafts(self, indices: torch.Tensor) -> torch.Tensor:
        return self.gpu_scalar.resolve_by_indices(indices, "num_correct_drafts")

    # seq_lens family. The "post-iter" seq_lens is derived from the previous
    # iter's seq_lens + the forward-driven finish/accept decision; storing it
    # to the gpu_scalar channel lets next iter's consumers (attention
    # backends, cuda graph runner, sampling) read a settled value with
    # cross-stream sync built in. Mirror methods exist for seq_lens_cpu and
    # orig_seq_lens which follow the same relay shape but on different
    # devices / dtypes.

    def store_seq_lens(self, future_indices: FutureIndices, value: torch.Tensor):
        self.gpu_scalar.store(future_indices, "seq_lens", value)

    def resolve_seq_lens(self, indices: torch.Tensor) -> torch.Tensor:
        return self.gpu_scalar.resolve_by_indices(indices, "seq_lens")

    def store_seq_lens_cpu(self, future_indices: FutureIndices, value: torch.Tensor):
        self.gpu_scalar.store(future_indices, "seq_lens_cpu", value)

    def resolve_seq_lens_cpu(self, indices: torch.Tensor) -> torch.Tensor:
        return self.gpu_scalar.resolve_by_indices(indices, "seq_lens_cpu")

    def store_orig_seq_lens(self, future_indices: FutureIndices, value: torch.Tensor):
        self.gpu_scalar.store(future_indices, "orig_seq_lens", value)

    def resolve_orig_seq_lens(self, indices: torch.Tensor) -> torch.Tensor:
        return self.gpu_scalar.resolve_by_indices(indices, "orig_seq_lens")

    # CPU value channel named relays. Producers (process_batch_result CPU
    # branches that decide finish / commit / etc.) call store_*; consumers
    # in the next iter call resolve_*. Each relay name is its own logical
    # slot dict on the shared cpu_value channel.

    def store_finished_status(self, future_indices: FutureIndices, values: list):
        """Per-req bool: did this req hit its stop condition after iter N?
        Consumer reads in iter N+1 ``filter_batch`` (replaces direct
        ``req.finished()`` reads where finish depends on forward output)."""
        self.cpu_value.store(future_indices, "finished", values)

    def resolve_finished_status(self, future_indices: FutureIndices) -> list:
        return self.cpu_value.resolve(future_indices, "finished")

    def store_kv_committed_delta(self, future_indices: FutureIndices, values: list):
        """Per-req int: number of KV positions to commit after iter N (for
        non-spec, 0 if finished else 1; for spec V2, accept_lens[i]-1).
        Consumer reads in iter N+1 cache management."""
        self.cpu_value.store(future_indices, "kv_committed_delta", values)

    def resolve_kv_committed_delta(self, future_indices: FutureIndices) -> list:
        return self.cpu_value.resolve(future_indices, "kv_committed_delta")

    def enqueue_release_action(self, action: Callable[[], None]):
        """Deferred CPU action (page release / pool free) that must run after
        forward N completes but before iter N+1's consumer reads the freed
        resource. Drained at the post-barrier point in event_loop_overlap."""
        self.cpu_action.enqueue(action)

    def drain_release_actions(self):
        self.cpu_action.drain()

    def stash_sampling_state(self, name: str, obj: Any):
        """Snapshot a state-object (e.g. SamplingBatchInfo penalizer state)
        for next-iter use. The state_obj channel is name-keyed; consumers
        fetch by the same name."""
        self.state_obj.put(name, obj)

    def fetch_sampling_state(self, name: str) -> Optional[Any]:
        return self.state_obj.get(name)

    def rotate_all(self):
        """Hook for per-iter slot reclamation across all channels.

        GPU channels are bounded by the circular ``_SlotAllocator`` wrap;
        CPU channels accumulate slot dicts without bound, so this method
        exists for the scheduler to call once per iter. No-op today until
        consumers settle into a steady-state subscription pattern.
        """
        # gpu_scalar / gpu_tensor: bounded by _SlotAllocator wrap; nothing to
        # do here. cpu_value: free slots older than the in-flight window.
        # cpu_action: drained at barrier in event_loop_overlap. state_obj:
        # owned references, dropped when scheduler clears them.
        pass

    # ------------------------------------------------------------------
    # Iter-pin ring: 2-iter Python ref retention for cross-stream tensor
    # lifetime that does not (yet) flow through a channel slot pool. The
    # canonical relay-driven design has every cross-stream tensor live in
    # a channel slot whose pool keeps it alive automatically; this ring
    # exists for the residual cases (e.g. a verify-phase ForwardBatch whose
    # internal tensors are read on the forward stream after the schedule
    # stream returns to mid-iter work) until those callsites are migrated.
    # ------------------------------------------------------------------

    def begin_iter_pin(self) -> int:
        """Rotate to the next slot in the 2-iter pin ring and reset it.
        Returns the slot index used by ``add_iter_pin``."""
        if not hasattr(self, "_iter_pin_ring"):
            self._iter_pin_ring = [None, None]
            self._iter_pin_ct = 0
        self._iter_pin_ct = (self._iter_pin_ct + 1) % 2
        self._iter_pin_ring[self._iter_pin_ct] = []
        return self._iter_pin_ct

    def add_iter_pin(self, *refs):
        """Pin ``refs`` for two iters (this slot survives until the second
        ``begin_iter_pin`` rotation overwrites it). Used by workers to keep
        Python refs to forward-stream-bound GPU tensors alive across the
        schedule-side return."""
        slot = getattr(self, "_iter_pin_ring", None)
        if slot is None:
            self.begin_iter_pin()
            slot = self._iter_pin_ring
        for ref in refs:
            slot[self._iter_pin_ct].append(ref)
