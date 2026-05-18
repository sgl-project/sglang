"""Relayer: scheduler-owned cross-iter relay services.

A relay is a value whose producer is iter N (forward kernel output, or
CPU branch in process_batch_result reading forward output) and whose
consumer is iter N+1. Three channels:

- gpu_scalar: per-req GPU scalar stacked into a circular buffer indexed
  by FutureIndices. Cross-stream sync via cuda event recorded on store
  and waited on resolve.
- cpu_value: per-req Python value (int/bool) indexed by slot.
- state_obj: cross-iter state objects (e.g. SamplingBatchInfo).

Plus an iter-pin ring for 2-iter Python ref retention of non-channel
cross-stream tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

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
    """Slot handle. ``indices`` is the 1-D int64 advanced-index tensor;
    ``interval`` is the equivalent slice. Slot 0 is the sentinel for the
    negative-encoding input_ids relay.
    """

    indices: torch.Tensor
    interval: Optional[slice] = None


class _SlotAllocator:
    """Circular slot allocator. Slots wrap modulo future_limit; the
    underlying buffer is sized to future_buffer_len = future_limit +
    headroom so in-flight slots have room.
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
    """Per-req GPU scalar relay.

    Named circular GPU buffers indexed by a shared slot allocator. Each
    buffer is lazily allocated on first store. Cross-stream sync: each
    store records a cuda event under the buffer name; resolve_* waits on
    that buffer's event only.
    """

    def __init__(self, allocator: _SlotAllocator):
        self._allocator = allocator
        self._buffers: Dict[str, torch.Tensor] = {}
        self._producer_events: Dict[str, torch.cuda.Event] = {}

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
            return
        if name not in self._buffers:
            self.ensure_buffer(name, value.shape[1:], value.dtype)
        self._buffers[name][intv] = value
        event = self._new_event()
        if producer_stream is None:
            event.record()
        else:
            event.record(producer_stream)
        self._producer_events[name] = event

    def _wait_producer_on(
        self, name: str, consumer_stream: Optional[torch.cuda.Stream]
    ):
        event = self._producer_events.get(name)
        if event is None:
            return
        if consumer_stream is None:
            consumer_stream = torch.get_device_module(
                self._allocator.device
            ).current_stream()
        event.wait(consumer_stream)

    def resolve_by_interval(
        self,
        future_indices: FutureIndices,
        name: str,
        consumer_stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        self._wait_producer_on(name, consumer_stream)
        return self._buffers[name][future_indices.interval]

    def resolve_by_indices(
        self,
        indices: torch.Tensor,
        name: str,
        consumer_stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        self._wait_producer_on(name, consumer_stream)
        return self._buffers[name][indices]

    def buffer(self, name: str) -> torch.Tensor:
        return self._buffers[name]

    def has_buffer(self, name: str) -> bool:
        return name in self._buffers


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


class StateObjChannel:
    """Name-keyed holder for cross-iter state objects (e.g. SamplingBatchInfo)."""

    def __init__(self):
        self._refs: Dict[str, Any] = {}

    def put(self, name: str, obj: Any):
        self._refs[name] = obj

    def get(self, name: str) -> Optional[Any]:
        return self._refs.get(name)


class Relayer:
    """Scheduler-owned cross-iter relay state.

    Channels: gpu_scalar (per-req GPU scalar), cpu_value (per-req Python
    value), state_obj (named cross-iter state object).
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

        self._gpu_allocator = _SlotAllocator(future_limit, future_buffer_len, device)

        self.gpu_scalar = GpuScalarChannel(self._gpu_allocator)
        self.cpu_value = CpuValueChannel(future_buffer_len)
        self.state_obj = StateObjChannel()

        if self.spec_algo.is_none():
            self.gpu_scalar.ensure_buffer("token_ids", torch.Size(()), torch.int64)

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
        """Reserve ``bs`` consecutive gpu_scalar slot indices."""
        return self._gpu_allocator.alloc(bs)

    def is_empty_slice(self, s: slice) -> bool:
        """``True`` for an empty interval (idle DP-attention zero-sized slot)."""
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
        """Forward-stream write: token_ids (non-spec) or spec V2 draft input."""
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

    def store_new_seq_lens(self, future_indices: FutureIndices, value: torch.Tensor):
        self.gpu_scalar.store(future_indices, "new_seq_lens", value)

    def resolve_new_seq_lens(self, indices: torch.Tensor) -> torch.Tensor:
        return self.gpu_scalar.resolve_by_indices(indices, "new_seq_lens")

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

    def store_finished_status(self, future_indices: FutureIndices, values: list):
        self.cpu_value.store(future_indices, "finished", values)

    def resolve_finished_status(self, future_indices: FutureIndices) -> list:
        return self.cpu_value.resolve(future_indices, "finished")

    def store_kv_committed_delta(self, future_indices: FutureIndices, values: list):
        self.cpu_value.store(future_indices, "kv_committed_delta", values)

    def resolve_kv_committed_delta(self, future_indices: FutureIndices) -> list:
        return self.cpu_value.resolve(future_indices, "kv_committed_delta")

    def stash_sampling_state(self, name: str, obj: Any):
        self.state_obj.put(name, obj)

    # ------------------------------------------------------------------
    # Iter-pin ring: 2-iter Python ref retention for cross-stream tensors
    # not yet routed through channel slots.
    # ------------------------------------------------------------------

    def begin_iter_pin(self) -> int:
        if not hasattr(self, "_iter_pin_ring"):
            self._iter_pin_ring = [None, None]
            self._iter_pin_ct = 0
        self._iter_pin_ct = (self._iter_pin_ct + 1) % 2
        self._iter_pin_ring[self._iter_pin_ct] = []
        return self._iter_pin_ct

    def add_iter_pin(self, *refs):
        """Pin ``refs`` for two iters (slot survives one rotation)."""
        slot = getattr(self, "_iter_pin_ring", None)
        if slot is None:
            self.begin_iter_pin()
            slot = self._iter_pin_ring
        for ref in refs:
            slot[self._iter_pin_ct].append(ref)
