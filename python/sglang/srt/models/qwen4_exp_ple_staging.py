"""Bounded pinned staging for PLE rows read by a CPU worker."""

import logging
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.models.qwen4_exp_ple_rows import PleRowSource

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)


class EagerHashKey(NamedTuple):
    mode: "ForwardMode"
    fast: bool
    fused: bool


class PleHostStaging:
    """Stage one batch at a time through two pinned host buffers.

    Graph buffers are allocated during warmup and retained for replay.
    Device copies, consumption, and chunk reuse must be ordered on the caller's
    CUDA stream. Callers switching streams must establish the dependency.
    """

    def __init__(
        self,
        source: PleRowSource,
        vocab_start: int,
        vocab_end: int,
        device_module=None,
        *,
        chunk_rows: int = 8192,
        pin_memory: bool = True,
    ):
        self.device_module = device_module or torch.get_device_module("cuda")
        self.source = source
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-rows")
        self._slots = []
        self._finalizer = weakref.finalize(
            self, self._close, self._pool, self._slots, source
        )
        self._graphs = {}
        self._eager = None
        self._pending = None
        self.chunk_rows = chunk_rows
        self._pin_memory = pin_memory
        self.verified_modes: set[EagerHashKey] = set()
        self.verified_replays: set[int] = set()
        self._contexts = {}
        self._replay_check = None
        self._scratch = None

    def buffer(self, rows: int, device: torch.device, *, graph: bool) -> torch.Tensor:
        if graph:
            if rows not in self._graphs:
                if not get_is_capture_mode():
                    raise RuntimeError(
                        f"PLE host staging has no captured buffer for {rows} rows"
                    )
                if self.device_module.is_current_stream_capturing():
                    raise RuntimeError("PLE graph buffer allocation during capture")
                self._graphs[rows] = torch.zeros(
                    (rows, self.source.row_bytes), dtype=torch.uint8, device=device
                )
            return self._graphs[rows]
        # Eager execution reuses one device chunk; graph buffers retain all rows.
        rows = min(rows, self.chunk_rows)
        if self._eager is None or len(self._eager) < rows:
            self._eager = torch.empty(
                (rows, self.source.row_bytes), dtype=torch.uint8, device=device
            )
        return self._eager[:rows]

    def _fetch(self, slot, ids):
        next_ids = ids[self.chunk_rows : 2 * self.chunk_rows]
        ids = ids[: self.chunk_rows]
        valid_next = next_ids[
            (next_ids >= self.vocab_start) & (next_ids < self.vocab_end)
        ]
        self.source.prefetch_rows(valid_next - self.vocab_start)
        host, event = self._slots[slot]
        # Do not overwrite this slot until its previous H2D copy finishes.
        event.synchronize()
        out = host.numpy()[: len(ids)]
        valid = (ids >= self.vocab_start) & (ids < self.vocab_end)
        if valid.all():
            self.source.fetch_rows(ids - self.vocab_start, out)
            return host[: len(ids)]
        out[~valid] = 0
        local = ids[valid] - self.vocab_start
        if self._scratch is None:
            self._scratch = np.empty(
                (self.chunk_rows, self.source.row_bytes), dtype=np.uint8
            )
        selected = self._scratch[: len(local)]
        self.source.fetch_rows(local, selected)
        out[valid] = selected
        return host[: len(ids)]

    @property
    def pending_rows(self) -> int:
        return 0 if self._pending is None else len(self._pending[0])

    def discard(self) -> None:
        self._replay_check = None
        pending, self._pending = self._pending, None
        if pending is not None:
            try:
                pending[2].result()
            except Exception:
                logger.exception("PLE host staging: discarded batch failed to fetch")

    def begin(self, ids, device, *, graph=False, prepare_ids=None):
        if self._pending is not None:
            raise RuntimeError("PLE staging batch has not been consumed")
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        output = self.buffer(len(ids), device, graph=graph)
        if not self._slots:
            for _ in range(2):
                host = torch.empty(
                    (self.chunk_rows, self.source.row_bytes),
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=self._pin_memory,
                )
                self._slots.append((host, self.device_module.Event(blocking=True)))

        def prepare_and_fetch():
            if prepare_ids is not None:
                ids[:] = prepare_ids().reshape(-1)
            first = ids[: self.chunk_rows]
            valid = first[(first >= self.vocab_start) & (first < self.vocab_end)]
            self.source.prefetch_rows(valid - self.vocab_start)
            return self._fetch(0, ids)

        # The worker fills ids; readers must wait for the fetch future.
        self._pending = ids, output, self._pool.submit(prepare_and_fetch), graph

    @property
    def pending_ids(self):
        if self._pending is None:
            return None
        self._pending[2].result()
        return self._pending[0]

    def capture_contexts(self, contexts):
        rows = len(contexts)
        if rows not in self._contexts:
            if self.device_module.is_current_stream_capturing():
                raise RuntimeError("PLE context buffer allocation during capture")
            self._contexts[rows] = torch.empty_like(contexts)
        # Capture a context copy for the first replay check at this token count.
        self._contexts[rows].copy_(contexts)

    def expect_replay(self, contexts):
        rows = len(contexts)
        if rows not in self.verified_replays:
            self._replay_check = rows, contexts.copy()

    def check_replay(self):
        if self._replay_check is None:
            return
        rows, expected = self._replay_check
        # Called on the model stream at the next prepare, after graph replay.
        # The blocking D2H copy therefore reads that replay's context write.
        actual = self._contexts[rows].cpu().numpy()
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"PLE replay contexts differ for {rows} tokens")
        self.verified_replays.add(rows)
        self._replay_check = None

    def verify(self, device_ids, key: EagerHashKey):
        host = self.pending_ids
        device = device_ids.reshape(-1).cpu().numpy()
        common = min(len(host), len(device))
        differences = np.flatnonzero(host[:common] != device[:common])
        if len(host) != len(device) or differences.size:
            first = int(differences[0]) if differences.size else common
            raise RuntimeError(
                f"PLE staged ids differ: {len(host)} host, {len(device)} device, "
                f"first differing index {first}"
            )
        self.verified_modes.add(key)

    def finish(self, consume=None):
        if self._pending is None:
            raise RuntimeError("PLE rows were not staged before forward")
        ids, staged, pending_fetch, graph = self._pending
        try:
            for chunk, start in enumerate(range(0, len(ids), self.chunk_rows)):
                future, pending_fetch = pending_fetch, None
                host = future.result()
                end = min(start + self.chunk_rows, len(ids))
                slot = chunk % 2
                if end < len(ids):
                    pending_fetch = self._pool.submit(self._fetch, 1 - slot, ids[end:])
                raw = staged[start:end] if graph else staged[: end - start]
                raw.copy_(host, non_blocking=True)
                # The fetch has returned, so this event can now track the new H2D copy.
                self._slots[slot][1].record()
                if consume is not None:
                    consume(raw, start, end)
        finally:
            self._pending = None
            # Join the fetch even if a consumer raised or the batch was empty.
            if pending_fetch is not None:
                try:
                    pending_fetch.result()
                except Exception:
                    logger.exception("PLE host staging: trailing fetch failed")
        return staged

    def close(self) -> None:
        self.discard()
        self._finalizer()

    @staticmethod
    def _close(pool, slots, source):
        pool.shutdown(wait=False)
        try:
            for _, event in slots:
                event.synchronize()
        except Exception:
            pass  # The device context may already be gone during finalization.
        source.close()
