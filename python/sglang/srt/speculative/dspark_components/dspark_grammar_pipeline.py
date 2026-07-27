"""Fail-closed, overlapped XGrammar mask construction for DSpark.

The pipeline deliberately has no scalar or Python traversal fallback:
XGrammar must provide the batched draft-tree traversal API or startup fails.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from xgrammar import BatchGrammarMatcher, GrammarMatcher

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar

logger = logging.getLogger(__name__)


class _RequestWithGrammar(Protocol):
    grammar: BaseGrammarObject | None


class _DraftInputWithGrammar(Protocol):
    grammar: BaseGrammarObject | None


class _GrammarMetricsCollector(Protocol):
    def observe_dspark_grammar_step(
        self,
        *,
        batch_size: int,
        active_matchers: int,
        outcome_counts: dict[str, int],
        phase_seconds: dict[str, float],
    ) -> None: ...

    def observe_dspark_grammar_gpu_gap(self, *, seconds: float) -> None: ...


@dataclass(slots=True)
class _GrammarBufferSlot:
    draft_tokens_cpu: torch.Tensor
    vocab_mask_cpu: torch.Tensor
    vocab_mask_device: torch.Tensor
    d2h_done: torch.cuda.Event
    mask_ready: torch.cuda.Event
    mask_consumed: torch.cuda.Event
    has_mask_consumer: bool = False


@dataclass(slots=True)
class _GrammarGpuTimingSlot:
    target_done: torch.cuda.Event
    mask_dependency_done: torch.cuda.Event
    reserved: bool = False
    target_recorded: bool = False
    dependency_recorded: bool = False
    emit: bool = False


@dataclass(slots=True)
class GrammarMaskStep:
    slot: _GrammarBufferSlot
    batch_size: int
    row_count: int
    requests: Sequence[_RequestWithGrammar]
    draft_input: _DraftInputWithGrammar
    buffer_wait_seconds: float
    gpu_timing: _GrammarGpuTimingSlot | None


@dataclass(slots=True)
class GrammarMaskResult:
    step: GrammarMaskStep
    grammar: XGrammarGrammar | ReasonerGrammarObject
    vocab_mask: torch.Tensor
    active_matchers: int
    traversal_seconds: float
    d2h_wait_seconds: float


class DSparkGrammarPipeline:
    """Double-buffered CPU traversal and async mask transfer for one DP rank."""

    def __init__(
        self,
        *,
        device: str,
        max_batch_size: int,
        chain_length: int,
        vocab_size: int,
        traversal_threads: int,
        metrics_collector: _GrammarMetricsCollector | None = None,
    ) -> None:
        if not hasattr(BatchGrammarMatcher, "batch_traverse_draft_tree"):
            raise RuntimeError(
                "DSpark grammar batching requires XGrammar BatchTraverseDraftTree"
            )
        if not str(device).startswith("cuda"):
            raise RuntimeError("DSpark grammar pipeline requires CUDA")
        if max_batch_size <= 0 or chain_length <= 0 or vocab_size <= 0:
            raise ValueError("invalid DSpark grammar buffer geometry")
        if traversal_threads <= 0:
            raise ValueError("grammar traversal thread count must be positive")

        self.device = device
        self.max_batch_size = max_batch_size
        self.chain_length = chain_length
        self.vocab_size = vocab_size
        self.mask_words = (vocab_size + 31) // 32
        self._device_module = torch.get_device_module(device)
        self._copy_stream = self._device_module.Stream()
        self._batch_matcher = BatchGrammarMatcher(max_threads=traversal_threads)
        self._metrics_collector = metrics_collector
        self._metrics_interval = int(
            os.environ.get("SGLANG_DSPARK_GRAMMAR_METRICS_INTERVAL", "16")
        )
        if self._metrics_interval <= 0:
            raise ValueError("grammar metrics interval must be positive")
        self._pending_outcomes = {
            "masked": 0,
            "thinking_only": 0,
        }
        self._next_token = torch.arange(1, chain_length + 1, dtype=torch.int64)
        self._next_token[-1] = -1
        self._next_sibling = torch.full(
            (chain_length,),
            -1,
            dtype=torch.int64,
        )
        self._slot_index = 0
        self._steps = 0
        self._traversal_seconds = 0.0
        self._traversal_max_seconds = 0.0
        self._d2h_wait_seconds = 0.0
        self._buffer_wait_seconds = 0.0
        self._active_matchers = 0
        self._gpu_timing_step = 0
        self._gpu_timing_cursor = 0

        mask_shape = (max_batch_size * chain_length, self.mask_words)
        self._slots = [
            _GrammarBufferSlot(
                draft_tokens_cpu=torch.empty(
                    (max_batch_size, chain_length),
                    dtype=torch.int64,
                    pin_memory=True,
                ),
                vocab_mask_cpu=torch.empty(
                    mask_shape,
                    dtype=torch.int32,
                    pin_memory=True,
                ),
                vocab_mask_device=torch.empty(
                    mask_shape,
                    dtype=torch.int32,
                    device=device,
                ),
                d2h_done=self._device_module.Event(),
                mask_ready=self._device_module.Event(),
                mask_consumed=self._device_module.Event(),
            )
            for _ in range(2)
        ]
        self._gpu_timing_slots = [
            _GrammarGpuTimingSlot(
                target_done=self._device_module.Event(enable_timing=True),
                mask_dependency_done=self._device_module.Event(enable_timing=True),
            )
            for _ in range(8)
        ]
        logger.info(
            "DSpark grammar pipeline enabled: max_bs=%d chain=%d "
            "vocab=%d threads=%d metrics_interval=%d double_buffer_bytes=%d",
            max_batch_size,
            chain_length,
            vocab_size,
            traversal_threads,
            self._metrics_interval,
            2
            * (
                max_batch_size * chain_length * torch.int64.itemsize
                + max_batch_size
                * chain_length
                * self.mask_words
                * (torch.int32.itemsize * 2)
            ),
        )

    def poll_gpu_timing_metrics(self) -> None:
        for slot in self._gpu_timing_slots:
            if not slot.dependency_recorded or not slot.mask_dependency_done.query():
                continue
            if slot.emit:
                elapsed_seconds = (
                    slot.target_done.elapsed_time(slot.mask_dependency_done) / 1000
                )
                if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
                    raise RuntimeError(
                        "DSpark grammar CUDA timing returned an invalid "
                        f"target-to-mask gap: {elapsed_seconds!r}"
                    )
                if self._metrics_collector is None:
                    raise RuntimeError(
                        "DSpark grammar CUDA timing has no metrics collector"
                    )
                self._metrics_collector.observe_dspark_grammar_gpu_gap(
                    seconds=elapsed_seconds
                )
            slot.reserved = False
            slot.target_recorded = False
            slot.dependency_recorded = False
            slot.emit = False

    def _reserve_gpu_timing(self) -> _GrammarGpuTimingSlot | None:
        self._gpu_timing_step += 1
        if (
            self._metrics_collector is None
            or self._gpu_timing_step % self._metrics_interval != 0
        ):
            return None
        self.poll_gpu_timing_metrics()
        for offset in range(len(self._gpu_timing_slots)):
            index = (self._gpu_timing_cursor + offset) % len(self._gpu_timing_slots)
            slot = self._gpu_timing_slots[index]
            if slot.reserved:
                continue
            slot.reserved = True
            self._gpu_timing_cursor = (index + 1) % len(self._gpu_timing_slots)
            return slot
        raise RuntimeError(
            "DSpark grammar CUDA timing ring exhausted before the GPU "
            "completed prior samples"
        )

    def begin(
        self,
        *,
        verify_ids_2d: torch.Tensor,
        requests: Sequence[_RequestWithGrammar],
        draft_input: _DraftInputWithGrammar,
    ) -> GrammarMaskStep:
        if verify_ids_2d.device.type != "cuda":
            raise RuntimeError("DSpark verify ids must be CUDA-resident")
        if verify_ids_2d.dtype != torch.int64 or not verify_ids_2d.is_contiguous():
            raise RuntimeError(
                "DSpark grammar batching requires contiguous int64 verify ids"
            )
        batch_size, chain_length = verify_ids_2d.shape
        if chain_length != self.chain_length:
            raise RuntimeError(
                f"DSpark grammar chain changed: expected {self.chain_length}, "
                f"got {chain_length}"
            )
        if batch_size > self.max_batch_size or len(requests) != batch_size:
            raise RuntimeError(
                f"DSpark grammar batch geometry is invalid: bs={batch_size}, "
                f"requests={len(requests)}, max={self.max_batch_size}"
            )

        slot = self._slots[self._slot_index]
        self._slot_index = (self._slot_index + 1) % len(self._slots)
        wait_started = time.perf_counter()

        current_stream = self._device_module.current_stream()
        with self._device_module.stream(self._copy_stream):
            # The copy stream is persistent, so this D2H is naturally ordered
            # after the slot's preceding mask H2D. Waiting on the host here
            # would serialize before target verify; the d2h_done event below
            # instead lets any residual slot reuse overlap target compute.
            self._copy_stream.wait_stream(current_stream)
            slot.draft_tokens_cpu[:batch_size].copy_(
                verify_ids_2d,
                non_blocking=True,
            )
            slot.d2h_done.record(self._copy_stream)
        buffer_wait_seconds = time.perf_counter() - wait_started

        return GrammarMaskStep(
            slot=slot,
            batch_size=batch_size,
            row_count=batch_size * chain_length,
            requests=requests,
            draft_input=draft_input,
            buffer_wait_seconds=buffer_wait_seconds,
            gpu_timing=self._reserve_gpu_timing(),
        )

    def mark_target_verify_enqueued(self, step: GrammarMaskStep) -> None:
        slot = step.gpu_timing
        if slot is None:
            return
        if not slot.reserved or slot.target_recorded or slot.dependency_recorded:
            raise RuntimeError("DSpark grammar CUDA target timing state is invalid")
        slot.target_done.record(self._device_module.current_stream())
        slot.target_recorded = True

    def _mark_mask_dependency(
        self,
        step: GrammarMaskStep,
        *,
        emit: bool,
    ) -> None:
        slot = step.gpu_timing
        if slot is None:
            return
        if not slot.reserved or not slot.target_recorded or slot.dependency_recorded:
            raise RuntimeError("DSpark grammar CUDA mask timing state is invalid")
        slot.mask_dependency_done.record(self._device_module.current_stream())
        slot.dependency_recorded = True
        slot.emit = emit

    @staticmethod
    def _resolve_native_matcher(
        grammar: BaseGrammarObject,
        draft_tokens: Sequence[int],
    ) -> tuple[GrammarMatcher, int] | None:
        if isinstance(grammar, XGrammarGrammar):
            return grammar.matcher, 0
        if not isinstance(grammar, ReasonerGrammarObject):
            raise RuntimeError("DSpark grammar batching requires the XGrammar backend")
        if grammar.enable_token_filter:
            raise RuntimeError(
                "DSpark grammar batching does not admit strict-thinking token filters"
            )
        inner = grammar.grammar
        if not isinstance(inner, XGrammarGrammar):
            raise RuntimeError("DSpark reasoning grammar has no XGrammar matcher")
        if grammar._is_generation():
            return inner.matcher, 0
        if not grammar._is_thinking():
            raise RuntimeError("reasoning grammar is in an invalid state")

        think_end_id = grammar.think_end_id
        for position in range(1, len(draft_tokens)):
            if int(draft_tokens[position]) == think_end_id:
                return inner.matcher, position
        return None

    def finish(
        self,
        step: GrammarMaskStep,
        *,
        grammar_barrier: Callable[[], dict[str, float]] | None,
    ) -> GrammarMaskResult | None:
        if grammar_barrier is None:
            raise RuntimeError("DSpark grammar overlap requires the scheduler barrier")
        if step.gpu_timing is not None and not step.gpu_timing.target_recorded:
            raise RuntimeError("DSpark grammar target completion was not recorded")

        finish_started = time.perf_counter()
        barrier_started = time.perf_counter()
        barrier_phase_seconds = grammar_barrier()
        barrier_seconds = time.perf_counter() - barrier_started
        if not isinstance(barrier_phase_seconds, dict):
            raise RuntimeError("DSpark grammar barrier must return phase timings")
        for phase, seconds in barrier_phase_seconds.items():
            if (
                not isinstance(phase, str)
                or not phase.startswith("barrier_")
                or isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds)
                or seconds < 0
            ):
                raise RuntimeError(
                    "DSpark grammar barrier returned invalid timing: "
                    f"{phase!r}={seconds!r}"
                )
        d2h_wait_started = time.perf_counter()
        step.slot.d2h_done.synchronize()
        d2h_wait_seconds = time.perf_counter() - d2h_wait_started

        cpu_mask = step.slot.vocab_mask_cpu[: step.row_count]
        matcher_resolution_started = time.perf_counter()
        matchers: list[GrammarMatcher] = []
        indices: list[int] = []
        root_positions: list[int] = []
        apply_grammar: XGrammarGrammar | ReasonerGrammarObject | None = None
        draft_tokens = step.slot.draft_tokens_cpu[: step.batch_size]
        # NumPy exposes the synchronized pinned tensor as a zero-copy CPU view.
        # Its scalar indexing avoids one PyTorch dispatcher trip per request
        # and draft position during reasoner transition detection.
        draft_token_rows = draft_tokens.numpy()

        for index, request in enumerate(step.requests):
            grammar = request.grammar
            if grammar is None:
                continue
            if apply_grammar is None:
                apply_grammar = grammar
            resolved = self._resolve_native_matcher(
                grammar,
                draft_token_rows[index],
            )
            if resolved is None:
                continue
            matcher, root_position = resolved
            matchers.append(matcher)
            indices.append(index)
            root_positions.append(root_position)
        matcher_resolution_seconds = time.perf_counter() - matcher_resolution_started

        if apply_grammar is None:
            raise RuntimeError(
                "batch.has_grammar was true but no request carried a grammar"
            )
        if not matchers:
            self._mark_mask_dependency(step, emit=False)
            self._record(
                batch_size=step.batch_size,
                active_matchers=0,
                outcome="thinking_only",
                traversal_seconds=0.0,
                d2h_wait_seconds=d2h_wait_seconds,
                buffer_wait_seconds=step.buffer_wait_seconds,
                phase_seconds={
                    "buffer_wait": step.buffer_wait_seconds,
                    "grammar_barrier": barrier_seconds,
                    **barrier_phase_seconds,
                    "d2h_wait": d2h_wait_seconds,
                    "matcher_resolution": matcher_resolution_seconds,
                    "native_traversal": 0.0,
                    "h2d_enqueue": 0.0,
                    "finish_total": time.perf_counter() - finish_started,
                },
            )
            return None

        traversal_started = time.perf_counter()
        completed = self._batch_matcher.batch_traverse_draft_tree(
            matchers,
            self._next_token,
            self._next_sibling,
            draft_tokens,
            cpu_mask,
            indices=indices,
            root_positions=root_positions,
        )
        traversal_seconds = time.perf_counter() - traversal_started
        if not all(completed):
            raise RuntimeError("native DSpark grammar traversal timed out")

        h2d_enqueue_started = time.perf_counter()
        current_stream = self._device_module.current_stream()
        with self._device_module.stream(self._copy_stream):
            if step.slot.has_mask_consumer:
                self._copy_stream.wait_event(step.slot.mask_consumed)
            step.slot.vocab_mask_device[: step.row_count].copy_(
                cpu_mask,
                non_blocking=True,
            )
            step.slot.mask_ready.record(self._copy_stream)
        current_stream.wait_event(step.slot.mask_ready)
        self._mark_mask_dependency(step, emit=True)
        h2d_enqueue_seconds = time.perf_counter() - h2d_enqueue_started

        step.draft_input.grammar = apply_grammar
        self._record(
            batch_size=step.batch_size,
            active_matchers=len(matchers),
            outcome="masked",
            traversal_seconds=traversal_seconds,
            d2h_wait_seconds=d2h_wait_seconds,
            buffer_wait_seconds=step.buffer_wait_seconds,
            phase_seconds={
                "buffer_wait": step.buffer_wait_seconds,
                "grammar_barrier": barrier_seconds,
                **barrier_phase_seconds,
                "d2h_wait": d2h_wait_seconds,
                "matcher_resolution": matcher_resolution_seconds,
                "native_traversal": traversal_seconds,
                "h2d_enqueue": h2d_enqueue_seconds,
                "finish_total": time.perf_counter() - finish_started,
            },
        )
        return GrammarMaskResult(
            step=step,
            grammar=apply_grammar,
            vocab_mask=step.slot.vocab_mask_device[: step.row_count],
            active_matchers=len(matchers),
            traversal_seconds=traversal_seconds,
            d2h_wait_seconds=d2h_wait_seconds,
        )

    def mark_consumed(self, result: GrammarMaskResult) -> None:
        result.step.slot.mask_consumed.record(self._device_module.current_stream())
        result.step.slot.has_mask_consumer = True

    def _record(
        self,
        *,
        batch_size: int,
        active_matchers: int,
        outcome: str,
        traversal_seconds: float,
        d2h_wait_seconds: float,
        buffer_wait_seconds: float,
        phase_seconds: dict[str, float],
    ) -> None:
        self._steps += 1
        self._active_matchers += active_matchers
        self._traversal_seconds += traversal_seconds
        self._traversal_max_seconds = max(
            self._traversal_max_seconds,
            traversal_seconds,
        )
        self._d2h_wait_seconds += d2h_wait_seconds
        self._buffer_wait_seconds += buffer_wait_seconds
        self._pending_outcomes[outcome] += 1
        if (
            self._metrics_collector is not None
            and self._steps % self._metrics_interval == 0
        ):
            self._metrics_collector.observe_dspark_grammar_step(
                batch_size=batch_size,
                active_matchers=active_matchers,
                outcome_counts=self._pending_outcomes.copy(),
                phase_seconds=phase_seconds,
            )
            self._pending_outcomes = {
                "masked": 0,
                "thinking_only": 0,
            }
        if self._steps % 1000 == 0:
            logger.info(
                "DSpark grammar stats: steps=%d active_per_step=%.2f "
                "traversal_avg_ms=%.3f traversal_max_ms=%.3f "
                "d2h_wait_avg_ms=%.3f buffer_wait_avg_ms=%.3f",
                self._steps,
                self._active_matchers / self._steps,
                self._traversal_seconds / self._steps * 1000,
                self._traversal_max_seconds * 1000,
                self._d2h_wait_seconds / self._steps * 1000,
                self._buffer_wait_seconds / self._steps * 1000,
            )
