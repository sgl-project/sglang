"""MLX overlap scheduling mixin for the SGLang scheduler.

Provides ``event_loop_overlap_mlx``, which pipelines MLX forward
passes by keeping two in-flight lazy graphs queued on the GPU while
the scheduler runs its CPU-side bookkeeping on the tokens of the
older one.  The lazy-graph primitives live in
``hardware_backend/mlx/tp_worker.py`` and ``model_runner.py``.

Each request's attention KV lives in per-request, per-layer
``ContiguousAttentionKVCache`` objects that ``MLXAttentionWrapper`` mutates
in place during the forward pass. Chained decodes reuse the same cache objects:
step N+1's graph reads step N's lazy writes via MLX's dependency tracking, so
the GPU runs both steps back-to-back with no idle gap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import mlx.core as mx

from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import resolve_forward_inputs
from sglang.srt.runtime_context import get_device
from sglang.srt.utils import DynamicGradMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.model_runner import (
        MlxPendingDecode,
        MlxPendingExtend,
        MlxPendingPrefill,
    )
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


@dataclass
class MlxPendingJob:
    """Unfinished MLX work and graphs queued on the GPU.

    Attributes:
        lazy_tokens: Lazily evaluated token IDs produced by the forward
            pass.  Unevaluated; calling ``.tolist()`` / ``.item()`` /
            ``mx.eval`` on it will block until the Metal kernel finishes.
            ``None`` for idle batches.
        prefills: MLX prefill state returned by the model worker — one
            entry per new request in an extend batch.  Used by
            ``finalize_mlx_result`` to commit per-request caches.  Empty
            list for pure-decode steps.
        extends: Chunked-prefill-continuation state, one entry per
            already-active request whose extend seq_len > 1.  Also empty
            for pure-decode steps.
        decode: Decode state covering full-decode mode AND mixed
            single-token decodes inside an extend batch.  Used as the
            chaining root by :meth:`async_chained_decode_mlx`.
        mode: One of ``"decode"``, ``"extend"``, ``"idle"`` describing
            which forward pass produced this job.  Drives finalise
            dispatch and whether chaining is safe.
        batch_copy: Snapshot of the :class:`ScheduleBatch` at launch
            time.  Decoupled from the live batch so
            ``process_batch_result`` can update request state without
            racing against the next scheduling decision.
        schedule_batch: The full scheduler batch.  Unlike ``batch_copy``,
            this keeps allocator/cache fields needed when a prefill batch
            becomes the next running decode batch.
        reqs: Snapshot of ``batch.reqs`` at launch time.  The overlap
            loop uses this to check ``req.finished()`` on the previous
            step's request list without holding a reference to the
            mutable batch object.
    """

    lazy_tokens: Optional[mx.array]
    prefills: list[MlxPendingPrefill]
    extends: list[MlxPendingExtend]
    decode: Optional[MlxPendingDecode]
    mode: str
    batch_copy: ScheduleBatch
    schedule_batch: ScheduleBatch
    reqs: List[Req]
    # False when the batch needs per-step CPU logit state (grammar vocab
    # masks / custom logit processors under --mlx-enable-sampling): step
    # N+1's mask needs token N materialized, so such batches must launch
    # fresh every step instead of chaining.
    chain_safe: bool = True
    # Captured at launch when batch.return_logprob, exactly like
    # Scheduler.run_batch does for the CUDA paths (the live values mutate
    # before output processing).
    extend_input_len_per_req: Optional[List[int]] = None
    extend_logprob_start_len_per_req: Optional[List[int]] = None


class SchedulerMlxOverlapMixin:
    """Mixin that adds MLX overlap scheduling to :class:`Scheduler`."""

    def _mlx_batch_chain_safe(self: Scheduler, batch: ScheduleBatch) -> bool:
        """False when per-step CPU logit state forbids chained decode.

        Grammar vocab masks and custom logit processors depend on the
        previous token being materialized; a chained step is built before
        that, so those batches launch fresh every step.
        """
        if not get_device().mlx_enable_sampling:
            return True
        sampling_info = batch.sampling_info
        if sampling_info is None:
            return True
        # batch.has_grammar, not sampling_info.grammars: the latter is only
        # populated at forward launch (see _build_logit_edit_rows).
        return not (batch.has_grammar or sampling_info.has_custom_logit_processor)

    def _finalize_mlx_pending_job(self: Scheduler, pending: MlxPendingJob):
        # Account for this completed forward step. The standard scheduler does
        # this inside run_batch(), but the MLX overlap loop bypasses run_batch,
        # so without this forward_ct never advances on MLX. That stalls the
        # watchdog liveness counter and, more importantly, breaks step-bounded
        # profiling: _profile_batch_predicate auto-starts/stops based on
        # forward_ct, so `--profile-steps` (and the server /start_profile
        # num_steps path) only takes effect once the counter moves here.
        self.forward_ct += 1
        self.profiler_manager._profile_batch_predicate(pending.schedule_batch)

        result = self.tp_worker.finalize_mlx_result(
            pending.prefills,
            pending.extends,
            pending.decode,
            pending.mode,
            pending.reqs,
        )
        result.extend_input_len_per_req = pending.extend_input_len_per_req
        result.extend_logprob_start_len_per_req = (
            pending.extend_logprob_start_len_per_req
        )
        if result.next_token_ids is not None:
            pending.batch_copy.input_ids = result.next_token_ids
            pending.schedule_batch.input_ids = result.next_token_ids
        self.last_batch = pending.schedule_batch
        self.process_batch_result(pending.batch_copy, result)

    @DynamicGradMode()
    def event_loop_overlap_mlx(self: Scheduler):
        """MLX-specific overlap loop modelled on ``mlx_lm.generate.generate_step``.

        At steady state we keep TWO in-flight MLX graphs queued on the
        GPU:

        * ``pending_curr`` — the step whose tokens we are about to block
          on and feed into the scheduler's bookkeeping.
        * ``pending_next`` — the step that was built on top of
          ``pending_curr``'s still-lazy output tokens via
          ``async_chained_decode_mlx`` and has already been handed to
          ``mx.async_eval``.  Because MLX tracks the full dependency
          graph, the GPU will execute ``pending_next`` back-to-back
          with ``pending_curr`` — there is no scheduling gap on the
          device.

        Bookkeeping timeline for a steady-state decode loop:

            iter k:
              build pending_next  (CPU graph build + mx.async_eval; cheap)
              block on pending_curr via .tolist() (wait only on curr's tokens)
              process_batch_result(pending_curr)   <-- GPU is running pending_next
              pending_curr = pending_next

        The chain is broken (we fall back to a "schedule + launch" step)
        whenever any of the following holds:

        * ``pending_curr`` is not a pure decode (e.g. prefill/extend).
        * The waiting queue has new requests that need prefill.
        * Any req in ``pending_curr`` just finished this iteration, so
          the composition for ``pending_next`` would need to shrink.

        When the chain breaks mid-flight we still finalise the
        already-launched ``pending_next`` normally (its tokens are
        valid for all surviving reqs).  With RadixCache-backed caches
        (#21509) there is no ``extract_cache`` step: per-request caches
        are the source of truth and are never merged into a shared
        batched buffer.
        """
        pending_curr: Optional[MlxPendingJob] = None
        pending_next: Optional[MlxPendingJob] = None

        def _launch_fresh(batch: ScheduleBatch) -> MlxPendingJob:
            # Materialize batch.input_ids from CPU staging (prefill) or the
            # FutureMap relay (decode) before the forward. With deferred input
            # materialization, get_next_batch_to_run leaves input_ids unset; the
            # CUDA paths call resolve_forward_inputs for this, but the MLX overlap
            # loop must do it too, otherwise async_forward_batch_generation_mlx
            # dereferences a None input_ids.
            resolve_forward_inputs(batch, self.future_map)
            lazy_tokens, prefills, extends, decode, mode = (
                self.tp_worker.async_forward_batch_generation_mlx(batch)
            )
            extend_input_len_per_req = None
            extend_logprob_start_len_per_req = None
            if batch.return_logprob:
                # Mirror Scheduler.run_batch's launch-time copy.
                extend_input_len_per_req = [
                    req.extend_range.length if req.extend_range is not None else 0
                    for req in batch.reqs
                ]
                extend_logprob_start_len_per_req = batch.extend_logprob_start_lens
            return MlxPendingJob(
                lazy_tokens=lazy_tokens,
                prefills=prefills,
                extends=extends,
                decode=decode,
                mode=mode,
                batch_copy=batch.copy(),
                schedule_batch=batch,
                reqs=list(batch.reqs),
                chain_safe=self._mlx_batch_chain_safe(batch),
                extend_input_len_per_req=extend_input_len_per_req,
                extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
            )

        def _launch_chained(prev: MlxPendingJob) -> MlxPendingJob:
            assert prev.decode is not None
            lazy_tokens, prefills, extends, decode, mode = (
                self.tp_worker.async_chained_decode_mlx(prev.decode)
            )
            # Composition is identical to prev: reuse a fresh batch copy
            # of the same underlying ScheduleBatch so process_batch_result
            # updates the same req objects with the new token.
            return MlxPendingJob(
                lazy_tokens=lazy_tokens,
                prefills=prefills,
                extends=extends,
                decode=decode,
                mode=mode,
                batch_copy=prev.batch_copy.copy(),
                schedule_batch=prev.schedule_batch,
                reqs=prev.reqs,
                chain_safe=prev.chain_safe,
                extend_input_len_per_req=prev.extend_input_len_per_req,
                extend_logprob_start_len_per_req=prev.extend_logprob_start_len_per_req,
            )

        while True:
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # 1. If pending_curr is a pure decode AND no new prefill is waiting,
            #    build pending_next on top of it NOW — before we block on curr.
            can_chain = (
                pending_curr is not None
                and pending_curr.mode == "decode"
                and pending_curr.decode is not None
                and pending_curr.chain_safe
                and not self.waiting_queue
            )
            if can_chain and pending_next is None:
                # Build + launch the chained step BEFORE we block on
                # pending_curr — this is the "no idle gap" trick.
                # GPU now has 2 steps queued.
                pending_next = _launch_chained(pending_curr)
                self.result_queue.append(pending_next)

            # 2. Finalize/process on pending_curr's tokens.  (GPU is already
            #    executing pending_next at this point.)
            if pending_curr is not None:
                self._finalize_mlx_pending_job(pending_curr)
                self.result_queue.popleft()
                pending_curr = None

            # 3. Decide whether pending_next is still valid (if no reqs finished)
            #    and promote it.
            finished_any = any(
                req.finished() for req in (pending_next.reqs if pending_next else [])
            )
            new_prefill_waiting = bool(self.waiting_queue)
            if (
                pending_next is not None
                and not finished_any
                and not new_prefill_waiting
            ):
                pending_curr = pending_next
                pending_next = None
                self.cur_batch_for_debug = pending_curr.schedule_batch
                self.last_batch = pending_curr.schedule_batch
                if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                    self.invariant_checker.self_check_during_busy()
                continue

            # 4. Chain is broken. Finalise pending_next (if any), then
            #    schedule fresh.
            if pending_next is not None:
                self._finalize_mlx_pending_job(pending_next)
                self.result_queue.popleft()
                pending_next = None
            plan = self.get_next_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            next_batch = plan.batch_to_run
            self.cur_batch_for_debug = next_batch
            if next_batch:
                pending_curr = _launch_fresh(next_batch)
                self.result_queue.append(pending_curr)
            else:
                self.on_idle()

            self.last_batch = next_batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()
