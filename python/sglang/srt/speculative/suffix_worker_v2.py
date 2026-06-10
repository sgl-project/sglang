"""SuffixWorkerV2 — M4 skeleton.

Glues M1 (linear chain → verify arrays), M2 (complete EagleVerifyInput),
M3b (SuffixV2DraftBuilder) into a Spec V2 worker that:

  - Reuses EAGLE V2's verify scaffold (prepare_for_v2_verify, target
    forward with is_verify=True, EagleVerifyInput.sample) verbatim
  - Replaces only the "draft" step: arctic suffix tree query instead of
    EAGLE draft model forward
  - Maintains per-rid token history for arctic add_active_response calls

Status: SKELETON. Not yet wired into sgl_suffix_plugin. End-to-end
integration test deferred until plugin registration + first server boot.

Open issues (to resolve before first server boot):

  [I1] How do we get the rid → accepted-tokens delta after each verify?
       Maintain our own `_req_tokens: Dict[rid, list[int]]` cache.
       After each step we read predict[accept_index] on CPU (sync on
       verify_done) and stash into `_pending_accepted`; next step drains
       it into the arctic tree before querying. CPU-sync cost accepted
       for M4; M6 will fold this into a CPU hook that runs after verify
       completes (no extra sync vs the existing batch.maybe_wait_verify_done).

  [I2] resolved: `mwb.reqs` is populated unconditionally by
       `ScheduleBatch.get_model_worker_batch()` (schedule_batch.py:2617).
       Use `[req.rid for req in mwb.reqs]`.

  [I3] resolved: prompts come from `req.origin_input_ids` (already on CPU,
       per-req) — no slicing of `mwb.input_ids` needed.

  [I4] capture_hidden_mode: we pass NULL (standalone SUFFIX, no MTP keep-up
       needed). For HYBRID V2 future use, this becomes FULL.

  [I5] Overlap event-stream: `plan_stream`, `_draft_done_event`,
       `record_stream` patterns from EAGLEWorkerV2.verify mostly transfer
       directly; we don't have a draft model forward to wait on, so the
       `_draft_done_event` synchronization may simplify.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

# sglang imports (resolved at runtime on pod where sglang is installed)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_info_v2 import fill_bonus_tokens
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.suffix_v2_draft_builder_full import SuffixV2DraftBuilder
from sglang.srt.speculative.suffix_v2_verify_input import (
    build_suffix_v2_eagle_verify_input,
)


class SuffixDraftWorker(BaseDraftWorker):
    """Stub draft worker for SUFFIX V2.

    SUFFIX doesn't use a separate draft model — drafts come from arctic
    suffix tree query handled inside SuffixWorkerV2 directly. BaseDraftWorker
    requires `draft()` and `draft_extend()` to exist as abstract methods, so
    we provide no-op implementations.
    """

    def draft(self, *args, **kwargs):
        # SUFFIX V2 doesn't run a draft model forward. Query happens inside
        # SuffixWorkerV2._decode_step via SuffixV2DraftBuilder.
        raise NotImplementedError(
            "SuffixDraftWorker.draft() is intentionally a stub — see "
            "SuffixWorkerV2._decode_step for the actual suffix tree query."
        )

    def draft_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "SuffixDraftWorker.draft_extend() is intentionally a stub."
        )


class SuffixWorkerV2(BaseSpecWorker):
    """SUFFIX speculative-decoding worker on the Spec V2 contract.

    Per-batch flow (decode):

      1. mwb.spec_info is the EagleDraftInput from prev step (carries
         bonus_tokens, new_seq_lens, verify_done event — resolved by
         FutureMap).
      2. CPU-sync on verify_done so we can read the prev step's accepted
         tokens (TODO I1: need them on CPU to update arctic tree).
      3. update_with_accepted(rid, full_history) — push delta into tree.
      4. query_batch(rids, current_tokens, K-1) → suffix draft tokens.
      5. build_suffix_v2_eagle_verify_input(...) → EagleVerifyInput
         (topk=1 linear chain).
      6. EAGLE V2 verify scaffold:
         a. prepare_for_v2_verify (allocates KV slots, sets up cuda graph)
         b. target_worker.forward_batch_generation(is_verify=True)
         c. verify_input.sample(batch, logits, vocab_mask)
      7. Build next_draft_input = EagleDraftInput(bonus_tokens, new_seq_lens,
         verify_done) and stash next step's expected accept tokens for I1.
      8. Return GenerationBatchResult.

    Per-batch flow (extend / prefill):

      1. Standard target forward (no spec).
      2. Initialize arctic cache state for new reqs (start_request).
      3. Build next_draft_input with bonus_tokens = next_token_ids.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.page_size = server_args.page_size
        self._target_worker = target_worker
        self._draft_worker = SuffixDraftWorker()

        # K = chain width per req (bonus slot + K-1 spec slots).
        self.K = server_args.speculative_num_draft_tokens
        assert self.K >= 2, "speculative_num_draft_tokens must be >= 2 for SUFFIX V2"
        self.K_minus_1 = self.K - 1

        # Model context length — used to cap each req's draft chain so the
        # verify step can't overshoot ctx_len and produce an overlap-schedule
        # zombie verify on the next iteration. See patch_nsa_zombie_overflow.py
        # for the underlying crash this prevents at the source.
        self._max_context_len = int(target_worker.model_runner.model_config.context_len)

        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # The suffix tree query backend. Wraps arctic SuffixDecodingCache.
        self.draft_builder = SuffixV2DraftBuilder(
            max_cached_requests=server_args.speculative_suffix_max_cached_requests,
            max_tree_depth=server_args.speculative_suffix_max_tree_depth,
            max_spec_factor=server_args.speculative_suffix_max_spec_factor,
            min_token_prob=server_args.speculative_suffix_min_token_prob,
        )

        # Per-rid token history (prompt + all accepted output so far). Used
        # both for arctic add_active_response (delta push) and as the
        # pattern context for the next query.
        # See [I1] in module docstring.
        self._req_tokens: Dict[str, List[int]] = {}

        # Stash the previous step's accepted tokens until the next step
        # consumes them. Populated at end of _decode_step, drained at start
        # of next _decode_step. See [I1].
        # Maps rid → list[int] (accepted tokens from prev step).
        self._pending_accepted: Dict[str, List[int]] = {}

        # Epoch-based GC. Scheduler doesn't give us a per-rid finish hook
        # and individual batches don't list all currently-active reqs
        # (prefill and decode are scheduled in SEPARATE batches), so we
        # can't prune based on a single batch's rid set. Instead we tag
        # each rid with the step number it was last seen and GC rids
        # unseen for `_gc_epoch_threshold` batches. The threshold is large
        # enough to ride out prefill↔decode interleaving and the longest
        # plausible quiescence (preemption / retraction-rerun) before the
        # scheduler officially drops the req.
        self._step_idx: int = 0
        self._rid_last_seen: Dict[str, int] = {}
        self._gc_epoch_threshold: int = 256

        # Debug-only periodic GC stats log; on with SUFFIX_V2_DEBUG=1.
        import os as _os

        self._debug = _os.environ.get("SUFFIX_V2_DEBUG", "0") == "1"
        # Per-phase timing log (heavy); on with SUFFIX_V2_TIMING=1.
        self._timing = _os.environ.get("SUFFIX_V2_TIMING", "0") == "1"

        # M6: deferred CPU work. End of each step builds a closure that
        # (a) sync's on this step's verify_done, (b) reads predict/accept_index
        # to CPU, (c) populates _pending_accepted. Start of next step drains
        # the closure BEFORE doing any other CPU work. The scheduler-side
        # work between worker returns (store_to_map, copy_to_cpu, output_ids
        # update, etc.) now overlaps with the GPU sync wait, instead of the
        # previous design where the sync blocked end-of-step return.
        self._deferred: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------------
    # BaseSpecWorker contract
    # ------------------------------------------------------------------
    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def clear_cache_pool(self):
        # Drain any pending deferred work first so the GPU tensor references
        # the closure holds are released before reset.
        self._drain_deferred()
        # Reset arctic cache + per-rid state on flush.
        for rid in list(self._req_tokens.keys()):
            self.draft_builder.stop_request(rid)
        self._req_tokens.clear()
        self._pending_accepted.clear()
        self._rid_last_seen.clear()
        # The token_to_kv_pool_allocator is shared with target_worker; it
        # handles its own clear via the scheduler's flush path.

    def _drain_deferred(self) -> None:
        """Run the previous step's deferred CPU work (GPU sync + read).
        Always safe to call — no-op when nothing is deferred."""
        if self._deferred is not None:
            commit, self._deferred = self._deferred, None
            commit()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def forward_batch_generation(
        self, mwb: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        import time as _t

        t_start = _t.perf_counter() if self._timing else 0.0
        self._step_idx += 1
        # M6: complete prev step's deferred CPU work (sync + read + populate
        # _pending_accepted). Runs after scheduler-side work has had a chance
        # to overlap with the GPU sync wait.
        self._drain_deferred()
        t_drained = _t.perf_counter() if self._timing else 0.0
        if mwb.forward_mode.is_idle():
            result = self._idle_step(mwb)
            self._maybe_publish(on_publish, result)
            return result
        # Touch last-seen epoch for every rid in this batch.
        if mwb.reqs:
            for req in mwb.reqs:
                self._rid_last_seen[req.rid] = self._step_idx
            # GC: prune rids unseen for many epochs (long-finished reqs).
            # Cheap O(n) sweep — n is bounded by max_running_requests.
            if self._req_tokens:
                self._gc_stale_rids()
        if mwb.forward_mode.is_extend() or mwb.is_extend_in_batch:
            result = self._extend_step(mwb)
            self._maybe_publish(on_publish, result)
            if self._timing:
                self._log_timing("EXTEND", t_start, t_drained, _t.perf_counter(), mwb)
            return result
        result = self._decode_step(mwb)
        self._maybe_publish(on_publish, result)
        if self._timing:
            self._log_timing("DECODE", t_start, t_drained, _t.perf_counter(), mwb)
        return result

    @staticmethod
    def _maybe_publish(on_publish, result: GenerationBatchResult) -> None:
        """Overlap-scheduling fence (spec-v2 contract). The scheduler passes
        on_publish=partial(future_map.publish, future_indices); calling it with
        this step's new_seq_lens lets the scheduler resolve seq_lens for the
        overlapped next batch. SUFFIX has no post-verify GPU work, so the
        verify-end fence is here (vs EAGLEWorkerV2 which fires it before its
        draft-extend tail)."""
        if on_publish is not None and result.new_seq_lens is not None:
            on_publish(result.new_seq_lens)

    def _log_timing(self, kind, t_start, t_drained, t_end, mwb) -> None:
        try:
            import logging

            logging.getLogger("sglang.spec.suffix_v2").info(
                "[SUFFIX_V2_TIME] step=%d kind=%s bs=%d drain=%.2fms total=%.2fms",
                self._step_idx,
                kind,
                len(mwb.reqs) if mwb.reqs else 0,
                (t_drained - t_start) * 1000,
                (t_end - t_start) * 1000,
            )
        except Exception:
            pass

    def _gc_stale_rids(self) -> None:
        threshold = self._step_idx - self._gc_epoch_threshold
        if threshold <= 0:
            return
        dropped = 0
        for rid in list(self._req_tokens.keys()):
            if self._rid_last_seen.get(rid, 0) < threshold:
                self.draft_builder.stop_request(rid)
                self._req_tokens.pop(rid, None)
                self._pending_accepted.pop(rid, None)
                self._rid_last_seen.pop(rid, None)
                dropped += 1
        if self._debug and (dropped or self._step_idx % 100 == 0):
            try:
                import logging

                logging.getLogger("sglang.spec.suffix_v2").info(
                    "[SUFFIX_V2_GC] step=%d dropped=%d kept=%d arctic_active=%d",
                    self._step_idx,
                    dropped,
                    len(self._req_tokens),
                    len(self.draft_builder.cache.active_requests),
                )
            except Exception:
                pass

    def _idle_step(self, mwb: ScheduleBatch) -> GenerationBatchResult:
        """Empty batch — scheduler still calls into the worker so it can
        keep its sampling loop fired. Forward through target, mirror EAGLE's
        idle path."""
        batch_output = self._target_worker.forward_batch_generation(mwb)
        idle_new_seq_lens = torch.empty(0, device=self.device, dtype=torch.int32)
        batch_output.next_draft_input = self._make_next_draft_input(
            bonus_tokens=torch.empty(0, device=self.device, dtype=torch.int32),
        )
        # Expose new_seq_lens for the overlap publish fence (empty for idle).
        batch_output.new_seq_lens = idle_new_seq_lens
        return batch_output

    def _make_next_draft_input(
        self,
        bonus_tokens: torch.Tensor,
    ) -> EagleDraftInput:
        """Build the next-step EagleDraftInput carried via FutureMap.

        FutureMap (overlap_utils.py _lazy_init_buf) unconditionally reads
        `draft_input.topk_p[0]` and `draft_input.topk_index[0]` to derive
        buffer shapes. SUFFIX V2 has no real per-step draft probabilities,
        so we pass dummy (bs, 1) zeros that round-trip safely through the
        buffer without being consumed.

        spec_need_hidden_states is patched to return False for SUFFIX, so
        we don't populate hidden_states.

        new_seq_lens / verify_done are NOT fields here: current upstream
        main carries new_seq_lens on GenerationBatchResult (published via
        on_publish, see _maybe_publish) and drives the post-verify CPU sync
        with a worker-local cuda event (see _decode_step) instead of an
        EagleDraftInput-stashed one.
        """
        bs = int(bonus_tokens.numel())
        return EagleDraftInput(
            topk_p=torch.zeros((bs, 1), device=self.device, dtype=torch.float32),
            topk_index=torch.zeros((bs, 1), device=self.device, dtype=torch.int64),
            bonus_tokens=bonus_tokens,
        )

    # ------------------------------------------------------------------
    # Extend path
    # ------------------------------------------------------------------
    def _extend_step(self, mwb: ScheduleBatch) -> GenerationBatchResult:
        import time as _t

        t0 = _t.perf_counter() if self._timing else 0.0
        # Standalone SUFFIX → NULL capture mode (no hidden states needed).
        mwb.capture_hidden_mode = CaptureHiddenMode.NULL
        batch_output = self._target_worker.forward_batch_generation(mwb)
        t_target = _t.perf_counter() if self._timing else 0.0

        # Initialize arctic cache state for any req we haven't seen.
        # [I3]: needs prompt tokens on CPU at this point.
        self._initialize_new_reqs(mwb)
        t_init = _t.perf_counter() if self._timing else 0.0

        # Bonus token for next decode step = the just-generated next_token_ids.
        bonus = batch_output.next_token_ids.to(torch.int32)
        new_seq_lens = (mwb.seq_lens + 1).to(torch.int32)

        # M6: defer the CPU read of bonus into _pending_accepted to the start
        # of the next step. Closure holds GPU tensor refs alive.
        rids_snapshot = self._extract_rids(mwb)
        pending = self._pending_accepted
        bonus_gpu = bonus

        def _commit_extend() -> None:
            bonus_cpu = bonus_gpu.cpu().tolist()
            for rid, tok in zip(rids_snapshot, bonus_cpu):
                pending[rid] = [int(tok)]

        self._deferred = _commit_extend

        batch_output.next_draft_input = self._make_next_draft_input(
            bonus_tokens=bonus,
        )
        # Expose new_seq_lens for the overlap publish fence.
        batch_output.new_seq_lens = new_seq_lens
        if self._timing:
            t_end = _t.perf_counter()
            import logging

            logging.getLogger("sglang.spec.suffix_v2").info(
                "[SUFFIX_V2_EXTEND_BREAKDOWN] step=%d bs=%d target=%.2fms init_reqs=%.2fms tail=%.2fms",
                self._step_idx,
                len(mwb.reqs) if mwb.reqs else 0,
                (t_target - t0) * 1000,
                (t_init - t_target) * 1000,
                (t_end - t_init) * 1000,
            )
        return batch_output

    def _initialize_new_reqs(self, mwb: ScheduleBatch) -> None:
        """Register any unseen rid with the arctic suffix cache.

        Reads prompts straight from `req.origin_input_ids` (already on CPU).
        """
        for req in mwb.reqs:
            rid = req.rid
            if rid in self._req_tokens:
                continue
            prompt = list(req.origin_input_ids)
            self.draft_builder.start_request(rid, prompt)
            self._req_tokens[rid] = prompt

    # ------------------------------------------------------------------
    # Decode path
    # ------------------------------------------------------------------
    def _decode_step(self, mwb: ScheduleBatch) -> GenerationBatchResult:
        import time as _t

        t0 = _t.perf_counter() if self._timing else 0.0
        prev_draft: EagleDraftInput = mwb.spec_info
        assert prev_draft is not None, "decode step requires prev spec_info"

        # M6: prev verify_done sync now happens inside the deferred closure
        # drained at top of forward_batch_generation, not here.

        rids = self._extract_rids(mwb)
        bs = len(rids)

        # 1. Drain pending accepted tokens from prev step → push into arctic tree.
        for rid in rids:
            accepted = self._pending_accepted.pop(rid, None)
            if accepted:
                self._req_tokens.setdefault(rid, []).extend(accepted)
                self.draft_builder.update_with_accepted(rid, self._req_tokens[rid])

        t_drain_pending = _t.perf_counter() if self._timing else 0.0

        # 2. Suffix tree query → linear draft tokens per req.
        current_tokens = [self._req_tokens[rid] for rid in rids]
        # Per-req draft cap = ctx_len - cur_seq_len - 1 (the -1 leaves room for
        # the bonus token without overshooting). Prevents zombie verify steps.
        seq_lens_list = mwb.seq_lens.cpu().tolist()
        max_remaining = [max(0, self._max_context_len - sl - 1) for sl in seq_lens_list]
        draft_tokens, draft_lens, valid_mask = self.draft_builder.query_batch(
            rids=rids,
            current_tokens=current_tokens,
            K_minus_1=self.K_minus_1,
            device=self.device,
            max_remaining_per_req=max_remaining,
        )
        t_query = _t.perf_counter() if self._timing else 0.0

        # 3. Build the linear EagleVerifyInput.
        bonus_tokens = prev_draft.bonus_tokens  # (bs,) int32 on device
        seq_lens_cpu = mwb.seq_lens.cpu()
        verify_input = build_suffix_v2_eagle_verify_input(
            bonus_tokens=bonus_tokens,
            suffix_draft_tokens=draft_tokens,
            seq_lens=mwb.seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        mwb.spec_info = verify_input

        # 4. Run the EAGLE V2 verify scaffold. Replicate the essential calls
        # from EAGLEWorkerV2.verify() — minus draft-tp-context, mamba,
        # MRoPE which we don't have for SUFFIX standalone.
        verify_forward_batch, can_run_cuda_graph = verify_input.prepare_for_v2_verify(
            self.req_to_token_pool,
            mwb,
            self._target_worker,
        )

        t_prepare = _t.perf_counter() if self._timing else 0.0
        forward_batch_output = self._target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output
        t_target = _t.perf_counter() if self._timing else 0.0

        # 5. Sample via EAGLE V2's verify_input.sample (handles grammar,
        # sampling_info, tree_greedy / target_only_speculative paths).
        vocab_mask = None  # grammar not yet supported in V2 SUFFIX
        predict, accept_lens, accept_index = verify_input.sample(
            mwb, logits_output, vocab_mask
        )
        # Clamp acceptance to the real suffix-match length. Short matches are
        # right-padded to K-1 with token-id 0 (query_batch returns draft_lens =
        # the real per-req match length). The verify chain mask is causal, so
        # every real position (chain depth <= draft_lens) is computed from clean
        # context and is safe to accept. A padding 0 at depth > draft_lens can
        # still be spuriously accepted when the target's own argmax there is 0,
        # which would advance seq_lens past the real output and feed a fake 0
        # into the arctic tree via the commit closure below. Cap at
        # bonus + draft_lens so only real positions ever count.
        accept_lens = torch.minimum(accept_lens, (draft_lens + 1).to(accept_lens.dtype))
        new_seq_lens = mwb.seq_lens + accept_lens

        # 6. Build next_draft_input via fill_bonus_tokens (same kernel EAGLE uses).
        if not mwb.forward_mode.is_idle():
            accept_tokens_gpu = predict[accept_index]  # (bs * K,) int32 padded with -1?
            bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
            fill_bonus_tokens[(bs,)](
                accept_tokens_gpu,
                accept_lens,
                bonus_tokens,
                self.K,
            )
        else:
            bonus_tokens = torch.empty(0, device=self.device, dtype=torch.int32)

        next_draft_input = self._make_next_draft_input(
            bonus_tokens=bonus_tokens,
        )

        # 7. M6: defer the CPU read of predict + accept_index. Closure captures
        # GPU tensors (kept alive via the closure ref) + a verify-done event +
        # rids snapshot, and populates _pending_accepted at start of next step.
        # Current upstream main no longer exposes verify_done on EagleDraftInput,
        # so record a worker-local event on the current stream after the verify
        # kernels are enqueued; synchronize() on it fences the CPU read.
        rids_snapshot = list(rids)
        pending = self._pending_accepted
        predict_gpu = predict
        accept_index_gpu = accept_index
        accept_lens_gpu = accept_lens
        verify_done_event = torch.get_device_module(self.device).Event()
        verify_done_event.record()

        def _commit_decode() -> None:
            verify_done_event.synchronize()
            ai_cpu = accept_index_gpu.cpu().tolist()
            pr_cpu = predict_gpu.cpu().tolist()
            al_cpu = accept_lens_gpu.cpu().tolist()
            for i, rid in enumerate(rids_snapshot):
                acc = []
                for j in ai_cpu[i]:
                    if j == -1:
                        break
                    acc.append(int(pr_cpu[j]))
                # Truncate to the clamped accept length so a spuriously
                # accepted padding-0 (see clamp after sample()) never enters
                # the suffix tree.
                acc = acc[: int(al_cpu[i])]
                if acc:
                    pending[rid] = acc

        self._deferred = _commit_decode

        if self._timing:
            t_end = _t.perf_counter()
            import logging

            logging.getLogger("sglang.spec.suffix_v2").info(
                "[SUFFIX_V2_DECODE_BREAKDOWN] step=%d bs=%d drain_pend=%.2fms query=%.2fms prepare=%.2fms target=%.2fms tail=%.2fms total=%.2fms",
                self._step_idx,
                bs,
                (t_drain_pending - t0) * 1000,
                (t_query - t_drain_pending) * 1000,
                (t_prepare - t_query) * 1000,
                (t_target - t_prepare) * 1000,
                (t_end - t_target) * 1000,
                (t_end - t0) * 1000,
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=self.K,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            routed_experts_output=forward_batch_output.routed_experts_output,
            indexer_topk_output=forward_batch_output.indexer_topk_output,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_rids(self, mwb: ScheduleBatch) -> List[str]:
        """Pull per-req rids from the batch.

        `mwb.reqs` is populated by ScheduleBatch.get_model_worker_batch
        (schedule_batch.py:2617) on every batch, so this is reliable.
        """
        return [req.rid for req in mwb.reqs]
