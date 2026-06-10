"""HybridSuffixMTPWorkerV2 — V2-API HYBRID worker (overlap scheduling).

Subclasses ``EAGLEWorkerV2`` and dispatches each batch among three
backends decided by ``HybridBackendSelector``:

  - **SUFFIX** — arctic suffix-tree draft + EAGLE-style verify at
    ``K = speculative_num_draft_tokens``. Followed by an optional MTP
    keep-up forward (under ``--speculative-hybrid-mtp-always-warm``) so a
    later MTP-picked step starts with warm draft state.
  - **MTP** — standard EAGLE V2 draft + verify (delegated to ``super``),
    ``K = speculative_num_steps + 1``.
  - **NONE** — plain target decode, no spec, ``K = 1``.

All three backends produce different verify input widths; dispatch to
three captured cuda graphs (main K_suffix, short_chain K_mtp, baseline
K=1) is handled transparently by ``ModelRunner._forward_raw`` via each
runner's width-based ``can_run``.

SUFFIX backend per-step flow:
  1. Pull current tokens per req (``req.origin_input_ids +
     req.output_ids``). In overlap mode this lags realized accepts by
     <= 1 step, which is corrected by the post-verify push below.
  2. Update arctic with the delta tokens; query for K-1 spec tokens per req.
  3. Build an ``EagleVerifyInput`` at K (linear chain, FULL hidden mode).
  4. Run ``self.verify(mwb)``.
  5. Post-verify: read realized ``accept_lens`` + accepted token ids from
     the batch result and push them back into arctic so the next peek
     reflects exact history (not the overlap-lagged ``req.output_ids``).
  6. Optionally fire ``_draft_extend_for_decode`` for MTP keep-up.

A V1 variant is intentionally not provided: SUFFIX V1's NGRAMWorker
subclass cannot live under the overlap scheduler, so a V1-only hybrid
has no upside.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.hybrid_backend_selector import (
    Backend,
    HybridBackendSelector,
)
from sglang.srt.speculative.suffix_v2_draft_builder_full import SuffixV2DraftBuilder
from sglang.srt.speculative.suffix_v2_verify_input import (
    build_suffix_v2_eagle_verify_input,
)


class HybridSuffixMTPWorkerV2(EAGLEWorkerV2):
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
        # Unlock K_suffix > K_mtp.
        #
        # The launch passes --speculative-num-draft-tokens K_suffix (the
        # wide SUFFIX chain, typically 16). EAGLE V2's draft_forward calls
        # torch.topk(score_list, K - 1) where score_list has num_steps × topk
        # columns, so K - 1 must be <= num_steps × topk (= num_steps for the
        # typical topk=1). Letting super().__init__ see K_suffix directly
        # would crash the draft cuda graph capture with "selected index k
        # out of range".
        #
        # Workaround: temporarily lower speculative_num_draft_tokens to
        # K_mtp = num_steps + 1 for the duration of super().__init__. The
        # main target cuda graph runner was already captured BEFORE
        # maybe_init_draft_worker at the real K_suffix (sglang init order:
        # model_runner.init_device_graphs → maybe_init_draft_worker), so we
        # keep that one. The *draft* cuda graph (EagleDraftCudaGraphRunner)
        # is captured INSIDE super().__init__ at the temporary K_mtp,
        # satisfying the topk invariant. After super returns, restore
        # K_suffix for our SUFFIX verify input construction.
        K_suffix = server_args.speculative_num_draft_tokens
        K_mtp = server_args.speculative_num_steps + 1
        assert K_suffix >= 2 and K_mtp >= 2
        if K_suffix != K_mtp:
            server_args.speculative_num_draft_tokens = K_mtp
        try:
            super().__init__(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                target_worker=target_worker,
            )
        finally:
            if K_suffix != K_mtp:
                server_args.speculative_num_draft_tokens = K_suffix

        # Two chain widths. SUFFIX path uses K_suffix (matches main cuda
        # graph); MTP path uses self.speculative_num_draft_tokens (set by
        # super to K_mtp).
        self.K_suffix: int = K_suffix
        self.K_mtp: int = K_mtp
        # Aliases for code paths that read self.K.
        self.K = K_suffix
        self.K_minus_1 = K_suffix - 1

        # Model context length — used to cap each req's SUFFIX draft chain so
        # the verify step can't overshoot ctx_len and produce an overlap-
        # schedule zombie verify (which manifests as NSA page-table OOB at
        # the attention backend).
        self._max_context_len = int(target_worker.model_runner.model_config.context_len)

        # SUFFIX-side state (mirrors SuffixWorkerV2).
        self.suffix_draft_builder = SuffixV2DraftBuilder(
            max_cached_requests=server_args.speculative_suffix_max_cached_requests,
            max_tree_depth=server_args.speculative_suffix_max_tree_depth,
            max_spec_factor=server_args.speculative_suffix_max_spec_factor,
            min_token_prob=server_args.speculative_suffix_min_token_prob,
        )
        self._req_tokens: Dict[str, List[int]] = {}
        self._pending_accepted: Dict[str, List[int]] = {}
        self._step_idx: int = 0
        self._rid_last_seen: Dict[str, int] = {}
        self._gc_epoch_threshold: int = 256

        # Deferred CPU work — same pattern as SuffixWorkerV2. Populated by
        # the SUFFIX path; MTP path stays managed by EAGLEWorkerV2 internals
        # (its own delay_sample_func mechanism).
        self._deferred: Optional[Callable[[], None]] = None

        # Per-batch backend chooser (SUFFIX / MTP / NONE). Self-contained —
        # only needs server_args, and choose() infers bs from len(suffix_scores).
        # ARCTIC_HYBRID_FORCE_BACKEND env var pins one backend for A/B testing.
        self.selector = HybridBackendSelector(server_args)

        # Cache of the current step's arctic query, populated by
        # _peek_suffix_scores and consumed by _decode_step_suffix to avoid
        # a second arctic query when SUFFIX is picked.
        # Layout: (step_idx, draft_tokens, draft_lens, valid_mask, bonus_cpu, current_tokens_list)
        self._cached_suffix_drafts = None

        # Debug
        import os as _os

        self._debug = _os.environ.get("HYBRID_V2_DEBUG", "0") == "1"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def clear_cache_pool(self):
        # Drain deferred so closure-held GPU refs release before reset.
        self._drain_deferred()
        for rid in list(self._req_tokens.keys()):
            self.suffix_draft_builder.stop_request(rid)
        self._req_tokens.clear()
        self._pending_accepted.clear()
        self._rid_last_seen.clear()
        # Parent's allocator/pool is shared with target; scheduler resets it.

    def _drain_deferred(self) -> None:
        if self._deferred is not None:
            commit, self._deferred = self._deferred, None
            commit()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def forward_batch_generation(
        self, mwb: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        self._step_idx += 1
        self._drain_deferred()

        if mwb.forward_mode.is_idle():
            # Just punt to EAGLE V2's idle handling (it fires on_publish).
            return super().forward_batch_generation(mwb, on_publish)

        if mwb.forward_mode.is_extend() or mwb.is_extend_in_batch:
            # EAGLE V2 extend = target prefill + MTP draft prefill. Drives MTP
            # state forward; SUFFIX side just needs the prompt registered.
            # Parent fires on_publish at the target-end fence.
            result = super().forward_batch_generation(mwb, on_publish)
            # Init arctic for any new rid (cheap; skipped for known rids).
            self._initialize_new_reqs(mwb)
            # Tag last_seen for epoch GC.
            if mwb.reqs:
                for req in mwb.reqs:
                    self._rid_last_seen[req.rid] = self._step_idx
            return result

        # DECODE
        if mwb.reqs:
            for req in mwb.reqs:
                self._rid_last_seen[req.rid] = self._step_idx
            if self._req_tokens:
                self._gc_stale_rids()

        # Peek SUFFIX + MTP scores, let selector choose backend. The peek is
        # side-effecting on arctic (start_request / add_active_response) so
        # the SUFFIX tree stays warm even on MTP-picked steps.
        use_scores = self._selector_uses_scores()
        # Diagnostic: at peek time, compare token-delta-pushed-to-arctic
        # against last step's actual accept_len. If delta < accept_len, the
        # bonus-append context fix is missing some accepted tokens (V2
        # overlap output_ids lag) — that under-feeds arctic tree and biases
        # ema_sfx low even when warmup data exists.
        if os.environ.get("HYBRID_V2_ARCTIC_DELTA_PROBE") and mwb.reqs:
            _last_acc = getattr(self, "_last_accept_per_rid", {})
            for req in mwb.reqs:
                rid = req.rid
                prev_len = len(self._req_tokens.get(rid, []))
                new_len = (
                    len(req.origin_input_ids)
                    + len(req.output_ids)
                    + (
                        1
                        if mwb.spec_info is not None
                        and hasattr(mwb.spec_info, "bonus_tokens")
                        else 0
                    )
                )
                delta = new_len - prev_len
                last_acc = _last_acc.get(rid, None)
                if last_acc is not None and self._step_idx % 50 == 0:
                    logging.getLogger("sglang.spec.hybrid_v2").info(
                        "[ARCTIC_DELTA_PROBE] step=%d rid=%s delta=%d "
                        "last_accept_len=%.2f shortfall=%s",
                        self._step_idx,
                        str(rid)[-6:],
                        delta,
                        last_acc,
                        max(0, last_acc - delta),
                    )
        suffix_scores = self._peek_suffix_scores(mwb) if use_scores else None
        mtp_scores = self._peek_mtp_scores(mwb) if use_scores else None
        backend = self.selector.choose(
            mwb,
            suffix_scores=suffix_scores,
            mtp_scores=mtp_scores,
        )

        # Cold-start: arrival-time decoder may have spec_info that isn't an
        # EagleDraftInput (e.g., after filter/restart). MTP needs warm
        # EagleDraftInput on mwb.spec_info; promote to SUFFIX if absent.
        if backend == Backend.MTP and not isinstance(mwb.spec_info, EagleDraftInput):
            backend = Backend.SUFFIX

        if backend == Backend.SUFFIX:
            result = self._decode_step_suffix(mwb, on_publish)
        elif backend == Backend.MTP:
            # Parent fires on_publish at its verify-end fence.
            result = super().forward_batch_generation(mwb, on_publish)
        elif backend == Backend.NONE:
            result = self._decode_step_none(mwb, on_publish)
        else:
            raise AssertionError(f"unknown backend {backend!r}")

        # Post-verify: push actual accepted tokens to arctic tree. Fixes the
        # bonus-only feedback bug — under V2 overlap, req.output_ids lags so
        # appending only the bonus to peek context misses (accept_len - 1)
        # spec-accepted tokens per step. Read them straight from result.
        K_used = (
            self.K_suffix
            if backend == Backend.SUFFIX
            else (self.K_mtp if backend == Backend.MTP else 1)
        )
        if (
            result is not None
            and result.accept_lens is not None
            and result.next_token_ids is not None
            and backend in (Backend.SUFFIX, Backend.MTP)
            and mwb.reqs
        ):
            try:
                accept_cpu = result.accept_lens.cpu().tolist()
                # predict tensor is flat (bs * K,) — chunk by K per req.
                predict_cpu = result.next_token_ids.cpu().tolist()
                for i, req in enumerate(mwb.reqs):
                    rid = req.rid
                    a = int(accept_cpu[i])
                    if a <= 0 or rid not in self._req_tokens:
                        continue
                    new_tokens = predict_cpu[i * K_used : i * K_used + a]
                    self._req_tokens[rid].extend(int(t) for t in new_tokens)
                    self.suffix_draft_builder.update_with_accepted(
                        rid, self._req_tokens[rid]
                    )

                # Update selector EMA with mean accept length.
                if accept_cpu:
                    self.selector.note_accept(
                        backend, sum(accept_cpu) / len(accept_cpu)
                    )
                    if os.environ.get("HYBRID_V2_ARCTIC_DELTA_PROBE"):
                        if not hasattr(self, "_last_accept_per_rid"):
                            self._last_accept_per_rid = {}
                        for req, a in zip(mwb.reqs, accept_cpu):
                            self._last_accept_per_rid[req.rid] = a
            except Exception as e:
                logging.getLogger("sglang.spec.hybrid_v2").warning(
                    "[HYBRID_V2_POST_VERIFY] push failed: %s", e
                )
        elif (
            result is not None
            and result.accept_lens is not None
            and backend == Backend.NONE
        ):
            # NONE path: K=1, only bonus generated. Still update arctic so
            # tree reflects every generated token.
            try:
                accept_cpu = result.accept_lens.cpu().tolist()
                predict_cpu = (
                    result.next_token_ids.cpu().tolist()
                    if result.next_token_ids is not None
                    else None
                )
                if predict_cpu is not None:
                    for i, req in enumerate(mwb.reqs):
                        rid = req.rid
                        if rid not in self._req_tokens:
                            continue
                        self._req_tokens[rid].append(int(predict_cpu[i]))
                        self.suffix_draft_builder.update_with_accepted(
                            rid, self._req_tokens[rid]
                        )
            except Exception:
                pass

        return result

    def _selector_uses_scores(self) -> bool:
        """Mirror V1's check — skip the peek when force/debug knobs short-circuit
        the choice anyway."""
        s = self.selector
        return not (
            getattr(s, "_forced", "") or getattr(s, "_debug_mode", None) is not None
        )

    def _peek_suffix_scores(self, mwb: ScheduleBatch) -> List[float]:
        """Per-req arctic-suffix-tree speculation score.

        Side effects (idempotent, matches V1):
          - register any new rid with arctic (start_request)
          - push delta tokens (origin + output + bonus) since last sighting
          - run arctic.speculate to populate the per-req draft + score

        We also stash the just-queried drafts on `self._cached_suffix_drafts`
        keyed by step_idx so _decode_step_suffix can reuse them WITHOUT a
        second arctic call (saves arctic CPU work on SUFFIX-picked steps).
        """
        rids = [req.rid for req in mwb.reqs]
        if not rids:
            self._cached_suffix_drafts = None
            return []

        # Context: use self._req_tokens[rid] as the authoritative token
        # history. Post-verify hook keeps it in sync with actually-generated
        # tokens (predict[accept_index]), avoiding the V2 overlap output_ids
        # lag that loses (accept_len-1) tokens per step. On cold start, init
        # from req.origin_input_ids + req.output_ids (prefill state).
        bonus_cpu = mwb.spec_info.bonus_tokens.cpu().tolist()
        current_tokens_list: List[List[int]] = []
        for req, b in zip(mwb.reqs, bonus_cpu):
            rid = req.rid
            if rid not in self._req_tokens:
                # Cold-start: prefill leaves output_ids = [first_token] which
                # equals this peek's bonus. Use bonus to avoid double-counting.
                init_tokens = list(req.origin_input_ids) + list(req.output_ids)
                # If output_ids doesn't yet contain the bonus, append it.
                if not init_tokens or init_tokens[-1] != int(b):
                    init_tokens.append(int(b))
                self.suffix_draft_builder.start_request(rid, init_tokens)
                self._req_tokens[rid] = init_tokens
            current_tokens_list.append(self._req_tokens[rid])

        # Per-req draft cap: ctx_len - cur_seq_len - 1 prevents zombie verify
        # at context boundary (see SuffixWorkerV2 for rationale).
        seq_lens_list = mwb.seq_lens.cpu().tolist()
        max_remaining = [max(0, self._max_context_len - sl - 1) for sl in seq_lens_list]
        draft_tokens, draft_lens, valid_mask, scores = (
            self.suffix_draft_builder.query_batch(
                rids=rids,
                current_tokens=current_tokens_list,
                K_minus_1=self.K_suffix - 1,
                device=self.device,
                return_scores=True,
                max_remaining_per_req=max_remaining,
            )
        )
        # Stash for _decode_step_suffix's reuse.
        self._cached_suffix_drafts = (
            self._step_idx,
            draft_tokens,
            draft_lens,
            valid_mask,
            bonus_cpu,
            current_tokens_list,
        )
        return scores

    def _peek_mtp_scores(self, mwb: ScheduleBatch) -> Optional[List[float]]:
        """Read MTP's confidence for the next draft token from prev step's
        EagleDraftInput.topk_p[:, 0]. Free byproduct of EAGLE V2's keep-up;
        cost is one [bs]-float D2H copy.

        Returns None on cold-start (no warm Eagle state) — selector treats
        that as 'no MTP confidence'.
        """
        spec_info = mwb.spec_info
        if not isinstance(spec_info, EagleDraftInput):
            return None
        tp = getattr(spec_info, "topk_p", None)
        if tp is None or tp.numel() == 0 or tp.ndim < 2:
            return None
        try:
            col0 = tp[:, 0].detach().cpu().tolist()
        except Exception:
            return None
        return col0

    # ------------------------------------------------------------------
    # SUFFIX backend
    # ------------------------------------------------------------------
    def _decode_step_suffix(
        self, mwb: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        """SUFFIX target verify with optional MTP keep-up.

        Builds an EagleVerifyInput from the arctic suffix tree (linear chain
        at K), runs the EAGLE V2 verify scaffold, then optionally fires MTP
        keep-up to populate next_draft_input.topk_p/hidden_states so a
        subsequent MTP step has warm state.

        on_publish (overlap fence) fires right after verify, BEFORE the MTP
        keep-up — same placement as EAGLEWorkerV2 (publish before its
        draft-extend tail) so the scheduler can prepare the next batch while
        keep-up runs.

        Tokens-context: read req.origin_input_ids + req.output_ids each call.
        In overlap mode this lags realized accepts by <= 1 step; the
        post-verify hook in forward_batch_generation pushes the missing
        accept_lens-1 tokens back into arctic to keep the tree exact.
        """
        rids = [req.rid for req in mwb.reqs]
        bs = len(rids)

        # Reuse drafts from _peek_suffix_scores if this step already queried
        # arctic. Else do the full peek inline (used when selector is in
        # force/debug mode and skipped the peek phase).
        cached = self._cached_suffix_drafts
        if cached is not None and cached[0] == self._step_idx:
            _, draft_tokens, draft_lens, valid_mask, bonus_cpu, current_tokens_list = (
                cached
            )
            self._cached_suffix_drafts = None  # consume
        else:
            # FORCE mode bypassed peek. Use authoritative _req_tokens (kept
            # in sync by post-verify hook). Cold-start init same as peek.
            bonus_cpu = mwb.spec_info.bonus_tokens.cpu().tolist()
            current_tokens_list = []
            for req, b in zip(mwb.reqs, bonus_cpu):
                rid = req.rid
                if rid not in self._req_tokens:
                    init_tokens = list(req.origin_input_ids) + list(req.output_ids)
                    if not init_tokens or init_tokens[-1] != int(b):
                        init_tokens.append(int(b))
                    self.suffix_draft_builder.start_request(rid, init_tokens)
                    self._req_tokens[rid] = init_tokens
                current_tokens_list.append(self._req_tokens[rid])
            # Per-req draft cap to prevent zombie verify at context boundary.
            seq_lens_list = mwb.seq_lens.cpu().tolist()
            max_remaining = [
                max(0, self._max_context_len - sl - 1) for sl in seq_lens_list
            ]
            draft_tokens, draft_lens, valid_mask = (
                self.suffix_draft_builder.query_batch(
                    rids=rids,
                    current_tokens=current_tokens_list,
                    K_minus_1=self.K_minus_1,
                    device=self.device,
                    max_remaining_per_req=max_remaining,
                )
            )

        if self._debug and self._step_idx % 10 == 0:
            import logging

            logger = logging.getLogger("sglang.spec.hybrid_v2")
            bonus_cpu = mwb.spec_info.bonus_tokens.cpu().tolist()
            drafts_cpu = draft_tokens.cpu().tolist()
            last_tokens = [t[-5:] for t in current_tokens_list[:2]]
            logger.info(
                "[HYBRID_V2_SUFFIX] step=%d bs=%d K=%d ctx_lens=%s draft_lens=%s "
                "bonus=%s drafts[0]=%s ctx_tail=%s arctic_active=%d",
                self._step_idx,
                bs,
                self.K,
                [len(t) for t in current_tokens_list],
                draft_lens.cpu().tolist(),
                bonus_cpu[:4],
                drafts_cpu[0] if drafts_cpu else None,
                last_tokens,
                len(self.suffix_draft_builder.cache.active_requests),
            )

        # 3. Build EagleVerifyInput. capture_hidden_mode=FULL so the target
        #    forward emits hidden_states for the MTP keep-up.
        bonus_tokens = mwb.spec_info.bonus_tokens
        verify_input = build_suffix_v2_eagle_verify_input(
            bonus_tokens=bonus_tokens,
            suffix_draft_tokens=draft_tokens,
            seq_lens=mwb.seq_lens,
            seq_lens_cpu=mwb.seq_lens_cpu,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        mwb.spec_info = verify_input

        # 4. Run EAGLE V2's verify scaffold (handles prepare_for_v2_verify,
        #    target forward, sample, fill_bonus_tokens, next_draft_input).
        batch_output = self.verify(mwb)
        # Overlap fence at verify-end, before the keep-up tail (mirrors
        # EAGLEWorkerV2's publish before draft-extend).
        if on_publish is not None and batch_output.new_seq_lens is not None:
            on_publish(batch_output.new_seq_lens)

        # 5. MTP keep-up. V2 path: slice predict/hidden/out_cache_loc to
        # K_target per req, call super's _draft_extend_for_decode which
        # uses uniform K_target chains. V1's variable-length flat-accept
        # approach doesn't fit V2's DSv4 kernel signatures (the fused
        # K-norm-rope kernel binds B from kv.size(0) and requires positions /
        # out_loc to share that B).
        self._run_keepup_after_suffix_v2(mwb, batch_output)
        ndi = batch_output.next_draft_input
        if ndi.topk_p is None:
            # Fallback: keep-up didn't run or failed silently. Fill placeholder
            # zeros for FutureMap.
            bs_actual = int(ndi.bonus_tokens.numel())
            ndi.topk_p = torch.zeros(
                (bs_actual, 1), device=self.device, dtype=torch.float32
            )
            ndi.topk_index = torch.zeros(
                (bs_actual, 1), device=self.device, dtype=torch.int64
            )
        if ndi.hidden_states is None:
            target_hidden = batch_output.logits_output.hidden_states
            if target_hidden is None:
                # Defensive: shouldn't happen with FULL capture_hidden_mode.
                hidden_dim = self._target_worker.model_runner.model_config.hidden_size
                ndi.hidden_states = torch.zeros(
                    (bs, hidden_dim),
                    device=self.device,
                    dtype=torch.bfloat16,
                )
            else:
                ndi.hidden_states = torch.zeros(
                    (bs, target_hidden.shape[-1]),
                    device=target_hidden.device,
                    dtype=target_hidden.dtype,
                )

        return batch_output

    # ------------------------------------------------------------------
    # Helpers (shared with future SUFFIX/NONE paths)
    # ------------------------------------------------------------------
    def _initialize_new_reqs(self, mwb: ScheduleBatch) -> None:
        for req in mwb.reqs:
            rid = req.rid
            if rid in self._req_tokens:
                continue
            prompt = list(req.origin_input_ids)
            self.suffix_draft_builder.start_request(rid, prompt)
            self._req_tokens[rid] = prompt

    # ------------------------------------------------------------------
    # NONE backend: trivial K=1 target decode, no spec
    # ------------------------------------------------------------------
    def _decode_step_none(
        self, mwb: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        """Plain K=1 target decode — selector picks this when spec overhead
        exceeds savings (typically at large bs).

        Builds an EagleVerifyInput with K=1 (just bonus, no spec slots).
        super.verify(mwb) dispatches to model_runner._forward_raw whose
        width-based runner pick lands on the baseline_chain runner
        (captured at K=1). accept_lens is always 1 (only bonus accepted).

        No MTP keep-up — NONE means we're skipping spec for this step. If
        the next step's selector picks MTP it sees no warm Eagle state and
        gets promoted back to SUFFIX by the cold-start guard in
        forward_batch_generation. Placeholder zeros for
        next_draft_input.{topk_p, topk_index, hidden_states} keep
        FutureMap's store_to_map happy.
        """
        bs = len(mwb.reqs)
        bonus_tokens = mwb.spec_info.bonus_tokens

        # K=1 chain: bonus only, no spec tokens. K_minus_1 = 0 → empty (bs, 0).
        empty_draft = torch.empty((bs, 0), dtype=torch.int32, device=self.device)
        verify_input = build_suffix_v2_eagle_verify_input(
            bonus_tokens=bonus_tokens,
            suffix_draft_tokens=empty_draft,
            seq_lens=mwb.seq_lens,
            seq_lens_cpu=mwb.seq_lens_cpu,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        mwb.spec_info = verify_input

        batch_output = self.verify(mwb)
        # Overlap fence at verify-end (no keep-up tail on the NONE path).
        if on_publish is not None and batch_output.new_seq_lens is not None:
            on_publish(batch_output.new_seq_lens)

        # Fill placeholder zeros for FutureMap (no keep-up, no real MTP state).
        ndi = batch_output.next_draft_input
        if ndi.topk_p is None:
            bs_actual = int(ndi.bonus_tokens.numel())
            ndi.topk_p = torch.zeros(
                (bs_actual, 1), device=self.device, dtype=torch.float32
            )
            ndi.topk_index = torch.zeros(
                (bs_actual, 1), device=self.device, dtype=torch.int64
            )
        if ndi.hidden_states is None:
            target_hidden = batch_output.logits_output.hidden_states
            if target_hidden is not None:
                bs_actual = int(ndi.bonus_tokens.numel())
                ndi.hidden_states = torch.zeros(
                    (bs_actual, target_hidden.shape[-1]),
                    device=target_hidden.device,
                    dtype=target_hidden.dtype,
                )

        return batch_output

    # ------------------------------------------------------------------
    # SUFFIX → MTP keep-up (V2-style, uniform K_target per req)
    # ------------------------------------------------------------------
    def _run_keepup_after_suffix_v2(self, mwb: ScheduleBatch, batch_output) -> None:
        """Uniform-K_target keep-up via super's _draft_extend_for_decode.

        Approach:
          1. Slice predict + hidden + out_cache_loc from K_src-wide verify
             to first K_target positions per req (= bs * K_target each).
          2. Cap accept_lens to K_target for select_index correctness inside
             _draft_extend_for_decode.
          3. Wrap into a synthetic batch_result with the sliced tensors.
          4. Call super's _draft_extend_for_decode under draft contexts.

        super's _draft_extend_for_decode internally:
          - constructs EagleDraftInput w/ num_tokens_per_req = K_target
          - prepare_for_extend_to_fill_draft_kvcache sets
            forward_mode = DRAFT_EXTEND_V2, batch.input_ids = predict
            (now bs * K_target), seq_lens += K_target, extend_seq_lens =
            [K_target for _ in range(bs)]
          - runs draft model forward (cuda graph or eager)
          - select_index = i * K_target + (capped_accept_lens[i] - 1)
          - softmax + fast_topk for next_draft_input.topk_p/topk_index
          - assigns hidden_states from draft output at select_index

        Lossy when accept_lens > K_target — we drop accepted positions
        beyond K_target. Rare in practice (typical accept_lens 1-3).
        """
        if batch_output is None:
            return
        accept_lens = batch_output.accept_lens
        if accept_lens is None or mwb.forward_mode.is_idle():
            return
        target_hidden = batch_output.logits_output.hidden_states
        if target_hidden is None:
            return
        if batch_output.next_token_ids is None or batch_output.next_draft_input is None:
            return

        K_src = self.K_suffix
        K_target = self.K_mtp
        bs = len(mwb.reqs)

        # Slice indices: for each req, take its first K_target positions of
        # the K_src-wide chain. Shape (bs * K_target,) flat.
        # idx = arange(bs)[:, None] * K_src + arange(K_target)[None, :]
        if K_src == K_target:
            # Fast path: no slicing needed.
            sliced_predict = batch_output.next_token_ids
            sliced_hidden = target_hidden
            sliced_out_cache_loc = mwb.out_cache_loc
        else:
            row = torch.arange(bs, device=self.device).unsqueeze(1) * K_src
            col = torch.arange(K_target, device=self.device).unsqueeze(0)
            sliced_idx = (row + col).flatten()
            sliced_predict = batch_output.next_token_ids[sliced_idx]
            sliced_hidden = target_hidden[sliced_idx]
            sliced_out_cache_loc = mwb.out_cache_loc[sliced_idx]

        capped_accept_lens = torch.clamp(accept_lens, max=K_target)

        # Build synthetic batch_result. We need:
        #   - next_token_ids: shape (bs * K_target,)
        #   - logits_output.hidden_states: shape (bs * K_target, hidden_dim)
        #   - accept_lens: capped to K_target
        #   - next_draft_input: SAME object as batch_output.next_draft_input
        #     (so super's _draft_extend_for_decode in-place writes propagate)
        import copy
        from dataclasses import replace

        fake_logits = copy.copy(batch_output.logits_output)
        fake_logits.hidden_states = sliced_hidden

        fake_result = replace(
            batch_output,
            next_token_ids=sliced_predict,
            accept_lens=capped_accept_lens,
            logits_output=fake_logits,
        )
        # Ensure next_draft_input is the SAME object (replace shallow-copies
        # the dataclass; field references are preserved).
        assert fake_result.next_draft_input is batch_output.next_draft_input

        # Patch mwb.out_cache_loc to the sliced version so the draft model's
        # KV write hits bs*K_target slots (matches super's assumption).
        out_cache_loc_backup = mwb.out_cache_loc
        mwb.out_cache_loc = sliced_out_cache_loc

        try:
            with (
                self._draft_worker.draft_tp_context(
                    self._draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self._draft_worker._draft_extend_for_decode(mwb, fake_result)
        finally:
            mwb.out_cache_loc = out_cache_loc_backup

    def _gc_stale_rids(self) -> None:
        threshold = self._step_idx - self._gc_epoch_threshold
        if threshold <= 0:
            return
        for rid in list(self._req_tokens.keys()):
            if self._rid_last_seen.get(rid, 0) < threshold:
                self.suffix_draft_builder.stop_request(rid)
                self._req_tokens.pop(rid, None)
                self._pending_accepted.pop(rid, None)
                self._rid_last_seen.pop(rid, None)
