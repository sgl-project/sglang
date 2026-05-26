"""ReqHandle: handle for a request submitted via ScriptedRuntime.

Hides the raw ``Req`` so test scripts cannot assert on scheduler
internals — preserves the harness's refactor-safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.runtime import ScriptedRuntime


ReqStatus = Literal["waiting", "running", "finished", "unknown"]


def _wishlist(name: str) -> NotImplementedError:
    """Build the standardised wishlist NotImplementedError for ReqHandle props."""
    return NotImplementedError(
        f"scripted_runtime: {name} is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


@dataclass(frozen=True, slots=True)
class ReqHandle:
    rid: str
    runtime: "ScriptedRuntime"

    @property
    def status(self) -> ReqStatus:
        return self.runtime._lookup_req_status(self.rid)

    # ============================================================
    # Wishlist properties (NotImplementedError stubs).
    # See 2026-05-26-round-5-de-skip-and-api-wishlist.md §5.3.
    # Each stub raises directly without going through runtime to
    # keep the runtime.py surface clean — real implementations will
    # delegate to runtime._lookup_req_* helpers.
    # ============================================================

    # === Lifecycle ===

    @property
    def finished(self) -> bool:
        """True iff the request has completed (stop / length / abort).

        Consumed by: test_abort_during_chunked_prefill (abort),
                     test_engine_fully_idle_after_drain (invariants),
                     test_chunked_prefill_finishes_with_correct_output_len (regression),
                     test_swa_chunked_req_early_return_no_double_free (hybrid_swa).
        """
        return self.runtime._lookup_finished(self.rid)

    @property
    def aborted(self) -> bool:
        """True iff the request was aborted (vs. natural finish).

        Consumed by: test_abort_during_chunked_prefill (abort),
                     test_force_retract_then_abort_same_yield (abort).
        """
        raise _wishlist("aborted")

    @property
    def output_tokens(self) -> List[int]:
        """Decoded output token ids accumulated so far.

        Consumed by: test_chunked_prefill_finishes_with_correct_output_len (regression),
                     test_pp_last_chunk_cross_mb_kv_correctness (pp).
        """
        raise _wishlist("output_tokens")

    @property
    def num_input_tokens(self) -> int:
        """Number of input tokens the engine saw (post-tokenization).

        Consumed by: test_chunked_prefill_input_len_invariant (invariants).
        """
        raise _wishlist("num_input_tokens")

    @property
    def finish_event_count(self) -> int:
        """How many times the engine fired the finalize-finish path for this req.

        Pre-fix double-finalize bugs would push this to >=2; the
        invariant is "exactly 1 after finished == True".

        Consumed by: test_no_double_finalize (regression),
                     test_abort_during_chunked_prefill (abort).
        """
        raise _wishlist("finish_event_count")

    @property
    def finish_reason(self) -> Optional[str]:
        """Reason the req finished: "stop" / "length" / "abort" / None (still running).

        Consumed by: test_finish_reason_length (regression),
                     test_finish_reason_abort (abort).
        """
        raise _wishlist("finish_reason")

    @property
    def error_message(self) -> Optional[str]:
        """Error message if the req failed; None otherwise.

        Consumed by: test_error_message_on_oom (regression).
        """
        raise _wishlist("error_message")

    @property
    def completion_time(self) -> Optional[float]:
        """Monotonic timestamp when the req finished; None if still running.

        Consumed by: test_completion_time_monotonic (regression).
        """
        raise _wishlist("completion_time")

    # === Chunked-state core ===

    @property
    def chunks_done(self) -> int:
        """Number of prefill chunks committed so far for this req.

        Source of truth for chunked-prefill progress assertions.

        Consumed by: test_chunked_oscillation_three_force_retracts (kv_pressure),
                     test_chunked_retract_at_chunk_first_mid_last (kv_pressure),
                     test_chunks_done_monotonic (regression).
        """
        raise _wishlist("chunks_done")

    @property
    def is_chunking(self) -> bool:
        """True iff the req is currently in the middle of chunked prefill.

        Reflects the scheduler's singular ``chunked_req`` slot — True iff
        this rid is the current chunked_req.

        Consumed by: test_is_chunking_transitions (regression),
                     test_chunked_req_slot_ownership (special_case),
                     test_swa_chunked_req_early_return_no_double_free (hybrid_swa).
        """
        return self.runtime._lookup_is_chunking(self.rid)

    @property
    def has_pending_chunk(self) -> bool:
        """Equivalent to ``chunks_done > 0 and not finished``.

        Source of truth is the engine's ``Req.has_pending_chunk`` field,
        not the synthesized expression — keeps tests robust if the
        engine changes the underlying definition.

        Consumed by: test_has_pending_chunk_invariant (invariants).
        """
        raise _wishlist("has_pending_chunk")

    @property
    def pending_middle_outputs(self) -> int:
        """Outputs queued from middle chunks that have not yet been flushed.

        Non-zero only when middle-chunk output stashing is in play.

        Consumed by: test_pending_middle_outputs_drains (regression).
        """
        raise _wishlist("pending_middle_outputs")

    @property
    def inflight_middle_chunks(self) -> int:
        """Number of middle chunks currently in flight for this req.

        Consumed by: test_inflight_middle_chunks_under_pp (pp).
        """
        raise _wishlist("inflight_middle_chunks")

    @property
    def chunked_req_scheduled_last_iter(self) -> Optional[bool]:
        """Per-req snapshot of ``_chunked_req_scheduled_last_iter``.

        Returns the scheduler flag if this req is the current chunked_req;
        ``None`` otherwise (req is not in the chunked slot this iter).

        Consumed by: test_chunked_req_scheduled_last_iter_flag (special_case),
                     test_swa_chunked_req_early_return_no_double_free (hybrid_swa).
        """
        return self.runtime._lookup_chunked_req_scheduled_last_iter(self.rid)

    @property
    def extend_input_len(self) -> int:
        """Current ``extend_input_len`` (chunk size for the next extend).

        Consumed by: test_extend_input_len_tracks_chunk_size (regression).
        """
        raise _wishlist("extend_input_len")

    @property
    def fill_ids_len(self) -> int:
        """Length of ``fill_ids`` (tokens accumulated in the req's buffer).

        Consumed by: test_fill_ids_grows_per_chunk (regression).
        """
        raise _wishlist("fill_ids_len")

    @property
    def remaining_prompt_tokens(self) -> int:
        """Prompt tokens not yet processed by extend.

        Consumed by: test_remaining_prompt_tokens_decreases (regression).
        """
        raise _wishlist("remaining_prompt_tokens")

    # === KV / row / lock ===

    @property
    def kv_pages(self) -> int:
        """Number of KV pages currently held by this req.

        Consumed by: test_kv_pages_returns_to_zero_after_finish (regression),
                     test_chunked_retract_at_chunk_first_mid_last (kv_pressure).
        """
        raise _wishlist("kv_pages")

    @property
    def kv_committed_len(self) -> int:
        """Number of KV positions committed (written) for this req.

        Consumed by: test_kv_committed_matches_chunks_done (invariants).
        """
        raise _wishlist("kv_committed_len")

    @property
    def kv_allocated_len(self) -> int:
        """Number of KV positions allocated (may exceed committed during over-alloc).

        Useful for detecting over-allocation bugs where allocated > committed
        persists past chunk boundaries.

        Consumed by: test_no_kv_over_allocation (regression).
        """
        raise _wishlist("kv_allocated_len")

    @property
    def prefix_indices_len(self) -> int:
        """Length of ``prefix_indices`` (radix prefix match length).

        Consumed by: test_prefix_indices_matches_radix_hit (radix).
        """
        raise _wishlist("prefix_indices_len")

    @property
    def host_hit_length(self) -> int:
        """Radix host (HiCache) prefix hit length, if any.

        Consumed by: test_host_hit_triggers_init_load_back (hicache).
        """
        raise _wishlist("host_hit_length")

    @property
    def row_idx(self) -> Optional[int]:
        """Row-pool index assigned to this req; None if unassigned.

        Consumed by: test_row_idx_assigned_on_admit (regression).
        """
        raise _wishlist("row_idx")

    @property
    def lock_refs(self) -> int:
        """Number of radix lock_refs held by this req.

        Consumed by: test_lock_refs_return_to_zero (regression),
                     test_lock_refs_during_chunked_prefill (regression).
        """
        raise _wishlist("lock_refs")

    @property
    def last_node_id(self) -> Optional[int]:
        """Identity (id) of the radix node the req's prefix terminates at.

        Consumed by: test_last_node_id_matches_radix_warmup (radix).
        """
        raise _wishlist("last_node_id")

    @property
    def init_load_back_count(self) -> int:
        """Number of HiCache init_load_back invocations for this req.

        Consumed by: test_host_hit_triggers_init_load_back (hicache).
        """
        raise _wishlist("init_load_back_count")

    @property
    def cached_tokens(self) -> int:
        """Radix prefix hit length on admit (cached tokens not re-extended).

        Consumed by: test_radix_prefix_hit_skips_extend (radix),
                     test_cached_tokens_reported (regression).
        """
        raise _wishlist("cached_tokens")

    @property
    def cumulative_kv_alloc_bytes(self) -> int:
        """Cumulative KV bytes allocated for this req over its lifetime.

        Useful for tests verifying that retract / re-admit cycles do
        not balloon allocation.

        Consumed by: test_cumulative_kv_alloc_bounded (kv_pressure).
        """
        raise _wishlist("cumulative_kv_alloc_bytes")

    @property
    def total_tokens(self) -> int:
        """``len(origin_input_ids)`` for this req (input prompt length).

        Consumed by: test_total_tokens_matches_prompt_len (regression).
        """
        raise _wishlist("total_tokens")

    @property
    def input_ids_equal_origin(self) -> bool:
        """True iff ``Req.input_ids`` still equals ``Req.origin_input_ids``.

        Pre-fix bugs mutated ``input_ids`` in place; this invariant
        guards against regression.

        Consumed by: test_input_ids_immutable (regression).
        """
        raise _wishlist("input_ids_equal_origin")

    def cached_tokens_snapshot(self) -> Dict[str, int]:
        """Return a breakdown of cached tokens by tier: ``{device, host, storage}``.

        Method, not property — the breakdown changes between calls and
        callers may want to snapshot multiple times.

        Consumed by: test_cached_tokens_breakdown_under_hicache (hicache).
        """
        raise _wishlist("cached_tokens_snapshot")

    # === Logprob ===

    @property
    def logprobs(self) -> Optional[Any]:
        """Captured logprob bundle (real dataclass TBD); None if not requested.

        Will eventually carry input/output token logprob values and the
        per-token top-k bundle.

        Consumed by: test_logprob_capture_smoke (sampling),
                     test_input_token_logprobs_match (sampling).
        """
        raise _wishlist("logprobs")

    @property
    def num_input_logprobs(self) -> int:
        """Number of input-token logprobs captured for this req.

        Consumed by: test_num_input_logprobs_matches_logprob_start_len (sampling).
        """
        raise _wishlist("num_input_logprobs")

    # === Disagg ===

    @property
    def disagg_send_state(self) -> Optional[str]:
        """Current state of the disagg KV-send state machine for this req.

        Consumed by: test_disagg_prefill_per_chunk_kv_send (disagg),
                     test_disagg_retract_resets_send_state (disagg).
        """
        raise _wishlist("disagg_send_state")

    @property
    def start_send_idx(self) -> int:
        """Index of the next KV chunk to send (disagg prefill side).

        Consumed by: test_disagg_prefill_per_chunk_kv_send (disagg).
        """
        raise _wishlist("start_send_idx")

    @property
    def tmp_end_idx(self) -> int:
        """Temporary end index used by overlap-mid-chunk disagg path.

        Consumed by: test_disagg_overlap_mid_chunk_tmp_end_idx (disagg).
        """
        raise _wishlist("tmp_end_idx")

    @property
    def kv_send_events(self) -> int:
        """Count of ``send_kv_chunk`` invocations for this req.

        Consumed by: test_disagg_prefill_per_chunk_kv_send (disagg).
        """
        raise _wishlist("kv_send_events")

    @property
    def kv_send_last_chunk_events(self) -> int:
        """Count of "last chunk" KV-send invocations specifically.

        Consumed by: test_disagg_last_chunk_send (disagg).
        """
        raise _wishlist("kv_send_last_chunk_events")

    @property
    def kv_send_partial_page_events(self) -> int:
        """Count of partial-page KV-send invocations.

        Consumed by: test_disagg_partial_page_send (disagg).
        """
        raise _wishlist("kv_send_partial_page_events")

    @property
    def decode_receive_cancelled(self) -> bool:
        """True iff the decode-side receive was cancelled for this req.

        Consumed by: test_disagg_retract_resets_send_state (disagg).
        """
        raise _wishlist("decode_receive_cancelled")

    @property
    def max_decode_alloc_reusing_len(self) -> int:
        """Max observed ``len(reusing)`` during decode-side alloc for this req.

        Consumed by: test_disagg_decode_alloc_reusing_bounded (disagg).
        """
        raise _wishlist("max_decode_alloc_reusing_len")

    # === Spec (EAGLE) ===

    @property
    def eagle_topk_p_captured(self) -> bool:
        """True iff EAGLE top-k probabilities were captured for this req.

        Consumed by: test_eagle_topk_p_captured (spec).
        """
        raise _wishlist("eagle_topk_p_captured")

    @property
    def eagle_topk_index_captured(self) -> bool:
        """True iff EAGLE top-k indices were captured for this req.

        Consumed by: test_eagle_topk_index_captured (spec).
        """
        raise _wishlist("eagle_topk_index_captured")

    @property
    def eagle_hidden_states_captured(self) -> bool:
        """True iff EAGLE hidden states were captured for this req.

        Consumed by: test_eagle_hidden_states_captured (spec).
        """
        raise _wishlist("eagle_hidden_states_captured")

    @property
    def spec_first_verify_prefix_len(self) -> Optional[int]:
        """Prefix length at the time of the first speculative verify.

        Consumed by: test_spec_first_verify_prefix_len (spec).
        """
        raise _wishlist("spec_first_verify_prefix_len")

    @property
    def spec_verify_count(self) -> int:
        """Number of speculative verify iterations performed for this req.

        Consumed by: test_spec_verify_count (spec),
                     test_spec_eagle_disagg_chunked (spec).
        """
        raise _wishlist("spec_verify_count")

    @property
    def spec_draft_state_cleared(self) -> bool:
        """True iff the spec draft state was cleared after this req finished.

        Consumed by: test_spec_draft_state_cleared (spec).
        """
        raise _wishlist("spec_draft_state_cleared")

    @property
    def spec_accept_rate(self) -> float:
        """Fraction of speculative draft tokens accepted for this req.

        Consumed by: test_spec_accept_rate_smoke (spec).
        """
        raise _wishlist("spec_accept_rate")

    # === SWA ===

    @property
    def swa_stash_double_free_count(self) -> int:
        """Count of stash-gate invariant violations for this req.

        Increments when ``stash_chunked_request`` is called on this req
        with ``Scheduler._chunked_req_scheduled_last_iter == False`` —
        the exact regression the flag was added to prevent. Invariant
        is "stays at 0".

        Consumed by: test_swa_chunked_req_early_return_no_double_free (hybrid_swa).
        """
        return self.runtime._lookup_swa_stash_double_free_count(self.rid)

    @property
    def swa_chunked_early_return_count(self) -> int:
        """Count of SWA early-returns from ``add_chunked_req`` for this req.

        Incremented when hybrid-SWA budget pressure forces
        ``schedule_policy.add_chunked_req`` to early-return without admitting
        the chunked_req to ``can_run_list``. Used by regression tests to
        verify the SWA early-return code path was actually exercised.

        Consumed by: test_swa_chunked_req_early_return_no_double_free (hybrid_swa).
        """
        return self.runtime._lookup_swa_chunked_early_return_count(self.rid)

    @property
    def swa_budget_overflow_count(self) -> int:
        """Count of SWA budget-overflow events for this req.

        Consumed by: test_swa_budget_overflow_guarded (hybrid_swa).
        """
        raise _wishlist("swa_budget_overflow_count")

    @property
    def swa_chunk_cache_first_two_evict_skips(self) -> int:
        """Count of "skipped evict on first two chunks" events for SWA chunk cache.

        Consumed by: test_swa_chunk_cache_first_two_no_evict (hybrid_swa).
        """
        raise _wishlist("swa_chunk_cache_first_two_evict_skips")

    # === HiSparse ===

    @property
    def hisparse_dma_in_flight(self) -> bool:
        """True iff a HiSparse DMA is currently in flight for this req.

        Consumed by: test_hisparse_dma_lifecycle (hisparse).
        """
        raise _wishlist("hisparse_dma_in_flight")

    @property
    def hisparse_staging_buffers_held(self) -> int:
        """Number of HiSparse staging buffers held by this req.

        Consumed by: test_hisparse_staging_buffers_released (hisparse).
        """
        raise _wishlist("hisparse_staging_buffers_held")

    # === LoRA ===

    @property
    def lora_path(self) -> Optional[str]:
        """LoRA adapter path the req was admitted with; None if no LoRA.

        Consumed by: test_lora_drainer_reject_then_retry (lora).
        """
        raise _wishlist("lora_path")

    # === Streaming ===

    @property
    def stream_events(self) -> List[Dict]:
        """List of stream events emitted for this req (each is a dict).

        Consumed by: test_stream_events_in_order (streaming).
        """
        raise _wishlist("stream_events")
