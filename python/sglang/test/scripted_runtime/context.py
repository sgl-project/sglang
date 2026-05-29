"""ScriptedContext: the object a test script drives.

Passed to the caller-provided script generator as its single argument
(``def my_script(t: ScriptedContext)``). Exposes the script-facing verbs
— submit requests, inject pressure, query scheduler state — and reaches
the live ``Scheduler`` through its :class:`ScriptedSchedulerHook`. The
hook owns the generator stepping and the scheduler-side lookups; this
object owns everything the script itself calls.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Union

from sglang.srt.managers.io_struct import (
    AbortReq,
    ContinueGenerationReqInput,
    FlushCacheReqInput,
    PauseGenerationReqInput,
)
from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )

logger = logging.getLogger(__name__)

START_REQ_ARRIVAL_TIMEOUT_S: float = 60.0


class ScriptedContext:
    """Script-facing control surface, created by :class:`ScriptedSchedulerHook`.

    Holds a back-reference to the hook (and through it the live
    ``Scheduler``) plus the script-side request bookkeeping. Every method
    here is called from the test script on the driver rank.
    """

    def __init__(
        self,
        *,
        scheduler_hook: "ScriptedSchedulerHook",
        tokenizer_recv_proxy: Optional["ScriptedTokenizerRecvProxy"],
    ) -> None:
        self._scheduler_hook = scheduler_hook
        self._scheduler = scheduler_hook._scheduler
        self._is_driver = scheduler_hook._is_driver
        self._tokenizer_recv_proxy = tokenizer_recv_proxy

        self._req_handles: dict[str, ScriptedReqHandle] = {}
        self._req_counter = 0
        self._http_threads: List[threading.Thread] = []

    @property
    def _generate_url(self) -> str:
        host = self._scheduler.server_args.host
        port = self._scheduler.server_args.port
        return f"http://{host}:{port}/generate"

    # ============================================================
    # Public API: called from test scripts on the driver rank.
    # ============================================================
    def start_req(
        self,
        *,
        prompt_len: int,
        max_new_tokens: int = 8,
        rid: Optional[str] = None,
        # === Wishlist kwargs (additive, see 2026-05-26-round-5-de-skip-and-api-wishlist.md §5.2) ===
        prompt_tokens: Optional[List[int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        ignore_eos: bool = False,
        min_new_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        return_logprob: bool = False,
        top_logprobs_num: Optional[int] = None,
        logprob_start_len: Optional[int] = None,
        priority: Optional[int] = None,
        lora_path: Optional[str] = None,
        session_id: Optional[str] = None,
        dp_rank: Optional[int] = None,
        return_hidden_states: bool = False,
        grammar: Optional[str] = None,
        stream: bool = False,
    ) -> ScriptedReqHandle:
        """Submit a synthetic request via a real ``/generate`` HTTP call.

        Fires the request asynchronously (a background thread streams and
        discards the response, so the scheduler never blocks on it), then
        drains the wrapped ``recv_from_tokenizer`` socket until the request
        carrying this ``rid`` has been buffered by the proxy. The request
        has arrived but is not yet handed to the scheduler — it is popped on
        the next ``yield`` (next ``recv_requests`` iteration), keeping each
        step deterministic.

        The first three parameters (``prompt_len``, ``max_new_tokens``,
        ``rid``) are fully implemented. All other keyword-only parameters
        are wishlist additions described in
        ``2026-05-26-round-5-de-skip-and-api-wishlist.md`` §5.2 — passing
        any of them to a non-default value raises ``NotImplementedError``
        until the engine-side wiring lands.
        """
        assert self._is_driver, "start_req is only callable from the driver rank"
        _check_start_req_wishlist_kwargs(
            prompt_tokens=prompt_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            ignore_eos=ignore_eos,
            min_new_tokens=min_new_tokens,
            stop=stop,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            return_logprob=return_logprob,
            top_logprobs_num=top_logprobs_num,
            logprob_start_len=logprob_start_len,
            priority=priority,
            lora_path=lora_path,
            session_id=session_id,
            dp_rank=dp_rank,
            return_hidden_states=return_hidden_states,
            grammar=grammar,
            stream=stream,
        )
        if rid is None:
            rid = f"scripted-{self._req_counter}-{uuid.uuid4().hex}"
            self._req_counter += 1

        self._post_generate_async(
            rid=rid, prompt_len=prompt_len, max_new_tokens=max_new_tokens
        )
        self._tokenizer_recv_proxy.wait_until_rid_arrived(
            rid, timeout_s=START_REQ_ARRIVAL_TIMEOUT_S
        )

        handle = ScriptedReqHandle(rid=rid, scheduler_hook=self._scheduler_hook)
        self._req_handles[rid] = handle
        return handle

    def pause_generation(self, *, mode: Literal["retract", "in_place"]) -> None:
        """Scripted-only entry point: bypasses HTTP server / tokenizer manager.

        Drives ``scheduler.pause_generation`` directly with the given mode.
        ``mode="abort"`` is intentionally not supported here: the scheduler
        has no abort branch inside ``pause_generation``; use
        :meth:`abort_all` instead.
        """
        assert self._is_driver, "pause_generation is only callable from the driver rank"
        self._scheduler.pause_generation(PauseGenerationReqInput(mode=mode))

    def continue_generation(self, *, torch_empty_cache: bool = False) -> None:
        """Scripted-only entry point: bypasses HTTP server / tokenizer manager.

        Resume after :meth:`pause_generation`. Defaults to skipping the
        empty_cache call to keep scripted runs deterministic.
        """
        assert (
            self._is_driver
        ), "continue_generation is only callable from the driver rank"
        self._scheduler.continue_generation(
            ContinueGenerationReqInput(torch_empty_cache=torch_empty_cache)
        )

    def abort_all(self) -> None:
        """Scripted-only entry point: bypasses HTTP server / tokenizer manager.

        Drives ``scheduler.abort_request(AbortReq(abort_all=True))`` —
        the real abort code path. ``pause_generation(mode="abort")`` does
        not abort in the scheduler today; this is the only way to reach
        the abort branch from a script.
        """
        assert self._is_driver, "abort_all is only callable from the driver rank"
        self._scheduler.abort_request(AbortReq(abort_all=True))

    def get_all_node_hit_counts(self) -> Dict[int, int]:
        """Map every non-root radix node id to its ``hit_count``.

        The chunked self-referencing guard skips ``_inc_hit_count`` on every
        ``cache_unfinished_req`` insert, so only the single non-chunked
        ``cache_finished_req`` insert bumps hit counts — touching each node on
        the committed path exactly once. On an otherwise-empty tree every node
        therefore sits at ``hit_count == 1``; without the guard the early
        prefix nodes would be re-bumped once per chunk and exceed 1.
        """
        hit_counts: Dict[int, int] = {}
        stack = list(self._scheduler.tree_cache.root_node.children.values())
        while stack:
            node = stack.pop()
            hit_counts[node.id] = node.hit_count
            stack.extend(node.children.values())
        return hit_counts

    def get_all_node_lock_refs(self) -> Dict[int, int]:
        """Map every non-root radix node id to its ``lock_ref``.

        A radix node is locked while a req still holds its KV path and
        released (``lock_ref`` decremented back to 0) once that req
        finishes. With no req in flight every node must therefore sit at
        ``lock_ref == 0``; a stash committed on an un-scheduled chunked
        req would ``inc_lock_ref`` a path without a matching release and
        leave a node locked after the engine drains.
        """
        lock_refs: Dict[int, int] = {}
        stack = list(self._scheduler.tree_cache.root_node.children.values())
        while stack:
            node = stack.pop()
            lock_refs[node.id] = node.lock_ref
            stack.extend(node.children.values())
        return lock_refs

    # ============================================================
    # Helpers
    # ============================================================

    def _post_generate_async(
        self,
        *,
        rid: str,
        prompt_len: int,
        max_new_tokens: int,
    ) -> None:
        """Fire a ``/generate`` request in a fire-and-forget background thread.

        The scheduler must never block waiting for the HTTP response, so the
        POST runs on its own daemon thread with ``stream=True`` and the body
        discarded. The request carries the caller-supplied ``rid`` so the
        proxy can match it on the socket. Token id 1 is BOS for most
        tokenizers; any valid token works since the harness does not validate
        decode quality.
        """
        # Local import keeps ``requests`` off the production scheduler load
        # path (this module is imported unconditionally by run_scheduler_process).
        import requests

        url = self._generate_url
        payload = {
            "input_ids": [1] * prompt_len,
            "sampling_params": {"max_new_tokens": max_new_tokens},
            "rid": rid,
            "stream": True,
        }

        def _run() -> None:
            try:
                with requests.post(url, json=payload, stream=True, timeout=600) as resp:
                    for _ in resp.iter_content(chunk_size=8192):
                        pass
            except Exception:  # noqa: BLE001 — fire-and-forget background thread
                logger.exception(
                    "scripted_runtime: /generate request rid=%s failed", rid
                )

        thread = threading.Thread(
            target=_run, name=f"scripted-generate-{rid}", daemon=True
        )
        self._http_threads.append(thread)
        thread.start()

    # ============================================================
    # Wishlist API (NotImplementedError stubs).
    # See 2026-05-26-round-5-de-skip-and-api-wishlist.md §5.1.
    # ============================================================

    # === Req lifecycle control ===

    def abort(self, r: ScriptedReqHandle) -> None:
        """Abort a single in-flight request immediately.

        Drives the engine's abort code path on a deterministic target
        without needing a client-side cancel. The request transitions to
        ``aborted=True``; downstream finalize / KV release must occur
        exactly once.

        Consumed by: test_abort_during_chunked_prefill (abort),
                     test_force_retract_then_abort_same_yield (abort),
                     test_abort_running_decode (regression).
        """
        raise NotImplementedError(
            "scripted_runtime: abort is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def force_retract(self, r: ScriptedReqHandle) -> None:
        """Force a single req to retract immediately, releasing its KV and rolling chunks_done back to 0.

        Independent of KV pressure — this is a deterministic test hook used to drive the retract code
        path without having to engineer a memory exhaustion scenario.

        Consumed by: test_chunked_oscillation_three_force_retracts (kv_pressure),
                     test_force_retract_then_abort_same_yield (abort),
                     test_chunked_retract_at_chunk_first_mid_last (kv_pressure).
        """
        raise NotImplementedError(
            "scripted_runtime: force_retract is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def retract_all(self) -> None:
        """Retract every currently-running request in one call.

        Bulk version of :meth:`force_retract`. Triggers the engine's
        engine-wide retract path; useful for stress / regression tests
        that need a clean slate without shutting down the engine.

        Consumed by: test_retract_all_then_resume (regression),
                     test_retract_all_during_chunked_prefill (regression).
        """
        raise NotImplementedError(
            "scripted_runtime: retract_all is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def pause_retract_all(self) -> None:
        """Invoke ``pause_generation(retract)`` engine-wide.

        Distinct from :meth:`retract_all`: this exercises the pause-style
        retract entry point used by the engine to handle external pause
        signals, not the per-req force_retract path.

        Consumed by: test_pause_retract_all_then_resume (regression).
        """
        raise NotImplementedError(
            "scripted_runtime: pause_retract_all is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def force_preempt(self, *, req: ScriptedReqHandle, by: ScriptedReqHandle) -> None:
        """Manually trigger priority preemption of ``req`` by ``by``.

        Bypasses the priority comparator so tests can drive the preempt
        code path without engineering specific priority value combos.

        Consumed by: test_priority_preempt_during_chunked_prefill (priority),
                     test_priority_preempt_releases_kv (priority).
        """
        raise NotImplementedError(
            "scripted_runtime: force_preempt is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    # === KV / pool pressure injection ===

    def exhaust_kv(self, *, leave_pages: int) -> None:
        """Consume KV pages until only ``leave_pages`` remain free.

        Deterministic KV pressure injection: occupies pages with dummy
        allocations so the next admit / extend hits the OOM branch
        without needing to engineer a real workload.

        Consumed by: test_chunked_retract_at_chunk_first_mid_last (kv_pressure),
                     test_kv_oom_triggers_retract (kv_pressure).
        """
        raise NotImplementedError(
            "scripted_runtime: exhaust_kv is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def exhaust_row_pool(self, *, leave_rows: int) -> None:
        """Consume row-pool entries until only ``leave_rows`` remain free.

        Row-pool analogue of :meth:`exhaust_kv` — drives the row-pool
        starvation branch independently of token-level KV pressure.

        Consumed by: test_row_pool_starvation_blocks_admit (kv_pressure).
        """
        raise NotImplementedError(
            "scripted_runtime: exhaust_row_pool is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def exhaust_lock_refs(self, *, leave_refs: int) -> None:
        """Consume lock-ref capacity until only ``leave_refs`` remain free.

        Lock-ref analogue of :meth:`exhaust_kv` — exercises the lock-ref
        exhaustion branch without needing a deeply-shared radix tree.

        Consumed by: test_lock_ref_exhaustion_blocks_admit (kv_pressure).
        """
        raise NotImplementedError(
            "scripted_runtime: exhaust_lock_refs is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def row_pool_used(self) -> int:
        """Return current row-pool occupancy (number of used rows).

        Read-only counter useful for invariant assertions like
        "row pool occupancy returns to baseline after all reqs finish".

        Consumed by: test_row_pool_occupancy_returns_to_baseline (invariants).
        """
        raise NotImplementedError(
            "scripted_runtime: row_pool_used is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    # === Radix operations ===

    def warmup_radix(self, *, prompt_tokens: List[int]) -> None:
        """Seed the radix cache with ``prompt_tokens`` without dispatching a real req.

        Pre-populates the prefix tree so subsequent :meth:`start_req` calls
        can hit a deterministic cached prefix length.

        Consumed by: test_radix_prefix_hit_skips_extend (radix),
                     test_radix_warmup_then_evict_roundtrip (radix).
        """
        raise NotImplementedError(
            "scripted_runtime: warmup_radix is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def evict_radix(self, *, prompt_tokens: List[int]) -> None:
        """Evict the radix entry that matches ``prompt_tokens`` (if present).

        Inverse of :meth:`warmup_radix`. Used to set up a "warm then
        evict then re-admit" cache-miss scenario.

        Consumed by: test_radix_warmup_then_evict_roundtrip (radix).
        """
        raise NotImplementedError(
            "scripted_runtime: evict_radix is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    # === LoRA ===

    def force_lora_drainer_reject(self, *, adapter: str) -> None:
        """Make the LoRA drainer reject ``adapter`` on its next admit attempt.

        Drives the drainer-reject branch without needing to engineer a
        realistic LoRA cache eviction race.

        Consumed by: test_lora_drainer_reject_then_retry (lora).
        """
        raise NotImplementedError(
            "scripted_runtime: force_lora_drainer_reject is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    # === Engine-wide actions ===

    def flush_cache(self) -> None:
        """Inject an engine-wide cache flush (radix tree + memory pools).

        Reaches every rank through the normal request-broadcast path, so it is
        safe under TP/PP. Like ``start_req``, the flush is visible on the next
        ``yield``, and the scheduler only honors it when the engine is idle
        (no in-flight reqs). ``run`` issues one before each script so a sub-
        script starts from a clean cache.
        """
        assert self._is_driver, "flush_cache is only callable from the driver rank"
        self._tokenizer_recv_proxy.inject(FlushCacheReqInput())

    def trigger_abort_on_waiting_timeout(self) -> None:
        """Simulate the watchdog firing on a stuck waiting-queue entry.

        Drives the watchdog-fire branch without waiting for the real
        timer; deterministic shortcut for the abort-on-timeout path.

        Consumed by: test_abort_on_waiting_timeout (regression).
        """
        raise NotImplementedError(
            "scripted_runtime: trigger_abort_on_waiting_timeout is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def shutdown(self) -> None:
        """Send an engine shutdown signal from inside the script.

        Lets a lifecycle test verify clean shutdown from the scripted
        side without relying on the outer ``execute_scripted_http_server``
        teardown.

        Consumed by: test_engine_shutdown_from_script (lifecycle).
        """
        raise NotImplementedError(
            "scripted_runtime: shutdown is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    # === State queries (read-only) ===

    def chunked_in_flight_count(self) -> int:
        """Return the number of requests currently mid-chunked-prefill.

        A request is counted while it has at least one finished chunk
        but has not yet completed its prefill.

        Consumed by: test_chunked_in_flight_invariant (invariants),
                     test_concurrent_chunked_reqs (multi_req).
        """
        raise NotImplementedError(
            "scripted_runtime: chunked_in_flight_count is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def get_chunked_req_rid(self) -> Optional[str]:
        """Return the rid currently held in ``Scheduler.chunked_req``, or None.

        Reflects the singular "current chunked req" slot in the scheduler;
        useful for special_case assertions about which req owns the slot.

        Consumed by: test_chunked_req_slot_ownership (special_case).
        """
        raise NotImplementedError(
            "scripted_runtime: get_chunked_req_rid is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def is_idle(self) -> bool:
        """Return True if the engine reported IDLE on the current iter.

        Single-iter snapshot — does not imply no waiting/chunked work.

        Consumed by: test_engine_idle_between_reqs (invariants).
        """
        raise NotImplementedError(
            "scripted_runtime: is_idle is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def is_fully_idle(self) -> bool:
        """Return True if engine is idle, has no chunked in-flight, and waiting queue is empty.

        Stronger than :meth:`is_idle` — useful as a quiescence gate
        between phases of a test.

        Consumed by: test_engine_fully_idle_after_drain (invariants),
                     test_chunked_req_slot_ownership (special_case).
        """
        raise NotImplementedError(
            "scripted_runtime: is_fully_idle is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def batch_size(self) -> int:
        """Return current ``running_batch.size()``.

        Consumed by: test_batch_size_under_kv_pressure (regression),
                     test_batch_size_after_retract (regression).
        """
        raise NotImplementedError(
            "scripted_runtime: batch_size is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def batch_composition(self) -> Dict[str, List[str]]:
        """Return a breakdown of current batch by forward-mode role.

        Shape: ``{"prefill": [...rids], "decode": [...rids], "chunked": [...rids]}``.

        Consumed by: test_batch_composition_chunked_plus_decode (multi_req),
                     test_batch_composition_invariants (invariants).
        """
        raise NotImplementedError(
            "scripted_runtime: batch_composition is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def batch_rids(self) -> Set[str]:
        """Return the set of rids currently in the batch (PP cross-mb dedup applied).

        For PP, deduplicates across micro-batches so a req in flight on
        both mbs only appears once.

        Consumed by: test_pp_cross_mb_dedup (pp),
                     test_batch_rids_invariant (regression).
        """
        raise NotImplementedError(
            "scripted_runtime: batch_rids is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def waiting_rids(self) -> Set[str]:
        """Return the set of rids currently in ``waiting_queue``.

        Consumed by: test_waiting_rids_after_kv_pressure (regression).
        """
        raise NotImplementedError(
            "scripted_runtime: waiting_rids is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def running_rids(self) -> Set[str]:
        """Return the set of rids currently in ``running_batch``.

        Consumed by: test_running_rids_after_retract (regression).
        """
        raise NotImplementedError(
            "scripted_runtime: running_rids is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def in_flight_other_mb_rids(self) -> Set[str]:
        """Return rids in micro-batches other than the current iter's mb.

        Non-empty only under PP (``pp_size > 1``); returns empty set
        otherwise.

        Consumed by: test_pp_in_flight_other_mb_visible (pp),
                     test_pp_cross_mb_dedup (pp).
        """
        raise NotImplementedError(
            "scripted_runtime: in_flight_other_mb_rids is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def list_active_reqs(self) -> List[str]:
        """Return rids of all requests the engine still owns.

        Includes waiting + running + chunked + cross-mb. Useful for
        leak-detection invariants.

        Consumed by: test_no_req_leaks_after_drain (invariants).
        """
        raise NotImplementedError(
            "scripted_runtime: list_active_reqs is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def forward_mode(self) -> str:
        """Return current ``ForwardMode`` name (e.g. "EXTEND", "DECODE", "MIXED", "IDLE").

        Consumed by: test_forward_mode_transitions (invariants).
        """
        raise NotImplementedError(
            "scripted_runtime: forward_mode is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def engine_stats(self) -> Dict[str, Any]:
        """Return a dict of internal engine counters and snapshots.

        Open-ended bag (radix hit_count, kv pool_free, mem stats, etc.).
        Specific keys are documented as they land.

        Consumed by: test_engine_stats_snapshot (invariants),
                     test_radix_hit_count_increments (regression).
        """
        raise NotImplementedError(
            "scripted_runtime: engine_stats is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def kv_pool_underflow_count(self) -> int:
        """Return the counter for "release_kv_cache called with token count > committed" near-misses.

        Pre-fix, the abort dual-queue bug would bump this counter; the
        invariant is "count stays at 0".

        Consumed by: test_no_kv_pool_underflow_under_abort (regression).
        """
        raise NotImplementedError(
            "scripted_runtime: kv_pool_underflow_count is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def lock_refs_snapshot(self) -> int:
        """Return the total ``lock_ref`` count summed across all radix nodes.

        Useful as a leak-detection invariant: should return to 0 once
        all reqs finish.

        Consumed by: test_lock_refs_return_to_zero (regression).
        """
        raise NotImplementedError(
            "scripted_runtime: lock_refs_snapshot is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def load_inquirer_num_pending_tokens(self) -> int:
        """Return ``LoadInquirer._get_num_pending_tokens()``.

        Used by router/dispatcher logic; tests verify it tracks the
        actual pending workload.

        Consumed by: test_load_inquirer_pending_tokens (regression).
        """
        raise NotImplementedError(
            "scripted_runtime: load_inquirer_num_pending_tokens is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def load_inquirer_snapshot(self) -> Dict[str, int]:
        """Return the full LoadInquirer dict snapshot.

        Open-ended dict mirroring the load-inquirer fields.

        Consumed by: test_load_inquirer_snapshot (invariants).
        """
        raise NotImplementedError(
            "scripted_runtime: load_inquirer_snapshot is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    # === Scheduler-path observability ===

    def last_admission_path(self) -> Optional[str]:
        """Return which admit branch fired on the last iter.

        One of: "new", "reuse", "chunked_resume", "tree_cache_resume", or
        ``None`` if no admit happened.

        Consumed by: test_admission_path_chunked_resume (special_case),
                     test_admission_path_tree_cache_resume (special_case).
        """
        raise NotImplementedError(
            "scripted_runtime: last_admission_path is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def last_scheduler_path(self) -> Optional[str]:
        """Return which top-level branch was taken in ``get_next_batch_to_run``.

        One of: "idle", "stash", "merge", "admit", ... (open-ended); ``None``
        if no scheduling decision was made on the last iter.

        Consumed by: test_scheduler_path_stash_then_merge (regression).
        """
        raise NotImplementedError(
            "scripted_runtime: last_scheduler_path is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def last_chunked_exclude_set_source(self) -> Optional[str]:
        """Return where the chunked_req_to_exclude set came from on the last iter.

        One of: "chunked_req", "last_batch.reqs", or ``None``.

        Consumed by: test_chunked_exclude_set_source (special_case).
        """
        raise NotImplementedError(
            "scripted_runtime: last_chunked_exclude_set_source is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    # === DP attention ===

    def dp_rank_max_pending(self, rank: int) -> int:
        """Return max pending tokens at the given DP rank.

        Only meaningful when ``dp_size > 1``.

        Consumed by: test_dp_rank_max_pending_balanced (dp_attention).
        """
        raise NotImplementedError(
            "scripted_runtime: dp_rank_max_pending is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )

    def dp_rank_is_idle(self, rank: int) -> bool:
        """Return True if the given DP rank is fully idle.

        Per-rank version of :meth:`is_fully_idle`. Only meaningful when
        ``dp_size > 1``.

        Consumed by: test_dp_rank_idle_invariant (dp_attention).
        """
        raise NotImplementedError(
            "scripted_runtime: dp_rank_is_idle is wishlist — see "
            "2026-05-26-round-5-de-skip-and-api-wishlist.md"
        )


def _check_start_req_wishlist_kwargs(
    *,
    prompt_tokens: Optional[List[int]],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    ignore_eos: bool,
    min_new_tokens: Optional[int],
    stop: Optional[Union[str, List[str]]],
    stop_token_ids: Optional[List[int]],
    repetition_penalty: Optional[float],
    frequency_penalty: Optional[float],
    presence_penalty: Optional[float],
    return_logprob: bool,
    top_logprobs_num: Optional[int],
    logprob_start_len: Optional[int],
    priority: Optional[int],
    lora_path: Optional[str],
    session_id: Optional[str],
    dp_rank: Optional[int],
    return_hidden_states: bool,
    grammar: Optional[str],
    stream: bool,
) -> None:
    """Raise NotImplementedError for any non-default wishlist kwarg.

    Centralised so each wishlist kwarg surfaces with a specific name in
    the error message, making the unimplemented-call site easy to find.
    """
    non_default_optionals: Dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "min_new_tokens": min_new_tokens,
        "stop": stop,
        "stop_token_ids": stop_token_ids,
        "repetition_penalty": repetition_penalty,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "top_logprobs_num": top_logprobs_num,
        "logprob_start_len": logprob_start_len,
        "priority": priority,
        "lora_path": lora_path,
        "session_id": session_id,
        "dp_rank": dp_rank,
        "grammar": grammar,
    }
    for name, value in non_default_optionals.items():
        if value is not None:
            raise NotImplementedError(
                f"scripted_runtime: start_req kwarg '{name}' is wishlist"
            )
    non_default_flags: Dict[str, bool] = {
        "ignore_eos": ignore_eos,
        "return_logprob": return_logprob,
        "return_hidden_states": return_hidden_states,
        "stream": stream,
    }
    for name, flag in non_default_flags.items():
        if flag:
            raise NotImplementedError(
                f"scripted_runtime: start_req kwarg '{name}' is wishlist"
            )
