# Optimistic Prefill for PD Disaggregation

## Goal Description

Reduce time-to-first-token (TTFT) in PD disaggregation mode by allowing prefill computation to begin before the bootstrap handshake with the decode server completes. Currently, the prefill server must wait for the decode server to finish preallocation and send back KV indices before any prefill work starts. With optimistic prefill, the prefill server immediately schedules the request for computation, then checks bootstrap readiness after each chunk completes. If bootstrap is not ready, the request's KV is cached to the radix tree, resources are released, and the request is placed back at the head of the waiting queue — where it gets a near-full prefix cache hit on retry and can batch alongside new requests.

A new server argument `--optimistic-prefill-retries <COUNT>` controls the retry budget: how many times we release-and-requeue before giving up and falling back to the normal bootstrap-wait path. COUNT=0 disables the feature (default). No separate pending-bootstrap queue is needed; all retry routing goes through the existing waiting queue.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Default `--optimistic-prefill-retries 0` preserves existing behavior exactly
  - Positive Tests (expected to PASS):
    - Run PD disagg prefill server with `--optimistic-prefill-retries 0` (or omitted); verify requests go through bootstrap queue → waiting queue flow identical to current behavior
    - Compare output tokens, logprobs, and transfer metrics against baseline — they must match
  - Negative Tests (expected to FAIL):
    - A request should NOT enter the waiting queue before bootstrap completes when retries=0

- AC-2: With retries > 0, requests enter prefill without waiting for bootstrap completion
  - Positive Tests (expected to PASS):
    - With `--optimistic-prefill-retries 1`, a new request begins prefill computation while bootstrap is still in progress (KVPoll.Bootstrapping)
    - The prefill forward pass executes and produces logits/next_token_id before bootstrap finishes
  - Negative Tests (expected to FAIL):
    - With retries > 0, a request should NOT be blocked in the bootstrap queue waiting for decode readiness before any prefill work starts

- AC-3: When bootstrap completes during or after prefill, KV transfer proceeds correctly
  - Positive Tests (expected to PASS):
    - Bootstrap completes inline: after a chunk (or 1-token retry extension), bootstrap poll returns WaitingForInput → sender is finalized, all KV is sent, request moves to inflight queue, transfer succeeds
    - Output tokens and logprobs match the non-optimistic baseline for the same input
    - EAGLE/spec metadata (topk_p, topk_index, hidden_states) is correctly transferred when spec decoding is active
  - Negative Tests (expected to FAIL):
    - KV transfer should NOT start before sender.init() is called with valid metadata_buffer_index and decode_prefix_len
    - send_kv_chunk should NOT be called with an uninitialized sender

- AC-4: When bootstrap is NOT ready after a chunk, request is released and requeued via waiting queue
  - AC-4.1: Mid-chunk stop (chunked prefill)
    - Positive Tests (expected to PASS):
      - After first chunk of a multi-chunk request, bootstrap not ready → KV is cached to radix tree via release_kv_cache, request state is reset, request is placed at waiting_queue[0]
      - On retry, PrefillAdder achieves prefix cache hit for the completed chunk, schedules the next chunk alongside other requests
      - Each retry decrements the retry counter
    - Negative Tests (expected to FAIL):
      - After release-and-requeue, the freed request should NOT hold any req_pool_idx or KV cache slots
      - Double-free assertions should NOT fire on retry (kv_committed_freed, kv_overallocated_freed must be reset)
  - AC-4.2: Final-chunk / single-chunk stop
    - Positive Tests (expected to PASS):
      - After full prefill completes and bootstrap not ready → same release-and-requeue. On retry, a 1-token extension runs (prefix cache hit for input_len - 1 tokens). This 1-token overhead is accepted.
      - Each retry decrements the retry counter
    - Negative Tests (expected to FAIL):
      - Output tokens should NOT be appended or grammar.accept_token called before bootstrap is confirmed ready
  - AC-4.3: Retry budget exhaustion
    - Positive Tests (expected to PASS):
      - When retry counter reaches 0, the request is placed back in the bootstrap queue (normal non-optimistic path) with same sender, to wait for bootstrap via the standard flow
      - Request eventually completes normally after bootstrap finishes
    - Negative Tests (expected to FAIL):
      - A request should NOT retry optimistically forever — it must fall back after budget exhaustion

- AC-5: No resource leaks under success, failure, and abort paths
  - Positive Tests (expected to PASS):
    - On successful optimistic prefill + transfer: req_pool_idx, metadata_buffer_index, KV indices, and sender are properly released
    - On bootstrap failure (KVPoll.Failed) at any point: KV cache released, request aborted, resources freed, metrics incremented
    - On request abort while in waiting queue with optimistic sender: sender aborted
  - Negative Tests (expected to FAIL):
    - After any terminal state (success/failure/abort), no dangling req_pool_idx, metadata buffer, or KV indices should remain allocated

- AC-6: Both normal and overlap schedulers work correctly with optimistic prefill
  - Positive Tests (expected to PASS):
    - `event_loop_normal_disagg_prefill`: bootstrap check in process_prefill_chunk and process_batch_result works correctly; release-and-requeue transitions correctly
    - `event_loop_overlap_disagg_prefill`: flag-based deferred release works correctly — process_prefill_chunk sets the stop flag, process_batch_result performs the actual release-and-requeue after result processing
  - Negative Tests (expected to FAIL):
    - In overlap mode, the request's req_pool_idx should NOT be freed in process_prefill_chunk (process_batch_result still needs it)

- AC-7: Unsupported configurations fail fast at startup
  - Positive Tests (expected to PASS):
    - `--optimistic-prefill-retries 1` with `--disaggregation-mode decode` → startup error
    - `--optimistic-prefill-retries 1` with Mamba/GDN hybrid models (detected via state_types/is_hybrid_ssm) → startup error
    - `--optimistic-prefill-retries 1` with SWA models (detected via is_hybrid_swa) → startup error
    - `--optimistic-prefill-retries 1` with pp_size > 1 → startup error
  - Negative Tests (expected to FAIL):
    - These unsupported configurations should NOT silently proceed and fail at runtime

- AC-8: Chunked prefill works correctly with optimistic prefill
  - Positive Tests (expected to PASS):
    - Multi-chunk request: after first chunk, bootstrap check. If not ready, stop chunking (clear chunked_req), release-and-requeue to waiting_queue[0]. On retry, prefix cache hit for completed chunk(s), next chunk scheduled.
    - Middle-chunk KV sends are skipped when sender is not initialized
    - On retry with prefix cache hit, KV from prior chunks is reused without recomputation (except 1 token)
  - Negative Tests (expected to FAIL):
    - Middle-chunk send_kv_chunk should NOT execute when sender is not initialized
    - After stopping a chunked request, the NEXT chunk should NOT be scheduled (chunked_req must be cleared)

- AC-9: Timing and observability are correct for optimistic path
  - Positive Tests (expected to PASS):
    - Optimistic prefill requests have valid timing stats (bootstrap_done_time may be after prefill_finished_time)
    - Optimistic retry events are counted in metrics
    - idle/liveness checks account for optimistic requests in waiting queue (via existing waiting queue length checks)
  - Negative Tests (expected to FAIL):
    - Timing assertions should NOT fire due to bootstrap_done_time being after prefill time

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The implementation includes the `--optimistic-prefill-retries` server argument, waiting-queue-head based retry (no separate pending queue), deferred KV send logic for both normal and overlap schedulers, release-and-requeue with full state reset, flag-based deferred release for overlap mode, proper abort/failure handling for all states, chunked prefill support with mid-chunk stop, return_logprob and EAGLE spec metadata support, startup validation for unsupported configurations, timing/metrics integration, and focused tests.

### Lower Bound (Minimum Acceptable Scope)
The implementation includes the server argument, waiting-queue-head retry for the normal (non-overlap) scheduler, release-and-requeue with state reset, startup validation gating off unsupported models/PP, and basic correctness tests. Overlap scheduler support and spec metadata handling may be simplified.

### Allowed Choices
- Can use: existing `poll_and_all_reduce_attn_cp_tp_group` for TP/CP consensus, existing `release_kv_cache` and `cache_finished_req` for cache management, existing `ReqToMetadataIdxAllocator` for metadata buffer allocation
- Can use: `Req.reset_for_retract()` as the base state-reset for retry, with additional disagg-specific field resets on top
- Can use: request fields to track optimistic state (optimistic flag, remaining retries)
- Can use: waiting_queue[0] insertion for retry (no separate data structure)
- Cannot use: dual membership in bootstrap queue + waiting queue simultaneously
- Cannot use: send_kv_chunk before sender.init()
- Cannot use: Mamba/GDN/SWA/PP with optimistic prefill in v1

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only.

### Conceptual Approach

```
# Modified request flow with optimistic prefill (no separate pending queue)

def _add_request_to_queue(req):
    if optimistic_prefill_retries > 0:
        # Create sender (same as bootstrap_queue.add but skip enqueue)
        create_sender(req)  # capacity check, sender construction, set max_new_tokens=1
        req.optimistic_prefill_remaining = optimistic_prefill_retries
        req.optimistic = True
        self.waiting_queue.append(req)  # skip bootstrap wait
    else:
        self.disagg_prefill_bootstrap_queue.add(req)  # existing path

# --- Mid-chunk bootstrap check (in process_prefill_chunk) ---
def process_prefill_chunk(self):
    if self.chunked_req and self.chunked_req.optimistic:
        maybe_cache_unfinished_req(self.chunked_req, self.tree_cache, chunked=True)
        poll = poll_bootstrap(self.chunked_req)
        if poll == WaitingForInput:
            # Bootstrap ready! Proceed with normal chunked flow
            _finalize_bootstrap(self.chunked_req)
            send_kv_chunk(self.chunked_req)  # send accumulated KV
            # continue chunking normally from here
        elif poll == Bootstrapping:
            if self.enable_overlap:
                # Overlap: can't free yet — process_batch_result still needs req_pool_idx
                self.chunked_req.optimistic_stop = True
                self.chunked_req = None  # prevent next chunk scheduling
            else:
                # Non-overlap: free immediately, requeue
                optimistic_release_and_requeue(self.chunked_req)
                self.chunked_req = None
        elif poll == Failed:
            release_kv_cache(self.chunked_req, self.tree_cache)
            abort(self.chunked_req)
            self.chunked_req = None

# --- Final-chunk bootstrap check (in process_batch_result_disagg_prefill) ---
# For optimistic requests reaching inflight_middle_chunks <= 0:
def handle_optimistic_final_chunk(req, next_token_id, ...):
    poll = poll_bootstrap(req)
    if poll == WaitingForInput:
        # Bootstrap ready! Finalize and proceed normally
        _finalize_bootstrap(req)
        req.output_ids.append(next_token_id)
        maybe_cache_unfinished_req(req, self.tree_cache)
        # ... logprobs, EAGLE metadata, grammar ...
        send_kv_chunk(req, last_chunk=True)
        self.disagg_prefill_inflight_queue.append(req)
    elif poll == Bootstrapping:
        # Not ready — release and requeue WITHOUT appending output
        maybe_cache_unfinished_req(req, self.tree_cache)
        optimistic_release_and_requeue(req)
    elif poll == Failed:
        release_kv_cache(req, self.tree_cache)
        abort(req)

# --- Overlap deferred release (in process_batch_result_disagg_prefill) ---
# For mid-chunk requests with optimistic_stop flag:
def handle_overlap_optimistic_stop(req):
    # req.optimistic_stop was set in process_prefill_chunk
    # Now that batch result is processed, safe to free
    optimistic_release_and_requeue(req)

# --- Release and requeue helper ---
def optimistic_release_and_requeue(req):
    release_kv_cache(req, self.tree_cache)  # cache to radix, free pool
    # Reuse the battle-tested retraction reset (schedule_batch.py:1304-1347)
    # which handles: prefix_indices, last_node, cache_protected_len,
    # kv_committed/allocated_len, kv_committed/overallocated_freed,
    # inflight_middle_chunks, extend_input_len, input logprob temp state,
    # mamba state, SWA state, routed_experts, indexer_topk, etc.
    req.reset_for_retract()
    # Additional disagg-specific resets (not covered by reset_for_retract):
    req.output_ids = array("q")       # retract preserves output_ids; we must clear
    req.start_send_idx = 0
    req.metadata_buffer_index = -1
    req.tmp_end_idx = 0
    req.hidden_states_tensor = None
    req.optimistic_stop = False
    # Do NOT destroy sender — kept for continued bootstrap polling
    req.optimistic_prefill_remaining -= 1
    if req.optimistic_prefill_remaining <= 0:
        # Budget exhausted: fall back to normal bootstrap-wait path
        self.disagg_prefill_bootstrap_queue.queue.append(req)
    else:
        # Still have retries: put at head of waiting queue
        self.waiting_queue.insert(0, req)

# --- Bootstrap finalization (extracted from pop_bootstrapped) ---
def _finalize_bootstrap(req) -> FinalizationResult:
    if metadata_allocator.available_size() == 0:
        return NO_METADATA  # non-terminal
    req.metadata_buffer_index = metadata_allocator.alloc()
    decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
    req.start_send_idx = decode_prefix_len
    num_pages = kv_to_page_num(len(req.origin_input_ids) - decode_prefix_len, page_size)
    req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
    req.time_stats.set_bootstrap_done_time()
    return READY
```

### Overlap Scheduler Detailed Flow

```
# Overlap event loop with optimistic prefill

Iteration N:
  process_prefill_chunk: (no chunked_req yet, or previous request)
  get_new_batch_prefill: creates batch with chunk 1 of optimistic request
  run_batch(batch_N):    chunk 1 runs on GPU
  process_batch_result:  processes PREVIOUS batch result (not chunk 1's)
  → chunked_req = optimistic request (needs more chunks)

Iteration N+1:
  process_prefill_chunk: handles chunked_req (optimistic request)
    → cache chunk 1's KV via maybe_cache_unfinished_req
    → bootstrap check: NOT ready
    → set req.optimistic_stop = True
    → clear self.chunked_req (prevents chunk 2 from being scheduled!)
  get_new_batch_prefill: does NOT include chunk 2 (chunked_req is None)
    → may schedule other requests from waiting queue
  run_batch(batch_N+1):  runs without the optimistic request
  process_batch_result(batch_N, result_N): processes chunk 1's result
    → sees req.inflight_middle_chunks > 0 (middle chunk)
    → sees req.optimistic_stop == True
    → skips KV send (sender not initialized)
    → calls optimistic_release_and_requeue(req)
      → release_kv_cache: cache to radix, free req_pool_idx
      → reset_req_for_retry
      → insert at waiting_queue[0]

Iteration N+2:
  PrefillAdder picks up the request from waiting_queue[0]
    → prefix cache hit for chunk 1's KV
    → schedules chunk 2 (alongside other requests)
  ...repeat bootstrap check...
```

Note: In overlap mode, when we stop in process_prefill_chunk, the NEXT chunk is prevented from being scheduled. The current batch runs without the request. The deferred release happens in process_batch_result of the SAME iteration (processing the PREVIOUS batch's result). This means NO extra chunks run beyond the one being processed — the "one extra chunk" concern does not apply because process_prefill_chunk runs BEFORE get_new_batch_prefill.

Overhead note: The optimistic stop path is actually cheaper than the normal chunked-send path. Normal path calls `send_kv_chunk` which does synchronous `.cpu()` on GPU tensors (heaviest CPU blocker). Optimistic stop skips `send_kv_chunk` entirely and instead calls `release_kv_cache`, which issues GPU work on the schedule stream without CPU sync and does not block the forward stream. No additional overlap dependency resolution is required beyond what already exists in the normal chunked prefill path (`maybe_cache_unfinished_req` already runs, `copy_done.synchronize()` is unavoidable either way).

### Relevant References
- `python/sglang/srt/disaggregation/prefill.py` — PrefillBootstrapQueue, event loops, send_kv_chunk, process_batch_result_disagg_prefill, process_prefill_chunk
- `python/sglang/srt/disaggregation/decode.py` — DecodePreallocQueue, DecodeTransferQueue (unchanged)
- `python/sglang/srt/managers/scheduler.py` — Scheduler._add_request_to_queue, event loop dispatch, is_fully_idle, abort handling
- `python/sglang/srt/server_args.py` — disaggregation server args, _handle_pd_disaggregation validation
- `python/sglang/srt/mem_cache/common.py` — release_kv_cache, maybe_cache_unfinished_req
- `python/sglang/srt/mem_cache/radix_cache.py` — cache_finished_req, cache_unfinished_req
- `python/sglang/srt/managers/schedule_batch.py` — Req class fields, prefix matching
- `python/sglang/srt/observability/req_time_stats.py` — timing fields and stats
- `python/sglang/srt/disaggregation/utils.py` — setup_state_kv_args, state types, poll helpers
- `python/sglang/srt/disaggregation/base/conn.py` — KVPoll states, sender/receiver interfaces

## Dependencies and Sequence

### Milestones

1. **Server Argument and Validation**: Add `--optimistic-prefill-retries` arg and startup validation
   - Phase A: Add the argument to `ServerArgs` with default 0
   - Phase B: Add validation in `_handle_pd_disaggregation()` (mode check, PP check)
   - Phase C: Add model trait validation in scheduler init (state_types, is_hybrid_swa, is_hybrid_ssm)

2. **Core Infrastructure**: Extract helpers and add request state
   - Phase A: Extract `_finalize_bootstrap()` helper from `PrefillBootstrapQueue.pop_bootstrapped()`
   - Phase B: Extract `create_sender()` method from `PrefillBootstrapQueue.add()`
   - Phase C: Add request state fields (optimistic flag, remaining retries, optimistic_stop flag)
   - Phase D: Implement `optimistic_release_and_requeue()` helper leveraging `Req.reset_for_retract()` plus disagg-specific field resets

3. **Optimistic Admission Path**: Modified request routing
   - Phase A: Modify `_add_request_to_queue()` to route optimistic requests directly to waiting queue
   - Phase B: In `process_prefill_chunk`, add bootstrap check for optimistic chunked_req
   - Phase C: In `process_batch_result_disagg_prefill`, add bootstrap check before irreversible side effects for final-chunk optimistic requests

4. **Overlap Scheduler Support**: Flag-based deferred release
   - Phase A: In `process_prefill_chunk` (overlap mode): set optimistic_stop flag, clear chunked_req
   - Phase B: In `process_batch_result_disagg_prefill` (overlap mode): detect optimistic_stop flag on middle-chunk requests, perform deferred release-and-requeue

5. **Abort, Idle, and Metrics**: Observability integration
   - Phase A: Update abort handling to handle optimistic senders in waiting queue
   - Phase B: Add timing fields for optimistic path
   - Phase C: Add retry metrics

6. **Testing**: Correctness verification
   - Phase A: Unit tests for _finalize_bootstrap, create_sender, state reset, release_and_requeue
   - Phase B: Integration tests for end-to-end optimistic prefill (normal + overlap schedulers)
   - Phase C: Edge case tests (abort, failure, chunked prefill, retries=0 baseline, retry budget exhaustion)

Dependencies: M1 → M2 → M3 → M4 → M5 → M6 (largely sequential, M5 can partially overlap M3-M4)

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Add `--optimistic-prefill-retries` to ServerArgs with default 0, add CLI arg registration | AC-1, AC-7 | coding | - |
| task2 | Add startup validation: mode check in _handle_pd_disaggregation, PP check | AC-7 | coding | task1 |
| task3 | Add model trait validation in scheduler init (state_types, is_hybrid_swa, is_hybrid_ssm) | AC-7 | coding | task2 |
| task4 | Extract `_finalize_bootstrap()` helper from pop_bootstrapped() with typed return values | AC-3 | coding | task1 |
| task5 | Extract `create_sender()` from PrefillBootstrapQueue.add() preserving all side effects | AC-2 | coding | task4 |
| task6 | Add request state fields (optimistic, optimistic_prefill_remaining, optimistic_stop) to Req | AC-2, AC-4 | coding | task1 |
| task7 | Implement `optimistic_release_and_requeue()` helper: `release_kv_cache` + `reset_for_retract()` + disagg-specific resets (output_ids, start_send_idx, metadata_buffer_index, tmp_end_idx, hidden_states_tensor, optimistic_stop) + requeue routing | AC-4, AC-5 | coding | task6 |
| task8 | Modify `_add_request_to_queue()` for optimistic admission: create_sender + direct waiting queue | AC-2 | coding | task5, task6 |
| task9 | Modify `process_prefill_chunk`: add bootstrap check for optimistic chunked_req (non-overlap path) | AC-4.1, AC-8 | coding | task7, task8 |
| task10 | Modify `process_batch_result_disagg_prefill`: add bootstrap check BEFORE irreversible side effects for optimistic final-chunk requests | AC-3, AC-4.2 | coding | task9 |
| task11 | Overlap support in `process_prefill_chunk`: set optimistic_stop flag, clear chunked_req | AC-6 | coding | task9 |
| task12 | Overlap support in `process_batch_result_disagg_prefill`: detect optimistic_stop on middle chunks, deferred release-and-requeue | AC-6 | coding | task11 |
| task13 | Handle retry budget exhaustion: fallback to bootstrap queue when remaining=0 | AC-4.3 | coding | task7 |
| task14 | Update abort handling: scan waiting queue for optimistic senders, call abort | AC-5 | coding | task10 |
| task15 | Add timing fields and metrics for optimistic retries | AC-9 | coding | task10 |
| task16 | Analyze existing test infrastructure for PD disagg and identify test patterns | AC-1 through AC-9 | analyze | task1 |
| task17 | Write unit tests for _finalize_bootstrap, create_sender, optimistic_release_and_requeue | AC-4, AC-5 | coding | task15, task16 |
| task18 | Write integration tests for end-to-end optimistic prefill (normal + overlap) | AC-2, AC-3, AC-6, AC-8 | coding | task17 |
| task19 | Write edge case tests (abort, failure, chunked prefill, retries=0 baseline, budget exhaustion) | AC-1, AC-5, AC-7, AC-8, AC-9 | coding | task18 |

## Claude-Codex Deliberation

### Agreements
- Extracting `_finalize_bootstrap()` from `pop_bootstrapped()` is necessary — the bootstrap finalization block (metadata alloc, decode_prefix_len, sender init) must be reusable
- Requests must NOT be in both bootstrap queue and waiting queue simultaneously — risk of double-scheduling
- Deferred KV send is the correct approach — sender.send() requires init() which requires bootstrap completion
- TP/CP consensus must use `poll_and_all_reduce_attn_cp_tp_group()`, matching existing patterns
- Gating PP and hybrid Mamba/GDN/SWA for v1 is appropriate given their separate consensus/state-transfer complexity
- `cache_finished_req` is a terminal operation — retry requires full state reset including kv_committed_freed/kv_overallocated_freed flags
- Metadata buffer exhaustion in `_finalize_bootstrap` must be non-terminal

### Resolved Disagreements
- **Dual-queue membership** (Round 1): Claude initially proposed adding to both bootstrap queue and waiting queue. Codex identified this causes double-scheduling. Resolved: optimistic requests go only to waiting queue via create_sender + direct enqueue.
- **Prefix cache "free reuse"** (Round 1): Claude described it as "free." Codex noted prefix matching caps at input_len-1, so at least one token is recomputed. Resolved: plan notes "near-complete prefix cache hit" with accepted 1-token overhead.
- **Inline vs event-loop polling** (Round 2): Claude proposed inline polling in process_batch_result. Codex noted this fails when traffic stops. Resolved (post-convergence): eliminated separate pending queue entirely. Retry requests go to waiting_queue[0], so the normal scheduling loop handles them.
- **Reset field types** (Round 2): Claude used Python list `[]` for output_ids reset. Codex noted Req uses `array("q")`. Resolved: use correct types.
- **Bootstrap check timing** (Round 2): Claude checked bootstrap AFTER appending output_ids. Codex recommended checking BEFORE irreversible side effects. Resolved: poll bootstrap first, skip output/grammar/logprob mutations if not ready.
- **_finalize_bootstrap return type** (Round 3): Claude used bool. Codex recommended typed outcomes (READY/NO_METADATA/FAILED). Resolved: use enum-like return values.
- **Separate pending-bootstrap queue** (Post-convergence, user feedback): User suggested using waiting_queue[0] insertion instead of a separate pending queue. This eliminates a data structure, keeps GPU utilized (retried requests batch with new requests), and uses the existing radix cache prefix-hit mechanism. Accepted with the tradeoff of 1-token recomputation per retry for fully-computed requests.

### Convergence Status
- Final Status: `converged`
- Rounds: 3 (Claude-Codex) + 1 (user refinement on queue design)
- All architectural decisions are settled. Remaining work is implementation-level.

## Pending User Decisions

- DEC-1: Flag naming
  - Decision Status: User chose `--optimistic-prefill-retries`

- DEC-2: Primary goal
  - Decision Status: User chose TTFT hiding

- DEC-3: v1 model/PP scope
  - Decision Status: User chose gate off Mamba/GDN/SWA + PP

- DEC-4: Return logprob and spec metadata support
  - Decision Status: User chose support both

- DEC-5: Fully-computed request handling
  - Decision Status: User chose accept 1-token overhead (no separate pending queue)

- DEC-6: Overlap scheduler deferred release
  - Decision Status: User approved flag-based approach; noted middle chunks don't have next_token_id (only final chunk does)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Key Implementation Constraints
- All polling must use `poll_and_all_reduce_attn_cp_tp_group()` for TP/CP consensus
- No separate pending-bootstrap queue — all retry routing uses waiting_queue[0] insertion
- The sender is shared between optimistic admission and later finalization — never destroy/recreate on retry
- For retry state reset, leverage `Req.reset_for_retract()` (schedule_batch.py:1304-1347) as the base — it already handles prefix_indices, last_node, cache_protected_len, kv_committed/allocated_len, kv_committed/overallocated_freed, inflight_middle_chunks, extend_input_len, input logprob temp state, mamba/SWA state, routed_experts, etc. On top, reset disagg-specific fields: output_ids (array("q") — retract preserves it; optimistic must clear), start_send_idx, metadata_buffer_index, tmp_end_idx, hidden_states_tensor, optimistic_stop. The caller pattern matches `ScheduleBatch.release_req()` which calls `release_kv_cache()` then `reset_for_retract()`.
- Skipping `send_kv_chunk` in the optimistic path actually reduces CPU overhead vs the normal chunked path, since `send_kv_chunk` contains synchronous `.cpu()` calls that block the CPU thread. The replacement `release_kv_cache` issues GPU work on the schedule stream without CPU sync and does not block the forward stream.
- In overlap mode: process_prefill_chunk sets the stop flag and clears chunked_req, but does NOT free req_pool_idx. The actual release happens in process_batch_result after the batch result has been consumed.
- Middle chunks do not produce next_token_id or logprobs — those only come from the final chunk. This simplifies the overlap deferred-release path for mid-chunk stops.

--- Original Design Draft Start ---

# Optimistic Prefill for PD (Prefill-Decode Disaggregation)

## Overview

The technique is called optimistic prefill. The idea is that for prefill we skip the bootstrap done confirm but directly go to the waiting queue. After finishing the first chunk, we check if bootstrap is done. If not, we call cache_finished_req to insert the kv cache and put the request back to the bootstrap queue.

## Server Argument

We can control it by a server arg like `--optimistic-prefill-limit <COUNT>`. This controls for how many times we skip the bootstrap done check.

- If COUNT is 1, it means after the first chunk if bootstrap is not finished we put it to the bootstrap queue and release cache.
- If COUNT > 1, we put the request aside but not release cache and we check it again after the next result processing time to be moved to transfer queue.
- If COUNT is 0, it means we disable this feature.

## Caveats

1. How to support overlap scheduler
2. For mamba and GDN, how to properly release during chunked-prefill

--- Original Design Draft End ---
