# Loop 3 Draft — Make DS Standalone Actually Serve

## Why this loop exists

Loop 2 (R0–R9) landed the **structural plumbing** for standalone Double Sparsity on DeepSeek-V3.2: typed-union → unified shape adapter, error containment, per-request summary transport, scheduler abort path, Prometheus counters, 150 passing unit tests. Loop 2 hit a circuit breaker at R9 because the remaining items required **hardware verification**, not more unit-test rounds.

This loop closes that gap. The structures from Loop 2 are not actually exercised end-to-end yet. Two structural prerequisites are missing for the DS path to serve real traffic, and we have no benchmark evidence that the path works on the 8×H200 node it's deployed on.

**Anchor:** start from Loop 2 R9 commit (`ba7d55d64`) on `dev/double-sparsity-standalone`. Loop 2's 150 unit tests must continue to pass; this loop adds hardware-level work on top.

## Hard scope — 3 items, no more

The Loop 2 retro identified a 2-vs-6 budget mismatch (1–2 closures per round vs 6 flagged gaps) and monotonic scope expansion as the dominant failure modes. This draft is deliberately small. Defer anything not on this list.

### M1 — Live PageSignatureTable population from the KV-write path
The DS selector reads page signatures to compute channel-mask scores. Loop 2 wired the API surface (`page_signature_write`, `refresh_current_page_signature`, `mark_populated`) but never called these from the real KV-write sites in `nsa_backend.py`. Right now `PageSignatureTable.valid_mask` is all-False at serve time, so the selector either skips DS or scores against stale/zero signatures.

Hook the writes at the existing `set_mla_kv_buffer` call sites in `nsa_backend.py`. Retract entries on KV-free (`req_to_token_pool` deallocation). The signature lifetime must follow the KV lifetime — not a bookkeeping layer on top.

### M2 — Per-request page ownership mask attached to ForwardBatch
The selector currently scores against the global PageSignatureTable without filtering for per-request page ownership. This means a request can be scored against pages belonging to other requests in the same batch. The result is selection that doesn't fall within the request's own KV pages.

Build `sparse_mask: [bs, max_pages]` from `req_to_token_pool.req_to_token`, `req_pool_indices`, `seq_lens`. Attach to `ForwardBatch.sparse_mask`. `retrieve_topk` consumes the mask before argmax.

### M3 — End-to-end DeepSeek-V3.2 (FP8) benchmark with DS enabled
This is the **done** criterion for the loop. Not "unit tests pass." A real serve with:

- `python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2 --enable-double-sparsity --ds-channel-config-path <path> ...`
- `bench_serving` against the launched server with a small but realistic workload (≥ 64 requests, mixed input lengths)
- Compare DS-on vs DS-off: tokens/sec, accept_rate, accept_length, end-to-end latency
- Sanity-check output quality on a few prompts (smoke test, not full eval)

The benchmark output goes into the round summary. If DS-on is slower than DS-off, that's a real result we must understand — not a failure of the loop, but a finding that constrains M1/M2 design.

## Acceptance Criteria

- **AC-1 (M1):** PageSignatureTable.valid_mask transitions False→True for newly written pages within the same forward step. Verified by both (a) a hardware-level test in `test/registered/unit/...` that runs a forward pass and inspects the mask, and (b) the M3 benchmark not crashing on selector reads.
- **AC-2 (M1):** Retraction on KV-free is verified — running 2× the page budget of requests through the same server doesn't leak signatures (mask stays bounded).
- **AC-3 (M2):** `sparse_mask` correctly excludes pages outside the per-request seq range. Verified with a multi-request batch test where requests have disjoint KV regions.
- **AC-4 (M2):** `retrieve_topk` honors `sparse_mask` — picks never land outside the request's KV range. Verified with a kernel-level test.
- **AC-5 (M3):** A successful benchmark run on this node, with numbers committed to the round summary (DS-on vs DS-off TPS, accept_rate, accept_length). DS-on must not crash; quality smoke test must not produce garbage.
- **AC-6 (regression):** All 150 Loop-2 unit tests continue to pass.

## Explicit non-goals (carried over from Loop 2, still deferred)

These were identified in Loop 2 R9 summary and remain deferred. Do not pull them into Loop 3:

- AC-8 captured-path zero-allocation (Triton kernel for value-domain assertions)
- AC-8 wrapper/multi-step backend metadata fixup
- M3-B perturbation negative (fixture redesign)
- Real two-rank TP divergence test (multi-process harness)
- `transform_index_page_table_decode_fast` 2048 hard-assert

If M3 produces evidence that any of these are blocking the benchmark, escalate as a Plan Evolution Log entry — don't silently expand.

## "Done" definition for the loop (single sentence)

A committed round summary showing a successful `bench_serving` run of DeepSeek-V3.2 FP8 with DS enabled on 8×H200, with measurable accept_rate and accept_length numbers, plus a comparison row against DS-disabled.

## Carry-forward lessons from Loop 2

- **BL-20260520-read-fields-before-abort-mutation**: capture batch-wide cursor spans BEFORE invoking abort helpers (`set_finish_with_abort` rewrites `req.origin_input_ids = [0]`).
- **BL-20260520-symbol-vs-test-fixture-drift**: test fixtures must reference live dataclass field names (use `forward_batch.rids` not `req_ids`; verify with `dataclasses.fields(ForwardBatch)`).
- Loop 2 R8→R9 regression cycle: the unit-test loop missed real-fixture bugs. This loop counts a unit test as **necessary but not sufficient** — the M3 benchmark is the actual signal.

## Hardware available for this loop

- Pod 1 (this node, rank-0): 8× H200, `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2` available.
- Pod 2 (rank-1, reachable via `ssh double-sparsity`): another 8× H200, same model storage.
- Both pods are in sync on `dev/double-sparsity-standalone @ ba7d55d64`.
- For Loop 3 M1+M2+M3 a single 8×H200 node is sufficient; multi-node work stays out of scope.

## Files of interest (so plan generation doesn't re-derive them)

- DS package: `python/sglang/srt/layers/attention/double_sparsity/`
- Page signature API (already exists, just unused at serve time): `python/sglang/srt/layers/attention/double_sparsity/page_signature_write.py`
- KV-write sites in NSA backend: `python/sglang/srt/layers/attention/nsa_backend.py` — search for `set_mla_kv_buffer` (Codex flagged ~L1383-1388, ~L1583-1588, ~L2108-2110 in Loop 2 review)
- DS selector + adapter: `python/sglang/srt/layers/attention/double_sparsity/selector.py`, `page_table_adapter.py`
- DeepSeek-V3 attention seam: `python/sglang/srt/models/deepseek_v2.py::DeepseekV2AttentionMLA._select_topk_indices`
- ForwardBatch dataclass (for sparse_mask attachment): `python/sglang/srt/model_executor/forward_batch_info.py`
- Existing 150-test suite: `test/registered/unit/layers/attention/test_double_sparsity_unit.py`
- Benchmark entry: `python/sglang/bench_serving.py`

## RLCR loop configuration

- **Anchor base branch:** `loop3-base` (to be created at `ba7d55d64`)
- **Working branch:** `dev/double-sparsity-standalone` (continues from R9)
- **Plan budget cap (advisory):** if a round closes < 1 AC and opens > 1 new gap, that's a stagnation signal — escalate after 2 such rounds, don't wait for the budget-9 circuit breaker like Loop 2 did.
- **Round budget:** ≤ 12 rounds (Loop 2 was 42 budgeted, hit stagnation at 9). With 3 ACs and tighter scope, 12 is generous.
