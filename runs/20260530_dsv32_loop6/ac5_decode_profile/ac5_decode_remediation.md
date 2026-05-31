# AC-5 strict-SLO remediation — decode-throughput first (Round 17)

Codex R16-review Required-Plan steps 2-3: treat the AC-5 strict miss as a **decode-throughput**
problem, profile the DS conc-16 decode hot path, and make the smallest code change to move conc-16
per-request TPS toward ≥ 30 — preserving the ABI lock (`indices.shape[-1] == dsa_index_topk == 2048`).
DS int8 / `mem_fraction_static=0.7` / radix-on, TP=8 node-0 (the verified AC-4/5/7/8 lifted point;
`max_total_num_tokens=396096`, int8 table, `disable_radix_cache=False` — see `get_server_info_ds*.json`).

## 1. Is conc-16 genuinely decode-bound, or a cold-flood artifact?  → GENUINELY decode-bound
The AC-5 directional run was `WARMUP=0` (cold flood), which the report flagged as depressing TPOT. To
separate real decode cost from prefill-interleave, I ran a **closed batch** (N parallel `/generate` of a
4096-token prompt, `ignore_eos`, no new arrivals → a clean decode batch of N, `#queue-req:0`). Server
steady `gen throughput / N` = pure per-request decode TPS (`ds_closed_batch_decode.txt`):

| batch | gen tok/s | per-req TPS | decode step |
|---:|---:|---:|---:|
| 1 | 39.66 | 39.7 | 25.2 ms |
| 8 | ~197 | 24.6 | 40.6 ms |
| 16 | ~278 | **17.4** | 57.6 ms |

conc-16 pure decode = **17.4 TPS/req ≈ the AC-5 cold-flood 17.6** → **not** a cold-flood artifact; a
genuine decode-cost deficit. per-req TPS = 1/step; step ≈ 23 ms fixed + ~2.15 ms/request. To reach 30
TPS/req the step must be ≤ 33.3 ms.

## 2. Where does the decode step go?  → the DS selection over-scans the full context width
The graph-safe decode selection (`retrieve_topk_graph_safe`) scores + top-k's over `max_seq_len`
columns **per layer (×61) per step**, and `ds_graph_state.max_seq_len = req_to_token.shape[1] =
model_config.context_len = 163840` (`dsa_backend.py:846/1161`). So a 4096-token request scores the
**entire 163840-slot context** every layer every step — a ~35× over-scan. The microbench
(`selection_width_microbench.json`, bs=16, seq_len=4096, 61 layers) isolates it:

| scan width | selection ms/step | (topk) | (score+misc) |
|---:|---:|---:|---:|
| 4608 (actual seq) | 8.56 | 6.15 | 2.42 |
| **163840 (real)** | **32.08** | 9.80 | 22.28 |

So **~32 ms of the 57.6 ms decode step is the selection, and ~23.5 ms of that is pure over-scan** (the
score kernel does work proportional to the full width even though only ~4096 positions are valid; it
masks the result but not the work). This is the dominant, removable DS-specific decode cost.

## 3. The fix — a numerically-identical, CUDA-graph-safe score-kernel early-exit
`_logical_score_kernel` (`selection_kernel.py`) now **skips token-blocks entirely past a request's
`seq_len`** (stores -inf and returns, instead of running the per-head signature loads + dot products
for the unused tail). Properties:
- **Bit-identical output** (those positions were already masked to -inf): `verify_early_exit.py` shows
  the selection (`out_indices`, `out_lengths`) is **identical** at width 4608 vs 163840 for layers
  0/7/30/60 → no recall/correctness change; **no flag needed** (transparent optimization).
- **CUDA-graph-safe**: the launch grid is unchanged (fixed at capture); only per-program work shrinks.
  No host sync, no dynamic shape, no ABI-lock touch (`top_k` stays 2048).
- **Preserves AC-8**: no context cap — long sequences still scan their full length (verified: a 64K
  request would scan its own ~70K, not be truncated).
- 281 DS unit tests pass; microbench selection @163840 **32.08 → 12.50 ms/step**.

## 4. End-to-end re-measure (patched, same operating point) — `ds_closed_batch_decode_patched.txt`
| batch | per-req TPS (before → after) | step (before → after) |
|---:|---:|---:|
| 1 | 39.7 → 40.9 | 25.2 → 24.5 ms |
| 8 | 24.6 → **32.6 ✓ (≥30)** | 40.6 → 30.8 ms |
| 16 | 17.4 → **27.1** | 57.6 → **36.9 ms** |

conc-16 decode step fell **20.7 ms** (matching the profiled over-scan savings 19.6 ms) → **27.1 TPS/req
(+56%)**; **conc-8 now passes ≥30**. Coherence smoke unchanged ("The capital of France is" →
" Paris. The capital of the United States" — no degeneration).

## 5. Status & the remaining lever (honest)
conc-16 per-req TPS is **27.1**, still short of the strict 30 (step 36.9 ms vs the 33.3 ms target,
~3.6 ms over). The residual is the **`torch.topk` over-scan**: topk #1 runs over the full 163840-wide
score row (the second topk is over 2048 = cheap). The microbench shows topk over 163840 vs 4608 ≈
+3.6 ms — exactly the remaining gap. Shrinking it is **capture-width-bound** (the captured topk shape is
fixed at 163840) and needs a more involved change — a seq-aware blocked/partial top-k (the
`DSGraphState.scratch_partial_*` buffers exist for this) or a bucketed selection width — **without**
capping context (AC-8). That is the next round's lever; then conc-32/64 TTFT tuning, then the full
AC-5 client re-run (NUM_PROMPTS=320, conc 16/32/64) with exact arrays + a fail-closed verifier.

**This round's mainline progress:** the AC-5 decode bottleneck is localized and the dominant component
(selection over-scan, ~20 ms/step at conc-16) is fixed, measured, and validated — conc-16 17.4 → 27.1
TPS/req, conc-8 now passing. The strict conc-16 ≥30 is not yet reached (the topk over-scan remains), so
AC-5 stays a live mainline blocker (DEC-3).

## Artifacts
- `closed_batch_decode.py` — closed-batch pure-decode profiler (reproducible).
- `ds_closed_batch_decode.txt` / `ds_closed_batch_decode_patched.txt` — before/after decode curves.
- `selection_width_microbench.py` + `.json` — selection cost vs scan width (the over-scan attribution).
- `verify_early_exit.py` — bit-identical equivalence (width 4608 == 163840) + the 32.08→12.50 ms timing.
- `get_server_info_ds.json` / `get_server_info_ds_patched.json` — operating-point sidecars (mem 0.7,
  int8, radix-on, max_total 396096), identical before/after the patch.
