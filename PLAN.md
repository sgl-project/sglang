# Double Sparsity Native Decode Pivot

## Summary
Continue on `dev/double-sparsity-v2`; do not restart from scratch. Keep the calibration code, K-label lifecycle, tests, and benchmark harness. Pivot away from the current v2 FA3 page-table sparse path as the main performance path.

Core decisions:
- Use a custom/native DoubleSparse sparse-decode kernel, seeded from PR #22992.
- Do not use FlashInfer generic sparse attention as the final sparse path.
- Change the benchmark goal from single-request-only latency to long-context decode throughput with concurrency sweep.
- Keep concurrency=1 as a reported diagnostic, not the sole pass/fail gate.

## Implementation Changes
Start the fresh Claude code session with:
```bash
git status --short --branch
sed -n '1,360p' benchmark/double_sparsity/HANDOFF.md
git fetch https://github.com/Jiminator/sglang.git dev/double-sparsity-reintro:refs/pr/22992
```

Build a new native DS decode path:
- Add a native sparse decode module under the attention Triton ops area, based on PR #22992’s `flash_decode_sparse_attention_fwd`.
- Wire `--enable-double-sparsity` to use native sparse decode by default for decode once `seq_len >= min_seq_len`.
- Keep dense FA3 for prefill/extend and short-sequence fallback.
- Keep the current K-label side cache from `DoubleSparsityAlgorithm`; do not move back to PR #22992’s KV-pool label-buffer design unless required.

Replace current sparse hot path behavior:
- Bypass `DSFlashAttentionAdaptor` for the native sparse decode path.
- Do not rewrite FA3 page tables per layer in the sparse path.
- Do not call the current stage-2 merge or torch union path in production native decode.
- Selection should output physical token ids directly for sparse attention.

Selection design:
- First specialize for the 70B TP=8 target where `H_kv_local == 1`.
- Score history tokens with `Q_label · K_label`, excluding the current token.
- Always append sink and recent tokens directly.
- Use one selected set per local KV head, shared by the local GQA query heads.
- Start with a simple exact score-kernel plus `torch.topk` path only if it removes the current stage-2/union/page-table overhead immediately.
- If profiling shows selection plus index prep is more than 10% of dense decode TBT or more than 5 ms/token end-to-end at the target benchmark, replace it with a fused streaming/block selector with runtime block count and no constexpr unroll.

Sparse attention requirements:
- Sparse attention must load K/V only for selected physical token ids.
- Runtime must scale with `max_selected`, not full `seq_len`.
- DS-on nsys should not show dense FA3 decode as the active sparse attention kernel.
- Keep `token_budget`, `recent_tokens`, `sink_tokens`, and `min_seq_len` as runtime knobs.

## Benchmark Changes
Change the headline benchmark to a long-context throughput sweep:
- Model: `meta-llama/Llama-3.1-70B-Instruct`
- Hardware: 8x H200, TP=8
- Contexts: 64K and 128K
- Output length: 512 first for iteration speed, then 1024 for final report
- Concurrency sweep: `1, 2, 4, 8, 16`
- Primary pass: DS-on median decode tok/s is at least 1.10x DS-off at one honest long-context concurrency point, with the best point clearly reported.
- Stretch pass: 1.30x DS-on/DS-off.
- Quality guard: NIAH DS-on accuracy must be no worse than DS-off by more than 0.02.

Update `benchmark/double_sparsity` so the comparison can report:
- Best concurrency point by DS-on/DS-off decode tok/s.
- Per-concurrency DS-off, DS-on, speedup, TBT p50/p95, and quality guard.
- A clear note that concurrency=1 is diagnostic if the win occurs at higher concurrency.

## Test Plan
Run existing unit coverage first:
```bash
pytest test/registered/unit/mem_cache/sparsity/ -q
```

Add or update tests for:
- Native selector returns valid physical ids, includes sink/recent, excludes invalid/padded positions, and preserves current decode self-token through recency.
- Native sparse decode matches dense attention on tiny fixtures when selected ids cover the full context.
- Native sparse decode approximates the torch reference when selected ids are top-k only.
- `H_kv_local == 1` fast path avoids stage-2 merge and union.
- Legacy FA3 page-table DS path remains either testable behind a debug flag or explicitly removed from production dispatch.

Performance validation:
- Run synthetic selector profiling at 32K, 64K, 128K.
- Run sparse attention microbench with fixed selected counts `512, 1024, 2048` and seq_len `32K, 64K, 128K`; attention time should be nearly constant across seq_len for fixed selected count.
- Run 70B TP=8 64K sweep first, then 128K sweep.
- Capture one nsys DS-on vs DS-off comparison at the winning point.

## Assumptions
- The branch may freely change implementation and benchmark definitions.
- The headline claim should be “Double Sparsity wins in long-context decode throughput,” not “Double Sparsity improves single-request latency.”
- FlashInfer remains useful as a reference/fallback, but not as the final sparse path because it does not eliminate DS-specific selection and per-layer metadata overhead.
- If the native path fails to beat dense even after removing page-table and union overhead, the next lever is benchmark concurrency/context/token-budget tuning before deeper kernel fusion.
