# Partial-page prefix reuse benchmark

This benchmark compares the same checkout with
`--enable-partial-prefix-reuse` disabled and enabled. It uses SGLang's
`sglang.benchmark.serving` client; the local adapter only constructs a
deterministic token-ID workload with a known partial-page LCP.

The cached topology contains an aligned prefix and a child spanning at least
two pages. A measured request matches the first page of that child, matches
exactly `R` tokens in its next page, and then diverges. No match metadata is
injected.

By default, `suffix_len = page_size + 1`, so baseline and reuse allocate the
same number of private pages. This isolates private-page KV copy cost from
page-boundary allocation effects.

Example:

```bash
python benchmark/radix_cache/run_partial_prefix_serving_matrix.py \
  --model-name qwen3-32b \
  --model-path /path/to/Qwen3-32B \
  --output-root /tmp/partial-prefix-p32 \
  --page-size 32 \
  --configs baseline reuse \
  --concurrencies 1 8 16 \
  --partial-lens 1 8 16 24 31
```

Run page sizes 16, 32, and 64 separately. The matrix driver starts each server,
performs excluded exact-shape warmups, invokes
`bench_partial_prefix_serving.py`, validates cached-token counts, and stores the
official serving metrics as JSONL. Reverse the configuration order for a paired
order-bias check:

```bash
--configs reuse baseline
```

Report request throughput and mean/median/P90/P99 TTFT and E2E latency, together
with the validated cached-token and computed-prefill-token counts.
