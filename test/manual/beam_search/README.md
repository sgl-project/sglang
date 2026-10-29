# Beam-search regression checks and service benchmark

This benchmark measures the complete `/generate` request against a running
SGLang server. It reports **returned** tokens/s, counting each returned beam once;
this is not model forward tokens/s. It also saves request parameters, responses,
server information, per-round timing, and ordered token/score comparisons.

## Regression tests

With the normal SGLang development dependencies installed:

```bash
python -m pytest -q test/registered/unit/beam_search \
  test/registered/unit/managers/test_scheduler_held_row_admission.py
python -m pytest -q test/registered/beam_search/test_beam_search_regressions.py
```

The GPU test launches a real model with a four-row pool, with overlap both off
and on. Atomic mixed-width batches `[4, 1]`, `[2, 1, 2]`, and `[1, 4]` must drain
successfully, then the server must accept another request. It also probes a real
model token and uses it as a stop token on the final allowed decode step. The
unit tests cover admission through the scheduler's prefill path, held and pending
rows, pipeline limits, per-candidate stop reasons, response text trimming, tail
layout, logits snapshot isolation, and fork/GC ownership.

On a fresh A100 environment after rebasing onto `38ff13ea7928`, the CPU suite
passed 37 tests and 18 subtests, and the real-model GPU test passed its eight
subtests across both overlap modes. Restoring only the base scheduler and
beam-group files made the targeted new regression checks report eight failures
(including subtest failures), reproducing mixed-width admission and final-step
stop errors. These are local/Colab results; upstream CI is separate.

## Reproduce a comparison

Use separate, clean checkouts/environments for the base and candidate, the same
model revision, and an otherwise identical server command. For example:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --dtype bfloat16 --tp-size 1 --page-size 1 \
  --max-running-requests 64 --max-total-tokens 32768 \
  --context-length 2048 --mem-fraction-static 0.6 \
  --attention-backend flashinfer --sampling-backend pytorch \
  --cuda-graph-max-bs-decode 64 --cuda-graph-backend-prefill disabled \
  --random-seed 42 --enable-tokenizer-batch-encode --enable-metrics \
  --decode-log-interval 100 --watchdog-timeout 120
```

Add `--disable-overlap-schedule` for the synchronous scheduler comparison. Run
this client against each server, copying it into the base checkout if needed:

```bash
python test/manual/beam_search/bench_beam_search.py \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --flush-cache --output base.json
# Restart with the candidate server, keeping all other arguments the same.
python test/manual/beam_search/bench_beam_search.py \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --flush-cache --compare base.json --output candidate.json
```

Defaults: eight simultaneous width-4 requests, then sixteen simultaneous requests
with widths `[1, 2, 4, 8] * 4`; all beams returned (`n = beam_width`), 64 output
tokens per beam, `ignore_eos=True`, two warmup rounds and three measured rounds.
Restart each variant twice and aggregate the six measured rounds. Run variants
on the same idle GPU. Prompt encoding, startup, JIT compilation, warmup, and cache
flush are outside the timer. Timing starts before issuing loopback HTTP requests
and ends after reading every response body, including scheduling, forward,
beam/KV operations, detokenization, and JSON transfer. p50 is the median request
latency; p95 uses sorted index `min(N - 1, floor(0.95 * N))`, then each statistic
is aggregated by its median across rounds. These small fixed batches do not
estimate production latency percentiles under a sustained arrival process.

Use `--input-mode text` to include server prompt encoding. Use `--prime-cache`
to flush and prime each prompt before each round. Use `--atomic-batch --best-only`
to force a heterogeneous batch through one HTTP request; its latency scope is
**the whole batch**, so its p50/p95 must not be compared to individual-request
latencies. The API uses a shared `n` for an atomic batch. Compare only identical
workloads, return counts, and generation lengths.

## Historical A100 measurements

[a100-results.json](a100-results.json) contains the 48 measured round records
behind the following table (eight variant/mode/workload combinations, six
samples each), plus their medians. The baseline was
`50d767a4473510b9e16da63f8abfce08800c982a`; the candidate had the production
changes in this PR. These measurements preceded the rebase onto
`38ff13ea7928ef76ce45459c63ba3035dd5b12d9`. They are historical evidence, not a
claim that the entire performance matrix was rerun on the rebased commit.
The latter base changed only an unrelated scheduler initialization argument in
the files touched by this PR. The JSON records the original candidate diff hash.

Environment: one NVIDIA A100-SXM4-40GB, driver 580.82.07, Python 3.12.3,
PyTorch 2.14.1+cu130, FlashInfer 0.7.2, Transformers 5.19.0; pinned Qwen model
revision above, BF16, TP=1. Server settings match the command above. Decode CUDA
graphs were enabled, prefill graphs disabled. The Rust tree extension was not
built, so the Python TreeCore fallback was used in both variants. Results are
not evidence for a Rust-tree build, other models, multiple GPUs, or other GPUs.

| Scheduler | Workload | Returned tokens/s, base → candidate | Change | p50 seconds, base → candidate | p95 seconds, base → candidate |
| --- | --- | --- | --- | --- | --- |
| Synchronous | Uniform | 1444.8 → 1550.7 | +7.3% | 1.415 → 1.318 | 1.416 → 1.319 |
| Synchronous | Mixed | 1636.5 → 2085.5 | +27.4% | 1.458 → 1.836 | 2.342 → 1.837 |
| Overlap | Uniform | 1364.5 → 1495.0 | +9.6% | 1.499 → 1.368 | 1.500 → 1.368 |
| Overlap | Mixed | 1557.6 → 1999.9 | +28.4% | 1.518 → 1.916 | 2.463 → 1.917 |

**Latency tradeoff:** correct mixed-width admission lets more groups run at once.
In this workload that improves aggregate throughput and p95, but p50 becomes
about 26% slower in both scheduler modes. The mixed gains include changed
batching, not just faster beam bookkeeping. Do not interpret them as a universal
latency improvement. Uniform-width gains offer a comparison without this
particular mixed-width admission change. No sustained-load or quality benchmark
was performed.

Separate deterministic-inference checks used raw-text requests, widths 2/4,
eight requests, one warmup and three measured rounds: synchronous throughput
885.3 → 938.3 returned tokens/s (+6.0%), overlap 862.9 → 890.1 (+3.2%). These
are a different workload/configuration and must not be compared to the default
BF16 table to estimate deterministic-mode overhead.

## BF16 correctness and reproducibility boundary

The admission fix intentionally changes batch composition. Default BF16 model
outputs can depend on batch shape and prefix-cache state, even with greedy
selection and a fixed seed. In the historical table, uniform-width ordered beam
outputs matched 48/48 request pairs in each scheduler mode; mixed-width ordered
beam lists matched only 30/96 pairs in each mode. This PR does **not** promise
bitwise-identical BF16 text across different schedules or cache states.

A separate clean-process investigation used 22 configurations and 156 atomic
batch cases on the pinned historical baseline/model:

- The unchanged baseline admitted the mixed workload as 11 + 5 requests at
  pool=64; the admission fix admitted all 16. Raising the unchanged baseline's
  pool to 128 (keeping decode graph maximum=64) reproduced the admission-only
  and full candidate's output, scores, and captured decode choices.
- With cache disabled and unchanged baseline code, changing the batch size
  already changed first-prefill hidden states and logits, before beam remapping
  or GC. With a fixed batch and baseline code, a primed Radix prefix also changed
  outputs; flushing restored the cold result. Disabling cache alone therefore
  does not guarantee schedule-independent BF16 output.
- Four performance changes separately, plus the stop fix, were compared against
  baseline with cache both enabled and disabled: 580/580 request pairs and
  6400/6400 traced decode steps matched. Full candidate versus admission-only
  also matched. Uninstrumented overlap comparisons matched 58/58 request pairs
  for full versus admission-only and 58/58 for full pool=64 versus baseline
  pool=128, with zero score difference.
- Replaying the LM head with the captured hidden state and actual weights
  reproduced the raw BF16 logits. Holding that hidden state fixed across tested
  batch sizes produced stable LM-head results. This localizes observed numerical
  divergence to the backbone before the LM head; it does not identify a specific
  attention/MLP kernel or establish a high-precision whole-model oracle.

No correctness regression from the performance-only changes was observed in
those cases. This is bounded test evidence, not proof of accuracy equivalence.
When token-for-token reproducibility is required, compare both variants with
`--enable-deterministic-inference` on the tested FlashInfer configuration and
verify the resulting server settings. That path uses batch-invariant operations
and disables Radix Cache; it is a different execution configuration. In the
historical deterministic checks, all 808 returned sequences matched in token
IDs and order, with zero difference in available sequence scores. The new GPU
regression test checks structural invariants and stop behavior, rather than
requiring exact default-BF16 output across different batch layouts.
