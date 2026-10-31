# A3 MTP3 shared-prefix host KV and metadata integration

## Paired branches

- Framework: [JinyanYi/sglang](https://github.com/JinyanYi/sglang/tree/feat/mtp3-prefix-kv-fused-metadata-20261031).
- Operators: [JinyanYi/sgl-kernel-npu](https://github.com/JinyanYi/sgl-kernel-npu/tree/feat/mtp3-prefix-kv-fused-metadata-20261031).

The framework branch preserves the existing HiSparse/MTP integration history and
adds shared-prefix host payload ownership and Q1/Q4 metadata fixes. The operator
branch retains the tested Q4 partial-chunk tail-mask fix at `6d28bd6`; this
framework change does not require another C++ operator modification.

## Behavior

Host KV payloads use native physical token/page IDs instead of temporary request
slots. Prefill writes at `out_cache_loc`; materialization and CopySFA resolve
source pages through native mappings. Radix cache reference counts and allocator
reuse therefore govern the retained prefix payload. Each request still owns its
own bounded HBM hot-cache residency map.

Accepted-token relocation mirrors the native physical KV move into host storage.
A temporary source snapshot preserves overlap and permutation semantics. Empty
elided device payload buffers are skipped during native moves.

Q4 pre-metadata prepares one request-union row with the full visible length.
Post-metadata accepts native host page tables. Both retain fixed-address graph
outputs. Q1 pre-metadata no longer tests the query-head count against one.

Existing capacity selection remains interval-based rather than a discrete list.
For long Q4 with TopK 2048, use C in [8192, 32640], aligned to 128 tokens. The
operator requires room for the four-route union. Sampled capacities do not
establish exhaustive coverage of every value.

## Validation on 2026-10-31

Environment: A3, 16 dies, TP16/DP1, GLM-5.2 W4A8, BF16 KV, CANN 9.1,
PyTorch 2.10, torch_npu 2.10.0.post6, DeepEP auto, NEXTN 3/1/4 and graph mode.

- 20 CPU regressions passed, including shared-prefix survival after request-slot
  reuse and overlapping accepted-token moves.
- 14 selected real NPU cases passed: Q1/Q4 metadata, eager and graph execution,
  padding changes, reset/rewrite/rollback, registered-host moves and mathematical
  attention references. Capacities 11392, 12288, 16384 and 16768 were sampled.
  This was not an exhaustive run of the parametrized NPU test matrix.
- The explicit distant-answer 65536-token retrieval check passed for cold1/hot2/
  hot4 (7 requests), then cold1/hot16 (17 requests). Each returned the expected
  answer, with acceptance length 3.0. The native control passed the same first
  seven requests. The auxiliary hot16 client's exit code was not retained;
  its complete response summary was retained.
- Separate clean timing and profiling replays completed all 16 requests with
  1024 output tokens each, with client exit code 0 in both runs.
- The profile contains 20 VERIFY markers and 19 complete commit-to-commit
  cycles: 21 target pre-metadata and 78 target post-metadata calls per cycle.
- Repository isort, selected Ruff lint, Ruff formatting and diff checks passed.

Run the CPU regression without an NPU installation:

```bash
python3 test/manual/npu/test_sparse_kv_offload_cache.py
```

Real-device tests are in `test_lim_mtp_manager.py`,
`test_lim_mtp_copy_metadata.py` and `test_lim_pre_metadata_graph.py`. Use an explicitly
assigned idle device and a rebuilt or compatible paired operator package.

## Performance evidence and limits

The workload used identical shared-prefix token IDs, 65536 input tokens and
1024 output tokens, temperature zero, seed 42, an excluded cold warmup and a
separate profiling replay. The client used the generated-shared-prefix token-ID
implementation from community snapshot `80afdc1280`; this feature branch's older
client must not silently replace that input generator. The result-only client
extension retained per-request acceptance lengths after timing.

HiSparse N16/C16, C16384, target/draft graph bucket16: 189.44 output tokens/s,
mean TPOT 56.88 ms, TTFT 20440.83 ms and ITL 56.88 ms. Native N6/C6, target
bucket8 and draft16: 226.28 tokens/s, TPOT 22.77 ms, TTFT 2564.97 ms and ITL
22.77 ms. Both clients completed every request with exit code 0. Completion
counts divided by request verify counts gave acceptance lengths 3.136895 and
3.247357 respectively, including bonus tokens. No end-to-end throughput gain
was established in this comparison.

HiSparse used ratio 0.89, native token pool 98304, chunk 1024, max-prefill 8192 and
prefill-max-requests 1. Native used ratio 0.82 and pool 196608. Process-scoped THP
was disabled. Host payload allocation was about 8435 MiB per target rank for the
98304-token pool. It did not retain the full attention KV payload in HBM.

Each formal request reused 65408/65536 prefix tokens (99.8046875%). The selected
KV C-pool hit rate was not measured. Serial cached-prefill admission introduced
long streaming gaps; client TPOT is not a pure steady-decode cycle measurement.

Mean complete profile cycles were 151.435869 ms for HiSparse BS16 and 72.273250 ms
for native BS6. Assuming exactly three output tokens per step gives 50.478623 ms /
316.965858 tokens/s and 24.091083 ms / 249.054803 tokens/s respectively. These are
conditional profile-derived values, not measured service throughput. Actual
acceptance was not paired to each profile step. Do not multiply system TPS by
TP16, or compare the different-batch operator deltas as isolated offload cost.

These tests do not replace a broad accuracy evaluation. Random-token prompts
can produce greedy sequence differences in both paths; the retrieval check and
operator references establish only the stated validation scope.
