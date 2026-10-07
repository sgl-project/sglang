# BF16 LiteTopK for DeepSeek-V4.1

The selector builds on [SGLang #40764](https://github.com/sgl-project/sglang/pull/40764)'s
LiteTopK design. This BF16 path runs independently of its standalone FP32 module.
It requires the BF16 histogram producer and matching scheduler from
[sgl-project/DeepGEMM #96](https://github.com/sgl-project/DeepGEMM/pull/96).
The producer core is adapted in #96 from
[deepseek-ai/DeepGEMM #462](https://github.com/deepseek-ai/DeepGEMM/pull/462).


`Bf16Dsv41DecodePlan` selects the exact BF16 top-512 of the decode and target-verify rows of DeepSeek-V4.1's ratio-1/2 index layers
on SM100: the paired DeepGEMM BF16 producer with an exact per-score histogram, then `bf16_exact/select.cuh`.
Exactness is relative to the BF16 logits, which can select different slots from the default FP32 producer.
Ties prefer the lower physical slot. Its output layout is that of
`topk_transform_paged_v2` with page size 128: int32 index-K pool slots, each row's min(length, 512) slots first in no
particular order, then `-1`; a row of at most 512 scores returns all its slots.

```python
import deep_gemm
from sglang.kernels.experimental.litetopk_decode.bf16 import Bf16Dsv41DecodeStorage

plan = Bf16Dsv41DecodeStorage(max_rows, device).plan(rows)
# The arguments of deep_gemm.fp4_paged_mqa_logits_bf16 (flattened rows, a row per
# request or draft token): q = (int8 [rows,1,32,64] packed e2m1, int32 [rows,1,32] packed ue8m0), index-K cache
# uint8 [pages,128,1,68], weights BF16 [rows,32], context_lens int32 [rows,1], block_table int32 [rows,pages].
# indices is int32 [rows]: adjacent equal request IDs share KV loads.
# n is a CPU hint, 1..6. Always pass the same hint and IDs to both calls.
schedule = deep_gemm.get_paged_mqa_logits_bf16_metadata(
    context_lens, 128, deep_gemm.get_num_sms(), indices=indices, tokens_per_request=n)
scores, slots = plan(q=q, kv_cache=kv_cache, weights=weights, context_lens=context_lens,
                     block_table=block_table, schedule_metadata=schedule, max_context_len=max_context_len,
                     indices=indices, tokens_per_request=n)
# or plan.scores(...) now and plan.select(scores=..., context_lens=..., block_table=..., out=...) later
```

The buffers take about 72 KB per row: the 4 KiB histogram, 16 bytes of hand-off state, 8 * `candidate_capacity`
bytes of candidates (64 KiB by default) and a 2 KiB output. Each call leaves every row's histogram and candidates at
zero and its hand-off state balanced, so the plans of a `Bf16Dsv41DecodeStorage` share its memory whatever their row
counts: a plan takes the storage's last `rows` hand-off states and first `rows` candidate lists.

It requires the paired sgl-gemm APIs `get_paged_mqa_logits_bf16_metadata` and
`fp4_paged_mqa_logits_bf16`. The older generic schedule is incompatible: its KV split differs.
SGLang's existing `SparseTableBackend` additionally requires the usual
`get_paged_sparse_mqa_logits_metadata` / `fp8_fp4_paged_sparse_mqa_logits`
extension. Those sparse APIs are already present in the paired sgl-gemm dev baseline.

### Serving hook

`SGLANG_OPT_DSV41_LITETOPK_DECODE=1` sends every `FullTopKIndexer.topk_decode` call and the top-512 of
`SparseTableBackend.publish_decode` through `Bf16Dsv41DecodePlan` (`srt/layers/attention/dsv4/v41_indexer/litetopk.py`).
With candidate filtering (eager decode and the `candidate_filtered` decode graphs) these are the ratio-2 layers and
the top-512 of the candidate source (layer 20 of DeepSeek-V4.1-Flash), whose block selection keeps reading the same
logits. The decode graphs that skip candidate filtering, `candidate_c2_all` (longest request up to 1024 tokens)
and `candidate_unfiltered` (up to 16384 tokens, DSpark verify included), run the candidate consumers through
`FullTopKIndexer.topk_decode` as well, so there the hook serves every index layer that runs, on rows of at most
16384 scores. Prefill is not covered.

At startup the hook probes CUDA SM100 and the paired APIs once; on an unsupported GPU or package
it logs the reason and `topk_transform_paged_v2` stays. Unset, this module is not imported
and the default path does what it did before (its call sites only test whether the hook is present).

- Routing: decode uses the CPU hint 1; target verify uses `spec_info.draft_token_num`.
  Hints 1..4 choose Q4 / three TMEM stages; 5/6 choose Q6 / two TMEM stages; 6 also
  swizzles histogram shared memory. Actual adjacent request-ID runs determine KV
  sharing, including ragged runs. Every row retains its own causal length.
  `init_forward_metadata_in_graph` rebuilds each ratio's paired schedule once per
  forward from live lengths and IDs, shared across layers and recorded in CUDA
  graphs. There is no device-to-host length read or schedule cached from warmup.
- Buffers: every row count is a plan on one `Bf16Dsv41DecodeStorage` sized for the largest batch so far. Graphs keep the
  storage they captured (SGLang captures the largest batch first, so one storage serves them all); a larger batch
  outside capture gets a storage of at least twice the rows, and the old one is freed unless a graph uses it. A row
  count first seen while a graph is captured uses the storage if it fits, otherwise that graph keeps the default
  top-k (logged once). The storage is allocated after SGLang sizes the KV pool and is not counted in
  `mem_fraction_static`: about 72 KB per row of the largest batch, e.g. 105 MiB for DSpark verify of 256 requests
  (6 rows each).
- Left to the default path, logged once: hints outside 1..6, more than 16384 query rows,
  row-chunked logits (very large batches under the logits memory budget),
  raw indices, index pages other than 128 slots, index queries other than next_n 1 with 32 heads of 128 dimensions,
  head weights other than FP32/BF16 `[rows, 32]`, a selection on another device.
  The fused Q packer's FP32 weights have already been rounded to BF16; converting
  them back for the paired API is lossless. Candidate block maxima accept BF16
  directly and emit FP32 keys for the existing block top-k.
- `SGLANG_DEBUG_DSV41_LITETOPK_CHECK=1` also runs SGLang's paged top-k on the same logits and counts on device the
  rows whose slot sets differ. The counts are read back at eager forwards only: every eager prefill, and eager decode
  calls at most every 10 s. Calls replayed by CUDA graphs are counted at the next eager forward, so a final short
  request flushes them; a decode-only (PD) instance whose batches all replay graphs never reports.
  `SGLANG_DEBUG_DSV41_LITETOPK_CHECK_FILE=<path>` appends each report to `<path>.TP<tp>_PP<pp>_DP<dp>_pid<pid>.jsonl`.
  A row can differ legitimately when its 512th score is tied, or when its scores differ only below FP16 resolution,
  where `topk_transform_paged_v2` is not exact.

Limitations: SM100 only, tuned for the 148-SM B200. Page-table rows within each
adjacent same-ID run must be identical, and causal lengths nondecreasing. The
histogram counts every live non-NaN BF16 score once. Output slots are exact for
those scores; this does not assert equality with the default FP32 index selection.

## Validation

`test/registered/unit/layers/attention/test_dsv41_litetopk_decode.py` covers
dispatch, fallback, storage lifetime and metadata forwarding. The dedicated
BF16 GPU suite covers n=1..6, PDL off/on, FullTopK and SparseTable serving,
dynamic graph replay, physical slots, reusable scratch and BF16 candidate maxima:

```sh
SGLANG_TEST_DSV41_BF16_EXACT=1 python -m pytest \
  test/registered/kernels/ops/attention/test_litetopk_decode_dsv41_bf16.py
```

The BF16 producer tests live in the paired DeepGEMM PR.
