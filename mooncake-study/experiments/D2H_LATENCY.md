# D2H Policy And Request Latency

P10 measurements now optionally join each native benchmark request to its
validated Mooncake publication. This rechecks the existing direct, KV staging,
teacher staging and combined staging policies after contiguous teacher row
selection stopped uploading/gathering full-vocabulary rows.

## Correct Token Timing

The native SGLang benchmark previously recorded token timing only when decoded
text was nonempty. Special tokens and incomplete byte sequences can produce
empty text while `meta_info.completion_tokens` advances. An initial real run
reported one successful request with zero TTFT and no ITLs; the new detail
validator rejected the phase. Its reported output length was the requested
length, so that old record alone does not establish the actual generated count.

`async_request_sglang_generate` now uses increasing completion counts for TTFT,
ITL and output length. A repeated final usage frame adds no token or ITL; an
empty trailer preserves previously decoded text. Requests with no generated
tokens report zero output tokens. E2E timing still includes final stream frames,
and TPOT remains `(E2E - TTFT) / (output_tokens - 1)`, as in the native aggregate.
Earlier reports retain their original results and client behavior; their latency
numbers cannot be retroactively corrected without original stream events.

## Request Evidence

`--request-details` selects a test-only wrapper around the existing native
client. It assigns unique request IDs, copies the input body, and buffers native
timing fields and SHA256 request identities. It saves no prompt/output text and
writes the records after measurement. Per-request metadata collection still
adds client work and is enabled equally for every policy in this comparison.

After producer exit, the driver validates every published manifest and tensor,
then joins `provenance.trace_id` to these identities. Duplicate/missing request
records, foreign publications, failed or truncated requests, nonfinite timing,
and discrepancies with native aggregate metrics fail the experiment. Reports
include all/published/not-published distributions and the ten worst TTFT/TPOT
requests with start offsets. The final collector also retains the full validated
publication identity set, so saved request records can reproduce every group
metric after Store teardown. Empty groups have no invented zero percentiles.

Publication membership alone is not causal latency attribution. An unsampled
request shares the serving batch with sampled requests; `not_published` can also
include failed or skipped capture. The producer counters and Catalog outcome
must be checked independently. Writer stage timers are aggregate wall time,
not GPU kernel, D2H or RDMA measurements.

## Reproduction

Use `h100-runtime-lock.json`, a local Qwen3-0.6B model and the resident worker
with its matching Python/Mooncake environment. Run each policy sequentially
from the same frozen checkout and choose a fresh output directory each time:

```bash
PYTHONPATH=python:test/registered/unit:test/registered/unit/training_capture \
  python -m unittest test_bench_sglang_streaming test_benchmark_latency -v

python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /models/Qwen3-0.6B --source-revision CHECKOUT_REVISION \
  --output-dir NEW_OUTPUT_DIRECTORY --num-prompts 512 \
  --input-len 16 --output-len 32 --concurrency 8 --capture-slots 16 \
  --ratios 0.1 --repeats 2 --request-details
```

| Policy | Additional Arguments |
| --- | --- |
| Direct | None |
| KV staging | `--kv-d2h-batch-tokens 16 --device-mib 16` |
| Teacher staging | `--teacher-d2h-batch-tokens 16 --device-mib 16` |
| Combined staging | `--kv-d2h-batch-tokens 16 --teacher-d2h-batch-tokens 16 --device-mib 16` |

Each policy uses two off/on/off rounds with overlap, decode graphs, eager
prefill, a real local TCP Store, an HTTP test Catalog and a GPFS recovery journal.
Warmup, startup, post-response drain and Store readback are excluded from client
timing. No tensor observer or profiler runs in the serving benchmark. The
workload is seeded fixed-length random token IDs, not production traffic.

## Policy Results

All four comparisons use frozen source `$LAB/sglang-d2h-latency-v4`, based on
`4433477c5867730fe72aafa96fd8abcb86d5d859`. Producer, driver, native client and
recorder hashes match across policies. Policy order is direct, KV16, teacher16,
then both16. Each completes two off/on/off rounds, for **12,288 timed requests**,
**393,216 output tokens** and **400 validated snapshots** in total.

| Policy | Mean Requests/s | Mean Throughput / Off Bracket | p99 TTFT ms, Rounds 0 / 1 | p99 TPOT ms, Rounds 0 / 1 | Device Staging Bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 75.25 | 83.92% | 67.26 / 59.05 | 3.483 / 3.559 | 0 |
| KV16 | 77.29 | 83.43% | 57.13 / 81.18 | 3.290 / 4.094 | 3,145,728 |
| Teacher16 | 75.56 | 81.49% | 79.27 / 58.80 | 3.495 / 3.423 | 263,168 |
| Both16 | 77.79 | 84.45% | 57.70 / 60.01 | 3.257 / 3.654 | 3,408,896 |

The throughput fraction is the arithmetic mean of the two per-round ratios,
each relative to the mean of its surrounding off phases. It is not a ratio of
pooled means. Combined staging is 3.37% higher in mean absolute throughput than
direct, but only 0.53 percentage points higher against the off brackets. Tail
improvement is inconsistent. Two rounds with a fixed policy order do not
establish significance or a production recommendation; both defaults remain one.

Every enabled phase selects, admits and publishes 50 of 512 requests. All have
identical capture forward counts, 29,491,200 KV bytes, 1,684,800 auxiliary bytes
and 700 payload objects. There are no measured capture failures, admission
backpressure, quarantined slots, Catalog errors or writer stage errors. All
eight actual Prometheus scrapes match producer status. Post-exit tensor
validation covers the complete payload, not just manifest presence.

The direct policy's first off phase has p99 TTFT 540.14ms. Its eight slowest
requests are indices 224-231, starting at 2.477s into measurement, with TTFT
539.82-575.42ms. This is not a first-request-only effect. Another direct off
phase has p99 TPOT 4.35ms and a worst-request ITL of 97.90ms; the combined
policy's first off phase reaches p99 TPOT 4.93ms. Off-bracket throughput drift
reaches 9.45%. These observations leave a serving/environment tail investigation
open; the records do not identify the responsible server operation.

Captured and uncaptured requests can share a slow batch: the direct policy's
first enabled phase has eight of its ten worst TPOTs at indices 120-127, with
both publication classes represented. Published mean TPOT is 2.35/2.38ms versus
1.91/1.93ms for not-published requests in the two direct rounds. These groups
describe association, not an isolated cost per captured request.

## Final Collector Validation

The final frozen source is `$LAB/sglang-d2h-latency-v5`. Compared with v4 it only
adds `published_trace_ids` to the post-readback summary and a JSON replay unit
test. Native streaming, per-request timing and producer code are identical.
The v4 comparison retains group metrics and worst-request identities, but not
the entire publication identity set; arbitrary published groups cannot be
recomputed from its request files alone. The final collector closes that gap.

All **10 final unit methods pass in 0.018s**, covering empty text, batched token
counts, repeated usage frames, final text preservation, zero-token output,
concurrent identity assignment, invalid records, empty groups, native TPOT and
JSON replay. Earlier five/nine-method runs overlap these tests and are not
additional distinct coverage. Five Python files pass Black. Four pass full
Ruff; native `serving.py` has the same 92 baseline findings and clean I/F checks.
No capture production module changes in this measurement patch.

The final collector then completes a separate 64-request-per-phase off/on/off
smoke run with 100% capture: **192 requests and 64 post-exit validated samples**.
An offline replay verifies all three request-file hashes and reproduces every
saved summary field using only those files and the retained publication IDs,
with no inference request. These timings are excluded from the policy comparison.
The final tests therefore validate **464 snapshots** in total.

All eleven worker jobs are terminal: nine complete and the two intermediate
failures described here remain retained. No additional H100 was allocated;
the resident worker resumed its idle load after the last run.

Commands, terminal job results, source/report/request/log hashes, per-round
statistics and limitations are retained in [d2h-latency.json](d2h-latency.json).
These runs do not establish production SLO, RDMA bandwidth, Catalog retention
or trained draft quality.

## Environment Recovery

An intermediate measurement failed when the shared 1 TiB user quota filled;
its truncated report is not a performance result. Before final measurements,
the unused downloaded `sglang_kernel-0.4.4-cp310-abi3-manylinux2014_x86_64.whl`
was backed up from `$LAB/wheels` to the agent-local directory
`/gpfs/user/fuxuanwei/mooncake-lab-archive/wheels`, then removed from shared
storage only after matching SHA256 verification. The backup is 615,071,908
bytes, SHA256 `558bc7035e3c0a795e8c3eca3cc29e189c7d29f1db2471a0a71b9156a8ee7fa1`.
It is not GPU-visible. The active runtime still uses kernel `0.4.6.post1` and
the same locked environment; no installed package changed.
