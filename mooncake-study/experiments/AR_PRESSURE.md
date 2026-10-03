# AR Capture Under KV Pool Pressure

Run the complete file with one CUDA GPU, matching SGLang dependencies, the
Mooncake SDK and `mooncake_master` on `PATH`:

```bash
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_ar_pressure.py -v -f
```

With two or four visible GPUs, run the corresponding distributed suites:

```bash
python test/registered/storage/test_training_capture_ar_pressure_distributed.py -v -f
python test/registered/storage/test_training_capture_ar_pressure_tppp.py -v -f
```

The fixture uses a real TCP Mooncake Store and an HTTP Catalog test double.
Four disjoint 16-token prompts each request 80 output tokens, for 384 total
path tokens against a 256-token KV pool. Scheduling conservativeness is 0.05
so all four requests can start; every individual request fits in the pool.
The test explicitly disables `SGLANG_TEST_RETRACT`. It neither invokes the
pause/retract API nor overrides the allocator's capacity decisions.

The single-rank and TP2 cases cross synchronous/overlap scheduling with
eager/Full decode CUDA graphs. PP2 and TP2/PP2 use synchronous eager and graph
execution. Attention uses Triton, prefill graphs are disabled, and decode
graph buckets are 1/2/4 requests. The model is BF16 Qwen3-0.6B with selected
layers 0/14/27, 128-token prefill chunks, 16-token bounded KV and teacher D2H batches,
64-token Store chunks and four capture reservations.

PP explicitly uses two-request microbatches. Each rank must observe all four
capture contexts active together before retirement, even though they belong
to separate pipeline microbatches. Non-PP serving keeps a four-request batch.

Test-only instrumentation wraps the original `ScheduleBatch.retract_decode`.
Before calling it unchanged, the wrapper records available capacity, required
next-decode capacity, capture lease IDs, committed slots and the most recent
online source frame. Afterward it records the retired requests and whether
their capture context/finalizer has detached. It does not choose which request
to retract or change the serving response.

Admission observations retain each rank's original capture ID before any peer
can invalidate it. Each rank records its own retraction events and source-frame
boundaries. Nonce-tagged responses to the real server-info request expose all
rank-local states; distributed admission waits for both free reservations and
the cohort service's admission-ready vote. No test hook modifies allocation,
retraction decisions, source KV, logits or publication.

The runtime test requires all of the following:

- Available KV capacity is below the real next-decode requirement at every
  observed retraction, and capacity increases after release.
- All pressure requests finish their full response, and every rank's HTTP
  metrics agree with per-request and scheduler-observed retraction counts.
- Each retired request's original capture lease reaches Catalog `FAILED`
  with `request_aborted_or_retracted` for one rank or `cohort_failed` for a
  distributed cohort; none becomes published or starts a second capture after
  resuming. Admission records must agree on request-to-capture IDs across all
  ranks. Each rank records exactly five admissions, including the fresh request.
- Surviving requests publish complete samples. A fresh short request is
  admitted and published after pressure; all four reservations recover with
  zero quarantined Host slots.
- On every rank, a later online source frame from a successful sample uses
  physical KV slots that belonged to a retired request before release.
- Non-PP graph cases execute a three-request decode batch using a captured
  graph; PP graph cases execute both two-request and one-request decode replay.
  Overlap cases observe the pending-result lookahead on every
  TP rank. Only the final PP stage has teacher predictions.
- After the producer exits, a new Store client validates every manifest and
  object digest. KV and raw top-128 values match the online source exactly;
  vocab IDs, full-vocabulary LSE (`rtol=atol=1e-6`), token IDs, masks, positions,
  teacher alignment and KV validity all pass without target recomputation.

This tests normal scheduler retraction under KV pool exhaustion, not a CUDA
allocator exception or a device failure. Source observers intentionally copy
and synchronize tensors, so this is correctness evidence, not a throughput or
latency measurement. Forced token outputs make the expected request boundaries
deterministic and do not measure model quality. Distributed P/D AR pressure,
DP/CP pressure, cross-node RDMA, simultaneous prefill graphs, saturated
Store/Catalog backpressure, production retention and consumer training remain
separate acceptance scopes.

## Verified H100 Results (2026-10-03)

The final frozen source passed all 12 cases:

| Suite | Cases | Scheduling | Elapsed |
| --- | --- | --- | --- |
| Single GPU | 4 | Synchronous/overlap, eager/graph | 187.011s |
| TP2 and PP2 | 6 | TP synchronous/overlap; PP synchronous; eager/graph | 371.960s |
| TP2/PP2 | 2 | Synchronous, eager/graph | 122.741s |

All 60 requests completed, producing 3,888 output tokens. Native pressure
retired 24 capture attempts; 36 complete snapshots remained readable after
producer exit, with 828 tensor objects and 33,032,352 tensor bytes checked.
The observers recorded 6,810 request/rank source frames and 1,500 rank-local
slot reuses. Slot reuse counts are summed across ranks, not globally unique
physical slots. Every rank observed four concurrent capture contexts, recovered
all four reservations and ended without quarantined Host buffers or queued work.

The first PP smoke run exposed an incorrect fixture assumption that four active
requests must share one batch. Native PP used separate microbatches. The final
fixture explicitly uses two-request PP microbatches and checks four simultaneous
capture contexts, plus both one-request and two-request decode graph replay.
The failed smoke and preliminary passes are retained for diagnosis and excluded
from the final totals. No production scheduler or capture behavior changed.

The [machine-readable report](capture-ar-distributed-pressure.json) retains
job commands, source hashes, per-case rank observations, results and limitations.
All 5,016 Python source/registered test files matched the final frozen runtime
checkout. The 32 archived artifacts are under
`/gpfs/user/fuxuanwei/mooncake-lab-archive/ar-pressure-distributed-20261003`;
the artifact manifest SHA-256 is
`e84228a77ba63a0385ed7b44d2e7230aaf0c80e921d8018c1b70548bd5e40281`.
The temporary four-H100 allocation was deleted after its worker exited. The
resident H100 worker and resumed idle task were separately verified live.
