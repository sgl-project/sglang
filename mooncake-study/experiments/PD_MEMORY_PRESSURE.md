# P/D Target-KV Draft Capture Under Memory Pressure

Run each complete file with matching SGLang dependencies, the Mooncake SDK and
`mooncake_master` on `PATH`. The pipeline file requires two CUDA devices:

```bash
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_dspark_pressure.py -v -f

PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_dspark_pp_pressure.py -v -f
```

The fixture starts real prefill and decode processes, Mooncake TCP KV transfer,
an independent Store segment and reader, and an HTTP Catalog test double. Both
serving sides load the synthetic target-KV DSpark draft. PP1 crosses synchronous
and overlap scheduling with eager and full verify CUDA graphs. PP2 runs
synchronous eager and graph execution with matching P/D stage counts.

## Natural Capacity Exhaustion

Each of four disjoint 16-token prompts requests 192 response tokens. Their 832
total path tokens exceed the 512-token decode KV pool; each individual request
fits. `--num-reserved-decode-tokens 16` permits the initial batch to start.
`SGLANG_TEST_RETRACT=0` disables debug-triggered retraction. A test-server barrier
holds the four tagged initial transferred requests until they can form a real
decode batch; resumed requests bypass the barrier. The allocator and retraction
decision run unchanged.

Single-rank capture leases are prepared asynchronously. The test waits for four
available leases before the initial burst, and at least one available lease
before the fresh recovery request. Cohort request admission uses its existing
distributed reservation path. A scheduler with no active capture records is
not sufficient proof that the asynchronous lease pool has finished warming up.

## Restore And Capture Invariants

The test-only server wrapper independently copies every local target layer's K
and V before calling the original request offload function. After the original
CPU-to-GPU restore completes, it gathers the logical prefix from the newly
allocated request slots and requires `torch.equal` against those saved tensors.
PP stages check their own layers and write bounded JSON metadata, not tensors,
to the restore log. This check covers all 28 target layers, including those not
selected for training snapshots.

The suite also requires:

- All four requests finish 192 response tokens, at least one request is
  retracted and at least one survives, and each stage's Prometheus retraction
  counter agrees with the request results.
- Restored requests have detached capture context/finalizer and retain the
  attempted marker. Their original capture IDs agree across stages and reach
  Catalog `FAILED`, with no corresponding publication.
- Each stage rebuilds the retired draft context from an empty projection
  state, covering the restored target prefix. P/D restore uses the normal CPU
  backup path; it does not ask P to generate the prefix again.
- Only surviving captures publish. A fresh short request is subsequently
  admitted and published; queues drain with no quarantined Host slots.
- A real four-request verify batch executes, and graph observations appear
  exactly in the graph cases.
- Surviving and fresh samples match the independent online source observations
  for selected KV, raw top-128 logits and vocab IDs, full-vocabulary LSE, token
  IDs, positions, masks, teacher alignment and terminal KV validity.
- Both producer process trees exit before the independent Store reader checks
  every published snapshot again, including object digests and contents.

The first test version treated an empty scheduler as a ready lease pool. Two
early PP1 runs exposed intermittent 3-of-4 capture admission. Waiting for the
configured capacity fixes that fixture race without weakening the four-sample
admission requirement or changing production backpressure.

An intermediate PP2 graph run also exposed a valid cohort ordering: one stage's
retraction invalidated its peer before that peer released locally. The peer
reported `peer_capture_failed` instead of `request_aborted_or_retracted`.
The final assertion accepts these two reasons, rejects other failure reasons,
and still requires exactly one failed capture per retired request. Capture IDs
are recorded at successful P/D handoff, before either stage can invalidate the
request, and checked against the actual failed Catalog leases after restoration.

## Verified Results

All five complete test files passed on H100 with the final production sources:

| Suite | Tests | Seconds | Post-exit snapshots |
| --- | ---: | ---: | ---: |
| P/D PP1 pressure | 4 | 337.714 | 12 |
| P/D PP2 pressure | 2 | 221.799 | 6 |
| Ordinary target-KV P/D regression | 4 | 327.410 | 24 |
| Hidden-input P/D regression | 6 | 389.911 | 36 |
| Pipeline target-KV P/D regression | 6 | 493.724 | 36 |
| Total | 22 | | 114 |

Each pressure case naturally retracted two requests. The six cases check 12
retired captures, 16 rank-local all-layer restores and 18 published snapshots.
Draft contexts rebuild through 124 and 164 restored prefix tokens. Four seed
snapshots used to construct the synthetic draft fixtures and all intermediate
runs are excluded from these totals.

Ordinary PP1 and hidden-input regressions ran from frozen sources before the
pressure-only capture-ID timing and peer-failure assertion refinement. Those
suites use no pressure-tagged requests; the shared launch helper and all
production sources match the final versions. Final PP1/PP2 pressure and PP
regression suites ran the exact final observer. Source and log hashes, per-case
observations and all earlier attempts are retained in
[the evidence JSON](pd-memory-pressure.json).

No production source changes were needed for these invariants. The temporary
two-H100 job was deleted after its model and Store processes exited. The
resident H100 has no queued experiment and resumed its 60% idle workload.

## Scope

The selected snapshot layers are 0/14/27, the target is BF16 Qwen3-0.6B, and
attention uses Triton. Prefill graphs are disabled. Observer copies synchronize
the device, and serving logit bias fixes response tokens; these tests establish
cache and snapshot correctness, not model quality or serving performance.

This is normal decode-pool exhaustion and request CPU backup/restore. It is
separate from the optional P/D KV offload manager, HiCache host-tier eviction,
CUDA allocator errors, rank failure and device loss. Combined TP/PP pressure,
asymmetric P/D pressure, cross-node RDMA pressure, production Catalog retention,
trained checkpoints and latency/throughput SLOs still need their own evidence.
