# Distributed P/D Prefill Graph Capture

These tests extend the [single-rank P/D matrix](PD_PREFILL_CAPTURE.md) to
matching TP2/PP1 groups with overlap and matching TP1/PP2 groups with synchronous
scheduling. P and D are independent worker groups sharing two GPUs. Each group
uses real Mooncake TCP KV transfer; D publishes the complete training snapshot
through a separate real TCP Store and an HTTP Catalog test double.

## Reproduction

Use two CUDA devices, the dependency overlay in `h100-runtime-lock.json`, and
`mooncake_master` on `PATH`:

```bash
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_pipeline_prefill_graph.py -v -f

PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_tensor_prefill_graph.py -v -f
```

Each file runs AR and static target-KV DSpark with Full, Breakable and
torch.compile piecewise P prefill graphs. DSpark loads its synthetic draft on
both P and D. D decode/verify graphs use Full. Piecewise uses the default eager
compiler with captured-input-address assertions. Each topology gets an
independent eager AR baseline for exact greedy token comparison.

## Checks

Every graph case must replay on every P rank. Intermediate PP outputs must be
`PPProxyTensors`, with every tensor trimmed to the live token count before
transport. Graph token/request padding must not enter stored samples. The
fixture also checks prefix reuse, chunked prefill, buffer reuse, NULL hidden
capture and empty auxiliary hidden output. D target forward entries are limited
to DECODE, TARGET_VERIFY and IDLE, excluding a second target prefill.

Each case produces ten complete samples and compares selected KV, raw top-128
values/IDs, full-vocabulary LSE, token IDs, loss masks, positions and terminal
validity with independent online observations. The manifest must describe D's
topology and complete ownership. A new Mooncake client reads and digest-checks
every object after both producer groups exit. DSpark additionally checks
target-KV projection, accepted prefixes and rejection.

Missing/stale first-teacher handoffs and a streamed abort are injected in every
case. Each fault request must replay on every P rank and produce teacher rows
on every TP rank of the last P stage. It must fail capture without publication,
release its reservations and leave no quarantined Host memory. Admission and
failure counters are checked against all submitted correctness requests.
Publication is checked against Catalog records and Store reads. The PP0 state
returned first by the server can report zero READY because the last-stage
auxiliary owner publishes the global manifest.

These are correctness workloads with bounded request batches, not saturation
tests. Distributed cohorts reserve lazily through their request router. The
single-rank draft-contract seed waits for preallocated capture capacity.

## Results

PP2 passes six tests in 761.683s; TP2 passes six in 634.442s. Each topology
validates 60 graph samples and ten eager baseline samples. Every graph case
records 16 P replay source frames across its two ranks, plus graph-backed
missing/stale/abort probes. All generated sequences equal their eager reference.
All 140 snapshots pass source comparison and post-exit Store digest checks.
The two synthetic-draft contract seeds are excluded from this total.

All 14 ten-sample runs report thirteen admissions, ten seals, two failed
handoffs, one failed running abort, zero admission backpressure and zero
quarantine in the inspected state. PP0's local READY count is zero as expected;
the Catalog and independent Store reader confirm ten global publications.

Both jobs used the same frozen source. The first PP startup compiled FlashInfer
kernels while health checks waited; it continued without a restart and passed.
No preliminary failed run or relaxed assertion was needed. Three changed/new
Python files pass Black and full Ruff checks. Commands, hashes, per-case state
and allocation cleanup are retained in `distributed-pd-prefill-capture.json`.

The temporary two-H100 allocation was deleted after both jobs completed, and
its pod is absent. The resident one-H100 worker remains allocated with its
idle workload and no experiment queued. The producer README now reflects the
implemented P/D and synchronous target-KV PP paths, their remaining capability
gates, and the rank-local meaning of publication counters.

## Scope

The model is BF16 Qwen3-0.6B, selected layers 0/14/27, FlashInfer target and
Triton draft. The target KV pool has 4096 tokens; prefill chunks have 128 tokens
and graph buckets are 16/32/64/128. KV and teacher staging use 16-row batches
within a shared 8 MiB device budget. Observation copies and synchronization
are test-only and cannot establish production performance.

Mixed TP/PP prefill, asymmetric P/D prefill, asynchronous PP, other model
families/speculative modes, Inductor, cross-node RDMA combinations, production
Catalog retention, trained quality and serving SLOs remain separate gates.
