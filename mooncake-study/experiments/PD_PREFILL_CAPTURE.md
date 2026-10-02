# P/D Prefill Graph Capture

P-side prefill graph replay supplies the first raw teacher row through the
existing fenced handoff. D owns the complete training snapshot: it exports
transferred target KV, appends accepted decode/verify data, and publishes the
manifest after the immutable Mooncake objects. KV-input DSpark reconstructs its
draft context from the received target prefix. No additional target prefill is
needed on D.

## Reproduction

Use the dependency overlay in `h100-runtime-lock.json`, a CUDA device with room
for two Qwen3-0.6B workers, and `mooncake_master` on `PATH`:

```bash
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_prefill_graph.py -v -f
```

The nine cases cover AR Full synchronous/overlap, AR Breakable/piecewise,
D-only target-KV DSpark with P Full, and P+D target-KV DSpark with Full
synchronous/overlap, Breakable and piecewise. Both roles use TP1/PP1, target
FlashInfer and, when enabled, draft Triton. D decode/target verify uses Full
graphs. Piecewise uses the default eager compiler with captured-input-address
assertions. P and D run in separate processes on the same GPU.

Before the graph cases, a separate AR run with prefill/decode graphs disabled
establishes exact greedy output references. Each case publishes ten samples:
one-token response, chunked 153-token prompt, prefix hit, unbiased rejection
probe, cached extension, another unbiased prompt and two two-request batches.
The fixture checks actual P replay, token padding, unused Full request slots,
cached prefix lengths, intermediate chunks without a prediction and input
buffer reuse. It also checks that hidden capture is NULL and no hidden-state
output is produced. DSpark must exercise both acceptance and rejection, and
its target-KV projection is independently observed.

The observer rejects D target model entries other than DECODE, TARGET_VERIFY
or IDLE. Raw logits are observed before serving bias; selected KV, raw top-128
scores and vocab IDs, full-vocabulary LSE, tokens, loss masks, positions and
terminal KV validity are compared with online observations. After both
producers exit, a new Store client reads every manifest and object and verifies
digests. These observer copies and synchronization are test-only.

Each case also injects missing and stale teacher handoffs and aborts a stream.
All three must be admitted to capture and fail without publishing; their P
requests must actually use prefill graphs and produce teacher rows. Distinct
13-token prompts keep these probes from becoming cached one-token fallbacks.

Correctness requests wait for sufficient single-rank capture reservations.
Publication completion and the background replacement of reservations happen
independently. The final counters must show zero admission backpressure and
exactly thirteen admissions per ten-sample case. Distributed cohorts reserve
lazily through their request router and do not use this availability wait.
This driver does not measure saturation or capture throughput.

## Results

The nine-case matrix passes in 940.343s. It validates 90 graph samples and ten
eager baseline samples, with eight P replay source frames per graph case for
successful requests. All ten runs have thirteen admissions, ten publications,
two rejected handoffs, one aborted capture and zero admission backpressure or
quarantined Host slots. A separate four-case P/D DSpark regression with prefill
graphs disabled passes in 432.918s and validates 24 complete snapshots.

The final-source AR regression passes two tests in 176.369s and validates ten
snapshots. Natural KV-pressure graph regressions pass synchronous and overlap
cases in 262.404s: each triggers two retractions, checks exact CPU KV restore,
excludes retired captures and validates three surviving/new snapshots after
producer exit. Across the four accepted jobs, 17 tests pass and 140 complete
snapshots pass post-exit readback. Contract seeds and partial successes from
failed preliminary jobs are excluded. All six submitted jobs are terminal, and
the resident H100 has resumed its idle workload with no queued experiment.

The initial matrix stopped after a batch produced only nine of ten expected
publications; the initial AR regression also lacked its final aborted capture.
Those logs do not establish whether the absent captures were admitted. The
fixture previously sent new requests as soon as publication completed, even
though replacement reservations arrive asynchronously. The corrected matrix
waits for reservation availability and asserts exact admission/failure counts.
Failed attempts are retained separately and excluded from acceptance totals.

Commands, source versions, log/result hashes and per-case counters are retained
in `pd-prefill-capture.json`. The matrix runs with the same validation path as
the final source; the later fixture change also enables capacity waits and exact
admission checks in the existing non-prefill regression cases. No production
capture code was changed for this P/D prefill validation.

## Scope

The runtime uses BF16 Qwen3-0.6B, selected layers 0/14/27, 128-token prefill
chunks, graph token buckets 16/32/64/128, a 4096-token target KV pool, 16-row
teacher/KV D2H staging and a shared 8 MiB staging budget. The draft is synthetic
and untrained. Transfer and Store both use real Mooncake TCP; Catalog is an
HTTP test double.

Distributed P/D prefill graphs, mixed TP/PP prefill, other model families and
speculative modes, Inductor, RDMA combinations, production retention, trained
quality and serving SLOs require separate validation. Existing P/D decode and
cross-node RDMA evidence does not certify these prefill combinations.
