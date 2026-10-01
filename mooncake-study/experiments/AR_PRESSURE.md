# AR Capture Under KV Pool Pressure

Run the complete file with one CUDA GPU, matching SGLang dependencies, the
Mooncake SDK and `mooncake_master` on `PATH`:

```bash
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_ar_pressure.py -v -f
```

The fixture uses a real TCP Mooncake Store and an HTTP Catalog test double.
Four disjoint 16-token prompts each request 80 output tokens, for 384 total
path tokens against a 256-token KV pool. Scheduling conservativeness is 0.05
so all four requests can start; every individual request fits in the pool.
The test explicitly disables `SGLANG_TEST_RETRACT`. It neither invokes the
pause/retract API nor overrides the allocator's capacity decisions.

The four cases cross synchronous/overlap scheduling with eager/Full decode
CUDA graphs. Attention uses Triton, prefill graphs are disabled, and decode
graph buckets are 1/2/4 requests. The model is BF16 Qwen3-0.6B with selected
layers 0/14/27, 128-token prefill chunks, 16-token bounded D2H batches,
64-token Store chunks and four capture reservations.

Test-only instrumentation wraps the original `ScheduleBatch.retract_decode`.
Before calling it unchanged, the wrapper records available capacity, required
next-decode capacity, capture lease IDs, committed slots and the most recent
online source frame. Afterward it records the retired requests and whether
their capture context/finalizer has detached. It does not choose which request
to retract or change the serving response.

The runtime test requires all of the following:

- Available KV capacity is below the real next-decode requirement at every
  observed retraction, and capacity increases after release.
- All pressure requests finish their full response, and HTTP metrics agree
  with per-request and scheduler-observed retraction counts.
- Each retired request's original capture lease reaches Catalog `FAILED`
  with `request_aborted_or_retracted`; none becomes published or starts a
  second capture after resuming.
- Surviving requests publish complete samples. A fresh short request is
  admitted and published after pressure; all four reservations recover with
  zero quarantined Host slots.
- A later online source frame from a successful sample uses physical KV slots
  that belonged to a retired request before release.
- Graph cases execute a three-request decode batch using a captured graph;
  overlap cases observe the pending-result lookahead.
- After the producer exits, a new Store client validates every manifest and
  object digest. KV and raw top-128 values match the online source exactly;
  vocab IDs, full-vocabulary LSE (`rtol=atol=1e-6`), token IDs, masks, positions,
  teacher alignment and KV validity all pass without target recomputation.

This tests normal scheduler retraction under KV pool exhaustion, not a CUDA
allocator exception or a device failure. Source observers intentionally copy
and synchronize tensors, so this is correctness evidence, not a throughput or
latency measurement. Forced token outputs make the expected request boundaries
deterministic and do not measure model quality. TP/PP/DP/CP pressure, simultaneous
prefill graphs, production Catalog retention and consumer training remain
separate acceptance scopes.
