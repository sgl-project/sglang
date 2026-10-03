# Distributed Capture Backpressure

## Behavior

Capture ingress previously observed only its own adaptive controller. A final
PP-stage publisher could remain stuck after every other owner had stored its
payload, while ingress kept a healthy local writer and continued admitting
captures. Non-ingress controllers also depended on request or optional latency
callbacks to refresh pressure. The retained pre-fix regression shows writer age
at one second, effective ratio one, and an available ticket despite the configured
20 ms stall threshold.

Every coordinator now observes local pressure in its background handoff loop.
The existing cohort state exchange validates and votes rank-local probabilities;
the minimum probability and all-rank readiness govern new tickets and Catalog
reservations. Recovery runs without new requests and does not feed the agreed
probability back into the local controller. Already issued tickets remain
bindable during an adaptive pause; publication, lease renewal and safe buffer
release retain their existing ownership rules.

The internal control protocol is version 5. All ranks must use matching code.
Snapshot schema and Store objects are unchanged. A spare header word carries an
exact binary64 probability; malformed/nonfinite/above-ceiling votes fail in the
collective validation phase before ranks choose an action.

`training_capture.admission` includes `local_effective_ratio` and
`cohort_effective_ratio`. Its `effective_ratio`, also used by the Prometheus
gauge, includes cohort readiness. Local reason and target fields remain local.

## Reproduce

Run from the SGLang checkout with its Python package on `PYTHONPATH`, the
matching CUDA/FlashInfer runtime, Mooncake SDK and `mooncake_master` available.
Set `TRAINING_CAPTURE_TEST_MODEL` to a local Qwen3-0.6B model when offline.

```bash
python -m unittest discover -s test/registered/unit/training_capture -p 'test_*.py' -v
python test/registered/storage/test_training_capture_cohort_backpressure.py -v -f
python test/registered/storage/test_training_capture_cohort_backpressure_tppp.py -v -f
python test/registered/storage/test_training_capture_pd_tp_pp_control.py \
  TestCombinedPDCaptureControl.test_graph_overlap_controls \
  TestDSparkCombinedPDCaptureControl.test_graph_overlap_controls -v -f
```

The first command requires one CUDA device for the full suite. Backpressure
tests require two and four devices respectively; combined P/D controls require
four. P/D producer groups share the devices in this correctness fixture.

The backpressure fixture performs eleven requests per topology:

1. Capture a healthy request and wait for full recovery.
2. Pause only the auxiliary owner's manifest write. Payloads, all-owner receipts,
   Catalog seal and the local durable journal precede this point. Require that
   all non-auxiliary writers are empty and retain local ratio one, while all
   ranks report effective ratio zero and no available capture ticket.
3. Complete eight inference requests while the gate remains held. Require no
   new captures, fixed Host allocation and no quarantined buffers. Wait within
   a fixed deadline for every rank's asynchronous Prometheus gauge to reach zero.
4. Release the gate, let the existing snapshot publish and wait for ratio one
   without new traffic. Capture one more request and check all owners recover.
5. Stop the producer and use a new Store client to read three complete snapshots.
   Compare token IDs, loss mask, positions/validity, selected-layer K/V, raw top128
   values/vocabulary IDs and full-vocabulary LSE against online source tensors.

The four-process unit scenarios also exercise a fractional peer vote, zero
admission, recovery and binding a previously issued ticket while paused. Invalid
NaN, negative and above-ceiling frame values must produce a common control
failure without releasing buffers whose transfer completion is uncertain.

## Evidence And Limits

The 2026-10-03 final runs passed all 284 unit methods (230.618s), the TP2/PP2
backpressure suite (two methods, 113.412s), combined TP2/PP2 backpressure (one
method, 58.975s), and combined AR/DSpark P/D controls (two methods, 362.728s).
Backpressure cases complete 33 requests, including 24 during a publication stall,
and validate nine post-exit snapshots / 162 objects / 2,177,784 bytes. P/D control
regression separately validates sixteen post-exit snapshots.

The machine-readable [report](cohort-backpressure.json) records final runs,
rank-local observations, exact job IDs, frozen source hashes and cleanup.
The archive retains the pre-fix failing regression and the initial test-harness
failures, as well as successful final runs. The subprocess CUDA bounds test now
sets its own module directory, so unittest discovery does not depend on the
launch directory. Metrics assertions wait for the existing one-second refresh.

This exercises a controlled manifest-call stall through the real TCP Mooncake
Store, using Qwen3-0.6B BF16, Triton, decode graphs and an HTTP test Catalog.
TP uses overlap; PP is synchronous. Prefill graphs are disabled. Source
observers and biased output tokens check preservation and continued generation;
their timing is not a serving SLO measurement. It does not certify saturated
RDMA capacity, cross-node pressure, production Catalog retention, SpecForge
consumption or trained draft quality.
