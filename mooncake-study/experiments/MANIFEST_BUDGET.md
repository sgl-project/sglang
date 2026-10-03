# Manifest Capacity Admission

The 32K coverage experiment produced approximately 1.45 MB of manifest JSON,
larger than the default 1 MiB registered manifest arena. Previously that mismatch
was detected in the background writer after KV/teacher collection. The producer
now checks a conservative size bound when binding a request to a reservation,
before constructing its capture context or enqueueing its capture copies.

`manifest_size_bound()` uses the maximum requested response length, current
provenance and validated global ownership layout. It encodes one sizing
descriptor per layer/component/canonical owner, multiplies by the maximum chunk
count, and includes auxiliary descriptors, array separators, field widths and
the complete metadata envelope. It includes replicated-head ownership and an
aux-only PP stage without counting redundant KV replicas. Work does not iterate
over token chunks or allocate payload tensors.

Both single-rank and cohort admission retire a rejected reservation through the
existing background lifecycle. Inference continues, no partial sample is
published, and later smaller requests remain eligible. The rank-local counter
`sglang:training_capture_events_total{event="admission_manifest_budget"}` reports
the reason separately from malformed requests and transport errors. No new
scheduler RPC, collective, CUDA synchronization or allocation is added.

The bound can reject requests whose eventual early response would fit. Increase
`manifest_buffer_bytes` within `max_host_bytes`, reduce the capture length limit,
or use larger storage chunks. The exact writer-side buffer check, semantic
validation and manifest-last publication remain mandatory. This estimate is not
a validation of data contents or a general bound for arbitrary external manifests.

## Verification

The sizing suite compares the bound with 312 complete manifests assembled by the
normal producer. Cases include full/final-missing KV, shortened replies, four
stop reasons, decimal-width boundaries, maximum-length sample/generation IDs,
escaped Unicode provenance, BF16/FP16, TP1/2/4 and an aux-only PP stage. A
2,147,483,647-token, one-token-chunk case verifies that sizing work stays bounded.
Coordinator tests check rejection before context construction, Catalog failure,
no Store payload, no quarantine, slot/ticket reuse and the dedicated metric.

| Suite | Methods | Seconds |
| --- | ---: | ---: |
| Sizing, coordinator, P/D, metrics, startup, protocol and writer regressions | 164 | 150.072 |
| Real Qwen3-0.6B capture admission through TCP Store | 1 | 86.021 |

The real-model test first publishes a 240-prompt/8-response sample with a
**16,734-byte** manifest using the normal arena. With a **12 KiB** arena, the same
request still returns its eight expected tokens but is excluded from capture.
Capture context admission, copy completion, snapshot construction and payload
Store calls are zero for that request; adaptive failure count and quarantine are
also zero. The four background reservations are available again. A subsequent
8-prompt/2-response request publishes an **8,469-byte** manifest in the same
process. Both accepted snapshots pass full tensor/content readback after their
producer exits, including exact token IDs and prompt/response masks.

This is single-rank synchronous eager serving with the HTTP test Catalog. Logical
TP/PP sizing and cohort unit tests do not constitute new multi-GPU inference,
RDMA, production retention, SLO or trained-draft quality acceptance.

## Reproduction And Evidence

Run in the pinned capture environment through the resident worker:

```bash
python -m unittest discover -s test/registered/unit/training_capture \
  -p test_manifest_budget.py -v
python test/registered/unit/training_capture/test_coordinator.py -v
python test/registered/unit/training_capture/test_cohort_coordinator.py -v
python test/registered/storage/test_training_capture_runtime.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  TestTrainingCaptureRuntime.test_manifest_budget_skips_capture_and_preserves_serving -v -f
```

[The evidence JSON](manifest-budget.json) records the terminal jobs and the
complete 5,048-Python-file source audit. The first unit submission started before
its source copy finished and failed to import a missing module; it ran no usable
test. The copy/overlay was completed and hashes checked before resubmission.
That failure is retained alongside the passing run. Formatting passes and the
eight touched Python files add no Ruff diagnostics relative to the baseline.

The 24-artifact archive is
`/gpfs/user/fuxuanwei/mooncake-lab-archive/manifest-budget-20261003`, with manifest
SHA-256 `9b40c6f8ef6395e349a4587016c0d7e6b3c256a5b2cf59ab27f2c9b928e86a5e`.
All model/Store/test processes exited and the resident H100 idle load resumed;
no additional GPU was allocated.
