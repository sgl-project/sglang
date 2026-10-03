# Bounded Native Store Reads

The Store adapter now exposes `get_tensors` for ordered, owned CPU tensors and
`verify_tensors` for bounded verification without retaining the payload. Native
`batch_get_into` is optional; its absence uses the existing single-object read.
The SDK returns byte counts for reads, unlike the zero status used for writes.

```python
objects = [
    (obj.key, obj.shape, DTYPES[obj.dtype], obj.sha256)
    for obj in manifest.objects
]
payloads = store.get_tensors(objects)
```

The caller has already validated the manifest and holds the relevant Catalog
authority. A trainer needs its read lease; a producer recovering publication
uses its fenced capture. The API does not acquire a lease, interpret windows,
perform target inference, or manage tensors retained across calls.

## Ownership And Budgets

Every shape and the aggregate receive size are checked before any allocation
or SDK read. The new destinations plus all quarantined registrations must fit
`max_receive_bytes`. Each destination owns independent CPU storage, including
when a key appears twice. Successful results preserve input order and have
already been unregistered. They are not a pinned asynchronous H2D pool.

A complete result vector must contain exactly one integer per request. A
negative count retains that destination and its registration until client
close. Missing results, non-integer counts or an SDK exception retain all
submitted destinations. Completed destinations are cleaned up even if another
item fails, a digest differs, or one unregister operation fails. A nonnegative
count with the wrong size is a contract failure; no partial batch is returned.
Read failures never initiate a fallback retry.

`verify_tensors` groups objects by the remaining receive budget and at most 64
objects. It discards the verified tensors before reading the next group.
Immutable batch-write retries, `SnapshotWriter.recover` and
`recover_partitions` use this path. All existing objects must verify before
missing write objects are submitted. Recovery keeps its journal and cannot
publish READY after an incomplete or corrupt read. A later recovery uses fresh
destinations and does not release an earlier uncertain receive buffer.

## Reproduction

Use the runtime locked in `h100-runtime-lock.json`, with the tested dependency
overlay recorded by the evidence JSON. From the SGLang worktree:

```bash
PYTHONPATH=python:test/registered/unit/training_capture python -m unittest \
  test_buffers test_snapshot_writer test_cohort_writer test_resources \
  test_coordinator -v

PYTHONPATH=python python \
  test/registered/storage/test_training_snapshot_mooncake.py -v -f
```

The Store suite starts its own TCP master and registered data segments. It
requires the Mooncake SDK/master binary, Gloo and CUDA for the existing
four-process collector factory case. One H100 is sufficient. The resident
worker automatically suspends its idle load while running these commands.

Focused regressions check aggregate limits, duplicate-key independence,
partial and ambiguous completion, registration/cleanup failure, malformed
lengths and digests, optional SDK fallback, and publication exclusion. The
real transport suite wraps the actual SDK only to count native calls. Its
independent readers validate complete tensor contents, not only SDK statuses.

The recovery case stores payloads on a separate data client, loses the mocked
Catalog publication response, closes the producer, and overwrites every source
tensor. A fresh client recovers through the real native batch read, retries the
exact publication metadata, leaves no registrations or journal entries, and
does not rewrite the already stored manifest. Catalog behavior remains a test
double; the payload path is the real Store.

## Scope

The final source passes 118 unit methods in 65.409 seconds and seven real Store
methods in 196.806 seconds. Five recorded independent readers validate 148
objects and 22,660 tensor bytes through native batches; the fresh-client recovery
adds one native verification batch. All 5,031 Python files match the final
storage snapshot. No new Ruff diagnostics are introduced.

The archive retains two unsuccessful attempts: the first unit command hit a
standard-library `test` import collision, and the initial full Store suite found
a stale startup assertion. That fixture expected preallocated reservations
while writer recovery was held, contrary to current all-rank readiness gating.
It now checks zero reservations and zero admission before releasing recovery.

This establishes native batch-read correctness over TCP. It does not establish
batch RDMA throughput, production retention, a trainer receive pool, SpecForge
manifest/window loading or serving SLOs. The SpecForge checkout still trains
its stock DSpark with hidden-state inputs; exporting that state under a KV
architecture name would not implement the intended training path.

Exact submissions, failures, runtime logs and source/archive digests are recorded
in [the evidence JSON](batch-store-reads.json).
