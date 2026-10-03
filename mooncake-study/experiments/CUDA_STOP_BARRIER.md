# CUDA Completion Before Store Shutdown

A failed per-request CUDA fence can leave real device work pending after the
writer retires the request into a quarantined Host slot. Previously,
`CaptureResources.close()` stopped the Store and marked the resource bundle
closed without a final CUDA barrier. A Store close only proves completion of
the Store transport; it cannot certify unrelated D2H or device staging work.

## Reproduction And Repair

The H100 regression creates a real single-rank coordinator with pinned Host
slots and a byte-addressed test Store. It queues delayed work on a non-default
CUDA stream, injects failure when the context creates its completion event,
aborts the request and waits for its slot to become quarantined. An independent
real CUDA event remains pending immediately before coordinator shutdown.

Both direct D2H and GPU staging reproduce Store close while that event is still
pending on the original implementation. The test keeps every buffer alive and
synchronizes in `finally`, including when the assertion fails; it does not
deliberately access freed memory or claim to reproduce data corruption.

Resource preparation now binds the capture device. KV owners use the exporter's
device; aux-only owners use the source pool device. An implicit CUDA index is
resolved during preparation so shutdown does not depend on the current device.
After callers stop their workers, resource close synchronizes that device,
closes Store transport, closes the journal and only then marks itself closed.
CPU and inactive resource bundles do not initialize CUDA during close.

A CUDA barrier failure preserves the open Store, journal and strong references
to the resource bundle. A Store close failure also preserves ownership. A later
close retries the barriers; successful close removes the retained reference.
This is a shutdown barrier, not a cancellation of GPU work or a timeout/recovery
mechanism for a hung device. Process teardown remains the final backstop.

The corrected CUDA regression requires its event to be complete at the exact
Store-close call and compares completed output values. No failed request may
publish. Quarantined slots remain non-reusable; successful transport shutdown
does not reopen the pool. External tensor references can still retain memory.

## Reproduction

Use the resident H100 and pinned capture environment, with `PYTHONPATH` pointing
at a frozen source tree and `OMP_NUM_THREADS=1`:

```bash
python test/registered/unit/training_capture/test_cuda_shutdown.py -v
python test/registered/unit/training_capture/test_resources.py -v -f
python test/registered/unit/training_capture/test_buffers.py -v -f
python -m unittest discover -s test/registered/unit/training_capture \
  -p 'test_*coordinator.py' -v -f
python -m unittest discover -s test/registered/unit/training_capture \
  -p 'test_*writer.py' -v -f
python test/registered/unit/training_capture/test_cuda_snapshot.py -v -f
python test/registered/storage/test_training_snapshot_mooncake.py -v -f
```

The resource suite verifies CUDA failure retention without keeping the exception
alive, successful retry, CUDA-before-Store ordering, Store failure retry,
aux-only rank device selection, CPU/inactive close, startup rollback and
four-process readiness failures. The original CUDA snapshot suite exercises
cross-stream staging, failed fences, source reuse and tail completion. The
real TCP Store suite checks independent readers, cohort ownership and recovery.

The first baseline direct-D2H attempt did not exercise pending work because
first-use allocation synchronized the device. Its staging case did reproduce
the ordering bug. Warming the allocator on the test stream fixes the direct-D2H
precondition; the second baseline run fails the ordering assertion in both
cases. The final test differs from that baseline only by explicit binding of
three local closure defaults for lint.

The initial aggregate unit runner lacked a `__main__` guard and failed when a
four-process test spawned children. A separately retained guarded runner
reruns the suite without changing production code or test assertions.

This validates a shutdown ownership invariant. It does not establish
long-duration serving memory stability, recovery from an actual damaged CUDA
device, RDMA outage behavior, production Catalog retention or training quality.

## Verified Results

All 144 distinct candidate test methods pass on the retained H100:

| Suite | Methods | Seconds |
| --- | ---: | ---: |
| Pending CUDA shutdown regression, two cases | 1 | 6.364 |
| Resource, buffer, coordinator and writer regression | 129 | 71.772 |
| Existing CUDA snapshot/source-reuse regression | 7 | 0.893 |
| Actual TCP Store and independent publication/readback | 7 | 197.017 |

At the Store-close boundary, the independent event is complete for both the
direct D2H and device-staged requests. Both requests remain unpublished and
their quarantined slots remain non-reusable. The resource tests also prove
that a failed global CUDA barrier prevents Store close and keeps buffers alive
even after the caller discards the exception and resource reference.

The [evidence JSON](cuda-stop-barrier.json) records all seven terminal jobs,
including the baseline failures and the initial runner error. All 5,032 Python
files match the frozen tested source. An AST comparison confirms the final CUDA
test matches the corrected baseline after normalizing its three closure argument
bindings. The resource implementation and both changed/new tests pass full Ruff;
the coordinator's comment-only change retains its nine baseline findings.

The 41-artifact archive is
`/gpfs/user/fuxuanwei/mooncake-lab-archive/cuda-stop-20261003`; its manifest
SHA-256 is `11e5f6f3e456b2a152dba6d012bcdd42f43eae61c822d97dc99134be53e5a68f`.
No additional GPU was allocated. Process inspection confirms only the resident
worker and idle load remain, with an empty experiment queue.
