# Publication Cleanup Recovery

The single-rank coordinator previously inferred publication progress from
`journal.has_pending(capture_id)` after a writer exception. That is insufficient:
`PublicationJournal.complete()` unlinks the file before directory fsync. If
cleanup raises after unlink, the Catalog may already have confirmed publication
while the coordinator reports sample failure and never counts READY.

A loopback HTTP Catalog probe reproduced this exact control-flow error. The
Catalog rejected the attempted failure, so the sample remained AVAILABLE, but
the producer's accounting and recovery decision were incorrect.

| Same Cleanup Failure | Failure Requests | READY Count | Catalog Publications |
| --- | ---: | ---: | ---: |
| Before | 1 | 0 | 1 |
| After | 0 | 1 | 1 |

The probe uses the actual coordinator, writer and HTTP client with a
byte-addressed Store test double. Cleanup failure is injected, not an observed
filesystem hardware fault. Native Store coverage is separate below.

## Recovery Boundary

`SnapshotWriter.write(..., on_prepared=callback)` invokes the callback with the
exact lease and encoded manifest after all payload WRITTEN acknowledgements and
before journal creation. Both inputs are immutable, including the lease's
original renewal fields. The single-rank coordinator retains them independently
of subsequent lease renewal and filesystem visibility.

An exception after this boundary leaves the reservation pending, pauses new
capture and retains the original transfer-completion state. The background
writer calls `recover_prepared(lease, manifest_bytes)`, which validates metadata,
restores the durable journal, checks the fence and every Store payload, and
replays manifest-last publication with a fresh registered buffer. Confirmation
increments READY once and permits healthy slots to be reused. The existing
startup journal path shares the Store verification/publication helper.

Publishing successfully cannot release a source with uncertain transfer
completion. Such an arena remains registered and quarantined even after READY.
No recovery step reads the old payload arena or reruns a target forward.
Failures before the prepared boundary still use normal sample failure handling.
If journal creation fails and the process exits before retry, the in-memory
metadata is lost; the Catalog's capture-expiry/GC policy remains responsible for
those orphan objects. This change does not add a cross-process guarantee before
durability or replace production retention authority.

## Validation

Coordinator fault tests run with direct and staged capture fixtures. They check
post-unlink directory-sync failure, journal creation failure, identical publish
retry bodies, frozen lease values across renewal, destroyed old payload bytes,
subsequent request admission, and successful publication without releasing an
uncertain manifest source. Existing cancellation/content validation tests check
that rejected samples never reach preparation.

The native TCP integration test writes a complete sample, confirms publish,
injects journal cleanup failure, closes the original Store producer, and clears
its tensors and manifest buffer. A fresh Store connection recovers using only
the retained lease/manifest metadata and stored payloads. It verifies identical
publication arguments, one native payload-read batch, no payload rewrite,
empty journal/receive registrations after success, and complete content readback.
The separate pre-existing test still checks restart recovery from a durable
journal after a lost publish response.

Use the pinned capture environment through the resident H100 worker:

```bash
python -m unittest discover -s test/registered/unit/training_capture \
  -p test_coordinator.py -v
python -m unittest discover -s test/registered/unit/training_capture \
  -p test_snapshot_writer.py -v
python test/registered/storage/test_training_snapshot_mooncake.py -v -f
```

This is fault-injection and transport evidence with Catalog test doubles.
It does not establish new model-serving, RDMA, production Catalog, MaaS SLO or
trained-draft quality acceptance.

## Results And Evidence

| Final Suite | Methods | Seconds |
| --- | ---: | ---: |
| Shared capture, P/D, startup, protocol and writer regression | 174 | 183.509 |
| Native Mooncake TCP integration | 9 | 237.598 |

[The evidence JSON](publication-cleanup.json) includes the before/after probe,
four completed jobs, two 5,624-file source audits and a relative-baseline lint
audit with no new findings. The source freeze completed and matched the working
tree before submission. The worker returned to its resident idle load and no
additional GPU was allocated.

The 24-artifact archive is
`/gpfs/user/fuxuanwei/mooncake-lab-archive/publication-cleanup-20261003`, with
manifest SHA-256
`e991e0b4c4d61e033eb638030415c651584d6d1fae8edcaa810a0b863cedb3fb`.
