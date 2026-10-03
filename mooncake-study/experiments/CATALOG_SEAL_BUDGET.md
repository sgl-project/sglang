# Catalog Seal Capacity And Cohort Retirement

A manifest fitting its registered Host arena can still exceed the Catalog HTTP
client's 8 MiB request-body limit after Base64 encoding. Previously the writer
registered and wrote payloads, received WRITTEN acknowledgements, and saved its
durable journal before discovering this local, non-retryable size mismatch.
Repeated recovery could not seal the same bytes.

The producer now checks both limits during single-rank and cohort admission.
The existing global sizing envelope bounds maximum sequence metadata and all
canonical owners; its manifest-byte bound also bounds the Base64 payload.
The seal descriptor and body builder are shared with the writer, so credential,
sequence, object-key, digest and idempotency fields are counted using the actual
wire serializer. Base64 uses exactly `4 * ceil(manifest_bytes / 3)` JSON bytes.
Sizing does not allocate a Base64 string, payload tensors or per-token chunks.

Each writer entry point also checks the exact size before its first Catalog
object registration, payload write or journal save. This protects direct callers
of `write`, `write_partition` and `publish_partitions`. The wire format and HTTP
limit remain unchanged. Increasing `manifest_buffer_bytes` cannot bypass the
Catalog limit; the raw manifest allowance is slightly less than 6 MiB.
Existing journals remain subject to the original recovery/reconciliation rules.

## Reproduced Failure

Both probe runs used the same 6,299,187-byte manifest and 8 MiB registered Host
arena, the real HTTP client and loopback test Catalog, and a byte-addressed Store
test double through the normal Store adapter and snapshot writer.

| Version | Payload Objects Written | Pending Journals | Published |
| --- | ---: | ---: | ---: |
| Before the seal check | 16 | 1 | 0 |
| After the seal check | 0 | 0 | 0 |

The probe isolates the local HTTP/journal failure. Native Mooncake transport is
covered separately by the integration suite, not by this probe's Store double.
Tests compare calculated lengths with actual HTTP request bodies across Base64
padding boundaries, exercise the largest accepted manifest size and the next
rejected byte, and check that all three writer entry points have no side effects.

## Cohort Rejection

The native Store suite adds four collector scenarios with four processes and
Gloo control messages:

- TP2/PP2 with all ranks rejecting oversized metadata before collection.
- TP4 replicated heads with canonical KV owners 0/2, auxiliary owner 1 and an
  inactive rank 3 that allocates zero payload bytes.
- TP2/PP2 with an inactive ingress PP stage; both ingress ranks allocate zero
  payload bytes and participate in rejection and ticket retirement.
- A rank-1-only metadata overflow after ranks 0/2/3 start actual CUDA copies.
  Rank 2's copy-completion gate holds retirement; all cohort slots remain
  unavailable until the gate drains, and no failed-sample payload is written.

Every scenario verifies FAILED Catalog state without registered/written objects,
no quarantine, rejection of the old ticket, and admission of two fresh samples.
The eight accepted samples each contain 32 objects and 4,532 tensor bytes; all
KV, token IDs, masks, raw top-128 scores/IDs and LSE are checked after the
producers exit. A further reader process exercises native batched reads. The
existing six collector cases and the remaining Store tests also run.

These are synthetic forward inputs in four processes sharing one resident H100.
They do not establish four-GPU model inference, RDMA, production Catalog
retention, MaaS SLO acceptance or training quality.

## Reproduction

Use the pinned capture environment on the resident worker:

```bash
python -m unittest discover -s test/registered/unit/training_capture \
  -p test_manifest_budget.py -v
python -m unittest discover -s test/registered/unit/training_capture \
  -p test_snapshot_writer.py -v
python test/registered/storage/test_training_snapshot_mooncake.py -v -f
```

The final source passes 168 regression methods in 149.979 seconds and all eight
native Store methods in 233.861 seconds. An earlier run of the cohort tests
before the seal fix also passed all eight Store methods in 233.643 seconds;
this is a repeated suite, not eight additional distinct tests. All eight touched
Python files pass formatting and introduce no Ruff findings relative to the
eleven baseline findings. The final audit compares 5,624 Python files with the
frozen execution tree.

Changed sources, source audits, the probe and the complete regression runner are
retained under
`/gpfs/user/fuxuanwei/mooncake-lab-archive/cohort-manifest-budget-20261003`.
[The evidence JSON](catalog-seal-budget.json) records all five terminal jobs and
the 31-artifact archive manifest SHA-256:
`f73ef9bab1658a4bfca6d57660e6cbd16258b0731a747f1b1714636786969fe1`.
The resident worker resumed its idle load after all tests; no additional GPU
was allocated.
