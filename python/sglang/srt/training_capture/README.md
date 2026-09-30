# Training Snapshot Producer

This package implements the tensor contract and producer-side data plane for
`maas_target_kv_v1`. It is under development. The request lifecycle and runtime
configuration are not wired into SGLang yet; installing this package alone does
not collect MaaS requests.

## Ownership

- `teacher.capture_teacher` reads the unpadded vocabulary before serving-side
  processors and returns independent top-128 IDs, raw values, and full-vocabulary
  LSE. All operations run on the calling CUDA stream.
- `SelectedLayerKVExporter` gathers the selected layers through logical-to-pool
  slot indices into independent temporaries and copies into pinned Host views.
  Enqueue before source-slot reuse. The caller must retain Host slots and record
  a completion event after the final copy. Layer IDs preserve contract order.
- `HostBufferPool` allocates fixed-capacity registered arenas once. Admission
  fails when all slots are held. A slot is reusable only after D2H and all Store
  calls finish. An uncertain transfer quarantines the slot.
- `build_snapshot` runs only after D2H completion. It retains the final partial
  chunk and produces request-owned objects, including tokens and masks. It
  never aliases or modifies a shared HiCache page.
- `SnapshotWriter` is synchronous and belongs on one background writer thread.
  It registers object identities with the Catalog, writes tensors, seals, writes
  the READY manifest, and publishes a metadata-only reference. The capture lease
  must already exist; the caller owns admission, heartbeats and failure handling.
- `PublicationJournal` persists exact manifest bytes before sealing. On restart,
  `recover()` confirms the fence and prepared state, retries the same manifest,
  and republishes with the same idempotency identity. It needs no target forward
  or saved tensor payload. Unknown outcomes leave the journal entry intact.

The Store adapter must have one owner thread after initialization. Stop that
thread and complete outstanding CUDA copies before `close()`. Store failures
keep strong references to registered source/receive memory until client close.
Do not unregister quarantined arenas based on an exception alone. Closing a
producer does not authorize deletion of published Store objects.

## Catalog Producer API

`HTTPCaptureCatalog` implements the producer half of the proposed SpecForge
Catalog API. It does not implement the Catalog server, distributor, consumer
leases, checkpoints, or GC. The endpoint is the API base URL. Requests include
`X-Training-Capture-Protocol: 1` and an optional Bearer credential supplied at
construction. Internal traffic ignores external-network proxy settings.

| POST Route | Request | Required Response |
| --- | --- | --- |
| `/captures:begin` | Immutable dataset/sample/generation identity, contract/owner metadata, reserved budget, idempotency key | `CaptureLease` |
| `/captures/{id}/heartbeat` | Capture ID and fence | Renewed `CaptureLease`, same identity/fence |
| `/captures/{id}/objects` | Fence, `phase=REGISTERED` or `WRITTEN`, descriptors, idempotency key | Same phase and sorted `accepted_object_ids` |
| `/captures/{id}/seal` | Fence, owner, sequence, total tensor bytes, manifest descriptor, exact `manifest_base64`, idempotency key | `state=PREPARED` or already `READY`, matching `manifest_sha256` |
| `/samples:publish` | Fence, dataset/sample/generation, contract, manifest key/digest/byte length, idempotency key | `state=AVAILABLE`, `publication_id`, `catalog_cursor` |
| `/captures/{id}/fail` | Fence, reason, idempotency key | Catalog failure receipt |

A `CaptureLease` contains `capture_id`, `fencing_token`, `dataset_id`,
`sample_id`, `generation_id`, `expires_in_seconds`, and `renew_after_seconds`.
TTL values are relative to the Catalog response, not shared machine clocks.
The producer should conservatively compute local deadlines from request start.

The Catalog must validate namespace, budget, owner coverage, descriptors,
checksums and the full decoded manifest at seal. Every retry must check the
current fence/state, including previously successful seals; a deleted sample
must not become writable again. Publishing atomically records the publication
and durable outbox event. Published captures cannot be reclaimed by the
unfinished-capture expiry path. Consumer retention and GC remain Catalog-owned.

The writer pre-registers the manifest alongside tensor identities. The manifest
descriptor has `object_id=manifest`, `kind=manifest`, key, byte length, digest
and owner. Its write receipt follows the manifest put. Catalog metadata stores
the exact manifest bytes to support recovery across ambiguous seal responses.

Only metadata enters the journal and HTTP API. Tokens, logits and KV bytes are
raw Store objects. Journal files should live on a durable volume; one process
owns a directory through a filesystem lock. A rejected fence leaves its entry
for Catalog/operator reconciliation and never triggers unconditional deletion.

## Verification

```bash
PYTHONPATH=python python3 -m pytest test/registered/unit/training_capture -v
PYTHONPATH=python python3 -m pytest test/registered/storage/test_training_snapshot_mooncake.py -v -s
```

The first command includes one CUDA ownership test. The second requires the
Mooncake SDK and `mooncake_master`. It starts an isolated master, writes through
`SnapshotWriter` and a registered arena, and reads/validates every object from a
different process over TCP. Its Catalog is a test double, not a real SpecForge
service. Every process it starts is cleaned up by the test.

The design schema remains in
`mooncake-study/training-data-contract/manifest.schema.json`. Produced manifests
are checked against that schema as well as cross-object and tensor-content
constraints. Strict SDK checks distinguish zero-status writes from byte-count
reads and reject clients without hard pin support.
