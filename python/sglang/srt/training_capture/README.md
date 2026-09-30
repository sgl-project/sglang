# Training Snapshot Producer

This package implements the tensor contract and producer-side data plane for
`maas_target_kv_v1`. It is under development. Runtime collection is opt-in through
`--training-capture-config`; see the implementation status document for verified
model/runtime combinations. The default configuration does not collect requests.

## Runtime Configuration

An example for an unquantized Qwen3-0.6B target with layers 0, 14 and 27:

```json
{
  "dataset_id": "dspark-qwen3",
  "model_id": "Qwen/Qwen3-0.6B",
  "producer_revision": "your-immutable-sglang-build-revision",
  "selected_layer_ids": [0, 14, 27],
  "catalog_endpoint": "http://catalog:8080/v1/training",
  "catalog_token_env": "CAPTURE_CATALOG_TOKEN",
  "journal_directory": "/var/lib/sglang/capture/worker-0",
  "store": {
    "local_hostname": "10.0.0.11:50052",
    "master_server_addr": "10.0.0.10:50051",
    "protocol": "tcp",
    "metadata_server": "P2PHANDSHAKE",
    "global_segment_size": 0,
    "local_buffer_size": 16777216,
    "rdma_devices": ""
  },
  "sample_ratio": 0.01,
  "max_sample_tokens": 8192,
  "max_inflight_samples": 4,
  "max_host_bytes": 536870912,
  "storage_chunk_tokens": 256
}
```

Use `--disable-overlap-schedule` for the initial ordinary AR collector. The
configuration is checked before weights load; unsupported TP/PP/DP, speculative,
PD, mixed-chunk, LoRA, quantized or embedding execution is rejected. Model/pool
binding additionally requires a local safetensors target, local tokenizer
artifacts, standard unscaled RoPE, full attention, and dense unquantized NHD
BF16/FP16 KV. Initial codecs recognize Qwen3, Qwen2 and Llama implementations;
recognition is not a substitute for per-model numerical verification.

Keep model files immutable during loading and serving. Weight/tokenizer file
digests and actual attention geometry, K norm, RoPE and output transforms bind
each sample. Optional `expected_weights_revision` and
`expected_tokenizer_revision` pin the computed artifact digests. Weight updates
or memory release disable capture and invalidate active attempts; start a new
producer with a newly bound identity after changing the model.

The Catalog endpoint must implement the protocol below. A separate Mooncake
data node owns storage when the producer uses `global_segment_size=0`; its
lifetime must exceed producer shutdown. A process-exclusive durable journal
directory is required per worker. RDMA uses the existing SDK's `protocol=rdma`
and `rdma_devices` settings and requires a separately validated NIC allocation.

Admission samples requests only once, excludes health checks and unsupported
request features, and requires capacity for prompt plus maximum response.
Background threads reserve/renew Catalog leases and publish snapshots. The
inference path never performs Catalog or Store network calls. Lack of a free
registered slot drops capture admission without delaying inference. A request
abort/retract fails the entire attempt; no partial READY sample is published.
Counters, Host slot states and the capture disable reason are exposed as
`training_capture` in SGLang's existing internal-state response.

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
  `recover()` confirms the fence, prepared state and each stored tensor digest,
  retries the same manifest,
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
PYTHONPATH=python python3 test/registered/storage/test_training_capture_runtime.py --model-path /models/Qwen3-0.6B -v
```

The first command includes one CUDA ownership test. The second requires the
Mooncake SDK and `mooncake_master`. It starts an isolated master, writes through
`SnapshotWriter` and a registered arena, and reads/validates every object from a
different process over TCP. Its Catalog is a test double, not a real SpecForge
service. Every process it starts is cleaned up by the test.

The third command adds actual SGLang prefill/decode, a prefix hit, a one-token
response and a serving logit bias. A test-only observer records each selected
layer's attention inputs and raw logits, independently of the exporter and KV
pool. Separate Store readback must match K/V and saved logits exactly, include
valid top-128 IDs, and match full-vocabulary LSE within 1e-5. The test verifies
readback after producer exit, then runs the same requests on normal serving with
decode CUDA graphs and compares every captured tensor. This currently covers
graph replay at batch size one.

A Transformers reference additionally checks teacher logits/LSE and reports KV
errors. BF16 intermediate KV is sensitive to the target implementation; the
optional `--assert-hf-kv` cross-engine gate currently fails for this fixture.
Its original tolerances are unchanged. See `mooncake-study/IMPLEMENTATION_STATUS.md`
and `mooncake-study/experiments/diagnose_qwen3_kv.py` for the reference-only
reproduction. Exact preservation of online tensors is the capture gate; DSpark
training/serving parity remains a separate work package.

Target recomputation exists only in development tests; the producer/reader API
never uses it. These tests use a Catalog double and do not validate production
consumer retention. The environment lock and resident-worker invocation are in
`mooncake-study/experiments/README.md`.

The design schema remains in
`mooncake-study/training-data-contract/manifest.schema.json`. Produced manifests
are checked against that schema as well as cross-object and tensor-content
constraints. Strict SDK checks distinguish zero-status writes from byte-count
reads and reject clients without hard pin support.
