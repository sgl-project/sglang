# Weight-cache correctness foundations (Phase 0A)

This opt-in package is the first diffusion fast-recovery implementation stage.
It does **not** enable a diffusion cache CLI, start a daemon, change existing
loaders, or modify SRT's legacy weight-cache transport. Those integrations follow
only after the preparation/admission and model-parity gates pass.

## State model

`snapshot_module()` walks registered parameters and buffers without losing
duplicate registration paths or invoking `state_dict` hooks. `StateManifest`
records:

- one bounded storage descriptor per original storage;
- tensor shape, dtype, stride and element offset;
- distinct storage groups and tensor-object groups;
- parameter/buffer kind, per-path buffer persistence and `requires_grad`;
- each module's training flag, including shared module paths.

Storage groups account for full storage bytes, not the sum of overlapping views.
Zero-byte storages remain distinct. Parameters and buffers sharing one *object*
across kinds, sparse/quantized layouts, unresolved conjugate views, parameter
subclasses and unsupported parameter metadata fail closed.

`import_state()` requires a meta module whose **finalized** schema matches the
manifest. It validates all registrations and storage bounds before changing the
module, constructs zero-copy typed views, restores exact object ties and buffer
persistence, and restores training flags without recursive `train()` side effects.
Scalar/container metadata and constructor-local callables on ordinary parameters
are retained; remote Python objects and derived tensor attributes are not imported.
Adapters remain responsible for derived state and for auditing immutable forward
behavior. `.eval()` and `requires_grad=False` are not write protection.

## One-shot CUDA IPC ownership

Use these primitives only between trusted processes in the same Linux PID
namespace, with visibility of the same physical CUDA device. The current CUDA
tests use PyTorch 2.13.0+cu130; private Torch IPC compatibility is version-checked.

1. The producer loads/finalizes a module, then constructs `CudaIpcExporter`.
   It retains the module and original storages, synchronizes producer CUDA writes,
   and creates a generation containing PID/start ticks, nonce, manifest digest,
   physical device UUID and Torch version.
2. The client validates its expected manifest/generation, then constructs
   `CudaIpcImporter`. This starts its producer watchdog **before any import** and
   translates physical UUID to the client's local CUDA ordinal.
3. For each fetch, the client requests a fresh `uuid.uuid4().hex`. The producer
   calls `export(request_id, generation=expected_generation)` once. Every nonempty
   storage gets a new counted Torch IPC send. Empty storage needs no send.
4. The transport delivers that `IpcDelivery` exactly once to the requesting
   consumer. The consumer calls `receive(delivery, meta_module, request_id=...)`.
   Reusing a request ID or locally re-importing a delivery is rejected. A retry
   requires a new request ID and new export, not replay of old payload bytes.
5. The service keeps the exporter and allocations alive until **all** consumers
   have stopped using them. Disconnecting a socket does not release ownership.
   `stop_admission()` rejects new fetches but frees no weights.

The allocation handle is opaque PyTorch metadata, not necessarily a raw 64-byte
CUDA handle. Its counted-send metadata is never cached as a reusable payload.
There is one import/release owner per storage send; tensor views do not create
extra sends. `expandable_segments` is rejected, including runtime configuration.

`CudaIpcImporter.close()` refuses while any imported C++ storage is still live,
including detached views or a partial-import exception traceback. Process-local
replay history is bounded. Do not move an exporter/importer across `fork`.

## Failure/resource policy

Producer death causes consumer SIGKILL via a pidfd watcher (PID/start-tick polling
fallback). This is bounded fail-stop detection, **not** safe continued execution
or an atomic barrier against arbitrary producer death. Graceful service shutdown
must stop admission, drain/terminate consumers, then release producer ownership.

Torch can retain send bookkeeping after partial delivery, failed import or fatal
consumer exit. Never guess which private counters to decrement. This package
uses a non-refundable per-generation budget, by default **128 delivery attempts**
and **65,536 storage-export reservations**. Empty storage reservations are counted
conservatively too. Rejections happen before another export; partial failures do
not refund reservations. `stats()` reports reservations, limits, failed exports,
cumulative event-bearing exports, unique weight bytes and admission state.

When exhausted, coordinate a consumer drain and producer restart. The defaults
are safety bounds, not tuned production lifetime promises. A future control
service must expose them and propagate errors; it must never silently reload disk
weights while a live producer occupies the GPU. This implementation does not
claim unlimited leak-free crash recovery.

## Identity helpers

`source_digest()` hashes the whole installed `sglang` Python package, including
reused SRT/kernel code. Deployment identity must additionally cover relevant
native/external dependencies or use a verified immutable build artifact ID.

The socket locator uses a 16-hex device hash and a 32-hex compatibility prefix.
The default path is 89 encoded bytes; paths above Linux's 107-byte pathname budget
are rejected. **Always compare the full compatibility digest at handshake**;
prefix collisions are not cache hits. Directory permissions/ownership, device
locking, peer credentials and atomic readiness belong to the subsequent service.

`checkpoint.py` builds portable SHA-256 content manifests for explicitly supplied
config/weight files, excluding the manifest itself:

```bash
python -m sglang.weight_cache_common.checkpoint /path/to/checkpoint config.json weights.safetensors
```

The daemon uses `verify_manifest()` against the frozen recipe's exact file set
and publishes a `VerifiedCheckpoint` receipt. The client re-reads the publisher's
manifest and calls `check_verified_stats()` without reading tensor bytes. This
requires trusted, stable checkpoint publication: **every rewrite must regenerate
the manifest**. Same-size edits within one filesystem timestamp tick can evade
stat checks; these checks do not make an actively rewritten directory immutable.
Local symlinks escaping the declared checkpoint root are rejected. HF snapshot
resolution/publication and component overrides are integrated in a later stage.

## Tests

```bash
python -m pytest test/registered/unit/model_loader/test_weight_cache_common.py -q
python -m pytest test/registered/model_loading/test_weight_cache_common_ipc.py -q -s
python -m pytest test/registered/unit/model_loader/test_weight_cache_protocol.py -q
```

The CUDA suite uses tiny synthetic modules, spawned processes and no checkpoints.
It checks alias/forward parity, import-only CUDA allocations separately from
forward workspaces, sequential/concurrent clients, independent send counters,
partial and fatal clients, producer death, and bounded repeated-attach resources.
The tests' counter inspection is a Linux diagnostic; production code never reads
or edits the shared counter file directly. Deliberately abandoned/fatal tests can
produce Torch's expected outstanding-reference warning when their producer exits.
