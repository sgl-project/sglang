# KVCR HiCacheStorage backend

This optional backend uses KVCR's indexed-region public API and NIXL/UCX
transport. Install KVCR 0.2.0 or later separately; selecting another
HiCache backend does not import KVCR.

## Ownership and data path

SGLang owns GPU memory, HiCache host pages, scheduling, and host-to-GPU copies.
KVCR owns its deposited DRAM cache:

- Backup: GPU -> HiCache host pages -> KVCR-owned DRAM.
- Restore: local or peer KVCR DRAM -> locked HiCache host pages -> GPU.

This is not a direct-HBM restore. Peer delivery does not require an
intermediate allocation in the target's KVCR DRAM cache. Unpinned SGLang host
pages are never offered as remote sources. Every rank stores its own shard,
including rank-replicated MLA KV.

## Configuration

Enable hierarchical caching and choose `--hicache-storage-backend kvcr`. KVCR
needs a `page_first` or `page_first_direct` host layout; a `layer_first`
request is switched to the page-first layout matching `--hicache-io-backend`.
Supply inline JSON through `--hicache-storage-backend-extra-config`, for
example:

```json
{
  "control_advertise_host": "<reachable-worker-host>",
  "control_port": 32100,
  "local_dram_bytes": 17179869184
}
```

This is an illustrative configuration, not a model-specific capacity
recommendation. Account separately for GPU cache, HiCache host staging, and
KVCR DRAM, which each rank allocates. Control ports must accommodate the rank
offsets and be reachable by peers. The control endpoint's TCP scheme does not
select the NIXL payload transport. The backend currently requires PP=1 and
DCP=1.

An optional positive integer `startup_timeout_seconds` requires a KVCR build
that accepts it. Omitting it preserves KVCR's normal startup budget. Transfer
and abandonment deadlines are separate from this initialization setting.

This version does not report backend transfer metrics. With
`--enable-storage-metrics`, only HiCache's own prefetch and backup counters are
recorded for KVCR.

## Remote hints

`enable_remote_hint` is off by default; without it, KVCR serves only its own
local DRAM. Add `"enable_remote_hint": true` to the extra config to turn it on.
When on, lookups and reads consume the request's `kv_hints`
(`kv.fetch` actions, version 1.0, copy mode). A hint names a source replica's
`tcp://` control endpoint, and each rank fetches from the same shard offset
there. Hints are caller-supplied and unauthenticated: enable them only where
the request source is trusted and every advertised endpoint is a peer you
intend to read from. For salted or adapter-keyed requests, only pages bound to
the request's own event IDs are fetched. A query is advisory; terminal delivery
decides which pages are usable.

For models with trailing/checkpoint pools (SWA, Mamba, DeepSeek V4 states),
peer reuse is effectively disabled in this version. A hinted page does not
prove that the peer holds the checkpoint state, so those pools count only local
hits, and every restorable prefix must end where the checkpoint is in local
DRAM. Peer pages cannot extend a prefix past that point. Confirming peer
checkpoint state is left to a follow-up.

## Failure handling

Successful or missed operations must reach terminal native completion before
HiCache can reuse their buffers. If native transfer state is uncertain, the
worker exits (`os._exit(1)`) because HiCache cannot quarantine pages that NIXL
may still access. Startup, progress and shutdown failures likewise must not
release registered memory while native access may remain possible.

## Validation

```bash
python test/registered/unit/mem_cache/test_kvcr_storage.py -v
python test/registered/unit/mem_cache/test_kvcr_endpoints.py -v
```

These CPU tests replace the native transport with a byte-copying fake. They do
not exercise NIXL, model accuracy or inference performance.
