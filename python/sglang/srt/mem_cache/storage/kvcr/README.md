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

With an explicit `control_advertise_host`, remote hints also report the
worker's control endpoint to a router (see below). Without
`publish_inventory`, Dynamo's router then treats the worker's GPU and HiCache
host pages as fetchable too, but a peer's KVCR serves only its own DRAM, so
many of those hints miss and the target recomputes. Use `publish_inventory`
with Dynamo's `kvcr` worker-selection policy.

For models with trailing/checkpoint pools (SWA, Mamba, DeepSeek V4 states),
peer reuse is effectively disabled in this version. A hinted page does not
prove that the peer holds the checkpoint state, so those pools count only local
hits, and every restorable prefix must end where the checkpoint is in local
DRAM. Peer pages cannot extend a prefix past that point. Confirming peer
checkpoint state is left to a follow-up.

## Inventory publication (Dynamo KV state agent)

With `"publish_inventory": true`, each rank reports what its KVCR DRAM
actually holds as KV events, so a KV-aware router (Dynamo's KV state agent)
can tell KVCR residency apart from SGLang's own GPU and HiCache host pages and
send `kv.fetch` hints that name a peer which really holds the prefix. It
requires remote hints and an explicit, peer-reachable, non-loopback
`control_advertise_host`; startup fails otherwise:

```json
{
  "enable_remote_hint": true,
  "publish_inventory": true,
  "control_advertise_host": "<reachable-worker-host>",
  "control_port": 32100
}
```

KV events must be on, with a ZMQ publisher in Dynamo format, e.g.
`--kv-events-config '{"publisher": "zmq", "endpoint": "tcp://*:5557",
"topic": "kv-events", "format": "dynamo"}'`. A wildcard bind is advertised on
`control_advertise_host`; the raw-event endpoint must be reachable from the
state agent's host as well. Startup fails for another format, a null or
connect-mode publisher, an IPC endpoint or a loopback bind.

The KVCR-owned events go out on the worker's ordinary KV event stream. Only
Dynamo's state agent reads their `ownership` today; sgl-router and
sgl-kv-indexer ignore it and would file KVCR DRAM pages as the worker's own
HiCache host pages. Do not combine `publish_inventory` with sgl-router KV
event routing.

What SGLang reports at startup (`scheduler_infos` of the engine):

- `kv_hint_sources`: one source per DP replica, from its first shard:
  `dp_rank`, `source_control_endpoint` (`tcp://host:port`), `capabilities`
  `["router_hint"]` whenever remote hints are on with an explicit
  `control_advertise_host`, plus `"kvcr_inventory"` and a `cache_owner_id`
  when publishing. The owner ID is Dynamo's `CacheOwnerId`,
  `<ns[:32]>:<ns[32:]>/0000000000000000/<uuid>`: the indexer domain is the
  64-hex storage namespace, the cluster field is fixed, and the incarnation is
  a per-process UUID, since KVCR's DRAM does not survive the process.
- `kv_event_sources`: per DP replica, the publisher's endpoint, `topic`
  (the configured topic plus `/owner/<sha256 of the owner>`), `block_size`
  and `format`.

Events are single-page `BlockStored`/`BlockRemoved` with `ownership: "kvcr"`
and medium `CPU_PINNED`, published from KVCR's native residency (not from an
acknowledged write). A page is stored once every attention TP x CP shard holds
it in every pool that must hold all pages (trailing SWA/Mamba/state pools are
not advertised), and removed as soon as any shard drops it. Event identities
come from the HiCache tree backups and prefetches that move a page; a page
resident only through a decode-side offload backup is not advertised.

End to end with Dynamo: start `python -m dynamo.kv_state_agent` in the
workers' namespace, then the frontend with the `kvcr` worker-selection policy,
which is what sends `kv.fetch` hints (with only `--router-mode kv`, the default
policy sends none):

```yaml
# worker-selection.yaml; add prefill/decode roles for disaggregated serving.
worker_selection:
  aggregated:
    type: kvcr
    parameters:
      shared_cache_multiplier: 0.5  # > 0 credits prefixes fetchable from a peer
```

```bash
python -m dynamo.frontend --router-mode kv --router-policy-config worker-selection.yaml
```

Keep the frontend's event-driven local indexer (no predict-on-route), which
fetch-source discovery requires. Then start `python -m dynamo.sglang` workers
with HiCache, this backend and the options above, KV events on. Each worker
attaches its owners to the state agent from this startup metadata. The router
then routes on, and hints from, actual KVCR residency, and a worker fetches
hinted pages from that peer's KVCR DRAM. See Dynamo's SGLang README
("KVCR-owned inventory") and router docs ("KV Fetch Hints").

Limits of this version: each worker runs on one node (`--nnodes 1`; peers may
be on other hosts), PP=1, no DCP, one tokenizer worker, and not replica DP
combined with attention DP; the backend must be configured at launch (no
runtime attach or detach). Inventory is process-local and starts empty: each
page is announced once, so pages stored before the worker attaches to the state
agent stay unknown to the router, and a dropped event (a ZMQ sequence gap)
stops the inventory until the worker restarts. Warm KVCR only after the worker
is attached.

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
