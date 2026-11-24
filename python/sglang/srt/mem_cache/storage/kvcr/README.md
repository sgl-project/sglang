# KVCR HiCacheStorage backend

This optional backend uses KVCR's indexed-region public API and NIXL/UCX
transport. KVCR is not an SGLang dependency: install KVCR 0.2.0 or later from
PyPI. An older KVCR (0.1.x) is rejected at startup. Selecting another HiCache
backend does not import KVCR.

KVCR 0.2.0 needs NIXL 1.5.x. The SGLang image already has NIXL with its CUDA 13
binary, so install KVCR there without dependencies; otherwise pip also adds the
CUDA 12 NIXL binary:

```bash
pip install --no-deps "kvcr>=0.2.0"
```

Elsewhere, `pip install "kvcr>=0.2.0"` also installs NIXL, msgspec and pyzmq.

## Ownership and data path

SGLang owns GPU memory, HiCache host pages, scheduling, and host-to-GPU copies.
KVCR owns its deposited DRAM cache:

- Backup: GPU -> HiCache host pages -> KVCR-owned DRAM.
- Restore: local or peer KVCR DRAM -> locked HiCache host pages -> GPU.

This is not a direct-HBM restore. Peer delivery does not require an
intermediate allocation in the target's KVCR DRAM cache. Unpinned SGLang host
pages are never offered as remote sources. By default every rank stores its own
shard, including rank-replicated MLA KV; `share_replicated_kv` keeps one copy
per replica instead (see Shared replicated KV).

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

Keys and defaults (ports are bases; each rank adds its DP/TP/CP offset):

- `control_advertise_host` (`127.0.0.1`): IPv4 address or host name peers and
  routers dial for this worker's control endpoint.
- `control_host` (`0.0.0.0` with remote hints, else `127.0.0.1`): IPv4 address
  or host name the control endpoint binds. KVCR 0.2.0's control channel is
  IPv4 only, so both hosts are rejected as IPv6 literals. The endpoint exists
  only with `enable_remote_hint` or `share_replicated_kv`; peers fetch through
  it without authentication, so bind it to the peer-facing interface where
  possible.
- `control_port` (`32100`): base control port.
- `nixl_listen_port` (`control_port + 1024`): base NIXL listen port. The
  default lies in Linux's ephemeral port range; prefer a base outside it, or
  reserve the ports (`ip_local_reserved_ports`).
- `local_dram_bytes` (8 GiB): KVCR DRAM per rank; each pool gets the same
  fraction of its host pages (bytes split in proportion to host bytes). For
  rank-replicated KV (MLA), every attention rank keeps its own copy of each
  page unless `share_replicated_kv` is on: TP x CP copies in DRAM and in backup
  traffic. The startup log names the multiplier.
- `local_dram_pool_pages` (`{}`): `{pool name: pages}`, an integer of at least
  1 per named pool, overriding that split.
- `operation_timeout_ms` and `abandon_timeout_ms` (derived): KVCR's deadlines,
  counted from each deposit or deliver call; KVCR requires abandonment to be
  at least twice the operation deadline. An operation not started by the
  operation deadline is a miss, and one in flight then fails. A transfer,
  local or remote, still in flight at the abandonment deadline is uncertain
  and ends the worker (see Failure handling). By default the operation
  deadline is the time one HiCache batch (128 pages of every pool, per rank)
  takes at 250 MB/s, and at least 5 s; abandonment is three times that, and at
  least 30 s. Setting only one derives the other, and the startup log prints
  both. Lower abandonment shortens a dead peer's stall; raise both for links
  slower than that rate.
- `enable_remote_hint` (`false`): see below.
- `share_replicated_kv` (`false`): one copy of rank-replicated KV per replica;
  see below.
- `extra_backend_tag` (unset): hashed into the storage namespace. The namespace
  keys on the served model name and the KV layout, not on the weights, so
  deployments that serve different weights under one served model name and
  share a router must set different tags.

An optional positive integer `startup_timeout_seconds` overrides KVCR's startup
budget; omitting it keeps KVCR's default. Unknown keys are ignored with a
warning.

This version does not report backend transfer metrics. With
`--enable-storage-metrics`, only HiCache's own prefetch and backup counters are
recorded for KVCR.

## Remote hints

`enable_remote_hint` is off by default; without it, KVCR serves only its own
local DRAM and binds no control endpoint (each rank still binds its NIXL
listen port). Add `"enable_remote_hint": true` to the extra config to turn it
on.
When on, lookups and reads consume the request's `kv_hints`
(`kv.fetch` actions, version 1.0, copy mode). A hint names a source replica's
`tcp://` control endpoint, and each rank fetches from the same shard offset
there. Hints are caller-supplied and unauthenticated: enable them only where
the request source is trusted and every advertised endpoint is a peer you
intend to read from. For salted or adapter-keyed requests, only pages bound to
the request's own event IDs are fetched. A query is advisory; terminal delivery
decides which pages are usable.

With an explicit `control_advertise_host`, remote hints also report the
worker's control endpoint to a router (see Router discovery).

For models with trailing/checkpoint pools (SWA, Mamba, DeepSeek V4 states),
peer reuse is effectively disabled in this version. A hinted page does not
prove that the peer holds the checkpoint state, so those pools count only local
hits, and every restorable prefix must end where the checkpoint is in local
DRAM. Peer pages cannot extend a prefix past that point. Confirming peer
checkpoint state is left to a follow-up.

## Shared replicated KV

With `"share_replicated_kv": true`, a deployment whose KV is replicated across
attention TP ranks (MLA: DeepSeek V2/V3/V4, GLM and Kimi DSA indexers) keeps
one copy of those pools per replica, on the rank HiCache elects as their writer
(TP rank 0), as HiCache's other backends do. The other ranks hold no KVCR DRAM
for them: they restore each page from the writer's KVCR on the same host,
through its control endpoint, and defer lookups to the writer (HiCache takes
the minimum hit across ranks). Pools HiCache keeps per rank (Mamba/KDA state,
MHA draft pools) stay per rank. It applies only with one node per worker
(`--nnodes 1`) and attention CP=1; otherwise, or when it is off, the startup log
warns that every rank keeps a copy.

Costs: each rank's control endpoint exists (bound to `127.0.0.1` unless remote
hints or `control_host` say otherwise), unauthenticated to anything that can
reach it. Every non-writer read is a peer pull from the writer: a restore of a
shared pool no longer runs TP local reads in parallel, but goes through the
writer's one KVCR source, which serves its own read and every sibling's pull of
the same pages at once (and remote replicas' ranks, with remote hints). On
DeepSeek-V2-Lite at TP2 with `sm` in `UCX_TLS`, restores took about 1.5 times
as long as with a copy per rank; the cost grows with TP. A replica reads pages
its writer lacks from the router-hinted replica only after the writer answers.
The default deadlines stay per rank (one batch at 250 MB/s); raise
`operation_timeout_ms` if the writer's link cannot serve TP batches at once
well within it. With KVCR 0.2.0, a writer whose progress stalls past
`operation_timeout_ms` stops serving peers until restart (logged as an error):
its replica then misses in L3 for the shared pools. Include `sm` in `UCX_TLS`
so same-host pulls use shared memory.

## Router discovery

With remote hints on and an explicit `control_advertise_host`, the engine
reports one `kv_hint_sources` entry per DP replica in `scheduler_infos`, from
its first shard: `dp_rank`, `source_control_endpoint` (`tcp://host:port`) and
`capabilities` `["router_hint"]`. A router such as Dynamo's reads it to name
this worker in `kv.fetch` hints; peers reach the other shards by rank offset
from that endpoint.

This version does not report what KVCR's DRAM holds, so such a router treats
the worker's GPU and HiCache host pages as fetchable too. A peer's KVCR serves
only its own DRAM, so many of those hints miss and the target recomputes.
Publishing KVCR residency to the router is left to a follow-up.

Limits of this version: with remote hints, each worker runs on one node
(`--nnodes 1`; peers may be on other hosts); PP=1, no DCP, and not replica DP
combined with attention DP.

## Failure handling

Successful or missed operations must reach terminal native completion before
HiCache can reuse their buffers. If native transfer state is uncertain, the
worker exits (`os._exit(1)`) because HiCache cannot quarantine pages that NIXL
may still access. Startup, progress and shutdown failures likewise must not
release registered memory while native access may remain possible. The `kvcr`
version and the options are checked when the backend is created; the per-pool
DRAM split and the control and NIXL listen ports when the cache attaches its
host pools, which fails the launch if a port is taken (the ports stay reserved
until KVCR starts). Memory registration happens on the first storage operation
(a draft sidecar pool may still register until then), where a failure
terminates the worker.

A peer that dies or hangs during a fetch is the main case of an uncertain
transfer. Nothing fails fast: the reading rank's prefetch IO waits until
`abandon_timeout_ms`, and then the worker exits, because KVCR cannot prove that
the source will not still write into the HiCache host pages. The fetch becomes
a miss only if the peer answers KVCR's probe before it has started writing, or
its write finishes within `abandon_timeout_ms` of the read. A peer that
restarted mid-transfer still leaves the pull uncertain: proving a restarted
source dead needs KVCR's guard, which this backend does not run. A source whose
own write is still running past its abandonment deadline exits as well.

## Validation

```bash
python test/registered/unit/mem_cache/test_kvcr_storage.py -v
python test/registered/unit/mem_cache/test_kvcr_endpoints.py -v
```

Most of these CPU tests replace the native transport with a byte-copying fake
and always run. `TestRealKVCR` runs only on Linux with KVCR and its NIXL
bindings installed, and is skipped otherwise: it starts real KVCR on host
memory and checks a backup, query and local restore round trip, and a shared
replicated rank restoring from its writer.
None of these tests cover peer reads across hosts, model accuracy or inference
performance.
