# TensorCast as L3 KV Cache

This document describes how to use TensorCast as the L3 storage backend for SGLang HiCache.

Related documentation:

- [TensorCast project](https://tensorcast.ai)
- [TensorCast README](https://github.com/tensorcast-ai/tensorcast/blob/main/README.md)
- [HiCache System Design and Optimization](https://docs.sglang.io/advanced_features/hicache_design.html)

## About TensorCast

TensorCast is a tensor state infrastructure layer that manages model weights, KV cache, checkpoints, and other tensor state as **distributed artifacts**. It separates control-plane scheduling from data-plane transfer:

- A central **Global Store** plans where artifacts live, how they move, and which replicas serve a given consumer, using artifact metadata, replica state, node load, and topology distance.
- Per-host **Store Daemons** own the local tensor memory and expose CUDA IPC handles for same-node zero-copy sharing;cross-node movement runs over RDMA or TCP P2P paths.

For SGLang HiCache, this means:

- Each SGLang serving host runs one Store Daemon. The Daemon owns the `HOST_SHARED` region(s) that back the L2 KV pool.
- One Global Store, shared by all Daemons in the cluster, brokers cross-host KV shards discovery and routing.

### TensorCast & SGLang HiCache

TensorCast serves as a high-performance L3 storage backend for SGLang HiCache. SGLang HiCache decides when to publish, prefetch, and evict; TensorCast handles cluster-wide page placement and the actual data movement, with two operating modes built on the same `HOST_SHARED` region model:

1. **Scratch-slab mode** (default).
    - HiCache L2 host pages live in ordinary host memory allocated by SGLang.
    - The TensorCast backend keeps one long-lived `HOST_SHARED` scratch slab per rank. KV pages are copied between L2 and the slab on every `put`/`get`.
2. **Allocator-backed direct mode** (recommended).
   - The L2 host pool itself is a `HOST_SHARED` region exported by the TensorCast Daemon. SGLang `mmap`s the region as the L2 buffer.
   - Page `put` and `get` skip the staging copy: the Daemon publishes directly from resident slot offsets and fetches directly into reserved destination slots.
   - Requires `--hicache-mem-layout page_blob_direct` (a page-major host layout introduced for this backend), and is opt-in via `host_allocator_enabled: true` in the backend extra config.

## Install TensorCast

**Method 1: with pip**

```bash
pip install tensorcast
```

**Method 2: from source**

See the [TensorCast build guide](https://github.com/tensorcast-ai/tensorcast/blob/main/docs/development/build-from-source.md).

## Deployment

When integrated with SGLang, the system has three components:

- The **Global Store** (one per cluster) — manages cluster-wide artifact
  metadata, placement decisions, and route resolution. Stateless across
  Daemons; recoverable from per-host Daemon inventory on restart.
- The **Store Daemon** (one per SGLang host) — owns local `HOST_SHARED` regions, runs the data-plane transfer engine, and registers with the Global Store. Each SGLang instance talks to the Daemon over its local gRPC endpoint.
- The **SGLang server** — runs serving and HiCache; its TensorCast HiCache backend connects to the worker-local Daemon. Each HiCache backend can only connect to a Daemon located on the *same* host, multiple backends (e.g., from multiple SGLang instances, or multiple ranks from one instance) can connect to the same local Daemon.

### Single-host deployment

For functional bring-up on a single host, all three components run on the same machine.

**Step 1: Start the Global Store**

```bash
tensorcast-cli global start \
    --config=python/sglang/srt/mem_cache/storage/tensorcast_store/configs/global_store_config.yaml
```

By default the Global Store listens on `0.0.0.0:50051`. Override with the `server.listen` block in the config.

**Step 2: Start the Store Daemon**

```bash
tensorcast-cli daemon start \
    --config=python/sglang/srt/mem_cache/storage/tensorcast_store/configs/store_daemon_config.yaml \
    --global-store-mode connect \
    --global-store-address 127.0.0.1:50051
```

The Daemon's gRPC endpoint defaults to `0.0.0.0:50052`. Each SGLang server will dial this endpoint as its `daemon_address`.

**Step 3: Verify both services are up**

```bash
tensorcast-cli global status
tensorcast-cli daemon status
```

**Step 4: Start the SGLang server with TensorCast enabled**

Scratch-slab mode (works with any `--hicache-mem-layout`, default for max compatibility):

```bash
python -m sglang.launch_server \
    --model-path <model-path> \
    --enable-hierarchical-cache \
    --hicache-storage-backend tensorcast \
    --hicache-storage-backend-extra-config '{"daemon_address": "127.0.0.1:50052"}'
```

Allocator-backed direct mode (zero-copy; requires `page_blob_direct`, recommended):

```bash
python -m sglang.launch_server \
    --model-path <model-path> \
    --enable-hierarchical-cache \
    --hicache-mem-layout page_blob_direct \
    --hicache-storage-backend tensorcast \
    --hicache-storage-backend-extra-config '{
        "daemon_address": "127.0.0.1:50052",
        "host_allocator_enabled": true
    }'
```

If `host_allocator_enabled=true` is passed without `page_blob_direct`, SGLang
fails closed at startup with a clear message rather than silently fall back.

**Step 5: Stop the services when done**

```bash
tensorcast-cli daemon stop
tensorcast-cli global stop
```

### Multi-host deployment

For real production runs, every SGLang host runs its own Daemon, and one designated host (any host in the cluster — typically a service host) runs the Global Store.

Topology:

```
                   ┌───────────────────────────┐
                   │       Global Store        │
                   │   (one per cluster)       │
                   └─────────────┬─────────────┘
                                 │ gRPC + HA registration
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
  ┌─────┴─────┐            ┌─────┴─────┐            ┌─────┴─────┐
  │  Daemon   │   RDMA P2P │  Daemon   │   RDMA P2P │  Daemon   │
  │ (host 0)  │◀──────────▶│ (host 1)  │◀──────────▶│ (host N)  │
  └─────┬─────┘            └─────┬─────┘            └─────┬─────┘
        │ gRPC                   │ gRPC                   │ gRPC
  ┌─────┴─────┐            ┌─────┴─────┐            ┌─────┴─────┐
  │  SGLang   │            │  SGLang   │            │  SGLang   │
  │ instance  │            │ instance  │            │ instance  │
  └───────────┘            └───────────┘            └───────────┘
```

Operationally this is the single-host flow repeated per host:

1. Start the Global Store on the chosen service host.
2. On every SGLang host: start the Store Daemon with
   `--global-store-mode connect --global-store-address <service-host>:50051`.
   Update the Daemon config so `high_availability.global_store_endpoints[0]`
   points at the service host.
3. On every SGLang host: launch SGLang with `daemon_address` set to the local Daemon endpoint (`127.0.0.1:<daemon-port>` is fine because the SGLang process and its Daemon share the host).

Each HiCache backend only talks to its own worker-local Daemon. Cross-host KV page resolution happens transparently through the Global Store + Daemon
RDMA P2P fabric.

For RDMA transport, set `communicator.enable_rdma: true` in the Daemon config and ensure the standard NCCL/IB environment is set in every Daemon and SGLang
launch (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_SOCKET_FAMILY`).

## Configuration

This subdirectory ships two starter configs under `configs/`:

- `global_store_config.yaml` — for the Global Store CLI.
- `store_daemon_config.yaml` — for each Store Daemon CLI.

These are templates. Copy them into your deployment and adjust the field in need.

### `--hicache-storage-backend-extra-config` reference

This is the JSON / YAML payload SGLang reads via
`--hicache-storage-backend-extra-config`. It can be passed inline on the command line or as `@/path/to/config.{json,toml,yaml}`.

**Required:**

| Key | Purpose |
|---|---|
| `daemon_address` | gRPC endpoint of the local Daemon (e.g. `"127.0.0.1:50052"`). |

**Required to enable allocator-backed direct mode** (otherwise scratch-slab is used):

| Key | Purpose |
|---|---|
| `host_allocator_enabled: true` | Switches to allocator-backed direct mode. **Must** be paired with `--hicache-mem-layout page_blob_direct`. |

**Optional — multi-tenant / multi-model isolation:**

| Key | Default | When to set |
|---|---|---|
| `namespace` | `sglang_hicache` | Several unrelated SGLang clusters share one TensorCast cluster. |
| `model_id` | falls back to `--served-model-name` | Several models share one Daemon and you want explicit page-identity scoping. |
| `model_version` | `"default"` | You roll multiple checkpoints / fine-tunes through the same Daemon and **must not** mix their KV pages. Pages with different `model_version` are distinct artifacts; same `model_version` are shared. |

**Optional — placement & lifetime:**

| Key | Default | When to set |
|---|---|---|
| `policy_profile` | `durable` | Pick a different TensorCast placement profile (`cache` / `durable` / `ha` / `cold` / `warm` / `pinned`). See [TensorCast docs](https://github.com/tensorcast-ai/tensorcast) for semantics. |
| `host_allocator_region_ttl_ms` | `0` (no TTL — region lives until SGLang exits) | You want the Daemon to reclaim the L2 host region automatically after a fixed lease, e.g. for canary-style restarts. |
| `host_allocator_region_name` | `sglang_tensorcast_host_pool` | Pure Daemon-side telemetry/debug label. Not a uniqueness key — the Daemon assigns its own `region_id`, so multiple ranks or processes sharing the default name do **not** collide. |

**Optional — prefetch tuning (shared with other HiCache backends):**

| Key | Default | When to set |
|---|---|---|
| `prefetch_threshold` | `256` (tokens) | Lower to make HiCache attempt storage hits for shorter prefixes. |
| `prefetch_timeout_base` / `prefetch_timeout_per_ki_token` / `prefetch_timeout_max` | see code defaults | Tighten or relax the linear per-prefetch timeout. |
| `hicache_storage_pass_prefix_keys` | `false` | Surface per-page prefix keys to the backend (advanced; usually leave off). |

All other keys accepted by `TensorcastHiCacheConfig` are internal and should
keep their defaults.

#### Configuration formats

The same payload can be passed three ways:

1. Inline JSON:
   ```bash
   --hicache-storage-backend-extra-config '{"daemon_address": "127.0.0.1:50052", "host_allocator_enabled": true, ...}'
   ```
2. JSON file:
   ```bash
   --hicache-storage-backend-extra-config @/etc/sglang/tensorcast.json
   ```
3. YAML or TOML file (recommended):
   ```bash
   --hicache-storage-backend-extra-config @/etc/sglang/tensorcast.yaml
   ```
