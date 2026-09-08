# TensorCast as an L3 KV Cache

This document describes how to use TensorCast as the L3 storage backend for
SGLang HiCache. The initial integration targets the Unified Radix Cache FULL
pool and uses TensorCast's public process-scoped
`RegionBackedArtifactSession`; SGLang does not construct daemon requests,
region layouts, or canonical TensorCast artifact IDs.

Related documentation:

- [TensorCast project](https://tensorcast.ai)
- [TensorCast repository](https://github.com/tensorcast-ai/tensorcast)
- [HiCache system design](https://docs.sglang.io/advanced_features/hicache_design.html)

## About TensorCast

TensorCast manages model state, KV caches, checkpoints, and other tensor state
as distributed artifacts. It separates cluster-wide discovery and routing from
host-local memory and data transfer:

- One **Global Store** manages artifact metadata, replica state, and routing.
- One **StoreDaemon** on each serving host owns local memory regions and moves
  artifact bytes locally or between hosts.
- Each SGLang rank attaches one process-scoped Session to its node-local
  StoreDaemon. The Session owns that rank's regions, RPC health, and transfer
  protocol details.

SGLang HiCache continues to decide when pages are published and prefetched.
TensorCast stores and moves the K/V fragments that make up those pages.

### Transfer modes

Both modes use the same artifact identity and HiCache operation path.

1. **Allocator-backed direct mode** is the default and recommended mode.
   - The TensorCast Session allocates the CPU tensors used to back the standard
     SGLang HostPool.
   - L2/L3 operations submit spans in those tensors directly, without an
     SGLang-side staging copy.
   - Separate HostPool allocations, including asymmetric MHA K and V tensors,
     become separate Session-managed regions automatically.
2. **Scratch mode** is the compatibility mode.
   - The standard SGLang allocator owns the HostPool tensors.
   - The Session lazily creates one fixed-capacity get arena and one
     fixed-capacity put arena and copies between them and the HostPool.
   - Operators must size `scratch.capacity_bytes` for the largest storage
     batch. Registration fails at startup if the configured capacity is too
     small.

## Requirements and Initial Scope

The initial integration requires:

- Linux and SGLang's CUDA backend;
- Unified Radix Cache with the primary FULL KV pool;
- `--hicache-host-memory-mode cache`;
- either `page_first` with the `kernel` I/O backend or
  `page_first_direct` with the `direct` I/O backend;
- a Global Store and node-local StoreDaemon started and ready before SGLang;
- `engine.cpu_shared_memory.enabled: true` in the StoreDaemon config; and
- a daemon-visible SGLang owner PID. Containers must share a PID namespace
  or provide an equivalent PID visibility arrangement.

The FULL path supports standard MHA, asymmetric MHA with separate K/V HostPool
tensors, and MLA. Currently TensorCast L3 does not support `buffer_only`, non-CUDA
platforms, `layer_first`, `page_head`, split-head layouts, DCP/attention CP,
packed MTP draft pools, or storage-v2 sidecar transfers such as Mamba and SWA.
There is no automatic fallback from allocator mode to scratch mode.

## Install TensorCast

Install an ABI-compatible TensorCast SDK and daemon:

```bash
pip install tensorcast
```

If the published wheel's Torch or CUDA build does not match the SGLang
environment, build TensorCast from source instead. See the
[TensorCast build guide](https://github.com/tensorcast-ai/tensorcast/blob/main/docs/development/build-from-source.md).

TensorCast is an optional SGLang dependency. Importing SGLang does not import
TensorCast unless this storage backend is selected.

## Deployment

The recommended deployment is operator-managed services plus SDK attachment:

- The operator starts one Global Store for the TensorCast cluster.
- The operator starts one StoreDaemon on every SGLang host and waits for it to
  become ready.
- SGLang ranks attach to the local daemon during HostPool construction.
- SGLang never starts, restarts, supervises, or stops TensorCast services.

### Single-host deployment

Run the following commands from the SGLang repository root. The checked-in
files under `configs/` are starter configurations and must be sized and secured
for the target deployment.

**Step 1: Prepare the environment**

```bash
export TC_CONFIG_DIR="$PWD/python/sglang/srt/mem_cache/storage/tensorcast_store/configs"
export TC_GLOBAL_SESSION=sglang-tensorcast-global
export TC_DAEMON_SESSION=sglang-tensorcast-daemon

# Large KV publishes can retain more than 1,024 artifact memfds.
ulimit -n 65535
```

**Step 2: Start the Global Store**

```bash
tensorcast-cli global start \
  --config "${TC_CONFIG_DIR}/global_store_config.yaml" \
  --gs-session "${TC_GLOBAL_SESSION}"
```

The checked-in config listens on port `50051`.

**Step 3: Start the StoreDaemon**

```bash
tensorcast-cli daemon start \
  --config "${TC_CONFIG_DIR}/store_daemon_config.yaml" \
  --global-store-mode connect \
  --global-store-address 127.0.0.1:50051 \
  --session "${TC_DAEMON_SESSION}"
```

The checked-in daemon config listens for SDK RPCs on port `50052` and uses
port `65090` for P2P transfers.

Its capability-token secret is fixed test material for local validation only.
A production copy must replace
`capability_tokens.active.secret` with independently generated secret material.
The Global Store and every daemon in a deployment must also use a consistent,
deployment-specific cluster identity.

**Step 4: Verify service readiness**

```bash
tensorcast-cli global status --gs-session "${TC_GLOBAL_SESSION}"
tensorcast-cli daemon status --session "${TC_DAEMON_SESSION}"
```

Do not start SGLang until both commands succeed. Session attachment additionally
checks that the daemon is ready, CPU shared memory is enabled, its local handle
service is reachable, and the endpoint is node-local.

**Step 5: Start SGLang in allocator mode**

Allocator mode is selected when `transfer_mode` is omitted:

```bash
python -m sglang.launch_server \
  --model-path <model-path> \
  --enable-hierarchical-cache \
  --hicache-host-memory-mode cache \
  --hicache-ratio 1 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend tensorcast \
  --hicache-storage-backend-extra-config '{
    "tensorcast": {
      "daemon_address": "127.0.0.1:50052"
    }
  }'
```

`page_first` plus `kernel` is also supported. Transfer mode controls L2/L3
movement; it does not silently select or change the HostPool layout or L1/L2
I/O backend.

**Scratch-mode alternative**

```bash
python -m sglang.launch_server \
  --model-path <model-path> \
  --enable-hierarchical-cache \
  --hicache-host-memory-mode cache \
  --hicache-ratio 1 \
  --hicache-mem-layout page_first \
  --hicache-io-backend kernel \
  --hicache-write-policy write_through \
  --hicache-storage-backend tensorcast \
  --hicache-storage-backend-extra-config '{
    "tensorcast": {
      "daemon_address": "127.0.0.1:50052",
      "transfer_mode": "scratch",
      "scratch": {
        "capacity_bytes": 4294967296
      }
    }
  }'
```

The 4 GiB value is only an example. Use the sizing rule below for the selected
model, rank topology, dtype, page size, and HostPool capacity.

**Step 6: Shut down in ownership order**

First terminate SGLang normally and allow its storage workers to join. Then
stop the StoreDaemon before the Global Store:

```bash
tensorcast-cli daemon stop --session "${TC_DAEMON_SESSION}"
tensorcast-cli global stop --gs-session "${TC_GLOBAL_SESSION}"
```

Do not stop the daemon while attached SGLang ranks are still running. The
StoreDaemon reclaims a rank's process-pinned stable region backing after that
rank process exits.

### Multi-host and multi-instance deployment

One Global Store may serve many hosts. Each host runs a StoreDaemon, and all
SGLang ranks on that host attach to its node-local endpoint:

```text
                         Global Store
                      (one per cluster)
                               |
               +---------------+---------------+
               |                               |
         StoreDaemon A  <----- P2P -----> StoreDaemon B
          /     |     \                       /     \
       rank 0 rank 1 rank N                rank 0 rank N
```

Multiple ranks and multiple SGLang instances may attach to the same local
daemon. Every rank owns a distinct process Session and one or more distinct
regions. SGLang appends `rank{world_rank}of{world_size}` to the configured
Session and region prefixes, while TensorCast makes concrete region names
PID-unique.

Instances reuse artifacts only when their artifact identity inputs agree,
including namespace, model ID, model version, FULL layout, dtype, page size,
and TP/PP rank topology. Each instance's rank 0 consumes the artifact shard
published by rank 0 of a compatible instance; ranks do not consume one
another's shards.

For cross-host traffic, configure every daemon to register with the same Global
Store and ensure its advertised and P2P addresses are reachable. Enable and
tune RDMA in the daemon config only when the host fabric and drivers support
it; otherwise configure the supported TCP transport.

## Configuration

The full extra-config value is a JSON object, or an `@path` reference to a
JSON, YAML, or TOML file. TensorCast-specific fields must be nested under
`tensorcast`:

```json
{
  "prefetch_threshold": 256,
  "prefetch_timeout_base": 1.0,
  "prefetch_timeout_per_ki_token": 0.25,
  "hicache_storage_pass_prefix_keys": false,
  "tensorcast": {
    "daemon_address": "127.0.0.1:50052",
    "namespace": "default",
    "transfer_mode": "allocator",
    "model_id": null,
    "model_version": "unversioned",
    "session_name_prefix": "sglang",
    "region_name_prefix": "sglang_tensorcast",
    "exists_timeout_s": 30.0,
    "transfer_timeout_s": null,
    "scratch": {
      "capacity_bytes": 16777216
    }
  }
}
```

### TensorCast fields

| Field | Default | Meaning |
|---|---|---|
| `daemon_address` | required | Numeric node-local `IP:port` for the StoreDaemon RPC endpoint. IPv6 must use `[address]:port`. Hostnames, DNS targets, and Unix socket URIs are rejected. |
| `namespace` | `default` | Byte-artifact namespace. Compatible publishers and consumers must use the same value. |
| `transfer_mode` | `allocator` | Selects `allocator` or `scratch` for the lifetime of the rank process. |
| `model_id` | SGLang storage model name | Optional stable model identity override. |
| `model_version` | `unversioned` | Immutable model/config revision. Production deployments should set a release- or content-specific value to prevent stale KV reuse after a model change. |
| `session_name_prefix` | `sglang` | Diagnostic Session prefix; SGLang appends the world-rank label. |
| `region_name_prefix` | `sglang_tensorcast` | Diagnostic region prefix; SGLang appends the world-rank label and TensorCast makes concrete names PID-unique. |
| `exists_timeout_s` | `30.0` | Positive metadata-only exists deadline in seconds. |
| `transfer_timeout_s` | `null` | Transfer deadline. The initial integration requires `null`; region get/put uses zero transparent retries. |
| `scratch.capacity_bytes` | `16777216` | Positive capacity in bytes for each scratch-direction arena. Ignored by allocator mode. |

### Generic HiCache fields

These optional fields remain at the top level of the extra config:

| Field | Default | Meaning |
|---|---|---|
| `prefetch_threshold` | `256` | Minimum prefix length in tokens before storage prefetch is attempted. |
| `prefetch_timeout_base` | `1.0` | Fixed portion of the Unified prefetch timeout in seconds. |
| `prefetch_timeout_per_ki_token` | `0.25` | Additional timeout in seconds per 1,024 tokens. |
| `hicache_storage_pass_prefix_keys` | `false` | Pass prefix keys to storage backends; TensorCast FULL v1 does not use them. |

### Scratch capacity

The exact minimum is known only after SGLang constructs and registers the
HostPool:

```text
maximum_batch_pages = min(128, host_pool.page_num)
page_bytes = sum(bytes of all registered fragments in one logical page)
minimum_capacity_bytes = maximum_batch_pages * page_bytes
```

For MHA, `page_bytes` includes both K and V. For MLA, it contains the single
combined KV fragment. The configured value applies independently to the lazy
get and put arenas, so budget up to `2 * scratch.capacity_bytes` of additional
host memory after both directions have been used.

The 16 MiB default is syntactically valid but may be too small for a real
model. An insufficient value fails rank startup with the configured and
required byte counts plus the calculated page geometry. Runtime scratch growth
is not supported.

### Supported layout matrix

| Host-memory mode | Transfer mode | Host layout | I/O backend | Result |
|---|---|---|---|---|
| `cache` | `allocator` | `page_first` | `kernel` | Supported; allocator is the default transfer mode. |
| `cache` | `allocator` | `page_first_direct` | `direct` | Supported and recommended for direct L1/L2 I/O. |
| `cache` | `scratch` | `page_first` | `kernel` | Supported with one copy at the L2/L3 boundary. |
| `cache` | `scratch` | `page_first_direct` | `direct` | Supported with one copy at the L2/L3 boundary. |
| `buffer_only` | either | any | any | Rejected. |
| `cache` | either | other layouts | any | Rejected initially. |

## Runtime and Failure Semantics

Session attachment is an early startup operation. A daemon readiness,
configuration, or allocator failure aborts rank startup; SGLang does not fall
back to another transfer mode.

During a supported FULL-v1 exists/get/put operation, a fatal Session or adapter
exception permanently disables TensorCast L3 for that rank. Subsequent calls
report no storage hits or false page results, while the SGLang worker continues
running and may recompute the missing prefix locally. Other ranks retain their own
rank-local Session health.

Transfer mode, daemon endpoint, and Session options cannot be changed at
runtime. A failed or terminated Session cannot reattach in the same rank
process. Restart the rank to establish a new Session.

`TensorcastStore.close()` terminates Session admission but does not directly
release process-pinned allocator or scratch regions. Their mappings remain
valid for the HostPool lifetime, and the daemon reclaims stable backing only
after the SGLang rank exits.

## Troubleshooting

- **Attach fails before model startup:** verify Global Store and StoreDaemon
  status, `cpu_shared_memory.enabled`, the numeric node-local daemon address,
  local handle socket access, and PID visibility.
- **`Too many open files` from `memfd_create`:** raise the StoreDaemon's
  inherited `RLIMIT_NOFILE`, for example with `ulimit -n 65535`, and restart
  the failed daemon and rank processes.
- **Scratch capacity is insufficient:** use the exact required byte count in
  the startup error and increase `scratch.capacity_bytes`; remember that get
  and put may allocate two arenas of that size.
