# KVCR direct linker

The KVCR linker transfers registered GPU KV pages directly to and from KVCR's
local DRAM or a hinted remote KVCR peer. It uses `deposit`, `query`, and `deliver`,
with at most four layer deliveries outstanding. Its owner thread continuously
polls KVCR, including while idle, so peer requests and transfer lifecycle events
keep progressing. Framework source-pin requests are declined; peers can retrieve
completed deposits in KVCR-owned DRAM.

For SWA/state layouts, ranks confirm a shared prefix boundary by fetching its
trailing window into KVCR DRAM with the complete expected layout. Those claims
remain held through GPU loading. A missing window retries a shorter shared
boundary; sparse state can require several probes. Full-prefix KV still uses
direct delivery, and layouts without trailing state skip this confirmation.

## Setup

Install KVCR with its NIXL/UCX dependencies in the serving environment. Use the
public API from KVCR revision `8a91e20` or a compatible revision that includes
the local-copy timeout lifecycle fix.

DRAM additions and removals carry `ownership="kvcr"` in the existing KV event
stream. Engine GPU events and cache clears retain the default ownership. Dynamo
needs its ownership-aware, versioned state-agent ingress to consume these events;
the legacy ingress rejects KVCR ownership. In-process DRAM does not survive engine
loss or runtime reset, but its inventory shares the retained cache-owner lifetime.
Ownership is not a durability claim.

For in-process Dynamo serving, PR #16827 (on top of #16117) supplies the
opt-in SGLang state-agent attachment. Its `--kvcr-state-agent-owner` option
currently accepts only single-rank aggregated direct-linker workers. It must
not be enabled for HiCacheStorage or TP>1, and is not an SGLang server flag.
The ordinary source-hint path and independently owned CPU inventory are
different integration contracts; a successful peer read alone does not prove
the latter.

This upstream port keeps loaded GPU slots private until the batch's forward
completes. Layer waits resolve without publishing partial cache contents;
the scheduler then waits for terminal native batch completion and aborts a
failed load before consuming its output. Queued, unstarted loads may be
cancelled, but active native transfers retain their slots until quiescence.
TP replicated-page dedup and process-wide GC tuning are excluded from this
initial integration. No new model accuracy or performance result is claimed.

## Worker configuration

An example worker configuration is:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B-FP8 \
  --trust-remote-code --tp-size 1 --page-size 64 \
  --attention-backend flashinfer --kv-cache-dtype bfloat16 \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend kvcr \
  --hicache-storage-backend-extra-config '{
    "dram_size_gb": 8,
    "control_host": "0.0.0.0",
    "advertise_host": "192.0.2.10",
    "control_port": 19500,
    "nixl_port": 20500,
    "operation_timeout_ms": 1000,
    "abandon_timeout_ms": 5000
  }'
```

Replace `192.0.2.10` with an address reachable by the other workers. These options
are strict; unknown names fail startup.

| Option | Default | Meaning |
| --- | --- | --- |
| `kvcr_service_socket_path` | `/run/kvcr/memory.sock` | Guard service socket; `null` explicitly selects in-process DRAM. |
| `guard_index` | `0` | Base Guard index; the global rank is added. |
| `compatibility_digest` | Model/layout/shard namespace | Must match the Guard service's deployment compatibility digest. |
| `dram_size_gb` | `1` | Fallback in-process DRAM GiB per rank, rounded down to complete pages across all pools. |
| `control_host` | `0.0.0.0` | Address on which KVCR binds peer control. |
| `advertise_host` | Resolved local hostname | Address peers use to reach this worker. |
| `control_port` | `19500` | Base control port; the global rank is added. |
| `nixl_port` | `20500` | Base NIXL port; the global rank is added. |
| `operation_timeout_ms` | `1000` | Positive operation deadline in milliseconds. |
| `abandon_timeout_ms` | `5000` | Failure-reporting deadline, at least twice the operation timeout. |
| `enable_telemetry` | `false` | Log aggregate KVCR timings, counters, and state at INFO every five seconds and on close/reset. |
| `trailing_capacity_divisor` | `1` | Positive integer reducing SWA-derived pools' page capacity in in-process DRAM; each retains at least one page. Guard capacities are configured separately. |
| `replicated_source_rank_routing` | `false` | Opt in to fetching replicated KV from the corresponding source rank; requires the source topology described below. |

## Guard memory and fallback

Workers first claim Guard-backed memory through KVCR's public API. A missing
socket or refused connection falls back to anonymous DRAM inside the serving
process. An incompatible/busy Guard, permission error, or interrupted claim
fails startup: these do not prove that falling back is safe. KVCR owns Guard
claiming, recovery, and release; SGLang does not allocate a second DRAM pool
when a claim succeeds.

Provision the service using KVCR's `python -m kvcr.kvcr_service` command with
`--socket-path`, `--pool-dir`, `--guard-count`, `--pool-sizes-gb`, and
`--compatibility-digest`. Startup logs include the requested digest and ordered
pool layouts (name and block bytes). The service's ordered pool capacities, not
`dram_size_gb`, size Guard memory. Allocate a distinct Guard index per worker;
reserve separate index ranges and ports for colocated replicas. A service has
one digest across its Guards. For nonreplicated TP shards, set the same explicit
deployment digest on the service and all workers; change it when the model,
dtype, layout, page size, or parallel topology changes. Storage keys still retain
their individual shard namespaces.

In fallback mode, pools receive the same page capacity by default. Increase
`trailing_capacity_divisor` to give sparsely stored SWA/state pools fewer pages
and reserve more of the budget for full-prefix pools. Set
`"kvcr_service_socket_path": null` to select this mode explicitly.

## Operational constraints

Enable `replicated_source_rank_routing` only when every hinted source is a
single-node worker with the same attention-TP size, advertises its attention
group's rank-zero control endpoint, and uses consecutive control ports. The
target must also use single-node, CP1/PP1 serving with rank-replicated pools.
Each target attention-TP rank then adds its rank to the advertised source port.
Leave this disabled for unknown or mixed source topologies; the original
advertised endpoint is used unchanged by default.

DRAM and GPU registrations remain owned until successful KVCR shutdown. A
shutdown error retains the mappings; an uncertain transfer keeps its affected
GPU pages unavailable for reuse until quiescence is established.

As with other external Linkers, `flush_cache` drains transfers but does not
clear Guard storage. Recovered KV is valid only for unchanged model weights and
layout. Online weight replacement requires a fresh model/cache identity or
restarting with cleared storage and matching peers; flush alone is insufficient.

Remote hints use KVCR's versioned `kv.fetch` envelope, with a reachable
`source_control_endpoint` and uint64 `block_hashes`. Hints are advisory. The
full-prefix lookup does not reserve source storage: capacity eviction between
lookup and delivery is an admitted load failure, including eviction caused by an older
offload. Failed admitted loads follow Linker's error path, without automatic
recomputation. Unsupported hint actions are ignored with a warning. The
runtime maps a storage hash's first 16 hexadecimal digits to the router hash.
With `cache_salt` or `extra_key`, existing KV events use a different hash chain;
those event hashes do not produce remote hits unless the router supplies the
corresponding storage hashes. Hints must target the matching source shard. The
adapter disables remote hints for layouts requiring shard-specific routing,
including CP/PP splits. Local reuse remains enabled.

## Manual checks and profiling

Install model weights and the normal SGLang test dependencies alongside the
KVCR/NIXL/UCX prerequisites above. Run the manual acceptance harness from the
repository root; `--model-path` accepts a local checkpoint:

```bash
python test/manual/cache/test_kvcr_linker.py --model qwen3 --case local
python test/manual/cache/test_kvcr_linker.py --model qwen3 --case peer --profile
```

This example uses TP1. Peer checks need two GPUs on one host and, when using
Guard, at least two Guards. Reserve control ports 19500 / 19600 and NIXL ports
20500 / 20600.
Start with fresh Guard pools for both replicas, especially the destination, so an
old local copy cannot masquerade as peer delivery. Use a new `--pool-dir` when
provisioning the Guard service for these checks.

Local cases require host-tier hits and compare cached decode logprobs with cold
prefill. Peer cases also compare greedy tokens with a cold source baseline,
with no inference requests to the source during peer checks.

`--output-dir` retains server logs, optional CPU/GPU profiles, and
`runtime_versions.json` with the installed KVCR version/source and available
repository commits/worktree changes. For profiling, align per-batch debug logs
(per-layer readiness, total batch drain time, and CPU `deliver()` submission time)
with the `--profile` GPU transfer and transformer-kernel timeline. Record the
overlap interval and bytes transferred; CPU readiness times and overlapping HTTP
requests alone cannot prove DMA/compute overlap. Use a system GPU trace if the
PyTorch profile does not expose NIXL transfers. The harness asserts neither
throughput nor overlap.

Run unit tests in separate processes, as CI does:

```bash
python -m pytest test/registered/unit/mem_cache/test_unified_cache_linker.py -q
python -m pytest test/registered/unit/mem_cache/test_linker_pool_assembler.py -q
python -m pytest test/registered/unit/mem_cache/test_kvcr_direct_linker.py -q
python -m pytest test/registered/unit/mem_cache/test_registry.py -q
```
