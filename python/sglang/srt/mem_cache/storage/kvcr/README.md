# KVCR as a HiCache L3 backend — reviewer's guide

Cross-instance KV reuse for SGLang: worker B serves a prefix that worker A
computed, over NIXL, steered by a dynamo router hint. This is a POC — it works
end to end, and the known gaps are listed at the bottom.

For the design rationale see `RFC_kvcr_hicache_backend.md` next to this file.
This document is only how to run it and what has been verified.

## What it is

KVCR (`nvidia-kvcr`, formerly `nvidia-kvcc` / "KV Cache Controller") is a
framework-neutral KV cache runner with its own DRAM tier and a NIXL data path.
This directory plugs it into SGLang as a
`HiCacheStorage` backend (`--hicache-storage-backend kvcr`), so:

- **offload** — SGLang's HiCache host tier writes pages into KVCR's DRAM tier
  (`batch_set_v2` → `deposit()`).
- **local fetch** — pages come back from that tier (`batch_get_v2` → `get()`).
- **remote fetch** — when the request carries a dynamo router hint naming
  another instance, KVCR pulls those pages from *that* instance's tier instead
  of recomputing them.

The router hint is the only thing that makes the remote case possible: SGLang
has no way to know which peer holds a prefix, and the destination cannot probe
a peer's residency.

## Versions this was built and tested against

| component | pin |
|---|---|
| KVCR | `nvidia-kvcr` 0.1.0, repo commit `873391ce97609c1caf8c785eebb78f7dfa58367d` |
| NIXL | 1.3.1 (KVCR's own pin) |
| dynamo | branch `linhu/kvcc-sglang-router-hint`, on top of router-hint PR #11695 |
| model used in every run below | Qwen3-8B, `--page-size 64` |

KVCR's Python API is not stable yet, and its version number does not track it:
the distribution has sat at `0.1.0` across every breaking change so far,
including the `kvcc` → `kvcr` package rename itself. Pin the repo commit, not
the version. A version skew shows up as an `AttributeError` or `TypeError` at
store construction, not as a silent misbehaviour.

## Running it

Two workers on two GPUs, one dynamo frontend, file-based discovery (no etcd, no
NATS). Per worker:

```bash
python3 -m dynamo.sglang \
  --model-path /path/to/Qwen3-8B \
  --served-model-name qwen3-8b \
  --mem-fraction-static 0.20 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-size 16 \
  --hicache-storage-backend kvcr \
  --hicache-storage-backend-extra-config '{
      "local_dram_bytes": 8589934592,
      "control_host": "127.0.0.1",
      "control_port": 25000,
      "control_advertise_host": "127.0.0.1",
      "enable_remote_hint": true}' \
  --kv-events-config '{"publisher": "zmq", "endpoint": "tcp://*:35000", "topic": "kv"}'
```

Worker 2 uses `control_port` 25001 and event port 35001. Both need
`DYN_DISCOVERY_BACKEND=file` and the same `DYN_FILE_KV` directory, and the
frontend must be started with the same two.

Three settings are load-bearing and easy to get wrong:

- **`--kv-events-config` is required.** The dynamo router builds hint candidates
  only from an event-driven index. Without KV events the router still routes,
  the hint is simply never populated, and every fetch is local.
- **The event endpoint must be a wildcard** (`tcp://*:PORT`). With
  `tcp://127.0.0.1:PORT` ZMQ *connects* instead of binding — the router index
  stays empty and the whole thing reports a clean, silent zero.
- **Keep the host pool at least as large as the device pool.** See "Known
  issues" below; this one stops offload permanently.

To confirm P2P actually happened, send the same long prompt to worker A then to
worker B and read `usage.prompt_tokens_details.cached_tokens` on B's response.
Non-zero means B served tokens it never computed. The backend also logs
cumulative counters:

```
KVCRStore remote path (cumulative): exists_with_hint=148 get_with_hint=148
  hinted_pages_requested=2608 hinted_pages_loaded=1680 deposit_pages_offered=...
```

`hinted_pages_loaded` is the only honest number — `batch_exists_v2` is
optimistic by design (it marks a page available whenever the hint covers it,
because the destination cannot verify the source still holds it), so
`exists_with_hint` alone proves nothing.

## What has been verified

**Functional**

- Two instances, dynamo-routed, hint-driven remote fetch: 4/4 runs, with the
  full causal chain in the logs (31 blocks × 64 = 1984 `cached_tokens`).
- TP=2. Found and fixed two silent per-rank bugs on the way (a bind collision,
  and both ranks dialing the same source port) — the symptom was correct-looking
  output built from the wrong shard, so the test compares generated text, not
  just token counts.
- DP>1, with the per-rank endpoint stride resolved on the dynamo side.
- Concurrency, against a same-worker concurrent control arm (a fresh-vs-cached
  comparison forks on its own and produces false failures).

**Correctness under failure** — a remote fetch that fails must degrade to
recompute, never admit wrong KV:

- Source frozen (SIGSTOP), source dead, source restarted: generated text matched
  the source-cached control arm word for word, 4/4, with the expected key
  present. No bad KV ever entered the radix tree.

**Capacity / sizing** — 4 configurations, one variable at a time, 95–130
distinct prefixes each. Established that the collapse we saw is the HiCache
sizing issue below, and *not* `local_dram_bytes`.

**Unit** — `test/registered/mem_cache/test_kvcr_*.py` and
`test_hicache_offload_stall.py`, 62 passing in-container.

## Known issues

1. **Host pool must be ≥ device pool, or offload stops permanently.**
   `HiRadixCache._update_host_leaf_status` admits a node to
   `evictable_host_leaves` only once the *device* tier has dropped it, so host
   pages are reclaimed only as a side effect of GPU eviction. If the device pool
   is the larger of the two, the host pool fills first, GPU eviction never fires,
   `write_backup` returns 0 forever, and L2 + L3 both stop for the life of the
   process. It does not recover as older entries age out.
   This is upstream HiRadixCache behaviour, not backend-specific — but it is
   silent, so this branch adds a warning (`HiCache host pool is full and nothing
   is evictable`). Diagnostic signature: `deposit_pages_offered` frozen while
   `exists_calls` keeps climbing.

2. **A peer restart costs ~34 s of degraded P2P** (KVCR-side; reported
   separately). Transient and self-healing, no wrong answers, no operator
   action. The restarted worker is only affected as a *source*; as a destination
   it works immediately.

3. **`local_dram_bytes` ≥ 32 GiB fails to start** — NIXL registration exceeds
   KVCR's 10 s progress-thread join timeout (`RuntimeError: KVCR progress thread
   did not start`). 14.9 GiB is fine; the threshold is somewhere between.

4. **The source offers no framework memory as a NIXL source.** `pin_adapter.py`
   declines every pin request: pinning a HiCache host page safely needs a
   residency index inside HiRadixCache that this backend does not have (that is
   the Shared-HiCache adapter — separate work). Everything served comes from
   KVCR's own tier, where its refcount holds the slot for the duration of the
   write. Cost is a miss, never a wrong result.

5. **Not benchmarked.** Every run above is a correctness run. No throughput or
   accuracy numbers against a real dataset yet.

## Where to look in the code

| file | what it holds |
|---|---|
| `kvcr_store.py` | the whole backend: the `HiCacheStorage` surface, deposit/get, the remote-hint path, counters |
| `router_hint.py` | parsing the dynamo hint and normalizing block hashes (the wire seam — a mismatch here silently makes every hint cover zero pages) |
| `pin_adapter.py` | the KVCR→framework pin callbacks, deliberately declining (see issue 4) |
| `kvcr_config.py` | `--hicache-storage-backend-extra-config` schema and timeouts |

Outside this directory the change is small — 13 files, ~190 lines, mostly
threading `kv_router_hint` from the request through the scheduler down to
`batch_exists`/`batch_get`. Note SGLang has **two** prefetch controller stacks
(`HiCacheController` and `HybridCacheController`); both had to be threaded or
the untouched one raises `TypeError`.
