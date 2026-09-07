# KVCR as a HiCache L3 backend — reviewer's guide

Cross-instance KV reuse for SGLang: worker B serves a prefix that worker A
computed, over NIXL, steered by a dynamo router hint. This is a POC — it works
end to end, and the known gaps are listed at the bottom.

For the design rationale see the PR description. This document is only how to
run it and what has been verified.

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
      "enable_remote_hint": true,
      "prefetch_timeout_base": 15.0}' \
  --kv-events-config '{"publisher": "zmq", "endpoint": "tcp://*:35000", "topic": "kv"}'
```

Worker 2 uses `control_port` 25001 and event port 35001. Both need
`DYN_DISCOVERY_BACKEND=file` and the same `DYN_FILE_KV` directory, and the
frontend must be started with the same two.

Four settings are load-bearing and easy to get wrong:

- **`prefetch_timeout_base` must be raised for the remote path.** The default
  linear budget is `base + per_ki_token × tokens/1024` = `1.0 + 0.25 ×
  tokens/1024`, which is 1.48 s for a 1984-token prefix. A *local* L3 read fits
  in that; a hinted *remote* fetch does not — measured 1.2–3.9 s for the same
  31 pages, because it crosses the control plane and a NIXL transfer. See
  "Why the default prefetch timeout is wrong for a remote fetch" below.

- **`--kv-events-config` is required.** The dynamo router builds hint candidates
  only from an event-driven index. Without KV events the router still routes,
  the hint is simply never populated, and every fetch is local.
- **The event endpoint must be a wildcard** (`tcp://*:PORT`). With
  `tcp://127.0.0.1:PORT` ZMQ *connects* instead of binding — the router index
  stays empty and the whole thing reports a clean, silent zero.
- **Keep the host pool at least as large as the device pool.** See "Known
  issues" below; this one stops offload permanently.

### Why the default prefetch timeout is wrong for a remote fetch

`hicache_storage_prefetch_policy` defaults to `timeout`, and that timeout is
linear in the fetch size:

```
timeout = prefetch_timeout_base + prefetch_timeout_per_ki_token * tokens / 1024
        = 1.0 + 0.25 * 1984 / 1024
        = 1.484 s          # for the 31-page prefix used by every test here
```

Those defaults were picked for a *local* L3 read. A hinted remote fetch is a
different order of magnitude — it dials the source's control plane, waits for
the source to pin, and then moves the pages over NIXL. Measured on the loopback
two-worker stack, for exactly those 31 pages: **1.2 s, 2.3 s, 2.4 s, 3.9 s**.
So the budget is under the *median*, and whether a fetch survives is a coin
flip on scheduler timing.

Losing that race is silent and looks nothing like a timeout:

1. `check_prefetch_progress` finds the budget exhausted and calls
   `terminate_prefetch`, which sets the terminated flag.
2. The transfer is still running on the storage thread. It completes normally,
   all pages retrieved, and calls `operation.increment(...)`.
3. `increment` returns `False` because the op is already terminated, so
   `completed_tokens` stays **0**.
4. The tree sees `completed_tokens == 0` and discards the fetch.

Nothing logs a warning: the backend succeeded, the transfer succeeded, and the
discard is an ordinary "nothing usable fetched" path. The KVCR counters make it
unmistakable once you look — `hinted_pages_requested == hinted_pages_loaded`
with no shortfall anywhere — while `cached_tokens` is 0. **Fully-loaded
counters next to a zero `cached_tokens` mean the result was thrown away after
arrival, and this timeout is the first thing to check.**

`prefetch_timeout_base: 15.0` is what every run below used. It is a ceiling on
a path that fails to a recompute, not a latency you pay: a fetch that is
genuinely never coming still ends the request correctly, just later. Tighten it
if you have measured your own remote fetch, but measure the tail, not the mean.

This is not specific to buffer mode. Both host-memory modes go through the same
`can_terminate_prefetch`, and both fail 4/4 at the default and pass 4/4 at 15 s.

### Host memory mode: `cache` vs `buffer_only`

Both are supported and verified. `--hicache-host-memory-mode` picks between
them:

- **`cache`** (default) — host RAM is a real L2 tier. Pages offloaded from the
  device stay resident and are served back on a later local hit, and KVCR sits
  under that as L3.
- **`buffer_only`** ([#34798](https://github.com/sgl-project/sglang/pull/34798))
  — host RAM is a transient staging buffer, never a tier. Writes stage device
  KV through op-owned bounces into storage and free them at the storage ack;
  reads fetch storage hits into bounces and publish into the device tree at
  prefill admission. KVCR becomes the only thing holding KV off-device.

The router hint works identically in both, which is why supporting both cost
nothing: the hint rides the *operation* (`StorageOperation.router_hint` →
`HiCacheStorageExtraInfo.extra_info`), not the host-memory tier, and buffer mode
reaches storage through the same `cache_controller.prefetch` /
`write_storage` entry points as cache mode.

Two constraints on `buffer_only`, both from upstream:

- It is only implemented for `UnifiedRadixCache`, which is selected by
  `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`, an **env var and not a CLI flag** — so
  the env gate has to be set together with the mode or `registry.py` raises at
  startup. Easy to miss when scripting a mode sweep.
- Not supported on decode instances (PD-disagg decode bypasses the buffer-mode
  pipeline) or with `--hicache-write-policy write_back`.

### Sizing the two tiers together

`--hicache-size` and `--mem-fraction-static` must be chosen as a pair. They are
squeezed from both sides by the two independent limits below, and both failure
modes are quiet:

- Too *small* a host tier relative to the device tier trips known issue 1 —
  offload stops for the life of the process.
- Too *large* a `local_dram_bytes` trips known issue 4 — the worker refuses to
  start with `KVCR progress thread did not start`.

The trap is that backing away from one walks into the other. Shrinking
`--hicache-size` to dodge the startup failure shrinks the *host* pool while the
device pool stays where `--mem-fraction-static` put it, and the host pool
silently becomes the smaller of the two. That reads as "P2P is broken on this
branch": the stack is healthy, both workers register, hints are advertised, and
every fetch returns nothing because nothing was ever deposited. Lower
`--mem-fraction-static` in the same step.

Read the two pool sizes back out of the worker log rather than computing them —
the device pool depends on model and dtype, and the warning only fires for the
inequality, not for a margin too thin to be useful:

```
KV Cache is allocated. ... #tokens: 46848        <- device pool
Allocating kv hierarchical KV host pool: 54272 tokens, 8.00 GB host memory.
```

Two configurations verified end to end on Qwen3-8B / 80 GB cards:

| `--mem-fraction-static` | `--hicache-size` | `local_dram_bytes` | device / host tokens |
|---|---|---|---|
| 0.20 | 16 | 8 GiB | fits, original run |
| 0.16 | 8 | 4 GiB | 46848 / 54272 |

A `WARN ... host pool (N tokens) is smaller than the device pool (M tokens)` at
startup means offload will never fire. It is worth treating as fatal.

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

**That line lags, and reading it too early will mislead you.** Counters are
flushed at most once per `_STATS_LOG_INTERVAL_S` (30 s), and only from inside
`_note` — so the last line in the log is not "the state now", it is the state as
of the last counted event that happened to fall after an interval boundary. A
single test run typically ends *before* its own counters are flushed, leaving a
stale snapshot from the previous run as the newest line. That is easy to read as
a failure that already happened.

Either run the test twice, or sleep 35 s and trigger one more request before
reading. Counters are cumulative per process, so a second run's line is a
superset — no state is lost by waiting, only by not waiting.

## What has been verified

**Functional**

- Two instances, dynamo-routed, hint-driven remote fetch: 4/4 runs, with the
  full causal chain in the logs (31 blocks × 64 = 1984 `cached_tokens`).
  Re-verified against `nvidia-kvcr` `873391c` after the rename, 2/2, counters
  showing no shortfall anywhere in the chain: source `deposit_pages_stored=64`,
  destination `hinted_pages_requested=62 hinted_pages_loaded=62`.
- **Both host-memory modes, 4/4 each** (`cache` and `buffer_only`), same
  two-instance hint-driven test, `prefetch_timeout_base: 15.0`. At the default
  timeout both modes fail 4/4 — the mode is not what decides it. Worth stating
  plainly: the earlier 2/2 `cache` result above was taken at the default and
  was luck, not a stable pass. Anything measured on this path before the
  timeout was raised should be re-run rather than trusted.
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

**Unit** — `test/registered/mem_cache/test_kvcr_*.py`, passing in-container
against `873391c`.

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
   `exists_calls` keeps climbing — or, if offload never started at all,
   `deposit_pages_offered` absent from the counter line entirely. Note this
   presents as a *remote-hint* failure even though nothing about the hint path
   is wrong; see "Sizing the two tiers together" above, and mind that backing
   away from issue 4 is the usual way to arrive here.

2. **The default prefetch timeout silently discards a successful remote fetch.**
   `1.0 + 0.25 × tokens/1024` is under the measured remote-fetch latency, and
   losing the race throws the fetch away *after* every page has arrived —
   `increment` is refused on a terminated op, `completed_tokens` stays 0, and
   nothing warns. Diagnostic signature: `hinted_pages_requested ==
   hinted_pages_loaded` with `cached_tokens=0`. Not backend-specific and not
   mode-specific; upstream's defaults are simply sized for a local L3 read. Set
   `prefetch_timeout_base` (see the section above). The right long-term fix is
   probably upstream — either a backend-declared default, or not discarding a
   transfer that completed between the termination decision and the increment.

3. **A peer restart costs ~34 s of degraded P2P** (KVCR-side; reported
   separately). Transient and self-healing, no wrong answers, no operator
   action. The restarted worker is only affected as a *source*; as a destination
   it works immediately.

4. **`local_dram_bytes` ≥ 32 GiB fails to start** — NIXL registration exceeds
   KVCR's 10 s progress-thread join timeout (`RuntimeError: KVCR progress thread
   did not start`). 14.9 GiB is fine; the threshold is somewhere between. The
   timeout (`_JOIN_TIMEOUT_SECONDS` in `kvcr/progress.py`) has been 10 s since
   at least kvcc `e3a816e`, so hitting this after a core bump is not evidence of
   a regression. When shrinking to get past it, shrink
   `--mem-fraction-static` too — otherwise you land on issue 1.

5. **The source offers no framework memory as a NIXL source.** `pin_adapter.py`
   declines every pin request: pinning a HiCache host page safely needs a
   residency index inside HiRadixCache that this backend does not have (that is
   the Shared-HiCache adapter — separate work). Everything served comes from
   KVCR's own tier, where its refcount holds the slot for the duration of the
   write. Cost is a miss, never a wrong result.

6. **Benchmarked only on two models.** Dense Qwen3-8B showed +53–59% qps and
   −50% TTFT, but only once the distinct working set exceeded the device pool;
   an FP8 MoE model showed +20% on the same harness. Do not extrapolate the
   dense number — and note both were taken before the KVCR rename.

## Where to look in the code

| file | what it holds |
|---|---|
| `kvcr_store.py` | the whole backend: the `HiCacheStorage` surface, deposit/get, the remote-hint path, counters |
| `router_hint.py` | parsing the dynamo hint and normalizing block hashes (the wire seam — a mismatch here silently makes every hint cover zero pages) |
| `pin_adapter.py` | the KVCR→framework pin callbacks, deliberately declining (see issue 5) |
| `kvcr_config.py` | `--hicache-storage-backend-extra-config` schema and timeouts |

Outside this directory the change is small — 13 files, ~190 lines, mostly
threading `kv_hints` from the request through the scheduler down to
`batch_exists`/`batch_get`. Note SGLang has **two** prefetch controller stacks
(`HiCacheController` and `HybridCacheController`); both had to be threaded or
the untouched one raises `TypeError`.
