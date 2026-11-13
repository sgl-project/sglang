---
title: "Hybrid Model Memory Pools"
metatags:
    description: "Split memory between the linear-attention state pool and the KV pool of a hybrid (Mamba / GDN / KDA) model with --mamba-full-memory-ratio, and size it for concurrency or for multi-turn prefix reuse."
---

Hybrid models mix full-attention layers with linear-attention layers (Mamba, Gated DeltaNet, KDA). SGLang keeps two separate pools for them, sized once at startup:

| Pool | Holds | Caps |
|---|---|---|
| **State pool** | The recurrent state of each linear-attention layer: working states for running requests and checkpoints kept for prefix reuse. | Running concurrency and retained checkpoints. Idle sessions also need a reusable checkpoint. |
| **KV pool** | Full-attention KV, one row per token. | How many tokens are cached or in flight: running contexts plus the prefixes kept for reuse. |

`--mamba-full-memory-ratio r` splits the memory left after weights: state pool : KV pool = `r`. The default is `0.9`; some models override it (see [Model defaults](#model-defaults)). `--max-mamba-cache-size N` pins the state pool to `N` slots instead, and the KV pool takes the rest.

## Read the split from the boot log

Each rank prints both pools:

```text
Mamba Cache is allocated. max_mamba_cache_size: 429, conv_state size: 1.00GB, ssm_state size: 28.55GB
KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 5890560, KV size: 32.89 GB
max_running_requests is capped to 85 by the mamba state cache (max_mamba_cache_size=429, 5 state slots per request)
```

A running request holds several state slots: 5 with the default `extra_buffer` strategy and the overlap scheduler, 4 with `--mamba-radix-cache-strategy extra_buffer_lazy`, plus speculative draft states. The last line appears when the state pool, not the KV pool, limits concurrency.

## Two ways to run out

- **Concurrency-bound** (many short requests): the state pool fills first and caps `max_running_requests` while KV is idle. Raise the ratio or pin a larger `--max-mamba-cache-size`.
- **Residency-bound** (multi-turn chat, agents): sessions return turn after turn, and reuse requires both their KV prefixes and matching recurrent checkpoints. State eviction can prevent reuse even when all KV fits, unless a usable host backup remains. If KV is the bottleneck, lower the ratio only after checking state capacity for running requests and retained checkpoints. If checkpoints are the bottleneck, increase their capacity or use supported host backup; lowering the ratio can make reuse worse.

A KV residency cliff appeared in this GLM-5.3-Flash AgentX trace replay (48 total sessions, peak concurrency 48, median concurrency 27, TP2, pools fixed at the same total memory):

| State slots / KV tokens | Prefix-cache hit rate | Output throughput |
|---|---|---|
| 429 / 5.89M (ratio 0.9) | 0.06 | 1x |
| 300 / 7.48M | 0.22 | 1.2x |
| 192 / 8.81M (`extra_buffer_lazy`) | 0.89 | 4.8x |

In this trace, the gap between peak and typical concurrency left space for idle-session checkpoints. These results do not establish that the same concurrency-based state budget is sufficient when sessions greatly outnumber running requests.

## Size the pools

1. Estimate the peak number of running requests `R`, the per-request slot count `S` from the boot log, and how many sessions you want to retain between turns.
2. Budget state slots for the peak simultaneous running-state and retained-device-checkpoint demand, plus margin. Count checkpoints already included in the running-request budget only once. `S × R` plus margin is a concurrency starting point, not a guarantee of session residency.
3. Pin `--max-mamba-cache-size` to that state budget and check that the remaining KV pool's `#tokens` covers the retained sessions' total context. For concurrency-bound traffic, start from the balanced ratio the [`compute-mamba-ratio`](https://github.com/sgl-project/sglang/blob/main/.agents/skills/compute-mamba-ratio/SKILL.md) skill computes from your average request length.
4. If you have safe memory headroom, consider raising `--mem-fraction-static`. Recheck both pool sizes; an explicitly pinned state-slot count stays fixed.
5. Verify on your traffic, including sessions far above concurrency: prefix reuse (`--enable-cache-report`, or `sglang:cached_tokens_total` with `--enable-metrics`), both pools' occupancy, and whether startup caps `max_running_requests`. Low active-state usage does not mean the state pool has room for more cached sessions.

For idle sessions, about one checkpoint per session is a starting estimate with `--mamba-max-states-per-path 1`, not a strict upper bound. This is a soft limit: tail, fork, locked, and device-leaf states can be preserved. Allow margin and measure retained checkpoints. Account separately for usable host backups and device space for restoring them.

The right ratio depends on the model and the deployment: it scales with the size of one state slot relative to one KV token, with the context per session and with the concurrency. Tuned values range from `0.1` to above `1`. Do not copy a ratio between models.

## Model defaults

| Model | Default | Notes |
|---|---|---|
| Most hybrid models | `0.9` | Generic starting point. Cookbook recipes pass an explicit value where it was tuned. |
| Inkling | `0.1` | |
| GLM-5.3-Flash | `0.25` with `extra_buffer_lazy` | Sized for multi-turn agent traffic. High Throughput recipes on GPUs where it would cap running requests pin `0.9`. |

An explicit `--mamba-full-memory-ratio`, `--max-mamba-cache-size` or `--mamba-radix-cache-strategy` always overrides the model default.
