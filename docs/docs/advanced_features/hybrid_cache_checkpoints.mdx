---
title: "Hybrid Model Prefix-Cache Checkpoints"
metatags:
    description: "Choose where the prefix cache of a hybrid (Mamba / GDN / KDA) model keeps recurrent-state checkpoints with --hybrid-cache-checkpoints-level, and what each level costs."
---

A hybrid model's linear-attention layers carry a recurrent state instead of per-token KV. The prefix cache can reuse a cached prefix only up to a position where it also kept that state, called a checkpoint. `--hybrid-cache-checkpoints-level` decides where checkpoints are kept.

| Level | Checkpoints | Best for |
|---|---|---|
| `2` (default) | Where two requests' prompts diverge, and where each request ends (on a PD prefill worker, where its prompt ends). A soft cap of one checkpoint per path, unless explicitly overridden. | Multi-turn chat, agents and tool loops, where each request extends the previous one's full context. |
| `3` | Also every prefill chunk and every `--mamba-track-interval` decode tokens. | Traffic that reuses the middle of an earlier prompt that no other request has diverged from yet. |

The per-path cap is a pruning target, not a strict state-slot bound. Tail, fork, locked, and device-leaf states can be preserved, so a path may retain multiple checkpoints. Size the state pool for running requests and retained checkpoints, and verify actual occupancy.

## What level 2 changes

- **Request-end checkpoints support continued conversations.** Level 2 preserves the checkpointing path used when the next turn extends the previous request's full context. Actual reuse still depends on checkpoint placement, KV and state capacity, eviction, and usable host backups; equal hit rates across levels are not guaranteed.
- **Fewer intermediate checkpoints can reduce host and storage writes.** HiCache, storage backends and the external linker receive published checkpoints, including fork and request-end states. Skipping intermediate checkpoints can reduce write traffic and pressure on other sessions' checkpoints; the benefit depends on the workload and cache configuration.
- **A late fork may need more recomputation.** If a request shares only the first part of an earlier prompt and lacks a matching checkpoint there, it resumes from the nearest usable earlier checkpoint, even if more KV is cached. Publishing a checkpoint at an eligible divergence point can help later siblings while that checkpoint and its KV remain available. Level 3 offers more intermediate checkpoints, but does not guarantee a checkpoint at every possible divergence position.
- **Ordinary requests defer intermediate cache publication.** Away from fork points and the PD prefill exception, a prompt joins the tree at request finish rather than chunk by chunk, limiting sharing with concurrent requests during prefill.

Models with both sliding-window and linear-attention layers keep checkpointing every prefill chunk at level 2.

## Measured

These historical measurements from [#49886](https://github.com/sgl-project/sglang/pull/49886) describe the listed models, traffic, and cache configurations. They are examples, not a guarantee of equal continuation hit rates, lower write traffic, or higher throughput. Compare both continuation reuse and late-fork recomputation on your workload before choosing a level.

| Workload (B300) | Level 3 | Level 2 |
|---|---|---|
| GLM-5.3-Flash, AgentX trace replay with HiCache: hit rate / output tok/min/GPU | 0.943 / 42,741 (per-path cap set to 1) | 0.961 / 45,442 |
| Qwen3.8-27B multi-turn sessions with HiCache: continuation hit / states written to host / host write bytes | 0.993 / 304 / 66.7 GB | 0.993 / 87 / 33.4 GB |
| Qwen3.8-Flash-Next, AgentX trace replay (cookbook TP1 MTP recipe): hit rate / output tok/min/GPU | 0.961 / 59,480 | 0.960 / 60,525 |

## Related options

- `--mamba-max-states-per-path N` is deprecated. When set, it still sets the soft per-path cap at either level (`-1` means unlimited).
- The size of the state pool that holds the checkpoints is set by `--mamba-full-memory-ratio` or `--max-mamba-cache-size`; see [Mamba Cache](/docs/advanced_features/server_arguments#mamba-cache).
