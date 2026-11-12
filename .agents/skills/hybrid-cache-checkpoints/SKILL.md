---
name: hybrid-cache-checkpoints
description: Choose, explain and verify --hybrid-cache-checkpoints-level for hybrid (Mamba / GDN / KDA) models -- where the prefix cache keeps recurrent-state checkpoints, what each level costs, how it interacts with HiCache, the linker and PD, and how to prove a level change on real traffic. Use when a user asks which level to run, why a prefix stopped hitting on a hybrid model, why host or L3 writes are high, or when reviewing a change to Mamba checkpoint placement.
---

# Hybrid cache checkpoint levels (`--hybrid-cache-checkpoints-level`)

A hybrid model can resume a prefix only from a position where the tree holds a recurrent-state checkpoint; KV alone is not enough. The level decides where those checkpoints are taken.

| Level | Checkpoints taken | Per-path cap |
|---|---|---|
| `3` | Every prefill chunk and every `--mamba-track-interval` decode boundary, plus fork points and request ends. | Unlimited (or `--mamba-max-states-per-path N`). |
| `2` (default) | Fork points (`mamba_branching_seqlen`) and request ends. A PD prefill worker treats the complete prompt as the request end. | 1 (or `--mamba-max-states-per-path N`). |

`--mamba-max-states-per-path` is deprecated; an explicit value still sets the cap at either level (`-1` = unlimited). Its old meaning is the cap only; level 2 additionally stops creating interior checkpoints.

## Where it lives

- `MambaComponent._defers_checkpoint` / `prepare_for_caching_req` (`mem_cache/unified_cache/components/mamba.py`): below level 3 a running request's checkpoint donates nothing and inserts nothing, except at its fork point or a PD prefill worker's complete prompt. `InsertParams.mamba_checkpoint_deferred` tells `cleanup_after_caching_req` to keep `mamba_last_track_seqlen`, so the finished insert still lands the prompt-end state.
- `exec.mamba.mamba_path_state_cap` (derived, `arg_groups/overrides.py::mamba_path_state_cap_of`) feeds the cap to both tree cores.
- A tree with sliding-window slots keeps checkpointing every chunk at level 2: prefill frees window rows only behind a checkpoint the tree takes.

## Choosing a level

- **Level 2** when reuse is continuation: multi-turn chat, agents, tool loops, where the next request extends the previous request's full context. Measured hit rates match level 3; host and L3 receive only request-end states, which cuts their write traffic and stops interior checkpoints from flushing other sessions' tails.
- **Level 3** when reuse forks from the *middle* of an earlier prompt that has not forked before: a shared system prompt or few-shot block whose first user arrives alone, regenerating from an earlier turn, exact re-sends of a finished prompt. Under level 2 the late arrival finds KV but no state at that depth and recomputes from the nearest state (often the root). Once a second request diverges there, the fork point gets a state and later siblings hit.
- At level 2 a prompt's KV joins the tree when its request finishes (the tree only inserts KV up to a state), so concurrent requests cannot share a long prompt while it is still prefilling.

## Measured (B300)

| Model / setting | Continuation hit | Late-fork hit | Mamba states written to host | Throughput |
|---|---|---|---|---|
| GLM-5.3-Flash, AgentX, HiCache, level 3 + cap 1 | 0.943 | | | 42,741 out tok/min/GPU |
| GLM-5.3-Flash, AgentX, HiCache, level 2 | 0.961 | | | 45,442 |
| GLM-5.3-Flash, AgentX, device, `extra_buffer_lazy` + cap 1 vs level 2 | 0.886 vs 0.888 | | | 34,814 vs 36,406 |
| Qwen3.8-27B probe, HiCache, level 3 vs level 2 | 0.993 vs 0.993 | 1.0 vs 0.0 | 304 vs 87 | host write bytes 66.7 GB vs 33.4 GB |
| Qwen3.8-Flash-Next, AgentX (cookbook B300 TP1 MTP, conc 24), level 3 vs level 2 | 0.961 vs 0.960 | | | 59,480 vs 60,525 |

## Verifying a level on a deployment

Prefix-cache claims need path evidence, not just a hit rate.

1. Launch with `--enable-metrics --enable-cache-report` (and HiCache / the linker as deployed), once per level, same flags otherwise.
2. Drive multi-turn sessions where each turn extends the previous turn's prompt plus output, with filler traffic between turns when you want the device copy gone. Then send a late fork: half of an earlier first prompt plus a new suffix.
3. Report per-turn `cached_tokens`, the late-fork `cached_tokens`, and the `/metrics` deltas: `sglang:hicache_backup_tokens_total{pool=...}`, `sglang:load_back_tokens_total{pool=...}`, `sglang:hicache_host_mamba_used_states`, `sglang:storage_prefetch_hit_tokens_total`.
4. Expected: continuation hits equal across levels; the mamba backup count drops at level 2 when prompts span several chunks; late-fork hits drop to the nearest earlier state at level 2.
5. Size the host pool from the boot line `Allocating kv hierarchical KV host pool: N tokens`: a small `--hicache-size` leaves the KV host pool smaller than the probe's footprint and every host hit becomes 0 at both levels, which looks like a bug and is not.

## Pitfalls

- `--mamba-max-states-per-path -1` is an explicit "unlimited", not "unset"; it overrides level 2's cap of 1.
- The PD prefill exception keys on `len(origin_input_ids)`: the insert that covers the whole prompt keeps its state.
- AgentX-tuned pool splits (see the `compute-mamba-ratio` skill) were measured at level 2; re-measure before reusing one at level 3.
