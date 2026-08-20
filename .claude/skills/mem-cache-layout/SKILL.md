---
name: mem-cache-layout
description: Where KV-cache allocators, device pools, and host pools live under `python/sglang/srt/mem_cache/`, and how to name them. Use when adding, moving, or reviewing any `BaseTokenToKVPoolAllocator`, `KVCache`, or `HostKVCache` subclass, when adding a module under `mem_cache/`, or when a `test_mem_cache_layout_ratchet` failure needs fixing.
---

# mem_cache — Layout and Naming

`mem_cache/` is being restructured from catch-all modules into per-family layer
packages ([issue #25371](https://github.com/sgl-project/sglang/issues/25371)). The
migration is incremental: `allocator/` is done, `pool_host/` is nearly done, `pool/`
has not started. `test/registered/unit/mem_cache/test_mem_cache_layout_ratchet.py`
enforces what has landed.

Read the issue for the full target tree before starting a move. This skill covers
what to do when *writing* code that touches the layout.

## Rule 1 — File a class by its layer, not by its feature

| Base class (transitive) | Layer | Home |
|---|---|---|
| `BaseTokenToKVPoolAllocator` | slot indexing | `mem_cache/allocator/<family>.py` |
| `KVCache`, `BaseSWAKVPool`, `ReqToTokenPool`, `MambaPool` | device storage | `mem_cache/pool/<family>.py` |
| `HostKVCache` | host mirror | `mem_cache/pool_host/<family>.py` |

`<family>` is the attention or state family: `mha`, `mla`, `dsa`, `mamba`, `swa`,
`hisparse`, `deepseek_v4`. A new quantization or layout variant of an existing family
is a new *file in that family's package*, not a new class in a catch-all module.

**`pool/` does not exist yet.** Until the first `pool/` PR lands, a new device-pool
class has no correct home, so the ratchet does not demand one — but it does refuse to
let the class land in `memory_pool.py` (Rule 3). If you need a new device pool now,
say so in review; the resolution is usually to create the family file under `pool/`
as part of your PR, which is also the roadmap's next step.

## Rule 2 — `Allocator` means two different things

- **Slot allocator** — a `BaseTokenToKVPoolAllocator` subclass. Answers "which KV
  slots are free". Lives in `allocator/`.
- **Host tensor allocator** — `HostTensorAllocator` and its subclasses
  (`ShmHostTensorAllocator`, `MooncakeHostTensorAllocator`, `UMBPHostTensorAllocator`).
  Answers "give me pinned host memory". Lives in `pool_host/common.py` and `storage/`.

Never file the second kind under `allocator/`. The ratchet classifies by base class
precisely so the name cannot mislead it — don't reintroduce the ambiguity by hand.

## Rule 3 — Don't add to a module scheduled for deletion

These modules are being emptied out; the ratchet freezes their class lists:

```
memory_pool.py          memory_pool_host.py      deepseek_v4_memory_pool.py
deepseek_v4_compress_state.py                    swa_memory_pool.py
base_swa_memory_pool.py hisparse_memory_pool.py  unified_memory_pool.py
multi_ended_allocator.py                         mamba_checkpoint_pool.py
dsa_cache_layer_split.py                         index_key_cache.py
```

Adding a class to any of them fails the ratchet. This is the rule that matters most in
practice: `memory_pool.py` grew from 2257 to 5042 lines *after* the restructure was
agreed, entirely through classes appended to a file everyone knew was being split.

## Rule 4 — No new top-level modules

`mem_cache/*.py` is a frozen list. A new module goes under `allocator/`, `pool/`,
`pool_host/`, `storage/`, or another existing package. Growing the pin is allowed but
needs a stated reason in review — that conversation is the point of the rule.

## Rule 5 — Names drop affixes that don't differentiate

If every file in a directory shares the role the directory already names, the affix
carries no information.

```
pool_host/mha.py                   not  pool_host/mha_pool_host.py
hybrid_cache/controller.py         not  hybrid_cache/hybrid_cache_controller.py
unified_cache/components/full.py   not  unified_cache/components/full_component.py
```

Keep a role affix only where same-directory siblings have *different* roles
(`_backend` next to `_controller`).

## Rule 6 — A family is a module; a module may be a package

One file per family is the default. Past ~1500 lines, the family becomes a package —
`pool/deepseek_v4/` already is, and `pool/mha/` will be (6 classes, ~1900 lines).
Splitting a family into a package is not a violation of "one file per family"; it is
the same rule at the next size.

## Rule 7 — Layers don't import upwards

- `pool/` and `pool_host/` must not import `allocator/`, `hybrid_cache/`, or
  `allocation.py`.
- `pool_host/` must not import `pool/` implementation classes.
- `allocator/` may hold a reference to the pool it allocates into; the reverse is
  forbidden.
- None of the three may import the construction layer (`kv_cache_configurator.py`,
  `kv_cache_builder.py`, `cache_init_params.py`, `allocation_sizing.py`,
  `kv_cache_dtype.py`, `kv_vmm_backing.py`). Construction depends on them.

## Fixing a ratchet failure

The failure message names the rule and the target file. Two shapes:

- **"belongs in mem_cache/<layer>/..."** or **"is slated for deletion but gained
  ..."** — move the code. Growing the pin instead needs an explicit reason in review.
- **"drop it from _LEGACY_HOMES" / "shrink _TOP_LEVEL_MODULES" / "drop its
  _SHRINKING_MODULES entry"** — you moved something; shrink the pin in the same PR.
  Every migration PR is expected to do this, and it is how the roadmap tracks progress.

## Moving code under this issue

Title the PR `[mem_cache][N/N]` and state whether it is a Mechanical Move. Mechanical
Moves must satisfy [`mechanical-refactor-verify`](../mechanical-refactor-verify/SKILL.md).
No compatibility shims — update every importer in the same PR, the way `allocator.py`
was deleted outright rather than left re-exporting.
