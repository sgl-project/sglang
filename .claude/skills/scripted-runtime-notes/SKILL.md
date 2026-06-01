---
name: scripted-runtime-notes
description: Requirements for the SGLang scripted runtime, chiefly when to add (vs not add) a harness API. Use for anything related to the scripted runtime.
---

# Scripted Runtime — Notes

Notes for anything related to the SGLang scripted runtime.

## When to Add an API

Tests read `r.req.*` and `t._scheduler.*` directly — there is no encapsulation boundary. A thin wrapper buys zero isolation; it only grows the surface.

Add an API only if it does real work:

1. **Control primitive** — drives the engine through a real path (`start_req`, `pause_generation`, `abort`, `evict_radix`, `exhaust_kv`). Reuse the real path; never hand-mutate state.
2. **Hook-backed** — value cannot be read from a snapshot; accumulate via `scheduler_hook.on_run_batch` or the recv proxy (`chunks_done`). Read-only; never monkey-patch; never add `*_count` to `srt/`.
3. **Multi-structure derivation, widely reused** — scans `chunked_req` + `waiting_queue` + `running_batch` + `last_batch` (`is_idle`, `status`, `batch_composition`).

Else: don't. Read `r.req.X` / `t._scheduler.X` in the test; inline single-use accessors.

Never:

- Weaken an assertion to fit a missing probe.
- Probe implementation details ("field non-None", "which branch ran") — assert the consequence, or delete the test.
