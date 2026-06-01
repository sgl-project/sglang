---
name: scripted-runtime-notes
description: Requirements for the SGLang scripted runtime, chiefly when to add (vs not add) a harness API. Use for anything related to the scripted runtime — any code under python/sglang/test/scripted_runtime/, python/sglang/test/scripted_runtime_chunked_helpers.py, or the scripted chunked tests test/{manual,registered}/chunked_prefill/test_scripted_*.py.
---

# Scripted Runtime — Notes

Notes for working on the SGLang scripted runtime: the harness under `python/sglang/test/scripted_runtime/`, `python/sglang/test/scripted_runtime_chunked_helpers.py`, and the scripted chunked tests under `test/{manual,registered}/chunked_prefill/test_scripted_*.py`.

## When to Add an API

The harness exposes accessors on `ScriptedContext` (`t`) and `ScriptedReqHandle` (`r`). The tests already read `r.req.*` and `t._scheduler.*` directly, so there is **no encapsulation boundary** — a thin wrapper over one field buys zero refactor isolation, it just grows the surface.

Add a new harness API only if it does real work — one of:

1. **Control primitive** — drives the engine through a real path (e.g. `start_req`, `pause_generation`, `abort`, `evict_radix`, `exhaust_kv`). Must reuse the engine's real retract/evict/allocator path, never hand-mutate internal state.
2. **Cross-time / hook-backed** — the value cannot be read from the current snapshot and must accumulate over steps via `scheduler_hook.on_run_batch` (`_batch_log`) or the tokenizer recv proxy (e.g. `chunks_done`). Read-only hook only: never monkey-patch, and never add `*_count` counters to `python/sglang/srt/` to make a test pass.
3. **Multi-structure derivation reused widely** — scans several scheduler structures at once (`chunked_req` + `waiting_queue` + `running_batch` + `last_batch`) and many tests need it (e.g. `is_idle`, `status`, `batch_composition`, `find_req_by_rid`).

Otherwise, do **not** add an API: read `r.req.X` / `t._scheduler.X` directly in the test. A thin single-field or single-test accessor should be inlined, not added.

Two anti-patterns to refuse:

- **Don't weaken assertions** to fit a missing probe. If the honest invariant needs cross-time observation, add a clean read-only hook (rule 2); if it doesn't, assert the real durable state.
- **Don't probe implementation details.** An accessor that only checks "some internal field is non-None / which branch ran" tests nothing real (it is usually true by construction). Assert the observable *consequence* instead, or delete the test.
