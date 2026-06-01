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
- Probe implementation details ("field non-None", "which branch ran") — assert the consequence.

## Other Tips

- **Engine-self-driven behavior: drive the real loop, don't call the private.** Never synchronously call a scheduler private (e.g. `scheduler._abort_on_waiting_timeout()`) from the harness/test — it runs at the wrong loop phase, bypasses the ordered `recv_requests` → `process_input_requests` injection, and can fire in states the real loop never reaches (e.g. while paused). For sweeps the engine runs itself (timeout/idle), enable the config/env and advance the loop with `yield`.
- **`py_compile` ≠ runs.** These are `test/manual/` GPU tests; nothing runs locally. A test calling a nonexistent `t.X()`, passing an unsupported `start_req` kwarg, or comparing `finished_reason == "str"` compiles but `AttributeError`s / is always-False at runtime. Sweep `t.X(` against the `ScriptedContext` API; `finished_reason` is an object → `isinstance(..., FINISH_LENGTH/FINISH_MATCHED_TOKEN/...)`.
- **`start_req` forwards a fixed kwarg set only** (`max_new_tokens`, `ignore_eos`, `return_logprob`, `logprob_start_len`, `top_logprobs_num`, `priority` int, `rid`, `dp_rank`, `prompt_token`, `lora_path`). Sampling params (`temperature`/`top_p`/`top_k`/`stop`/`seed`/`min_new_tokens`) are NOT supported.
- **`chunks_done` = `0 if prompt_len <= chunk_size else ceil(prompt_len/chunk_size)`** — the final tail iteration counts (`257`/`256` → `2`). A radix prefix hit shrinks effective new tokens (radix is lora-aware), making the count non-deterministic — then assert a lower bound + comment, not an exact value.
- **Magic numbers and timing-dependent asserts need a GPU run.** Reason against source + `py_compile`, but exact counts and "released after X" timing are only proven by a real run.
