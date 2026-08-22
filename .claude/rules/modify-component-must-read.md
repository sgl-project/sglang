# Must-Read Skills Before Modifying Components

Before modifying the following components, read the listed skill first.

- **Speculative decoding code** (anything under `python/sglang/srt/speculative/`, related attention backends, scheduler accumulators, IPC fields, observability metrics, or CLI flags) → [`speculative-naming`](../skills/speculative-naming/SKILL.md)
- **`Scheduler` / `TokenizerManager` / `ModelRunner` `__init__`** (`python/sglang/srt/managers/scheduler.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/model_executor/model_runner.py`) → [`large-class-style`](../skills/large-class-style/SKILL.md)
- **Any edit to a frozen core file** (currently `python/sglang/srt/model_executor/model_runner.py`) → [`large-class-style`](../skills/large-class-style/SKILL.md)
- **Environment variables** (adding, renaming, or reviewing any `SGLANG_*` env var, migrating a legacy `SGL_*` alias, or touching `python/sglang/srt/environ.py`) → [`env-var-conventions`](../skills/env-var-conventions/SKILL.md)
- **Scripted runtime** (anything related to the scripted runtime) → [`scripted-runtime-notes`](../skills/scripted-runtime-notes/SKILL.md)
- **KV-cache allocators / device pools / host pools** (adding, moving, or renaming any `BaseTokenToKVPoolAllocator`, `KVCache`, or `HostKVCache` subclass, or adding a module under `python/sglang/srt/mem_cache/`) → [`mem-cache-layout`](../skills/mem-cache-layout/SKILL.md)
