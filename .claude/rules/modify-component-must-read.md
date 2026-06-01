# Must-Read Skills Before Modifying Components

Before modifying the following components, read the listed skill first.

- **Speculative decoding code** (anything under `python/sglang/srt/speculative/`, related attention backends, scheduler accumulators, IPC fields, observability metrics, or CLI flags) → [`speculative-naming`](../skills/speculative-naming/SKILL.md)
- **`Scheduler` / `TokenizerManager` / `ModelRunner` `__init__`** (`python/sglang/srt/managers/scheduler.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/model_executor/model_runner.py`) → [`large-class-init-style`](../skills/large-class-init-style/SKILL.md)
- **Environment variables** (adding, renaming, or reviewing any `SGLANG_*` env var, migrating a legacy `SGL_*` alias, or touching `python/sglang/srt/environ.py`) → [`env-var-conventions`](../skills/env-var-conventions/SKILL.md)
- **Scripted runtime test harness** (adding or reviewing any `ScriptedContext` / `ScriptedReqHandle` accessor under `python/sglang/test/scripted_runtime/`, or writing the scripted chunked tests `test/{manual,registered}/chunked_prefill/test_scripted_*.py`) → [`scripted-runtime-notes`](../skills/scripted-runtime-notes/SKILL.md)
