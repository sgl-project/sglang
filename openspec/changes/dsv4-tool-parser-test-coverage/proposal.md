## Why

DeepSeek-V4-Flash is deployed in production with DSPARK speculative decoding, but its DSML streaming tool-call parser has only **6 unit tests** — compared to 16 for Qwen3Coder, 25 for Inkling, and 9 for DeepSeek-V32. There are **zero server-level SSE integration tests** for any DeepSeek detector (only llama3/pythonic have E2E coverage). A production regression showed ~7-8% of agentic streaming turns produce complete prose but **silently drop the tool call** — the agent stops, and the user must manually intervene. Multiple known parser bugs remain unmerged in `main` (#31786, #33813, #32332, #32167, #30481), and the serving-code gap that allows partial DSML openers to be silently dropped at stream end has no test coverage at all.

## What Changes

- Add a **DSV4 DSML streaming robustness** unit test suite covering: tokenizer-level BPE token-boundary splitting, stream-end edge cases (partial opener in buffer, EOS mid-opener), malformed input recovery, parameter-type conversion (int/bool/array/anyOf/null), bare-invoke without wrapper, self-closing invoke, multiple sequential calls, and the `bot_token` override fragility (V4 overrides V32's `function_calls` to `tool_calls`; if the override is lost, only the secondary `<｜DSML｜invoke` check saves detection).
- Add **serving-code integration tests** that exercise the full pipeline through the real `_process_tool_call_stream` method and the real finish_reason emission code path in `serving_chat.py`. These tests pin the exact condition under which `finish_reason` stays `"stop"` instead of being rewritten to `"tool_calls"` — without copying production logic into test helpers.
- Add **regression tests** for known DSML parser bug classes from GitHub history, including: preamble loss (#31786/#33813), malformed JSON recovery (#32332), final-delta args loss (#32167), fence leak (#29426), max_tokens truncation (#30527), and bare-invoke parsing (#23786).
- Add a **DSPARK truncation simulation** test category with 8 representative opener-split patterns (down from 128 exhaustive patterns — the failure mode is "buffer assembly breaks at a split boundary," which 8 boundary-class representatives cover as well as 128) plus truncation at each mid-opener position.
- Add a **production-bug reproduction test**: feed prose + partial DSML opener across 3 deltas, signal stream end, and assert the full integrated pipeline (detector → has_tool_calls → finish_reason → unstreamed-args recovery) produces: prose emitted, no tool_calls, finish_reason="stop". This is the exact failure path from the ~7-8% production regression.
- Establish a **tokenizer-aware test infrastructure** that uses the actual DeepSeek-V4-Flash tokenizer to generate BPE-accurate chunk boundaries (current tests use V3.2 tokenizer; V4 may have different token boundaries for DSML markers).

## Capabilities

### New Capabilities
- `deepseek-dsml-streaming-robustness`: Behavioral requirements for the DeepSeek-V4 DSML streaming tool-call parser, covering streaming delta assembly, stream-end recovery, parameter parsing, and serving-code integration (finish_reason rewrite, has_tool_calls flag, unstreamed args recovery).

### Modified Capabilities
_(none — this change adds test coverage for existing behavior; no spec-level behavior changes)_

## Impact

- **Test files**: New test file(s) under `test/registered/unit/function_call/` (CPU, `base-a-test-cpu`); optional new E2E test under `test/registered/openai_server/function_call/` (GPU, requires DSV4 model access).
- **Source files**: No production code changes — purely test coverage. However, tests may surface bugs that require subsequent fix PRs.
- **Dependencies**: Tests using the real tokenizer require `deepseek-ai/DeepSeek-V4-Flash-0731` (or V3.2 as fallback) to be downloadable in CI. CPU-only tests should stub the tokenizer where possible.
- **CI**: New unit tests register on `base-a-test-cpu` (est_time ~15s). E2E tests (if added) register on `base-b-test-1-gpu-large` with DSV4 model.
- **Known limitation**: The #33397 variant where DSML arrives in `reasoning_content` instead of `content` is a model/template-level issue, not a parser bug. Unit tests cannot reproduce it. Documented in spec References but not testable here.
