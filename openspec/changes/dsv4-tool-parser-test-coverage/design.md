## Context

The existing DSV4 detector test coverage lives in two places:
1. `test/registered/unit/function_call/test_function_call_parser.py` — `TestDeepSeekV4Detector` class, 6 tests, uses V3.2 tokenizer with interval=1 chunking
2. `.worktree/dsv4-9303e26-extra/test/registered/unit/function_call/test_deepseekv32_detector.py` — 10 preamble-preservation tests, only in production worktree (not main)

The test framework uses `unittest.TestCase` (or `CustomTestCase` for CI retry), `register_cpu_ci(suite="base-a-test-cpu")`, and reusable helpers: `_collect_streamed_tool_calls`, `tool_calls_by_index` reassembly, tokenizer-split chunking. The best-covered detector (Qwen3Coder, 16 tests) includes a systematic per-type parameter conversion suite that DSV4 lacks.

The serving code (`serving_chat.py`) has three critical integration points that no DeepSeek test exercises: `has_tool_calls[index]` flag (set unconditionally at `has_tool_calls[index] = True` in the tool-call yield loop of `_process_tool_call_stream`), `finish_reason` rewrite (in `_stream_generator`: `if has_tool_calls.get(idx, False) and finish_reason_type == "stop": final_finish_reason = "tool_calls"`), and `_check_for_unstreamed_tool_args` (checks `prev_tool_call_arr` and `streamed_args_for_tool` for trailing args). These are only tested E2E via llama3/pythonic in `test_openai_function_calling.py`.

## Local Development Environment

The Python package lives under `python/` (not repo root). The CPU variant uses `python/pyproject_cpu.toml` (package name `sglang-cpu`, `torch==2.12.0`, no `sgl-kernel`/`sgl-deep-ep`/`sgl-deep-gemm`).

### Setup (one-time)

```bash
# 1. Create .venv with Python 3.12 (NOT 3.14 — triton/compat issues)
uv venv .venv --python 3.12

# 2. Install sglang-cpu editable (same pattern as docker/xeon.Dockerfile)
cd python
cp pyproject.toml pyproject.toml.bak
cp pyproject_cpu.toml pyproject.toml
# macOS: triton==3.7.0 has no macOS wheels — remove it
sed -i '' '/triton==/d' pyproject.toml   # macOS (BSD sed)
# sed -i '/triton==/d' pyproject.toml     # Linux (GNU sed)
uv pip install -e . --python ../.venv/bin/python
mv pyproject.toml.bak pyproject.toml      # restore original
cd ..
```

### Running tests

```bash
# Option A: direct python call (recommended — no uv project discovery needed)
.venv/bin/python -m pytest test/registered/unit/function_call/test_deepseekv4_streaming_robustness.py -v

# Option B: uv run with UV_NO_SYNC (avoids uv trying to sync against main pyproject.toml)
UV_NO_SYNC=1 uv run --project python python -m pytest test/registered/unit/function_call/ -v

# Option C: activate venv then run
source .venv/bin/activate
python -m pytest test/registered/unit/function_call/ -v
```

**Why `UV_NO_SYNC=1`**: `uv run` by default syncs the project environment against `pyproject.toml`. Since the main `python/pyproject.toml` lists GPU-only deps (`sgl-kernel`, `sgl-deep-ep`), syncing would fail on macOS CPU. `UV_NO_SYNC=1` tells uv to skip syncing and just use the existing `.venv`. There is no `uv.lock` in the repo, so `--frozen` and `--locked` are not applicable.

## Goals / Non-Goals

**Goals:**
- Achieve test coverage parity with Qwen3Coder (16+ tests) for DSV4, then exceed it with DSPARK-specific and serving-code integration tests
- Cover all 9 requirement groups (REQ-1 through REQ-9) from the spec
- Use BPE-accurate tokenizer splitting (V4-Flash or V3.2 fallback)
- Pin the serving-code gap (partial opener silent drop) as a known-behavior test
- Provide regression tests for 7 known bug classes from GitHub history

**Non-Goals:**
- Fixing the parser bugs themselves (this change is test-only; fixes are separate PRs)
- Testing the mHC fix or model output distribution (requires GPU A/B, out of scope for unit tests)
- Testing the opencode client's SSE handling (client-side, not sglang's responsibility)
- Refactoring the existing test file structure (new tests go in a new file or extend the existing class)
- PRE/POST A/B comparison via monkey-patching (archaeology — the #33813 patch is deployed; testing old behavior has no guard value)
- Concurrency-corruption reproduction (#33397 Tier 1/3) — not reproducible in unit tests

## Decisions

### D1: New file vs. extend existing `test_function_call_parser.py`

**Decision**: Create a new file `test/registered/unit/function_call/test_deepseekv4_streaming_robustness.py`.

**Rationale**: The existing `test_function_call_parser.py` is already 5184 lines with 205 tests across 19 detector classes. Adding 20+ DSV4-specific tests would further bloat it. A separate file follows the pattern of `test_hunyuan_detector.py`, `test_hermes_detector.py`, etc. — model-specific detector tests get their own file.

### D2: Serving-code integration test approach — test the REAL code, not a copy

**Decision**: Exercise the real `_process_tool_call_stream` method on a `ServingChat` instance (or a thin subclass overriding only I/O), NOT a copied helper function.

**Rationale**: The original proposal (D2 in the prior draft) recommended extracting the finish_reason rewrite logic (the `if has_tool_calls.get(idx, False) and finish_reason_type == "stop"` check in `_stream_generator`) as a test helper `_compute_final_finish_reason`. This is a **mirror test** — it copies 3 lines of production code into a test helper and asserts the helper's output. Per the repo's own unit-test admission rules (`.claude/rules/unit-test-admission.md`): "Mirror tests that restate the implementation logic as assertions" are **Not admissible**. If someone changes the production code but not the helper, the test passes while production is broken.

Instead, the test should:
1. Create a `FunctionCallParser` with `tool_call_parser="deepseekv4"`
2. Feed deltas through `parse_stream_chunk`
3. Call the real `_process_tool_call_stream` (or a thin wrapper) to observe `has_tool_calls` flag propagation
4. Call the real `_check_for_unstreamed_tool_args` at stream end
5. Observe finish_reason emerging from the real emission code path (note: `_process_tool_call_stream` only sets `has_tool_calls[index]`; the finish_reason rewrite is in the outer `_stream_generator` method — see D8)

This avoids GPU requirements (no server launch) while testing the actual production logic. The serving-code methods are async generators; the test collects their yields synchronously via `asyncio.run` or an event loop helper.

**Alternative considered**: Full server launch with `popen_launch_server`. Rejected for unit tests — too slow, requires GPU. Kept as an optional E2E test (REQ-10).

### D3: Tokenizer selection

**Decision**: Use `deepseek-ai/DeepSeek-V4-Flash-0731` tokenizer if available; fall back to `deepseek-ai/DeepSeek-V3.2`.

**Rationale**: The existing tests already use V3.2. V4-Flash may have different BPE token boundaries for DSML markers (confirmed: markers are NOT special tokens). Tests should use the V4 tokenizer when possible, but V3.2 is acceptable since the detector code is shared (V4 inherits V32). The tokenizer is only used for generating chunk boundaries, not for parsing.

### D4: Opener-split pattern count — 8 representative, not 128 exhaustive

**Decision**: Test 8 representative opener-split patterns instead of all 128 exhaustive combinations.

**Rationale**: The failure mode is "buffer assembly breaks at a split boundary." There are only a few classes of split boundary:
- (a) Single-token chunks (interval=1) — tests every possible single boundary
- (b) Split inside `<｜DSML｜` (between `<` and `｜`)
- (c) Split between `｜DSML｜` and `invoke`
- (d) Split inside `invoke` (between `inv` and `oke`)
- (e) Split inside `name="..."` (between ` name` and `="`)
- (f) Two-token chunks (interval=2)
- (g) Whole opener in one delta
- (h) Split at the 4th token boundary (mid-name attribute)

128 subTest cases all test the same invariant ("all splits → detected"). The marginal coverage from pattern #9 to #128 is zero — no new failure mode is exercised. Per the repo's test admission rules: "One strong case beats several weak ones: each additional case must guard a distinct failure mode."

### D5: Test registration

**Decision**: Register on `base-a-test-cpu` with est_time=15s. If E2E tests are added, register separately on `base-b-test-1-gpu-large`.

**Rationale**: All unit tests are CPU-only (no server, no model weights). The tokenizer download may add a few seconds. Following the existing pattern in `test_function_call_parser.py` which registers on both `base-a-test-cpu` and `base-c-test-cpu`.


### D8: ServingChat instantiation — factory function

**Decision**: Tests that need to call `_process_tool_call_stream` must use a factory function `_make_test_serving_chat(tokenizer)` rather than a "thin subclass overriding only `__init__`".

**Rationale**: `_process_tool_call_stream` accesses `self.tokenizer_manager.tokenizer`, `self._effective_tools()`, `self._get_history_tool_calls_cnt()`, `self._process_tool_call_id()`, and `self.tool_call_parser`. A thin subclass overriding only `__init__` does not cover these dependencies. The factory function must:

```python
def _make_test_serving_chat(tokenizer):
    sc = ServingChat.__new__(ServingChat)
    # tokenizer_manager: MagicMock with .tokenizer set to the real DeepSeek tokenizer
    sc.tokenizer_manager = MagicMock()
    sc.tokenizer_manager.tokenizer = tokenizer
    # tool_call_parser: must be "deepseekv4"
    sc.tool_call_parser = "deepseekv4"
    # Stub methods that _process_tool_call_stream calls
    sc._effective_tools = lambda request=None: []
    sc._get_history_tool_calls_cnt = lambda request: 0
    sc._process_tool_call_id = lambda call_item, cnt: f"call_{cnt}"
    # content dict template (what _process_tool_call_stream reads)
    # content = {"meta_info": {"id": "test-id"}, ...}
    return sc
```

The `content` dict template passed to `_process_tool_call_stream`:
```python
content = {
    "meta_info": {
        "id": "test-stream-id",
        "completion_tokens": 0,
        "reasoning_tokens": 0,
    }
}
```

Tests call `_process_tool_call_stream(parser, content, request, index=0)` and collect yielded SSE chunks. The `request` can be a simple mock with `.model = "test-model"` and `.stream_options = None`.

## Risks / Trade-offs

- **Tokenizer download dependency**: Tests require downloading the DeepSeek tokenizer in CI. If the model is gated or unavailable, tests will fail. Mitigation: skip tests gracefully if tokenizer download fails, and use V3.2 as fallback (widely available).
- **Serving-code coupling via real method calls**: Tests that call `_process_tool_call_stream` directly are coupled to the method's signature and async-generator behavior. If the method is refactored, the test breaks. This is **desirable** — a breakage means the test caught a real change in the serving-code path, which is exactly what we want. The test should import the real `ServingChat` class and call real methods, not reimplement them.
- **E2E test limitations**: The E2E server test (REQ-10) cannot reproduce the intermittent production bug (load-dependent, ~7-8% failure rate). Its value is limited to SSE format validation. The production-bug reproduction is covered by REQ-3.5 at the unit level. The E2E test is optional and gated on GPU availability.

## D7: Handling tests for known-unfixed bugs

Tests fall into three categories with different handling:

1. **Already-fixed regressions** (tasks 5.1, 5.4, 5.6 — #33813/#29426/#23786): Test asserts the **correct** behavior. Passes on current code. Would fail on pre-fix code. Committed now.

2. **Known-unfixed bug tests** (tasks 5.2, 5.3 — #32332/#32167): These bugs are **not yet fixed** in `main`. Per the repo's unit-test-admission rules: "Bug regression. Guards a bug that actually happened. Before committing, verify the case fails on the pre-fix code and passes on the fix." Since there is no fix yet, these tests cannot be committed as passing tests. **SGLang does NOT use `xfail` or `expectedFailure`** — zero occurrences in the entire repo. These tests must be written as part of the **fix PR** — test + fix together. In this spec, these tasks are **design-only**: they define what the test should assert, but implementation is deferred to the fix PR.

3. **Behavior-pinning tests** (task 3.2, task 5.5, REQ-3.5 — production gap): These test the **current** behavior as an architectural invariant. Task 3.2 pins partial opener → `finish_reason="stop"` (passes now; fails if stream-end recovery is added). Task 5.5 pins `finish_reason="length"` staying `"length"` when `has_tool_calls=True` but `finish_reason_type != "stop"` (passes now; fails if the rewrite condition is widened). Both pass now. When someone adds stream-end recovery or changes the rewrite condition, the test will fail — forcing the fix author to update the test. This falls under admission categories 2 (derived property) and 3 (critical-path bookkeeping), not category 1 (bug regression). Committed now.

**SGLang convention**: No `xfail`, no `expectedFailure`, no "temporarily failing" tests in `main`. Bug regression tests come with their fix. Behavior-pinning tests document and guard current architecture.
