# BitLesson Knowledge Base

This file is project-specific. Keep entries precise and reusable for future rounds.

## Entry Template (Strict)

Use this exact field order for every entry:

```markdown
## Lesson: <unique-id>
Lesson ID: <BL-YYYYMMDD-short-name>
Scope: <component/subsystem/files>
Problem Description: <specific failure mode with trigger conditions>
Root Cause: <direct technical cause>
Solution: <exact fix that resolved the problem>
Constraints: <limits, assumptions, non-goals>
Validation Evidence: <tests/commands/logs/PR evidence>
Source Rounds: <round numbers where problem appeared and was solved>
```

## Entries

<!-- Add lessons below using the strict template. -->

## Lesson: mla-config-rope-dim-derivation
Lesson ID: BL-20260527-mla-config-rope-dim-derivation
Scope: calibrate.py, any code reading DeepSeek MLA config fields
Problem Description: Deriving qk_rope_head_dim as `head_dim - qk_nope_head_dim` fails when head_dim is itself derived from `hidden_size // num_heads`. DeepSeek-V3.2 has hidden_size=7168, num_attention_heads=128 → head_dim=56, not 128+64=192. The derived value is negative, making full_mla_q_width=None and silently skipping all Q hooks; "Calibration hooks did not fire" is only raised much later.
Root Cause: DeepSeek MLA configs have qk_rope_head_dim as an explicit field but no head_dim field. Using hidden_size // num_heads as head_dim is valid for non-MLA models but wrong for MLA where per-head dims don't pack to exactly hidden_size / num_heads.
Solution: Always read config.qk_rope_head_dim directly first. Only fall back to head_dim - qk_nope_head_dim for configs that lack the field. Validate the fallback is positive and raise a clear error with all derived values if it is not.
Constraints: Applies to any code reading MLA per-head dimensions from HuggingFace configs (calibrate.py, model dimension inference code).
Validation Evidence: test_dsv32_real_config_shape_q_hook_fires uses hidden_size=32, num_heads=4 (hidden_size//num_heads=8 ≠ qk_nope+qk_rope=12); 188 passed.
Source Rounds: 13

## Lesson: reshape-before-slice-mla
Lesson ID: BL-20260527-reshape-before-slice-mla
Scope: calibrate.py, dsa_backend.py, any code consuming MLA projection outputs (kv_b_proj, q_b_proj)
Problem Description: Flat-slicing `tensor[..., :H*nope_dim]` before reshape on MLA projection outputs selects V or RoPE columns from later heads, producing silently wrong noPE activations. The output shape is correct; only the values are wrong. Triggered when tensor has shape `[T, H*(nope+suffix)]` and the intent is to extract only the noPE prefix per head.
Root Cause: MLA projections pack per-head blocks sequentially: `[K0_nope|V0|K1_nope|V1|...]` for kv_b_proj, `[Q0_nope|Q0_rope|Q1_nope|Q1_rope|...]` for q_b_proj. Flat-slicing the first `H*nope_dim` columns crosses head boundaries and includes suffix columns from head 0 as head 1's noPE values.
Solution: Reshape to `[-1, H, nope_dim + suffix_dim]` FIRST, then slice `[..., :nope_dim]`. Wrapped as `_extract_mla_nope_prefix(tensor, num_heads, nope_dim, suffix_dim)` in calibrate.py. The same fix was applied to dsa_backend.py in Round 3 (line ~1436: `kv_proj_out.reshape(T, H_local, nope_dim + v_head_dim)[..., :nope_dim]`).
Constraints: Only applies to MLA projections (kv_b_proj, q_b_proj). Standard attention projections (k_proj, q_proj) have no suffix dimension and use direct reshape.
Validation Evidence: `test_mla_k_extraction_ignores_v_columns` and `test_mla_q_extraction_ignores_rope_columns` use sentinel V/RoPE values of 100.0; both tests fail under flat-slice and pass under reshape-then-slice. 185 passed, 0 failed in Round 11.
Source Rounds: 3 (dsa_backend.py fix), 11 (calibrate.py fix + sentinel tests)

## Lesson: torch-topk-aliasing-corrupts-input
Lesson ID: BL-20260527-torch-topk-aliasing-corrupts-input
Scope: selection_kernel.py, any allocation-free CUDA-graph-safe pipeline using torch.topk with `out=` argument
Problem Description: When `torch.topk(input=A, k, ..., out=(values=B, indices=A))` aliases the output indices tensor with the input tensor, the read is corrupted. Symptom observed in Round 17: input int64 tensor `[3, 1]` produced sorted ascending output values `[0, 1]` instead of `[1, 3]`. The bug only manifests in step-3 of the topk pipeline (the "ascending sort via smallest-first topk" trick) when an earlier step's output indices buffer is reused as both the input AND the (throwaway) output of the next topk call.
Root Cause: PyTorch's topk implementation does not enforce input/output non-aliasing for the `indices` output channel. The dispatcher reads input and writes output indices interleaved in the same buffer.
Solution: Allocate a dedicated `scratch_throwaway_idx` tensor (int64 [max_bs, max_top_k]) for the discarded gather indices of the second `torch.topk` call. Never alias topk's input with its output indices buffer, even when the indices are not used afterward.
Constraints: Applies to any allocation-free pipeline that uses `torch.topk(..., out=...)` more than once on overlapping buffers. The values output may safely alias other unrelated tensors but the indices output must not alias the input.
Validation Evidence: `test_retrieve_topk_graph_safe_per_request_valid_masks_position` fails with `sorted([0, 1]) != [1, 3]` under aliasing; passes after introducing `scratch_throwaway_idx`. 197 passed, 0 failed in Round 17.
Source Rounds: 17

## Lesson: ds-metadata-via-forward-context
Lesson ID: BL-20260527-ds-metadata-via-forward-context
Scope: deepseek_v2.py::_select_topk_indices, any DS hook that needs to read the active attention backend or its `forward_metadata` (DSAMetadata) — applies to other model files that gate on DS too.
Problem Description: Production publishes the active attention backend through `ForwardContext` (set by `cuda_graph_runner.py` and `model_runner.py`). Real `ForwardBatch` has NO `attn_backend` field and the CUDA-graph capture machinery constructs a fresh local `ForwardBatch` without DS-specific fields. Looking up `forward_batch.attn_backend.forward_metadata.<X>` silently returns None (or, with `MagicMock`, a fake auto-attribute) and the code falls through to a per-call lazy allocation (`torch.empty_like(...)`, `torch.empty(...)`). The bug is invisible in unit tests that mock `forward_batch.attn_backend` because the synthetic path mirrors the dead branch; only a `ForwardContext`-based probe exposes it.
Root Cause: Two source-of-truth paths for the attention backend exist in this codebase. `forward_batch.attn_backend` is documented on some unit-test fixtures but never set by `cuda_graph_runner.py` or `model_runner.py` in production. The DS path was historically written against the synthetic path.
Solution: Resolve DS metadata in `_select_topk_indices` in exactly this order: (1) `forward_batch.<field>` (set by `dsa_backend.init_forward_metadata` for dynamic non-graph forwards), (2) `has_forward_context() and get_attn_backend().forward_metadata.<field>` (real CUDA-graph capture/replay), (3) optionally a last-resort lazy allocation for CPU unit tests that synthesize forward_batch. Never look up `forward_batch.attn_backend.<X>` — it does not exist on production ForwardBatch and any unit test that mocks it is testing a dead path. When writing unit-test stubs for the backend, use `SimpleNamespace(use_mha=False, forward_metadata=...)` rather than `MagicMock()` so `getattr(backend, 'forward_metadata', None)` returns the intended None (or the explicit stub) instead of a polluting auto-attribute. Required regression: a CUDA test that publishes the field only through `ForwardContext`, spies `torch.empty_like` (call_count must stay 0), and asserts the returned tensor's `data_ptr` aliases the metadata buffer.
Constraints: Applies to any DS read-side hook that consumes per-batch attention-backend state during a forward pass. Does not apply to one-time bind-time data (those go through `server_args._*` attrs set by `finalize_double_sparsity_bind()`).
Validation Evidence: `test_select_topk_indices_uses_metadata_ds_topk_indices_out_via_forward_context` (spy on `torch.empty_like` + `data_ptr` aliasing); `test_select_topk_indices_zero_allocs_production_path` with the manual `forward_batch.ds_topk_indices_out` pre-set removed; 201 passed, 0 failed in Round 20. Symptom probe before fix: spy.call_count = 1 (one lazy `torch.empty_like` per call); after fix: 0.
Source Rounds: 18 (introduction of the synthetic path), 19 (partial fix for ds_graph_state only), 20 (full fix for both DS metadata fields).

## Lesson: shell-json-into-python-source
Lesson ID: BL-20260527-shell-json-into-python-source
Scope: any dev script that captures a tool's JSON output (e.g. `curl /get_server_info`, `kubectl get -o json`, `aws ... --output json`) and then needs to emit a metadata sidecar via inline Python.
Problem Description: A bash script splices the captured JSON variable directly into a Python heredoc as source code:
```bash
SERVER_ARGS_JSON="$(curl -s ...)"
python3 - <<PYEOF
import json
print(json.dumps({"server_args": ${SERVER_ARGS_JSON}}))
PYEOF
```
Works fine in development against fixtures of strings + ints + dicts of those, then fails in production with `NameError: name 'true' is not defined` because real JSON contains `true` / `false` / `null` (valid JSON tokens, NOT valid Python identifiers). The crash happens AFTER the expensive task succeeds, so the operator gets the benchmark JSONL but no reproducibility metadata, and only notices when the AC-11 audit fails weeks later.
Root Cause: JSON and Python have overlapping but unequal value-literal grammars. JSON `true|false|null` and Python `True|False|None` are different tokens; JSON numbers and Python numbers agree by accident on the integer subset but disagree on `Infinity` / `NaN`. Direct shell interpolation into a Python heredoc treats the JSON text as source code, so any non-trivially-overlapping value crashes.
Solution: Pass the captured JSON to Python as DATA, never as source. Two safe patterns: (a) export the variable and read it via `os.environ.get(...)`, then `json.loads(...)` inside Python; (b) pipe the JSON on stdin and `json.load(sys.stdin)`. Prefer (a) when the heredoc also needs other shell vars. Always handle the empty/malformed cases (record `{}` plus an error string so the operator can diagnose offline). Use `<<'PYEOF'` (quoted heredoc delimiter) so bash leaves the Python source alone. Extract the writer into a standalone helper (e.g. `development/_bench_meta_writer.py`) so it's directly testable: a registered test can feed `{"a": true, "b": null}` and assert valid output.
Constraints: Applies to any dev/CI/operator script that bridges shell capture and Python emission of metadata. Same risk exists in reverse (Python → shell) for similar reasons; the fix there is `json.dumps` not f-string.
Validation Evidence: Round 23 sidecar writer crashed in `benchmark.sh:71-83` when `/get_server_info` returned `{"disable_radix_cache": true, "kv_events": null}`. Round 24 extracted `development/_bench_meta_writer.py` with `json.loads` from `SERVER_ARGS_JSON` env var; 10 registered tests in `test_bench_meta_writer.py` exercise realistic, empty, malformed, and non-object inputs. 229 passed, 0 failed.
Source Rounds: 23 (introduced), 24 (fixed + locked).

## Lesson: importlib-dataclass-sys-modules
Lesson ID: BL-20260527-importlib-dataclass-sys-modules
Scope: any CI test that loads a sibling Python module by file path via `importlib.util.spec_from_file_location(...)` when the sibling file defines `@dataclass`-decorated classes.
Problem Description: setUpClass fails with `AttributeError: 'NoneType' object has no attribute '__dict__'` at the moment the loaded module's `@dataclass` decorator runs. Stack trace points into `/usr/lib/python3.12/dataclasses.py:749` (`sys.modules.get(cls.__module__).__dict__`). The decorator needs to look up the owning module by name to resolve forward-reference annotations, but with `spec_from_file_location("_foo", path)` the module is not in `sys.modules` until it has finished executing — and the dataclass decorator runs DURING execution.
Root Cause: Python's `dataclasses` module assumes any class it decorates has already been registered in `sys.modules[cls.__module__]`. Standard `import` registers the module before executing the body. `importlib.util.spec_from_file_location` + `spec.loader.exec_module(mod)` does NOT — it executes first, then the test code is responsible for registering. So the moment Python hits an `@dataclass`-decorated class inside the loaded file, `sys.modules.get("_foo")` returns None, and `.<...>__dict__` raises AttributeError.
Solution: Register the module in `sys.modules` BEFORE `spec.loader.exec_module(mod)`:
```python
spec = importlib.util.spec_from_file_location("_foo", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["_foo"] = mod          # <-- before exec
spec.loader.exec_module(mod)
```
This matches what `importlib.import_module` does internally. The same pattern is needed whenever the loaded module defines anything that introspects `sys.modules` during class body evaluation (dataclasses, typing.get_type_hints with forward refs, attrs, pydantic, etc.).
Constraints: Specific to CI tests that bridge "registered" and "manual" / file-path-only modules. The pattern recurs whenever a registered test exercises helpers from a module that is intentionally not on the import path (e.g. test/manual/ without __init__.py).
Validation Evidence: Round 25 AC-12 helper test (`test/registered/unit/manual/test_ac12_helpers.py`) hit this on first run with 11 ERROR results; all 11 turned PASSED after adding the `sys.modules["_ac12"] = mod` line before `exec_module`. The failing path was the `@dataclass _NIAHRunResult` definition inside `test/manual/test_double_sparsity_v32.py`.
Source Rounds: 25.

## Lesson: conservative-llm-output-parser
Lesson ID: BL-20260527-conservative-llm-output-parser
Scope: any test harness or eval that extracts a single-token answer (A-D, yes/no, JSON literal) from a free-form LLM response under temperature=0. Currently applied in `test/manual/test_double_sparsity_v32.py::_parse_mmlu_letter`; same pattern recurs in any future evaluator that reads short structured answers.
Problem Description: A naïve parser that "returns the first occurrence of an A-D character anywhere in the response" silently mis-scores common model completions. `"Answer: B"` returns `"A"` (the A in "Answer"). `"Awful but B is right"` returns `"A"` (the A in "Awful"). The defect is invisible without a counterexample test because most short A=A / B=B / C=C / D=D answers happen to coincide with the first A-D character. With `max_new_tokens` ≥ 4 (common for chat templates), the model frequently emits an answer-prefix and the parser converts every B/C/D answer into A silently. This pattern recurs across every LLM-eval harness Codex has reviewed in this loop.
Root Cause: LLM completions at `temperature=0` are NOT pure single-token answers — they contain answer-introducer prefixes, formatting punctuation, and explanation text. Scanning for "first occurrence of a valid answer character" ignores structure: the *position* and *boundary* of the character matters, not just its presence.
Solution: Two-tier conservative parser, regex-driven, with a hard fall-through to `None`:
```python
_LEADING = re.compile(r'^[\s\(\[\{<"\'`]*([A-Da-d])(?!\w)')
_MARKER  = re.compile(r'(?i)(?:answer\s*[:\-]?|answer\s+is|the\s+answer\s+is|option|choice)\s*[\(\[\{<"\'`:.]*\s*([A-Da-d])(?!\w)')

def parse(response):
    s = response.strip()
    if not s: return None
    m = _LEADING.match(s)
    if m: return m.group(1).upper()
    m = _MARKER.search(s)
    if m: return m.group(1).upper()
    return None
```
Critical details:
- `(?!\w)` after the captured letter forbids matching the first letter of a longer word like "Answer" / "Awful".
- Marker scan is case-insensitive (`(?i)`) so "answer is C" works alongside "Answer: B".
- Conservative: narrative text without a marker returns `None` rather than guessing.
- Unit tests must include the Codex-required counterexamples: `"Answer: B"`, lowercase `"b"`, `"(C)"`, `"D."`, narrative-no-marker, empty string.
Constraints: This is for "first valid token wins" eval patterns (MMLU, BoolQ, simple yes/no, etc.). For evaluations that need full structured output (JSON, multi-step reasoning), use a JSON parser or structured-output API instead.
Validation Evidence: Round 27 commit `faa41438e` — 13 registered regressions in `test_ac12_helpers.py` lock all of: `Answer:B→B`, `b→B`, `(C)→C`, `D.→D`, `[A]→A`, "answer is C"→C, "The answer is D."→D, "option B"→B, "Choice (A)"→A, narrative-no-marker→None, ""→None, whitespace-only→None, leading-punct→B. 271 unit tests pass total (was 254).
Source Rounds: 26 (introduced the broken first-char parser), 27 (fixed + locked).
