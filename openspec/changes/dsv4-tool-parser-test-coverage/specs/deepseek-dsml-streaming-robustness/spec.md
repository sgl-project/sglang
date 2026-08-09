## Overview

The DeepSeek-V4 DSML streaming tool-call parser must robustly handle all streaming delta patterns, tokenizer boundary splits, stream-end edge cases, and serving-code integration points. This spec defines the behavioral requirements that the test suite must verify.

**Scope**: `DeepSeekV4Detector` (inheriting `DeepSeekV32Detector`), `FunctionCallParser.parse_stream_chunk`, and the serving-code finish_reason/has_tool_calls/unstreamed-args pipeline in `serving_chat.py`.

**DSML markers**: The model emits `<｜DSML｜tool_calls>...</｜DSML｜tool_calls>` wrappers containing `<｜DSML｜invoke name="...">...</｜DSML｜invoke>` blocks. `DeepSeekV4Detector.__init__` overrides `bot_token` to `<｜DSML｜tool_calls>` (matching the V4 model), but inherits `has_tool_call` from V32 which also checks the secondary `<｜DSML｜invoke` marker — if the override were accidentally removed, `bot_token` would revert to V32's `<｜DSML｜function_calls>` and only the secondary check would save detection. DSML markers are regular BPE tokens (NOT special tokens); the 8-token opener `<｜DSML｜invoke name="bash">` has 4 internal split points.

---

## Requirements

### REQ-1: Preamble Preservation Across All Delta Splits

The streaming parser MUST preserve all prose text that precedes a DSML tool-call marker, regardless of how the text is split across streaming deltas.

- **REQ-1.1**: When a single delta contains both prose and a complete DSML opener, the prose MUST be emitted as `normal_text` and the tool call MUST be detected.
- **REQ-1.2**: When prose and the DSML opener arrive in separate deltas, the prose MUST be emitted immediately (not buffered until the opener completes).
- **REQ-1.3**: The total `normal_text` emitted across all deltas of a stream MUST equal the `normal_text` returned by `detect_and_parse` for the same complete input, at every possible split point. (Derived property — guards against "looks equivalent" streaming rewrites.)
- **REQ-1.4**: Preamble preservation MUST work for both wrapped (`<｜DSML｜tool_calls>`) and bare (`<｜DSML｜invoke` without wrapper) opener forms.
- **REQ-1.5**: A literal `<` in plain prose (not part of a DSML marker) MUST NOT be swallowed or buffered.

### REQ-2: Tokenizer-Level Delta Assembly

The streaming parser MUST correctly assemble DSML markers that are split across BPE token boundaries, as produced by speculative decoding (DSPARK).

- **REQ-2.1**: The 8-token opener `<｜DSML｜invoke name="...">` MUST be correctly assembled when split at 8 representative boundary patterns: (a) single-token chunks (interval=1), (b) split inside `<｜DSML｜` (between `<` and `｜`), (c) split between `｜DSML｜` and `invoke`, (d) split inside `invoke` (between `inv` and `oke`), (e) split inside `name="..."` (between ` name` and `="`), (f) two-token chunks (interval=2), (g) whole opener in one delta, (h) opener split exactly at the 4th token boundary. These 8 patterns cover all boundary classes; exhaustive 128-pattern enumeration is unnecessary since the failure mode is "buffer assembly breaks at a split boundary" and these 8 represent every class of split.
- **REQ-2.2**: Delta splitting MUST NOT cause false-positive DSML detection (e.g., a `<` in prose followed by `｜DSML｜` in the next delta should not be treated as a DSML opener unless it actually precedes an invoke).
- **REQ-2.3**: Tests MUST use the actual DeepSeek-V4-Flash tokenizer (or V3.2 as fallback) to generate BPE-accurate chunk boundaries, not arbitrary string-level splits.
- **REQ-2.4**: Streaming at interval=1 (single-token chunks) MUST produce the same tool-call detection result as non-streaming `detect_and_parse`.

### REQ-3: Stream-End Edge Cases

The streaming parser and serving code MUST handle stream termination gracefully when the DSML opener is incomplete.

- **REQ-3.1**: When the stream ends with a partial DSML opener in the detector's `_buffer` (e.g., `<｜DSML｜invoke name="ba`), the partial text MUST remain in `_buffer` (not cleared). The test MUST assert `_buffer` contains the partial opener after stream end. This pins the current behavior — the partial opener is neither emitted as `normal_text` nor explicitly flagged; it stays buffered.
- **REQ-3.2**: When the stream ends with a partial opener in `_buffer` and `current_tool_id == -1` (no tool was ever named), `has_tool_calls[index]` MUST remain `False` and `finish_reason` MUST remain `"stop"` — this is the current behavior. The test pins this invariant so that any future stream-end recovery change is detected.
- **REQ-3.3**: `_check_for_unstreamed_tool_args` MUST return `None` when `prev_tool_call_arr` is empty (no tool was ever named). This test pins the exact condition under which stream-end recovery does NOT fire.
- **REQ-3.4**: When the stream ends with a COMPLETE tool call (closing `</｜DSML｜invoke>` present), all tool-call deltas (name + arguments) MUST have been emitted, and `finish_reason` MUST be rewritten to `"tool_calls"`.
- **REQ-3.5** (Production-bug reproduction): The full integrated pipeline — detector `parse_stream_chunk` → `has_tool_calls` flag → `_check_for_unstreamed_tool_args` → finish_reason emission — MUST be tested end-to-end with the specific failure scenario: prose + partial DSML opener arriving across 3 separate deltas, then stream end. The test MUST assert: (a) prose is emitted as content, (b) no `tool_calls` in output, (c) `finish_reason="stop"`, (d) `_check_for_unstreamed_tool_args` returns None. This is the exact failure path from the ~7-8% production regression under DSPARK speculative decoding.

### REQ-4: Serving-Code Integration (finish_reason Pipeline)

The serving code MUST correctly rewrite `finish_reason` from `"stop"` to `"tool_calls"` when and only when the detector emitted at least one tool call with a `.name` field.

- **REQ-4.1**: Tests MUST exercise the real `_process_tool_call_stream` method on a `ServingChat` instance (or a thin subclass that overrides only I/O), NOT a copied helper function. The finish_reason rewrite logic (in the `_stream_generator` method, `if has_tool_calls.get(idx, False) and finish_reason_type == "stop": final_finish_reason = "tool_calls"`) is 3 lines; copying it into a test helper creates a mirror test that passes when production diverges. Note: `_process_tool_call_stream` sets `has_tool_calls[index]` but does NOT compute `final_finish_reason` — the finish_reason rewrite is in the outer streaming handler and can only be tested E2E (REQ-10) or by extracting it into a named method (a production code change, out of scope for this test-only spec).
- **REQ-4.2**: `has_tool_calls[index]` MUST be set to `True` for every `call_item` yielded by `_process_tool_call_stream`, regardless of whether `.name` is set (the assignment is unconditional at `has_tool_calls[index] = True` in the tool-call yield loop). This pins the actual production behavior — the finish_reason rewrite that depends on `has_tool_calls` is tested via E2E (REQ-10).
- **REQ-4.4**: When `has_tool_calls[index]` is `False` and `finish_reason_type == "length"`: `final_finish_reason` MUST be `"length"` (no rewrite).

### REQ-5: Parameter Parsing and Type Conversion

The parser MUST correctly parse both XML-format and JSON-format parameters inside invoke blocks, with proper type conversion.

- **REQ-5.1**: XML parameters with `string="true"` MUST be returned as strings.
- **REQ-5.2**: XML parameters with `string="false"` MUST be JSON-parsed (integers, booleans, arrays, nested objects).
- **REQ-5.3**: Direct JSON body (`{ "key": "value" }`) inside invoke MUST be parsed correctly.
- **REQ-5.4**: Integer parameters MUST be returned as `int`, not string.
- **REQ-5.5**: Boolean parameters MUST be returned as `bool`, not string.
- **REQ-5.6**: Array parameters MUST be returned as `list`, not string.
- **REQ-5.7**: `anyOf` with `null` type MUST accept `null` values.
- **REQ-5.8**: Nested object parameters MUST be parsed recursively.
- **REQ-5.9**: Malformed JSON in parameters MUST fall back to raw string value, not crash.

### REQ-6: Multiple Sequential Tool Calls

The parser MUST handle multiple invoke blocks within a single stream, with correct tool-index assignment and argument streaming.

- **REQ-6.1**: Two sequential invokes in one wrapped section MUST produce two distinct tool calls with incrementing `tool_index`.
- **REQ-6.2**: Each tool call MUST receive its own name-delta (with `.name` set) before argument-deltas.
- **REQ-6.3**: Multiple invokes split across various delta boundaries MUST all be detected, with no tool call lost.
- **REQ-6.4**: A mix of self-closing (`<｜DSML｜invoke name="x"/>`) and long-form invokes in the same block MUST both be parsed correctly.

### REQ-7: bot_token Mismatch Resilience

The parser MUST correctly detect tool calls via both the primary `bot_token` check and the inherited secondary `<｜DSML｜invoke` check. V4 overrides `bot_token` to `<｜DSML｜tool_calls>` in `__init__`, but inherits `has_tool_call` from V32 which includes the secondary check as defense-in-depth.

- **REQ-7.1**: `has_tool_call` MUST return `True` for text containing `<｜DSML｜tool_calls>` (the primary `bot_token` check works because V4 overrides `bot_token` in `__init__`).
- **REQ-7.2**: `has_tool_call` MUST return `True` for text containing a bare `<｜DSML｜invoke` without any wrapper.
- **REQ-7.3**: `_dsml_section_start` (when present) MUST find the DSML boundary even when `bot_token` is absent, using the fallback markers. (Does not exist in `main`; introduced by OPEN PR #33813.)
- **REQ-7.4**: A test MUST document and pin the fragility of the `bot_token` override: if V4's `__init__` override were accidentally removed, `bot_token` would revert to V32's `<｜DSML｜function_calls>` and the primary check would fail — only the inherited secondary `<｜DSML｜invoke` check would save detection. The test verifies that `has_tool_call` returns `True` for `<｜DSML｜tool_calls>` (primary check works with the override) and documents this fragility in a docstring.

### REQ-8: Regression Tests for Known Bug Classes

The test suite MUST include regression tests for every known DSML parser bug class identified from GitHub history.

- **REQ-8.1**: Preamble loss regression (#31786/#33813): prose before tool call in same delta MUST be preserved.
- **REQ-8.2**: Malformed JSON recovery (#32332): malformed JSON in invoke body MUST not crash the parser; it MUST fall back to raw text or empty params. The test MUST assert that the malformed JSON does NOT leak the raw DSML buffer into `normal_text` (the actual #32332 symptom).
- **REQ-8.3**: Final-delta argument loss (#32167): the last argument delta before stream end MUST be emitted, not lost.
- **REQ-8.4**: Fence leak (#29426): raw DSML markers MUST NOT leak into `normal_text` when the parser successfully detects a tool call.
- **REQ-8.5**: Max-tokens truncation (#30527): when `finish_reason="length"` mid-invoke, the partial tool call MUST be emitted with whatever arguments were parsed so far. The finish_reason stays `"length"` (NOT rewritten to `"tool_calls"` — the rewrite only fires when `finish_reason_type == "stop"`).
- **REQ-8.6**: Bare invoke without wrapper (#23786): a bare `<｜DSML｜invoke` without `<｜DSML｜tool_calls>` wrapper MUST be detected and parsed.
- **REQ-8.7**: Empty-content delta before tool call (#29441): an empty `normal_text=""` delta followed by a tool-call delta MUST NOT break downstream clients. The test MUST verify the serving-code emission path does not emit a separate SSE chunk with `content: ""` when normal_text is empty.

### REQ-9: DSPARK Truncation Simulation

The test suite MUST simulate DSPARK verify-truncation patterns that split the opener across verify boundaries.

- **REQ-9.1**: 8 representative ways to split the 8-token opener across deltas (listed in REQ-2.1) MUST produce a detected tool call when the full opener eventually arrives.
- **REQ-9.2**: Truncation at each of the 7 mid-opener positions (opener incomplete, stream ends) MUST produce the correct failure mode: `has_tool_calls=False`, `finish_reason="stop"`, partial opener in `_buffer`.

### REQ-10: Server-Level SSE Integration (E2E, Optional)

When feasible with GPU resources, the test suite SHOULD include an end-to-end server test that exercises the full streaming pipeline with a real DeepSeek-V4 model.

- **REQ-10.1**: A streaming request with tools MUST produce SSE chunks containing: content deltas (prose), tool_call deltas (name + arguments), and a final chunk with `finish_reason="tool_calls"`.
- **REQ-10.2**: The same prompt sent in non-streaming mode MUST produce the same tool call (name + arguments) as the streaming mode.
- **REQ-10.3**: The test SHOULD capture raw SSE and verify the delta ordering: content before tool_calls, name before arguments, finish_reason last.
- **REQ-10.4**: This test MAY be gated on GPU availability and model access; it SHOULD be registered as `base-b-test-1-gpu-large` or nightly.
- **REQ-10.5**: This test CANNOT reproduce the intermittent production bug (load-dependent, ~7-8% failure rate under concurrent DSPARK). Its value is limited to SSE format validation. The production-bug reproduction is covered by REQ-3.5 at the unit level.
- **REQ-10.6**: The E2E test MUST verify the finish_reason rewrite: when tool calls are present and `finish_reason_type == "stop"`, `final_finish_reason` MUST be `"tool_calls"`; when `finish_reason_type == "length"`, `final_finish_reason` MUST stay `"length"`. This is moved from REQ-4.2 because `_process_tool_call_stream` only sets `has_tool_calls[index]` — the finish_reason rewrite is in the outer `_stream_generator` method and cannot be observed at the unit level.

---

## References

### A. User-Reported Failure Symptoms (from GitHub Issues)

The following are actual user-reported symptoms from production deployments, organized by failure class. Each entry includes the issue number, reporter, environment, and the exact user-visible symptom.

#### Symptom Class 1: DSML Markup Leaks into Content

**#17593** ( RonaldJEN, DeepSeek-V3.2, 8×H200, v0.5.7, CLOSED):
> "Tool call responses are incorrectly parsed and placed in the content field as plain text instead of being structured in the tool_calls field."
> Actual output: `content: "tool_call_name\nget_weather\ntool_call_arguments\n{\"city\": \"北京\", \"unit\": \"celsius\"}\ncalculate\ntool_call_arguments\n{\"expression\": \"123*456\"}"`, `tool_calls: null`
> Expected: `content: null`, `tool_calls: [{name: "get_weather", ...}, {name: "calculate", ...}]`
> Does NOT occur with vLLM using the same model.

**#14695** (momaek, DeepSeek-V3.2, 8×H200, v0.5.6, EAGLE speculative decoding, CLOSED, high priority):
> "Model occasionally produces malformed tool call output during streaming. Instead of emitting the expected DSML format with ｜DSML｜ markers, the model sometimes outputs a raw XML-like snippet without the ｜DSML｜function_calls wrapper."
> Intermittent — most of the time output is correct DSML, but sometimes one response is malformed.
> SSE output shows prose ("我来帮您查询上海当前的天气情况。\n\n\n上海\nc\nelsius") then `finish_reason: "stop"` — tool call never emitted.

**#30480** (AG0708, hermes/qwen25, v0.5.x, OPEN):
> "Non-streaming responses leak raw tool-call markup into content when a tool call is truncated by max_tokens."
> Non-streaming: full raw markup in `content`. Streaming: content truncated + partial tool-call deltas.
> "Clients running agent loops (Continue, LangChain, AI SDK, etc.) see prompt-injection-looking garbage in `message.content` for non-streaming calls whenever a tool call hits the token budget."
> One truncated call discards every valid call before it (parallel tool calls → all dropped).

**#32332** (enrico9034, DeepSeek-V4, v0.5.15.post1, OPEN PR):
> "Non-string tool-call parameter streams a value that is not valid JSON (e.g. free-form or non-Latin prose) → `_partial_json_loads` raises `MalformedJSON` → exception escapes → returns entire raw DSML buffer as `normal_text`, leaking tool-call markup into the visible stream. Buffer never clears, so every subsequent decode token re-parses and re-emits the whole growing buffer."
> Production symptom: "bursts of `Error in parse_streaming_increment: Unexpected character З` (~90 in 75 minutes), clients receiving the same DSML block repeated on every token, proxies with mid-stream repetition detectors (LiteLLM) killing the live stream → dropped connections."

#### Symptom Class 2: Tool Call Completely Missing (prose complete, no tool call)

**#33397** (manueldomke, DeepSeek-V4-Flash-0731, 2×H200, v0.5.15.post1 + v0.5.16, OPEN):
> **Tier 2 symptom**: "DSML markup appears as visible text, or a complete, well-formed DSML block is streamed as `reasoning_content` while `tool_calls` never arrives, so the tool parser never sees it and the turn ends `finish_reason: stop` with empty content and no tool calls."
> Byte-level capture (1203 upstream chunks, one turn): `<｜DSML｜tool_calls>` / `<｜DSML｜invoke name="Bash">` / closers arrive **exclusively in `delta.reasoning_content`** — **no `delta.tool_calls` anywhere**, `content` deltas empty, final chunk `finish_reason: "stop"`.
> Initially read as a reasoning/tool parser classification bug; possibly a corrupted or dropped `<｜DSML｜tool_calls>` token.
> **Testability note**: The `reasoning_content` variant is a model/template-level issue (DSML emitted in the wrong channel), not a parser bug. Unit tests of the tool-call parser cannot reproduce it because the parser only sees `content` deltas. Documented here for completeness; not covered by REQ-* requirements.

**#14695** (also this class): Model emits prose ending with partial content ("上海\nc\nelsius") then `finish_reason: "stop"` — tool call opener never appears in the stream.

#### Symptom Class 3: Output Corruption Scaling with Concurrency

**#33397** (same issue, Tier 1 and Tier 3):
> **Tier 1**: "Grammatically fluent but semantically wrong words, drifting register, occasional wrong-language insertions (mixed German/English traffic, corrupted turns swap between the two mid-sentence). Easy to dismiss as model quality until harder tiers show up."
> **Tier 3**: "Total corruption — token soup. Verbatim: 'Allen Kolleginnen und Kollegen großzügig nuts. Hier die Rücksicht genommen.' (grammatical German with semantically random words, English word 'nuts' spliced in). Another: 'React was gebraten' ('React was fried'). Model sometimes notices and apologises: 'das war Unsinn — meine letzte Antwort war fehlerhaftes Zeug'."
> Load correlation: corruption clusters at 8-9 concurrent requests per DP rank. Single-request smoke tests clean.
> vLLM 0.26.0 on same node/checkpoint: no corruption.
> **Testability note**: Concurrency-dependent corruption is not reproducible in unit tests. Out of scope for this spec.

**#33163** (yiminghub2024, DeepSeek-V4-Flash, OPEN):
> "Deepseek-v4-flash toolcall error when switching runner_backend from marlin to flashinfer_mxfp4."
> Toolcall works correctly with `--moe-runner-backend marlin`, breaks with `flashinfer_mxfp4`.

#### Symptom Class 4: Empty Content Chunk Breaks Downstream Clients

**#29441** (florianfeigl, Qwen3.6-35B-A3B-FP8, v0.5.x, OPEN):
> "Streaming response includes an initial SSE chunk with empty content and no tool_calls before the actual tool call chunks arrive: `data: {"choices":[{"delta":{"content":""}}]}` — the middle chunk is the problem."
> "AI SDK's OpenAI provider (used by OpenCode, Open WebUI, and any code using @ai-sdk/openai) interprets an SSE chunk with `content: ""` and no tool_calls as a completed text turn. The client truncates the stream at that point, so tool calls are never delivered to the frontend."
> Workaround: HTTP proxy that filters empty-content SSE chunks.

#### Symptom Class 5: finish_reason Incorrectly "stop" When Tool Call Present

**#17558 / #17141 / #17320** (multiple users, various models):
> `finish_reason: "stop"` returned when a tool call was actually present in the response. The serving code's `has_tool_calls[index]` flag was not set because the detector never emitted a name-bearing ToolCallItem, so the finish_reason rewrite from "stop" to "tool_calls" did not fire.
> This is the exact serving-code gap identified in our investigation (REQ-3, REQ-4).

---

### B. GitHub Historical Parser Issues & PRs (sgl-project/sglang)

#### DSML / DeepSeek-specific PRs (primary target)

| # | Title | State | Streaming? | Bug / Edge Case |
|---|-------|-------|------------|------------------|
| #33813 | Preserve streaming preamble before DSML tool calls for all marker forms | OPEN | Yes | `parse_streaming_increment` returns `normal_text=""` on tool-call branch, dropping prose in same delta. Adds `_dsml_section_start` for bare `<｜DSML｜invoke`. Cherry-picked to production branch as `793e155cd2`. |
| #31786 | Fix: preserve content before DeepSeek streaming tool calls | OPEN | Yes | Earlier version of #33813; handles `find(bot_token)` but misses bare `<invoke>`. |
| #30480 | ISSUE: non-streaming leaks cross with streaming | — | Both | Cross-contamination between streaming/non-streaming parse paths. Fix PR: #30481 (hermes, qwen25). |
| #32332 | Catch MalformedJSON, reset buffer | OPEN | Yes | Malformed JSON in invoke body crashes parser; should catch and reset buffer. |
| #32167 | Final delta args lost (speculative/EAGLE) | OPEN | Yes | Last argument delta before stream end lost under speculative decoding. |
| #31009 | Think tags split across stream chunks | OPEN | Yes | Reasoning parser splits `</think>` tags across deltas incorrectly. |
| #29426 | DeepSeek-V3 fence leak | MERGED | Both | Raw DSML markers leak into `normal_text` content. |
| #30527 | Recover DSv32 calls on max_tokens truncation | OPEN | Yes | `finish_reason="length"` mid-invoke loses partial tool call. |
| #23786 | Bare invoke blocks without function_calls wrapper | MERGED | Both | Bare `<｜DSML｜invoke` without `<｜DSML｜tool_calls>` wrapper not detected. |
| #15217 | Streaming content loss deepseekv32 | MERGED | Yes | Streaming drops content under certain delta patterns. |
| #31351 | Don't leak partial bot_token bytes in V3/V3.1 | MERGED | Yes | Partial `<｜DSML｜` bytes leak as normal_text when opener is incomplete. |
| #24199 | DSV3.2 tool calling fix | MERGED | Both | Initial V3.2 encoding/format fixes. |
| #25229 | Fall back legacy structural tag for parallel calls | MERGED | Both | Structural tag fallback for parallel tool call scenarios. |
| #34050 | Default the tool-call parser when dsv4/dsv32 encoding | OPEN | N/A | Auto-detect tool-call parser from model encoding. |

#### Cross-detector issues (patterns applicable to DSV4)

| # | Title | Bug / Pattern |
|---|-------|---------------|
| #33181 | Inkling leaks tool name when turn opens with tool call | Tool name leaks into content on first-turn tool call. Fix: #34051. |
| #31912 | ReDoS PythonicDetector | Regex denial-of-service in Pythonic detector. Fix: #32107. |
| #33901 | GLM47Moe: only first of multiple calls in final streaming increment | Last `parse_streaming_increment` call loses all but first tool call. |
| #29441 | Empty content chunk before tool call breaks AI SDK | Empty `normal_text=""` delta before tool_call delta breaks downstream clients. |
| #23363 | KimiK2 streaming multi-turn hang | State not cleared between turns causes hang. |
| #19662 | Streaming doesn't support arguments with tool_choice=auto | tool_choice=auto + streaming drops arguments. |
| #23863 | Function name validation in streaming detectors | Invalid function names not validated in streaming path. |
| #17558 / #17141 / #17320 | finish_reason stop issue | `finish_reason="stop"` when tool call was present (serving-code gap). |

#### vLLM shared parser lineage (vllm-project/vllm)

| # | Title | Bug / Pattern |
|---|-------|---------------|
| #48931 | DSV4 leaks raw DSML, missing START token | DSML fragments leak into content when START token is missing. |
| #36654 | DSV3.2 frequent tool call failures | High failure rate under concurrent streaming. |
| #40801 / #40800 | DSV4 leaks DSML fragments in auto+streaming | tool_choice=auto + streaming leaks raw DSML. |
| #48702 | Streaming args truncated, non-string coalesce | Non-string argument types truncated in streaming. |
| #42878 | Fake-stream args not incremental deltas | Arguments not sent as incremental deltas in fake-stream mode. |
| #42747 | Invokes parser despite tool_choice=none | Parser invoked even when tool_choice=none, wasting cycles. |
| #47986 | Unwrap with wrong tool schema, parallel | Wrong schema unwrap for parallel tool calls. |

### B. Existing Test File Inventory (sglang main branch)

#### CPU unit tests — `test/registered/unit/function_call/` (13 files, 428 tests)

| File | Tests | Detector(s) |
|---|---|---|
| `test_function_call_parser.py` | 205 | 19 classes: Inkling(25), Pythonic(14), Mistral(5), BaseFormat(6), Llama32(7), KimiK2(6), **DSv3(1)**, **DSv32(9)**, **DSv4(6)**, Qwen3Coder(16), GptOss(2), Glm4Moe(10), Glm47Moe(12), JsonArrayParser(6), Lfm2(20), GigaChat3(~20), GetStructureConstraint(9), Qwen25(4), Gemma4(~14) |
| `test_hunyuan_detector.py` | 42 | HunyuanDetector |
| `test_poolside_v1_detector.py` | 28 | PoolsideV1Detector |
| `test_kimik3_structural_tag.py` | 27 | KimiK3Detector + xgrammar |
| `test_minimax_m3_detector.py` | 23 | MinimaxM3Detector |
| `test_normalize_json_schema_types.py` | 23 | Schema alias util |
| `test_mistral_detector.py` | 18 | MistralDetector (overlaps parser file) |
| `test_json_schema_constraint.py` | 18 | Constraint util |
| `test_llama32_detector.py` | 15 | Llama32Detector (overlaps) |
| `test_minicpm5_detector.py` | 14 | MiniCPM5Detector |
| `test_hermes_detector.py` | 12 | HermesDetector |
| `test_unknown_tool_name.py` | 2 | `SGLANG_FORWARD_UNKNOWN_TOOLS` env |
| `test_parallel_tool_calls.py` | 1 | JsonArrayParser |

#### GPU E2E integration tests

| File | Tests | Coverage |
|---|---|---|
| `test_openai_function_calling.py` | 14 | llama3 + pythonic only; streaming/non-streaming, finish_reason, tool_choice |
| `test_anthropic_tool_use.py` | 10 | llama3 via Anthropic SSE; raw `data:` parsing, event ordering |

**Critical gap**: NO server-level SSE integration test exists for any DeepSeek detector.

#### Production worktree only (not in main)

| File | Tests | Coverage |
|---|---|---|
| `test_deepseekv32_detector.py` | 10 | Preamble preservation (boundary sweep, fixed-size chunking, literal `<`, streaming invariant). Only in `.worktree/dsv4-930326-extra` (cherry-picked with #33813). |

### C. DSV4 Test Coverage Gap Analysis

**Current DSV4 coverage**: 6 tests in main + 10 tests in production worktree = **16 total**.

**Best-covered detector**: Qwen3Coder (16 tests) — systematic per-type parameter conversion, streaming, edge cases, structural tags. Inkling has 25 tests but focuses on canonical-framing semantics.

**Missing coverage dimensions** (what DSV4 lacks that Qwen3Coder/Inkling have):
1. Per-type parameter conversion (int/bool/array/anyOf/null) — DSV4 has none
2. Malformed input recovery — DSV4 has none
3. Multiple sequential calls with streaming — DSV4 has limited
4. Stream-end edge cases (partial opener) — no detector has this
5. Serving-code integration (finish_reason, has_tool_calls) — no detector has this
6. DSPARK truncation simulation — no detector has this
7. bot_token override fragility documentation — no detector has this
8. Server-level SSE E2E — only llama3/pythonic have this

### D. DSML Tokenization Reference

Confirmed from pod `arena-sglang-dsv4-flash-predictor-84f96577d8-84wns`:
- DSML markers are **regular BPE tokens**, NOT special tokens. `tokenizer_config.json` has zero added tokens containing 'invoke', 'tool_call', or 'DSML'.
- `<｜DSML｜invoke name="bash">` (opener) = 8 BPE tokens: `['<', '｜DSML｜', 'inv', 'oke', ' name', '="', 'bash', '">']`
- `<｜DSML｜tool_calls>` (wrapper) = 6 tokens
- `</｜DSML｜invoke>` (closer) = 5 tokens
- `<｜DSML｜parameter name="command" string="true">` = 11 tokens
- **bot_token override**: V4's `__init__` overrides `bot_token` to `<｜DSML｜tool_calls>` (matching the V4 model). V32's `bot_token` is `<｜DSML｜function_calls>`. `has_tool_call` checks `self.bot_token in text or "<｜DSML｜invoke" in text` — the secondary `<｜DSML｜invoke` check is inherited from V32 as defense-in-depth. If the V4 override were removed, only the secondary check would save detection.
