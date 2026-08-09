## Tasks

### Phase 1: Unit Tests — Detector Streaming Robustness (CPU, REQ-1, REQ-2, REQ-5, REQ-6, REQ-7)

- [x] **1.1** Create `test/registered/unit/function_call/test_deepseekv4_streaming_robustness.py` with boilerplate: imports (`DeepSeekV4Detector`, `FunctionCallParser`, `Tool`, `Function`, `CustomTestCase`, `register_cpu_ci`), `register_cpu_ci(est_time=15, suite="base-a-test-cpu")`, shared helpers (`_make_tools`, `_invoke`, `_wrapped_call`, `_collect_streamed_tool_calls`, `_stream`).
  - Verify: file is discoverable by `python test/run_suite.py --hw cpu --suite base-a-test-cpu --dry-run`.

- [x] **1.2** Implement `TestPreamblePreservation` (REQ-1): prose + complete tool call in one delta (wrapped + bare); boundary sweep at every split point for both forms; fixed-size chunking (1, 3, 7); streaming matches `detect_and_parse` invariant at every split; literal `<` in prose not swallowed; literal `<` before tool call splits at DSML marker not first `<`. MUST feed deltas one at a time through `parse_stream_chunk`, accumulating `normal_text`, then separately call `detect_and_parse` on full text for the streaming-vs-non-streaming invariant.
  - Verify: all tests pass on current `main` code; all tests pass on production worktree code.

- [x] **1.3** Implement `TestTokenizerLevelSplitting` (REQ-2): load tokenizer (V4-Flash or V3.2 fallback); encode a full tool-call turn; test the 8 representative opener-split patterns from REQ-2.1 (single-token, split inside `<｜DSML｜`, split before `invoke`, split inside `invoke`, split inside `name=`, two-token chunks, whole opener, split at 4th token boundary); feed through parser; assert tool call detected with correct name + params. Also test bare-invoke form at interval=1. Chunk boundaries MUST be generated via `tokenizer.encode`/`tokenizer.decode`, NOT string slicing — the test verifies BPE-accurate splits.
  - Verify: tool call detected in all 8 split-pattern variants; params match non-streaming result.

- [x] **1.4** Implement `TestParameterTypeConversion` (REQ-5): XML params with `string="true"` → string; `string="false"` → JSON-parsed (int, bool, array, nested object); direct JSON body; `anyOf` with null; malformed JSON falls back to raw string. Mirror the Qwen3Coder type-conversion test pattern.
  - Verify: each type assertion matches expected Python type.

- [x] **1.5** Implement `TestMultipleSequentialCalls` (REQ-6): two invokes in one wrapped section; each gets distinct `tool_index`; name-delta before arg-deltas; mix of self-closing and long-form; various delta split patterns.
  - Verify: `_collect_streamed_tool_calls` returns 2 calls with correct names and params.

- [x] **1.6** Implement `TestBotTokenOverride` (REQ-7): assert `has_tool_call` returns True for `<｜DSML｜tool_calls>` wrapper (primary check works because V4 overrides `bot_token` in `__init__`); assert `has_tool_call` returns True for bare `<｜DSML｜invoke` (secondary check inherited from V32); document in a docstring that if the `__init__` override were removed, `bot_token` would revert to V32's `<｜DSML｜function_calls>` and only the secondary check would save detection; if `_dsml_section_start` exists, assert it finds the boundary via fallback markers when `bot_token` is absent.
  - Verify: all assertions pass; test documents the override fragility.

- [x] **1.7** Implement `TestFalsePositiveDSMLDetection` (REQ-2.2): feed `<` in one delta, `｜DSML｜` in next delta (NOT followed by `invoke`), assert no tool call detected and text emitted as `normal_text`.
  - Verify: no tool call detected; text flows through as normal content.

### Phase 2: Stream-End Edge Cases (CPU, REQ-3)

- [x] **2.1** Implement `TestStreamEndPartialOpener` (REQ-3.1, REQ-3.2): feed prose + partial opener (`<｜DSML｜invoke name="ba`) across 2-3 deltas, then signal stream end. Assert: `has_tool_calls` is False, `_buffer` contains partial opener, `prev_tool_call_arr` is empty, `current_tool_id` is -1.
  - Verify: produces `has_tool_calls=False`; partial opener remains in buffer (not silently dropped).

- [x] **2.2** Implement `TestUnstreamedToolArgsRecovery` (REQ-3.3): create `FunctionCallParser` with `deepseekv4`; feed deltas leaving partial opener in buffer; call the real `_check_for_unstreamed_tool_args` method (import from `ServingChat`, do NOT copy logic); assert it returns None when `prev_tool_call_arr` is empty.
  - Verify: recovery returns None; partial opener remains in buffer.

### Phase 3: Serving-Code Integration (CPU, REQ-4)

- [x] **3.1** Implement `TestHasToolCallsViaRealCodePath` (REQ-4.1, REQ-4.2, REQ-4.4): use the `_make_test_serving_chat(tokenizer)` factory (see design.md D8) to create a `ServingChat` instance and call the real `_process_tool_call_stream` method. Feed deltas producing a name-bearing `ToolCallItem` → assert `has_tool_calls[idx]=True` (the assignment is unconditional for every `call_item`, regardless of `.name` — this pins the actual production behavior). Also test `finish_reason_type="length"` with `has_tool_calls=False` → stays `"length"` (observed via E2E only — see note below).
  - Note: `_process_tool_call_stream` sets `has_tool_calls[index]` but does NOT compute `final_finish_reason`. The finish_reason rewrite (`if has_tool_calls.get(idx, False) and finish_reason_type == "stop": final_finish_reason = "tool_calls"`) is in the outer `_stream_generator` method and can only be tested E2E (REQ-10/REQ-10.6) or by extracting it into a named method (a production code change, out of scope for this test-only spec). Do NOT assert finish_reason in this unit test.
  - Verify: `has_tool_calls[idx]` is True for every `call_item` yielded by `_process_tool_call_stream`; finish_reason is NOT asserted at the unit level.

- [x] **3.2** Implement `TestFullPipelineProductionBugReproduction` (REQ-3.5, REQ-3.1-3.3 + REQ-4 combined): feed prose + partial DSML opener across 3 separate deltas through `parse_stream_chunk`, then signal stream end. Run the full integrated pipeline: collect `(normal_text, calls)` from each delta → check `has_tool_calls` flag → call real `_check_for_unstreamed_tool_args` → observe finish_reason. Assert: (a) prose emitted as content, (b) no `tool_calls` in output, (c) `finish_reason="stop"`, (d) `_check_for_unstreamed_tool_args` returns None. This is the exact failure path from the ~7-8% production regression.
  - Verify: the test reproduces the production failure scenario at the unit level. If someone "fixes" the stream-end recovery to fire when it shouldn't, this test catches it.

### Phase 4: DSPARK Truncation Simulation (CPU, REQ-9)

- [x] **4.1** Implement `TestOpenerSplitRepresentative` (REQ-9.1, REQ-2.1): generate the 8 representative split patterns from REQ-2.1. For each, feed deltas (prose + opener split + body + close) through parser. Assert tool call detected with correct name.
  - Verify: all 8 patterns produce a detected tool call.

- [x] **4.2** Implement `TestOpenerTruncationMidOpener` (REQ-9.2): for each of the 7 mid-opener truncation positions, feed prose + partial opener (tokens 1..k where k=1..7), then signal stream end. Assert `has_tool_calls=False`, `_buffer` contains partial opener, `finish_reason` stays "stop".
  - Verify: all 7 truncation positions produce the correct failure mode.

### Phase 5: Regression Tests for Known Bug Classes (CPU, REQ-8)

- [x] **5.1** Implement `TestPreambleLossRegression` (REQ-8.1, #31786/#33813): prose + opener in same delta; assert prose preserved (not `normal_text=""`). Tag as regression test with PR number in docstring.
  - Verify: test passes on POST code; would fail on PRE code (prose would be lost).

- [ ] **5.2** *(DEFERRED to fix PR for #32332 — bug is unfixed in `main`; SGLang does not use `xfail`)* Design for `TestMalformedJSONRecovery`: feed invoke with malformed JSON body (`{"key": "val`); assert parser does not crash; assert it falls back to raw string or empty params. CRITICAL: also assert that the raw DSML buffer does NOT leak into `normal_text` (the actual #32332 symptom — buffer never clears, every subsequent token re-emits the whole DSML block). **Do NOT commit this test until the fix PR.** The task description serves as the spec for the fix PR author.
  - Verify (when fix is ready): no exception raised; result has `calls` with partial or empty params; `normal_text` does not contain DSML markers.

- [ ] **5.3** *(DEFERRED to fix PR for #32167 — bug is unfixed in `main`)* Design for `TestFinalDeltaArgLoss`: feed tool call where the last argument chunk is the final delta before stream end; assert the last argument chunk is emitted (not lost). Call real `_check_for_unstreamed_tool_args` to verify the recovery path catches the trailing args. **Do NOT commit this test until the fix PR.**
  - Verify (when fix is ready): `streamed_args_for_tool` matches full params after all deltas.

- [x] **5.4** Implement `TestFenceLeak` (REQ-8.4, #29426): feed complete tool call; assert `normal_text` does NOT contain any DSML markers (`<｜DSML｜`, `</｜DSML｜`, `invoke`, `parameter`).
  - Verify: `normal_text` is clean prose only.

- [x] **5.5** Implement `TestMaxTokensTruncation` (REQ-8.5, #30527): feed tool call where stream ends mid-body (after opener, before closing tag) with `finish_reason="length"`; assert partial tool call is emitted with whatever args were parsed.
  - Verify: `has_tool_calls=True`, `finish_reason="length"` (NOT rewritten — the rewrite only fires for `finish_reason_type=="stop"`). This is a behavior-pinning test (D7 category 3).

- [x] **5.6** Implement `TestBareInvokeWithoutWrapper` (REQ-8.6, #23786): feed bare `<｜DSML｜invoke` without `<｜DSML｜tool_calls>` wrapper; assert tool call detected.
  - Verify: `has_tool_call` returns True; `detect_and_parse` returns calls.

- [x] **5.7** Implement `TestEmptyContentDeltaBeforeTool` (REQ-8.7, #29441): feed an empty `normal_text=""` delta followed by a tool-call delta; call the real `_process_tool_call_stream` and collect yielded SSE chunks. Assert: no SSE chunk with `content: ""` is emitted to the client (the `if normal_text:` guard in `_process_tool_call_stream` must suppress empty content). This tests the SERVING-CODE emission path (the `if normal_text:` guard in `_process_tool_call_stream`), not just the detector.
  - Verify: no yielded chunk has `delta.content == ""`.

### Phase 6: E2E Server Test (GPU, OPTIONAL, REQ-10)

- [x] **6.1** Implement `TestDeepSeekV4StreamingE2E` in `test/registered/openai_server/function_call/test_dsv4_streaming.py`: launch server with `--tool-call-parser deepseekv4 --model deepseek-ai/DeepSeek-V4-Flash-0731`; send streaming request with tools; capture raw SSE; assert delta ordering (content → tool_calls → finish_reason); assert `finish_reason="tool_calls"`.
  - Gate: `register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")`. Skip if model unavailable.
  - Limitation: cannot reproduce the intermittent production bug (load-dependent). Value is SSE format validation only.
  - Verify: SSE stream contains content deltas, tool_call deltas, and correct finish_reason.

- [x] **6.2** Implement `TestDeepSeekV4StreamVsNonStream` (REQ-10.2): send same prompt in streaming and non-streaming modes; assert both produce the same tool call (name + arguments).
  - Verify: streaming and non-streaming results match.
