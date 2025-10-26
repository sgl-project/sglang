# Unified Harmony Pipeline Design (GPT‑OSS Chat + Responses)

## Goals
- One pipeline that supports GPT‑OSS Harmony models for both Chat Completions and Responses API.
- Keep the existing gRPC router stages and reuse them where possible.
- Build requests via Harmony (messages → token ids) before worker selection.
- Parse outputs via Harmony channels (analysis/commentary/final), not via separate text parsers.
- Integrate MCP tool loop for Responses; basic tool handling for Chat.
- Preserve current behavior for non‑Harmony models (no regressions).

## Non‑Goals
- Replacing the existing non‑Harmony chat pipeline for non‑GPT‑OSS models.
- Introducing model‑agnostic Harmony semantics to all models.
- Changing storage schemas for conversations/responses.

## High‑Level Flow

- Harmony detection: model id/name determines Harmony mode (gpt‑oss family).
- If Harmony:
  - Build request with Harmony encoder → `input_ids`.
  - Select worker using the built text (user input summary) after Harmony build.
  - Build gRPC `GenerateRequest` (inject Harmony stop token ids as needed).
  - Execute request, stream or collect results.
  - Parse outputs with Harmony, expose channels:
    - `analysis` → reasoning
    - `commentary` → tool calls
    - `final` → content text
  - For Responses: run MCP loop until no more tool calls.
- If non‑Harmony: unchanged current pipeline.

## Request Types
- Chat Completion (OpenAI compatible): existing `ChatCompletionRequest` remains.
- Responses API: existing `ResponsesRequest` remains.
- Internally, Harmony requests produce `input_ids` and use the same `Generate` execution path.

## Key Components

- `HarmonyDetector` (new): decides if a model uses Harmony.
  - Simple rule matching (e.g., contains `gpt-oss`, `gpt-4o`, `gpt-4.5`, `gpt-5`).

- `HarmonyBuilder` (new):
  - Converts Chat/Responses payloads → Harmony messages.
  - Encodes messages via Harmony encoding to `token_ids`.
  - Computes a concise text snippet (e.g., last user text) for policy‑based worker selection.
  - Adds Harmony stop token ids for assistant actions (`<|return|>`, `<|call|>`) to sampling params.

- `HarmonyParserAdapter` (new):
  - Wraps `openai_harmony::StreamableParser` for both non‑streaming and streaming.
  - Emits structured channel outputs; no separate Reasoning/Tool parser classes.
  - Maps channels to protocol shapes:
    - Chat: `reasoning_content`, `tool_calls`, `content` deltas or final.
    - Responses: `ResponseOutputItem::*` entries (web search, function call, text, etc.).

- MCP Tool Loop (existing, extended):
  - Responses: orchestrates tool execution and re‑prompting when Harmony indicates tool calls.
  - Continues to rebuild Harmony messages incrementally and re‑execute until no tool calls.

## Pipeline Integration (Stages)

Files for reference:
- `src/routers/grpc/context.rs`
- `src/routers/grpc/pipeline.rs`
- `src/routers/grpc/processing.rs`
- `src/routers/grpc/streaming.rs`
- `src/routers/grpc/responses/*`

### Stage 0: Detection
- When receiving Chat/Responses, call `HarmonyDetector` with model id/name.
- Store a Harmony flag in the request context (e.g., `ctx.state.response.harmony_mode = true`).

### Stage 1: Build (Harmony or Legacy)
- If Harmony:
  - Responses: `HarmonyBuilder.build_from_responses(request)` → `{ token_ids, stop_ids, selection_text, harmony_messages }`.
  - Chat: `HarmonyBuilder.build_from_chat(request)` → same outputs (harmony_messages derived from chat messages/tools).
  - Save `token_ids` and `selection_text` in `PreparationOutput`.
  - Create `StopSequenceDecoder` only for Generate fallback or non‑Harmony paths; Harmony uses its own parser.
- Else (non‑Harmony): keep existing `PreparationStage` behavior.

### Stage 2: Worker Selection
- Use `selection_text` produced by Harmony or `processed_messages.text` from legacy path.
- Policies remain unchanged.

### Stage 3: Client Acquisition
- Unchanged.

### Stage 4: Request Building
- Harmony: construct `proto::GenerateRequest` via `build_plain_generate_request` using `input_ids`.
  - Inject additional stop token ids for Harmony assistant actions if not already present.
- Legacy: unchanged (chat template path).

### Stage 5: Dispatch Metadata
- Unchanged.

### Stage 6: Execution
- Unchanged (single or PD depending on mode/policy).

### Stage 7: Response Processing
- Harmony, non‑streaming:
  - Collect `GenerateComplete` messages; for each, feed `output_ids` to `HarmonyParserAdapter`.
  - Build final response:
    - Chat: `ChatCompletionResponse` with `reasoning_content` and `tool_calls` mapped from Harmony channels.
    - Responses: list of `ResponseOutputItem` built from parsed messages and any remaining buffered state.
- Harmony, streaming:
  - Maintain per‑index `StreamableParser` state.
  - On each chunk, feed token ids to parser; emit SSE events:
    - Chat: use `ChatCompletionStreamResponse` deltas: `reasoning_content`, `tool_calls`, `content`.
    - Responses: use `responses/streaming` event emitter to produce `response.output_*` events; accumulate for final `response.completed`.
- Legacy paths: unchanged (`processing.rs` / `streaming.rs`).

## Chat vs Responses Details

### Responses API (Harmony)
1. Build Harmony messages from `ResponsesRequest` (instructions, input items, previous response chain, tools). Handle:
   - System/developer messages, reasoning effort, and built‑in tool descriptions when MCP allows.
   - Complex input: include tool outputs/items from previous responses.
2. Select worker with `selection_text` (e.g., last user text segment).
3. Build `GenerateRequest` via `input_ids`.
4. Execute gRPC `generate` (single or PD).
5. Collect from gRPC stream(s).
6. Parse with `HarmonyParserAdapter` to produce `ResponseOutputItem`s per channel:
   - `analysis` → `ResponseReasoningContent`
   - `commentary` → function/tool calls (`mcp_*`, `function_tool_call`, browser/python/container actions)
   - `final` → `response.text`
7. Tool handling + MCP:
   - Detect tool calls in parsed output; execute via MCP using existing manager and record items.
   - Build new Harmony messages with tool outputs; re‑encode and repeat from step 2 until no more tool calls or limits reached.
8. Persist and return. For streaming, interleave emitted events while accumulating final state for storage.

Constraints/Notes:
- Logprobs: not supported for Harmony (align with vLLM). Validate and reject if requested.
- Stop token ids: ensure Harmony assistant action tokens are included.

### Chat Completions (Harmony)
1. Build Harmony messages from Chat payload (messages/tools/tool_choice), including developer message for custom tools if present.
2. Select worker with `selection_text` (last user message text).
3. Build `GenerateRequest` via `input_ids`.
4. Execute gRPC `generate`.
5. Collect response from gRPC.
6. Parse with `HarmonyParserAdapter` and map to Chat:
   - `analysis` → `reasoning_content`
   - `commentary` → `tool_calls` (OpenAI format: id, name, arguments JSON)
   - `final` → assistant `content`
7. Handle tools and reasoning in both streaming and non‑streaming. No MCP loop here (Chat API remains OpenAI tool‑call semantics only).

## Mapping and Semantics

- Channel → Chat mapping:
  - `analysis` → `choice.delta.reasoning_content` (streaming) / `choice.message.reasoning_content` (final)
  - `commentary` tool calls → `choice.delta.tool_calls` / `choice.message.tool_calls`
  - `final` → `choice.delta.content` / `choice.message.content`
- Channel → Responses mapping:
  - Reasoning → `ResponseOutputItem::Reasoning`
  - Tool calls/actions → `ResponseOutputItem::McpCall`/`McpListTools`/`FunctionToolCall`/web search items, etc.
  - Final → `text`
- Finish reasons: derive from backend `finish_reason` and/or Harmony matched stops; tool calls force `tool_calls` finish reason for Chat.

## Interfaces and Data Flow

- `SharedComponents` additions:
  - Optional Harmony encoding provider or adapter.
- `RequestContext` additions:
  - `harmony_mode: bool`
  - `preparation.selection_text: Option<String>`
  - Optional `ResponseState.harmony` (parser state for streaming per choice), or maintained per‑index in `StreamingProcessor`.
- `RequestPipeline`:
  - Guarded branches inside Preparation/RequestBuilding/ResponseProcessing based on `harmony_mode`.

## Error Handling
- Input validation rejects Harmony + logprobs.
- If Harmony encoding fails, return 400 with clear message.
- If Harmony parser fails mid‑stream, send SSE error event and terminate.

## Observability
- Log model type, Harmony on/off, worker type (single/PD), and selected policy.
- Count reasoning tokens, tool output tokens, final tokens (align with vLLM counters).
- Emit warnings when MCP tool execution fails but continue the loop unless fatal.

## Compatibility
- Non‑Harmony models: existing tokenizer + chat template path remains intact.
- Existing `processing.rs` already supports token‑based reasoning parsing hook; Harmony adapter can plug into it for Chat fallback.
- `streaming.rs` gains a Harmony branch that bypasses generic text parsers and feeds token ids directly.

## Implementation Plan

- Add `harmony` module under `src/routers/grpc/`:
  - `detector.rs` – model detection logic.
  - `builder.rs` – Chat/Responses → Harmony messages → `input_ids` and `selection_text`.
  - `parser.rs` – streamable parser adapter for non‑streaming and streaming.
  - `types.rs` – shared structs/enums (channel mapping, message wrappers).
- Extend `SharedComponents` (optional) to inject Harmony encoder.
- Update `RequestContext`/`PreparationOutput`:
  - Add `selection_text: Option<String>`; set by Harmony or legacy builder.
  - Add `harmony_mode` flag (or detect per request type each stage).
- Update `PreparationStage` and `RequestBuildingStage` to branch on Harmony:
  - Build `input_ids`; inject Harmony stop token ids.
- Update `ResponseProcessingStage`/`StreamingProcessor`:
  - If Harmony: use `HarmonyParserAdapter` for channels → Chat/Responses shapes.
  - Else: current tool/reasoning parsers.
- Responses:
  - Reuse existing MCP tool loop; detect tool calls from Harmony outputs.
  - For streaming, emit `response.output_*` events as Harmony channels are parsed.
- Validation: reject Harmony+logprobs; ensure `tool_choice` is `auto` for Responses (aligning with vLLM limitation).

## Testing Plan
- Unit tests for detector, builder (encoding parity), parser (channel routing), and mapping.
- Golden tests vs known Harmony transcripts (analysis/commentary/final → expected API shapes).
- Responses MCP loop: single and multi‑iteration with mock MCP manager.
- Streaming tests: chunking boundaries, partial tool args, reasoning streams.

## vLLM Alignment
- Use StreamableParser only; no separate ReasoningParser/ToolParser for Harmony.
- Add Harmony assistant action stop tokens to sampling params.
- Responses: treat Harmony context as the source of truth for routing; `parse_output_message()` equivalent mapping.

## Open Questions
- Do we want to support Chat API MCP loop? (Proposed: no — keep MCP confined to Responses.)
- Where to source Harmony encoder: crate feature vs. optional runtime dependency.
- How much of `responses/streaming.rs` should be specialized for Harmony vs. keep transforming Chat SSE?

## Migration Notes
- Ship behind a config flag at first (per‑model opt‑in) to reduce risk.
- Keep legacy path intact; only activate Harmony branch for detected models.
- Incrementally land: detector + builder first (non‑stream), then streaming, then MCP loop.
