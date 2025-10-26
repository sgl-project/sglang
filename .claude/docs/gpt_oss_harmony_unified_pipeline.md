# GPT‑OSS Harmony Unified Pipeline (Chat + Responses)

## Objectives
- One pipeline that supports GPT‑OSS Harmony models for both Chat Completions and Responses API.
- Build requests via Harmony (messages → token ids) before worker selection; reuse existing gRPC pipeline stages.
- Parse outputs via Harmony channels (analysis/commentary/final), not separate text parsers.
- Integrate MCP tool loop for Responses; preserve OpenAI tool semantics for Chat.
- Keep non‑Harmony models on current path with zero regressions.

## Context and References
- Current gRPC pipeline and context:
  - `src/routers/grpc/context.rs`, `pipeline.rs`, `processing.rs`, `streaming.rs`, `router.rs`
  - `src/routers/grpc/responses/*` (conversions, handlers, streaming, tool loop)
- vLLM Harmony behavior (gpt‑oss):
  - Uses HarmonyContext + StreamableParser; no separate Reasoning/Tool parsers
  - Token‑level processing; channels: `analysis` (reasoning), `commentary` (tool), `final` (text)
  - `parse_output_message()` routes Harmony messages to Responses items (and Chat mapping)
  - Adds assistant action stop tokens to sampling params; disables logprobs for Harmony

## High‑Level Design

- Harmony detection per request decides the branch.
- If Harmony model:
  1) Construct Harmony messages from payload (Chat or Responses).
  2) Render to `input_ids` with Harmony encoding; compute a concise `selection_text` for routing policy.
  3) Select worker using `selection_text` (post‑build).
  4) Build `proto::GenerateRequest` using `input_ids` (no chat template), injecting Harmony assistant action stop token ids.
  5) Execute (single or PD) and collect/stream responses.
  6) Feed output token ids into StreamableParser; route parsed messages per channel.
  7) Chat: map channels to OpenAI fields; Responses: map to `ResponseOutputItem` and run MCP tool loop until stable.
- If non‑Harmony model: current pipeline unmodified.

## Pipeline Mapping To sgl-router

We leverage existing stages with minimal branching.

- Stage 0 (Detection)
  - Add `harmony_mode: bool` to `RequestContext` or compute per stage via a detector.
  - Detector: check model id/name (e.g., contains `gpt-oss`).

- Stage 1 (Preparation)
  - Harmony branch (new):
    - Chat: build Harmony messages from `ChatCompletionRequest` (messages, tools/tool_choice).
    - Responses: build Harmony messages from `ResponsesRequest` (instructions, input items, prev response chain, tools).
    - Render Harmony messages to `token_ids` (prompt); compute `selection_text` (e.g., last user text segment).
    - Create/record Harmony assistant action `stop_token_ids` to pass downstream.
    - Store into `PreparationOutput`:
      - `original_text: None`, `token_ids`, `selection_text: Some(...)`, and (for Chat) a preserved filtered request if needed.
  - Legacy branch: existing `PreparationStage` logic (chat template, tokenizer encode, tool constraints, stop decoder setup).

- Stage 2 (Worker Selection)
  - Use `selection_text` if set (Harmony), else `processed_messages.text`.
  - Unchanged policy mechanics.

- Stage 3 (Client Acquisition)
  - Unchanged.

- Stage 4 (Request Building)
  - Harmony branch: build `GenerateRequest` via `build_plain_generate_request` using Harmony `token_ids` and optional cache salt; inject assistant action stop token ids.
  - Legacy branch: current build path (chat template text and multimodal inputs).

- Stage 5 (Dispatch Metadata)
  - Unchanged.

- Stage 6 (Execution)
  - Unchanged (Single or Dual dispatch). We pass through response streams or collect completes.

- Stage 7 (Response Processing)
  - Harmony non‑streaming:
    - For each `GenerateComplete`, feed `output_ids` into `HarmonyParserAdapter` (token → messages with channels).
    - Chat: build `ChatCompletionResponse` with:
      - `analysis` → `reasoning_content`
      - `commentary` → `tool_calls` (id, name, JSON args)
      - `final` → assistant `content`
    - Responses: produce `Vec<ResponseOutputItem>` via routing function (`parse_output_message()` equivalent), then apply `parse_remaining_state()` for incomplete trailing content.
  - Harmony streaming:
    - Maintain per‑index StreamableParser state inside `StreamingProcessor`.
    - On each token chunk, process via parser; emit SSE deltas:
      - Chat SSE: map deltas to `ChatCompletionStreamResponse` (role on first; then content/reasoning/tool_calls; finish + usage).
      - Responses SSE: emit `response.output_*` events via a dedicated emitter using parsed channels; accumulate final state for completion.
  - Legacy: existing `processing.rs` and `streaming.rs` behavior.

## Responses API Specifics (Harmony)

1) Build Harmony from `ResponsesRequest`:
   - If new conversation: include system (reasoning effort), developer (tools) when custom tools present; otherwise keep minimal.
   - If continuing: splice previous messages; for a prior final message, drop intermediate analysis messages per vLLM logic.
   - Append new input: strings or structured items including past tool outputs.
2) Worker selection uses `selection_text` (last user input text) computed during Harmony message build.
3) Construct `GenerateRequest` with Harmony `input_ids` (no chat template), add Harmony action stop tokens.
4) Execute gRPC generate.
5) Collect completes or stream chunks.
6) Parse via Harmony:
   - `analysis` → ResponseReasoningItem
   - `commentary` + `functions.*` → ResponseFunctionToolCall (arguments already JSON)
   - `commentary` + `python` / `browser.*` / `container.*` → treat as reasoning output items (web search calls where applicable)
   - `final` → ResponseOutputMessage (text)
7) MCP loop:
   - Detect tool calls; execute via `McpManager` and active sessions (per tool). Add tool outputs to the Harmony message sequence.
   - Re‑render to token ids; re‑select worker; rebuild request; execute; parse; repeat until no more tool calls.
   - Background/streaming paths: preserve cancellation support (store client + grpc_request_id, as current pipeline already does in `execute_chat_for_responses`).
8) Persist final response to storage when `store=true` (unchanged persistence code paths).

Validation and constraints:
- Logprobs unsupported for Harmony; reject at handler level (align vLLM).
- For Responses, Harmony supports only `tool_choice=auto` (align vLLM); reject other modes for Harmony models.
- Ensure Harmony assistant action stop tokens always included.

## Chat Completions (Harmony)

- Request build:
  - Convert OpenAI Chat payload to Harmony messages (messages + optional tool schema context in developer message when tools present).
  - Render to `input_ids`; compute `selection_text`.
- Execution and parsing:
  - Same as Responses but return Chat schema:
    - `analysis` → `reasoning_content`
    - `commentary` → `tool_calls`
    - `final` → `content`
  - Finish reason: `tool_calls` if any tool calls are present; otherwise backend finish reason. Usage computed from counts.
- No MCP loop in Chat API (keep legacy OpenAI tool semantics only).

## Data/Types Integration

- `SharedComponents`
  - Optionally include a Harmony encoding provider/adapter (to avoid coupling to tokenizer chat template).

- `RequestContext`
  - Add `harmony_mode: bool` (or derive on the fly via per‑request detector).
  - In `PreparationOutput` add `selection_text: Option<String>` to support policy selection from Harmony builds.
  - `ResponseState` streaming can keep per‑index parser handles in `streaming.rs` (no change to `RequestContext` needed).

- `RequestPipeline` branching points
  - Preparation, RequestBuilding, ResponseProcessing/Streaming handle Harmony vs. legacy.

## Metrics and Usage

- Track Harmony token usage similar to vLLM:
  - `prompt_tokens`, `output_tokens`, `cached_tokens`
  - `reasoning_tokens` (count when current channel is `analysis` or `commentary`)
  - `tool_output_tokens` derived from per‑turn deltas (optional; can be Phase 2)
- Chat: fill `Usage` as today; include completion tokens details if applicable.
- Responses: populate usage details structure when emitting streaming and final responses.

## Error Handling
- Harmony message rendering failures → 400 with clear message.
- Parser failures during stream → SSE error event and termination (stream marked completed).
- MCP execution failures → warnings and appropriate status in Responses; continue loop unless fatal.

## Implementation Plan

- New module `src/routers/grpc/harmony/`:
  - `detector.rs` – model detection by id/name; configurable list/prefixes.
  - `builder.rs` – Chat/Responses → Harmony messages → `input_ids` and `selection_text`; inject assistant action stop tokens.
  - `parser.rs` – StreamableParser adapter (non‑stream/stream) that exposes parsed messages with channels and routing helpers.
  - `types.rs` – shared Harmony types (Message wrapper, channel enums) mirroring openai‑harmony structures.
- Modify pipeline stages:
  - `PreparationStage`: if Harmony, call builder; set `selection_text` and `token_ids`.
  - `RequestBuildingStage`: if Harmony, use `build_plain_generate_request` with `input_ids`, inject Harmony stop tokens.
  - `ResponseProcessingStage` and `streaming.rs`: add Harmony path that feeds token ids into StreamableParser and maps outputs.
- Responses handlers:
  - Validate Harmony constraints (no logprobs; tool_choice=auto).
  - Use existing MCP tool loop; drive iterations based on Harmony‑detected tool calls.
  - For streaming, add a direct Harmony responses emitter (optional), or continue to transform Chat SSE as today.

## Open Questions
- Should we emit native Responses SSE directly for Harmony instead of transforming Chat SSE? (Lower overhead; recommended phase 2.)
- Centralize token accounting for Harmony across non‑stream/stream to match vLLM per‑turn stats.
- How to configure Harmony detection (static list vs. registry hints from workers)?

## Alignment With vLLM
- StreamableParser is the only parsing mechanism; channels drive routing.
- Inject Harmony assistant action stop tokens to sampling params.
- Disallow logprobs and restrict Responses tool_choice to `auto` in Harmony.
- Worker selection after Harmony build using `selection_text`.

