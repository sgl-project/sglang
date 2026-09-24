# Vendored patch: `openai-protocol` v1.0.0

This is a locally-patched copy of the `openai-protocol` crate (pinned to
`=1.0.0` in `sgl-model-gateway/Cargo.toml`, upstream:
<https://github.com/lightseekorg/smg>), applied via `[patch.crates-io]`
in `sgl-model-gateway/Cargo.toml`.

## Why

Fixes <https://github.com/sgl-project/sglang/issues/30781>:
`sgl-model-gateway` (the Rust router / "SGLang Model Gateway") rejected
`/v1/responses` requests whose `tools[].type` was anything other than
`function`, `web_search_preview`, `code_interpreter`, or `mcp` — e.g.
`type: "custom"`, which the OpenAI Codex CLI (and other clients) send.

The root cause: `openai_protocol::responses::ResponseToolType` only had
4 variants, while the Python server's
`sglang.srt.entrypoints.openai.protocol.RESPONSE_TOOL_TYPES` (the source
of truth the backend actually validates against) has 12. Requests that
sglang's own Python server accepts fine were being rejected by the
router with a 400 before they ever reached a worker.

Upstream `openai-protocol` did add the missing variants starting in
v1.8.0, but as part of a larger, source-incompatible redesign of
`ResponseTool` (flat struct-with-enum-tag -> tagged enum with
per-variant payload types) that would require rewriting ~35 call sites
across `sgl-model-gateway`'s gRPC/OpenAI routers. To fix the bug without
that unrelated churn, this vendored copy keeps the v1.0.0 API shape
as-is and only extends `ResponseToolType` (see `src/responses.rs`) with
the missing variants (`web_search`, `file_search`, `image_generation`,
`computer_use_preview`, `local_shell`, `custom`, `namespace`,
`tool_search`), plus a `#[serde(other)]` catch-all `Unknown` variant so
future Python-side additions degrade to "still proxied" instead of
"hard 400" again.

## Maintenance

Delete this directory and the corresponding `[patch.crates-io]` entry
once `sgl-model-gateway` is upgraded to consume a version of
`openai-protocol` (>= 1.8.0) that includes these variants natively, at
which point the downstream call sites will also need updating for the
`ResponseTool` API redesign mentioned above.
