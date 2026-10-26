//! Anthropic Messages adapter shared by the HTTP routes and the gRPC
//! service: an `api.v1` Anthropic request becomes one OpenAI chat request
//! served by the chat core, and the chat reply or stream is re-shaped into
//! Anthropic blocks and events. `entrypoints/anthropic/serving.py` is the
//! reference for every translation here.

use std::pin::Pin;

use axum::http::StatusCode;
use dynamo_parsers::{ToolChoice as DynamoToolChoice, ToolDefinition};
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionToolChoiceOption, CompletionUsage,
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    FinishReason as OpenAIFinishReason,
};
use futures::{Stream, StreamExt};
use sglang_api_types::api::v1 as api;
use sglang_api_types::api::v1::{
    anthropic_content::Value as ContentValue, anthropic_content_block::Block,
    anthropic_content_delta::Delta, anthropic_search_source::Value as SearchSource,
    anthropic_stream_event::Event,
};
use sglang_processor::dynamo_tool_choice;
use tonic::{Code, Status};

use super::core::CoreError;
use super::handlers::{CoreCall, CoreHandle};
use super::http::openai::chat::{
    SamplingDefaults, chat_event_stream, chat_sampling, prepare_chat_prompt, starts_in_reasoning,
    unary_chat_response,
};
use super::http::openai::pd_routing::PDRoutingFields;
use super::http::openai::{ChatFormatter, ChatTemplateKwargs, unix_seconds_u32};
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::request::GenerateRequest;

pub(crate) type Events = Pin<Box<dyn Stream<Item = api::AnthropicStreamEvent> + Send + 'static>>;

/// The per-listener facts an Anthropic call needs; both transports build one
/// from their own state.
pub(crate) struct Anthropic<'a> {
    pub(crate) core: &'a CoreHandle,
    pub(crate) chat_formatter: Option<&'a ChatFormatter>,
    pub(crate) server_args: &'a ServerArgs,
}

/// A failure in Anthropic terms: the HTTP status, the documented
/// `error.type` for it, and a message a client may see (5xx messages are
/// replaced, as Python does, so no internal detail leaks).
#[derive(Debug)]
pub(crate) struct AnthropicFailure {
    pub(crate) status: StatusCode,
    pub(crate) kind: &'static str,
    pub(crate) message: String,
}

impl AnthropicFailure {
    fn invalid(message: impl Into<String>) -> Self {
        Self::from_status(StatusCode::BAD_REQUEST, message)
    }

    pub(crate) fn from_status(status: StatusCode, message: impl Into<String>) -> Self {
        let message = if status.is_server_error() {
            "Internal server error".to_string()
        } else {
            message.into()
        };
        Self {
            status,
            kind: error_type(status),
            message,
        }
    }

    fn from_core(error: CoreError) -> Self {
        Self::from_status(error.http_status(), error.to_string())
    }

    pub(crate) fn envelope(&self) -> api::AnthropicErrorResponse {
        api::AnthropicErrorResponse {
            r#type: "error".into(),
            error: Some(api::AnthropicError {
                r#type: self.kind.into(),
                message: self.message.clone(),
            }),
        }
    }

    pub(crate) fn grpc_status(&self) -> Status {
        let code = match self.status {
            StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => Code::InvalidArgument,
            StatusCode::NOT_FOUND => Code::NotFound,
            StatusCode::TOO_MANY_REQUESTS => Code::ResourceExhausted,
            StatusCode::SERVICE_UNAVAILABLE => Code::Unavailable,
            StatusCode::GATEWAY_TIMEOUT => Code::DeadlineExceeded,
            _ => Code::Internal,
        };
        Status::new(code, self.message.clone())
    }
}

/// Python's `ERROR_TYPE_MAP` plus the two the HTTP exception handler adds.
fn error_type(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        408 => "request_timeout_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 => "overloaded_error",
        _ => "api_error",
    }
}

fn message_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4().simple())
}

/// One submitted chat generation and what its reply shaping needs.
struct Submitted {
    call: CoreCall,
    model: String,
    parser: Option<String>,
    reasoning_parser: Option<String>,
    starts_in_reasoning: bool,
    tools: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
}

fn is_server_tool(tool: &api::AnthropicTool) -> bool {
    tool.r#type.as_deref().is_some_and(|kind| {
        ["web_search_", "computer_", "bash_", "text_editor_"]
            .iter()
            .any(|prefix| kind.starts_with(prefix))
    })
}

fn content_text(content: &api::AnthropicContent) -> Option<&str> {
    match &content.value {
        Some(ContentValue::Text(text)) => Some(text),
        _ => None,
    }
}

fn content_blocks(content: &api::AnthropicContent) -> &[api::AnthropicContentBlock] {
    match &content.value {
        Some(ContentValue::Blocks(list)) => &list.items,
        _ => &[],
    }
}

/// Python's `_extract_system_text`: the trimmed text of a string or of the
/// text blocks, or nothing.
fn system_text(content: &api::AnthropicContent) -> Option<String> {
    if let Some(text) = content_text(content) {
        let text = text.trim();
        return (!text.is_empty()).then(|| text.to_owned());
    }
    let texts: Vec<&str> = content_blocks(content)
        .iter()
        .filter_map(|block| match &block.block {
            Some(Block::Text(text)) => Some(text.text.trim()),
            _ => None,
        })
        .filter(|text| !text.is_empty())
        .collect();
    (!texts.is_empty()).then(|| texts.join("\n"))
}

/// Python's `_text_from_search_result`.
fn search_text(block: &api::AnthropicSearchResultBlock) -> String {
    let mut parts = Vec::new();
    if let Some(title) = block.title.as_deref().filter(|t| !t.is_empty()) {
        parts.push(format!("Title: {title}"));
    }
    let source = match block.source.as_ref().and_then(|s| s.value.as_ref()) {
        Some(SearchSource::Reference(text)) => Some(text.clone()),
        Some(SearchSource::Object(object)) => object.url.clone().or_else(|| object.text.clone()),
        None => None,
    };
    if let Some(source) = source.filter(|s| !s.is_empty()) {
        parts.push(format!("Source: {source}"));
    }
    let content: Vec<&str> = block
        .content
        .iter()
        .filter(|item| item.r#type.as_deref() == Some("text"))
        .filter_map(|item| item.text.as_deref())
        .filter(|text| !text.is_empty())
        .collect();
    if !content.is_empty() {
        parts.push(format!("Content: {}", content.join("\n")));
    }
    parts.join("\n")
}

fn unsupported(what: &str) -> AnthropicFailure {
    AnthropicFailure::invalid(format!("{what} is not supported by the Rust server"))
}

/// Python's `_convert_tool_result_content`: the tool messages' contents
/// (one per text / reference group) and the flattened text.
fn tool_result_contents(
    content: Option<&api::AnthropicContent>,
) -> Result<(Vec<serde_json::Value>, String), AnthropicFailure> {
    let Some(content) = content else {
        return Ok((
            vec![serde_json::Value::String(String::new())],
            String::new(),
        ));
    };
    if let Some(text) = content_text(content) {
        return Ok((
            vec![serde_json::Value::String(text.to_owned())],
            text.to_owned(),
        ));
    }
    let mut texts = Vec::new();
    for item in content_blocks(content) {
        match &item.block {
            Some(Block::Text(text)) if !text.text.is_empty() => texts.push(text.text.clone()),
            Some(Block::Text(_)) => {}
            Some(Block::Image(_)) => return Err(unsupported("image content")),
            Some(Block::ToolReference(_)) => return Err(unsupported("tool_reference content")),
            Some(Block::SearchResult(result)) => {
                let text = search_text(result);
                if !text.is_empty() {
                    texts.push(text);
                }
            }
            _ => {}
        }
    }
    let text = texts.join("\n");
    // Without images or references every part is text, so one tool message
    // carries the joined text (Python groups parts the same way).
    Ok((vec![serde_json::Value::String(text.clone())], text))
}

impl Anthropic<'_> {
    /// Python's `_convert_to_chat_completion_request`: the OpenAI chat
    /// request plus the chat-template kwargs the thinking switch sets.
    fn to_chat_request(
        &self,
        request: &api::AnthropicMessagesRequest,
    ) -> Result<(CreateChatCompletionRequest, Option<ChatTemplateKwargs>), AnthropicFailure> {
        if request.model.is_empty() {
            return Err(AnthropicFailure::invalid("Model is required"));
        }
        if request.max_tokens <= 0 {
            return Err(AnthropicFailure::invalid("max_tokens must be positive"));
        }
        let mut messages: Vec<serde_json::Value> = Vec::new();

        // The system prompt: `system` first, then inline system messages
        // (merged, as for chat encoders without inline-system support).
        let mut system_parts = Vec::new();
        if let Some(system) = &request.system
            && let Some(text) = system_text(system)
        {
            system_parts.push(text);
        }
        for message in &request.messages {
            if message.role == "system"
                && let Some(text) = message.content.as_ref().and_then(system_text)
            {
                system_parts.push(text);
            }
        }
        if !system_parts.is_empty() {
            messages
                .push(serde_json::json!({"role": "system", "content": system_parts.join("\n")}));
        }

        for message in &request.messages {
            match message.role.as_str() {
                "system" => continue,
                "user" | "assistant" => {}
                role => {
                    return Err(AnthropicFailure::invalid(format!(
                        "messages.role: unexpected value {role:?}"
                    )));
                }
            }
            let role = message.role.as_str();
            let Some(content) = &message.content else {
                return Err(AnthropicFailure::invalid("messages.content is required"));
            };
            if let Some(text) = content_text(content) {
                messages.push(serde_json::json!({"role": role, "content": text}));
                continue;
            }
            let blocks = content_blocks(content);
            let mut parts: Vec<serde_json::Value> = Vec::new();
            let mut tool_calls: Vec<serde_json::Value> = Vec::new();
            let mut reasoning_content = None;
            if role == "assistant" {
                reasoning_content = self.thinking_history(blocks)?;
            }
            for block in blocks {
                match &block.block {
                    Some(Block::Thinking(_)) | Some(Block::RedactedThinking(_)) => {}
                    Some(Block::Text(text)) => {
                        parts.push(serde_json::json!({"type": "text", "text": text.text}));
                    }
                    Some(Block::Image(image)) if image.source.is_some() => {
                        return Err(unsupported("image content"));
                    }
                    Some(Block::Image(_)) => {}
                    Some(Block::SearchResult(result)) => {
                        let text = search_text(result);
                        if !text.is_empty() {
                            parts.push(serde_json::json!({"type": "text", "text": text}));
                        }
                    }
                    Some(Block::ToolUse(tool_use)) => {
                        let id = if tool_use.id.is_empty() {
                            format!("call_{}", uuid::Uuid::new_v4().simple())
                        } else {
                            tool_use.id.clone()
                        };
                        tool_calls.push(serde_json::json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": tool_use.name,
                                "arguments": tool_use.input.clone().unwrap_or_else(|| "{}".into()),
                            },
                        }));
                    }
                    Some(Block::ToolResult(result)) => {
                        let (contents, text) = tool_result_contents(result.content.as_ref())?;
                        let tool_call_id = result
                            .tool_use_id
                            .clone()
                            .or_else(|| result.id.clone())
                            .unwrap_or_default();
                        if role == "user" {
                            // Flush what came before so the wire order stays
                            // user(pre), tool, user(post).
                            flush_user_parts(&mut messages, &mut parts);
                            for content in contents {
                                messages.push(serde_json::json!({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": content,
                                }));
                            }
                        } else {
                            parts.push(serde_json::json!({
                                "type": "text",
                                "text": format!("Tool result: {text}"),
                            }));
                        }
                    }
                    Some(Block::ToolReference(_)) => {
                        return Err(unsupported("tool_reference content"));
                    }
                    None => return Err(AnthropicFailure::invalid("messages.content: empty block")),
                }
            }
            let mut message = serde_json::json!({"role": role});
            if let Some(reasoning) = reasoning_content {
                message["reasoning_content"] = serde_json::Value::String(reasoning);
            }
            if !tool_calls.is_empty() {
                message["tool_calls"] = serde_json::Value::Array(tool_calls.clone());
            }
            if !parts.is_empty() {
                message["content"] = collapse_parts(&parts);
                messages.push(message);
            } else if !tool_calls.is_empty() {
                messages.push(message);
            } else if role == "user" {
                // A user turn made only of tool results: its tool messages
                // are already out.
            } else {
                // An empty assistant turn keeps the role alternation strict
                // templates check.
                message["content"] = serde_json::Value::String(String::new());
                messages.push(message);
            }
        }

        let mut body = serde_json::json!({
            "messages": messages,
            "model": request.model,
            "max_tokens": request.max_tokens,
            "stream": request.stream_or_default(),
        });
        if let Some(temperature) = request.temperature {
            body["temperature"] = temperature.into();
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = top_p.into();
        }
        if !request.stop_sequences.is_empty() {
            body["stop"] = serde_json::json!(request.stop_sequences);
        }

        // Tools: server tools are skipped (no implementation here); a custom
        // tool becomes an OpenAI function tool.
        let mut tools = Vec::new();
        for tool in &request.tools {
            if is_server_tool(tool) {
                tracing::info!(
                    name = %tool.name,
                    kind = ?tool.r#type,
                    "skipping built-in Anthropic server tool: no native support"
                );
                continue;
            }
            if tool.defer_loading == Some(true) {
                return Err(unsupported("tools[].defer_loading"));
            }
            let mut schema: serde_json::Value = match &tool.input_schema {
                Some(schema) => serde_json::from_str(schema).map_err(|error| {
                    AnthropicFailure::invalid(format!("tools[].input_schema: {error}"))
                })?,
                None => {
                    return Err(AnthropicFailure::invalid(
                        "tools[].input_schema is required",
                    ));
                }
            };
            let Some(object) = schema.as_object_mut() else {
                return Err(AnthropicFailure::invalid(
                    "input_schema must be a dictionary",
                ));
            };
            object
                .entry("type")
                .or_insert_with(|| serde_json::Value::String("object".into()));
            tools.push(serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description.clone().unwrap_or_default(),
                    "parameters": schema,
                },
            }));
        }
        let has_tools = !tools.is_empty();
        if has_tools {
            body["tools"] = serde_json::Value::Array(tools);
        }
        match &request.tool_choice {
            Some(choice) => match choice.r#type.as_str() {
                "none" => body["tool_choice"] = "none".into(),
                "auto" if has_tools => body["tool_choice"] = "auto".into(),
                "any" if has_tools => body["tool_choice"] = "required".into(),
                "tool" if has_tools => {
                    let name = choice.name.clone().unwrap_or_default();
                    let known = body["tools"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .any(|tool| tool["function"]["name"] == name);
                    if !known {
                        return Err(AnthropicFailure::invalid(format!(
                            "tool_choice references tool {name:?} but it is not in the forwarded tools list (server-side Anthropic tools cannot be selected)"
                        )));
                    }
                    body["tool_choice"] =
                        serde_json::json!({"type": "function", "function": {"name": name}});
                }
                "any" | "tool" => {
                    return Err(AnthropicFailure::invalid(format!(
                        "tool_choice={:?} requires at least one custom tool; all supplied tools were server-side Anthropic built-ins which the OpenAI-compatible backend cannot invoke",
                        choice.r#type
                    )));
                }
                "auto" => {}
                other => {
                    return Err(AnthropicFailure::invalid(format!(
                        "tool_choice.type: unexpected value {other:?}"
                    )));
                }
            },
            None if has_tools => body["tool_choice"] = "auto".into(),
            None => {}
        }

        // Thinking: `disabled` turns reasoning off, anything else turns it
        // on, through the template's `enable_thinking` switch. The budget
        // and `display` are accepted and not enforced, as in Python.
        let mut kwargs = None;
        if let Some(thinking) = &request.thinking {
            let enabled = match thinking.r#type.as_str() {
                "enabled" => {
                    match thinking.budget_tokens {
                        None => {
                            return Err(AnthropicFailure::invalid(
                                "thinking.budget_tokens is required when thinking.type is 'enabled'",
                            ));
                        }
                        Some(budget) if budget < 1024 => {
                            return Err(AnthropicFailure::invalid(format!(
                                "thinking.budget_tokens must be >= 1024 (got {budget})"
                            )));
                        }
                        Some(budget) => tracing::warn!(
                            budget,
                            "Anthropic thinking.budget_tokens is accepted but not enforced"
                        ),
                    }
                    true
                }
                "disabled" => {
                    if thinking.budget_tokens.is_some() || thinking.display.is_some() {
                        return Err(AnthropicFailure::invalid(
                            "thinking.budget_tokens and thinking.display are not allowed when thinking.type is 'disabled'",
                        ));
                    }
                    false
                }
                "adaptive" => {
                    if thinking.budget_tokens.is_some() {
                        return Err(AnthropicFailure::invalid(
                            "thinking.budget_tokens is not allowed when thinking.type is 'adaptive'",
                        ));
                    }
                    true
                }
                other => {
                    return Err(AnthropicFailure::invalid(format!(
                        "thinking.type: unexpected value {other:?}"
                    )));
                }
            };
            if thinking.display.as_deref() == Some("omitted") {
                tracing::warn!(
                    "Anthropic thinking.display='omitted' is accepted but reasoning text is still emitted"
                );
            }
            if self.server_args.reasoning_parser.is_none() {
                if enabled {
                    return Err(AnthropicFailure::invalid(
                        "Anthropic thinking is not supported for models without a reasoning parser",
                    ));
                }
            } else {
                kwargs = Some(ChatTemplateKwargs::from([
                    (
                        "enable_thinking".to_string(),
                        serde_json::Value::Bool(enabled),
                    ),
                    ("thinking".to_string(), serde_json::Value::Bool(enabled)),
                ]));
            }
        }
        if let Some(config) = &request.output_config {
            if let Some(effort) = &config.effort {
                tracing::info!(%effort, "Anthropic output_config.effort is accepted and not applied");
            }
            if let Some(budget) = &config.task_budget {
                tracing::info!(total = budget.total, kind = %budget.r#type, "Anthropic output_config.task_budget hint");
            }
        }
        if !request.betas.is_empty() {
            tracing::info!(betas = ?request.betas, "Anthropic request opted into betas; no-op locally");
        }

        let chat_request: CreateChatCompletionRequest = serde_json::from_value(body)
            .map_err(|error| AnthropicFailure::invalid(format!("invalid request: {error}")))?;
        Ok((chat_request, kwargs))
    }

    /// Python's `_convert_assistant_thinking_blocks`: prior-turn thinking
    /// becomes `reasoning_content` when a reasoning parser frames it, and is
    /// dropped with a warning otherwise; redacted thinking cannot be replayed.
    fn thinking_history(
        &self,
        blocks: &[api::AnthropicContentBlock],
    ) -> Result<Option<String>, AnthropicFailure> {
        if blocks
            .iter()
            .any(|block| matches!(block.block, Some(Block::RedactedThinking(_))))
        {
            return Err(AnthropicFailure::invalid(
                "Anthropic redacted_thinking history is not supported",
            ));
        }
        let parts: Vec<&str> = blocks
            .iter()
            .filter_map(|block| match &block.block {
                Some(Block::Thinking(thinking)) if !thinking.thinking.is_empty() => {
                    Some(thinking.thinking.as_str())
                }
                _ => None,
            })
            .collect();
        if parts.is_empty() {
            return Ok(None);
        }
        if self.server_args.reasoning_parser.is_none() {
            tracing::warn!(
                blocks = parts.len(),
                "dropping prior-turn thinking history: no reasoning parser is configured"
            );
            return Ok(None);
        }
        Ok(Some(parts.join("\n")))
    }

    /// Render, resolve sampling, and submit one chat generation, as the
    /// OpenAI chat route does for `n = 1`.
    async fn submit(
        &self,
        request: api::AnthropicMessagesRequest,
        stream: bool,
    ) -> Result<Submitted, AnthropicFailure> {
        let Some(formatter) = self.chat_formatter else {
            return Err(AnthropicFailure::invalid(
                "this model has no usable chat template",
            ));
        };
        let (mut chat_request, kwargs) = self.to_chat_request(&request)?;
        if chat_request.messages.is_empty() {
            return Err(AnthropicFailure::invalid("messages cannot be empty"));
        }
        let tool_choice = dynamo_tool_choice(&chat_request.tool_choice);
        let tools = sglang_processor::chat_tool_definitions(
            chat_request.tools.as_deref(),
            &chat_request.messages,
        )
        .map_err(|error| AnthropicFailure::invalid(error.to_string()))?;
        let tools_enabled = !tools.is_empty() && tool_choice != DynamoToolChoice::None;
        let parser = tools_enabled
            .then(|| self.server_args.tool_call_parser.clone())
            .flatten();
        if tools_enabled && parser.is_none() {
            return Err(AnthropicFailure::invalid(
                "tool calls require --tool-call-parser",
            ));
        }
        let reasoning_parser = self.server_args.reasoning_parser.clone();
        let tools = (!tools.is_empty()).then_some(tools);
        let prompt = prepare_chat_prompt(formatter, &mut chat_request, kwargs.as_ref())
            .map_err(AnthropicFailure::invalid)?;
        let mut sampling = chat_sampling(
            &chat_request,
            SamplingDefaults::CHAT,
            parser.as_deref(),
            &tool_choice,
            tools.as_deref().unwrap_or_default(),
            chat_request.parallel_tool_calls,
            self.server_args,
        )
        .map_err(AnthropicFailure::invalid)?;
        if let Some(top_k) = request.top_k {
            sampling.top_k = top_k;
        }
        let routing = PDRoutingFields {
            bootstrap_host: request.bootstrap_host,
            bootstrap_port: request.bootstrap_port,
            bootstrap_room: request.bootstrap_room,
            routed_dp_rank: request.routed_dp_rank,
            disagg_prefill_dp_rank: request.disagg_prefill_dp_rank,
        }
        .normalize(1, 1)
        .map_err(|error| AnthropicFailure::invalid(error.to_string()))?;
        let uses_tool_call_structural_tag = sampling.structural_tag.is_some();
        let starts_in_reasoning = starts_in_reasoning(reasoning_parser.as_deref(), &prompt);
        let native = GenerateRequest {
            rid: Rid::new(),
            text: Some(prompt),
            // The rendered template owns its special tokens.
            skip_special_tokens: true,
            sampling_params: sampling,
            stream,
            bootstrap_host: routing.bootstrap.bootstrap_hosts[0].clone(),
            bootstrap_port: routing.bootstrap.bootstrap_ports[0],
            bootstrap_room: routing.bootstrap.bootstrap_rooms[0],
            routed_dp_rank: routing.routed_dp_rank,
            disagg_prefill_dp_rank: routing.disagg_prefill_dp_rank,
            ..Default::default()
        };
        let call = self
            .core
            .generate(native)
            .await
            .map_err(AnthropicFailure::from_core)?;
        Ok(Submitted {
            call,
            model: chat_request.model,
            parser,
            reasoning_parser,
            starts_in_reasoning,
            tools,
            tool_choice: chat_request.tool_choice,
            uses_tool_call_structural_tag,
        })
    }

    /// `POST /v1/messages` without `stream`.
    pub(crate) async fn messages(
        &self,
        request: api::AnthropicMessagesRequest,
    ) -> Result<api::AnthropicMessagesResponse, AnthropicFailure> {
        let submitted = self.submit(request, false).await?;
        let response = unary_chat_response(
            vec![(0, submitted.call)],
            message_id(),
            submitted.model,
            unix_seconds_u32(),
            false,
            submitted.parser,
            submitted.reasoning_parser,
            submitted.starts_in_reasoning,
            submitted.tools,
            true,
            None,
        )
        .await
        .map_err(|(status, message)| AnthropicFailure::from_status(status, message))?;
        Ok(convert_response(response))
    }

    /// `POST /v1/messages` with `stream`: the Anthropic event sequence over
    /// the OpenAI chat stream, exactly as Python re-shapes it. Failures
    /// after the first event ride the stream as an `error` event.
    pub(crate) async fn messages_stream(
        &self,
        request: api::AnthropicMessagesRequest,
    ) -> Result<Events, AnthropicFailure> {
        let model = request.model.clone();
        let submitted = self.submit(request, true).await?;
        let chunks = chat_event_stream(
            vec![(0, submitted.call)],
            message_id(),
            submitted.model,
            unix_seconds_u32(),
            false,
            true,
            submitted.parser,
            submitted.reasoning_parser,
            submitted.starts_in_reasoning,
            submitted.tools,
            submitted.tool_choice,
            submitted.uses_tool_call_structural_tag,
            true,
            None,
        );
        Ok(Box::pin(anthropic_events(chunks, model)))
    }

    /// `POST /v1/messages/count_tokens`: the rendered prompt's token count.
    pub(crate) async fn count_tokens(
        &self,
        request: api::AnthropicCountTokensRequest,
    ) -> Result<api::AnthropicCountTokensResponse, AnthropicFailure> {
        let Some(formatter) = self.chat_formatter else {
            return Err(AnthropicFailure::invalid(
                "this model has no usable chat template",
            ));
        };
        let messages_request = api::AnthropicMessagesRequest {
            model: request.model,
            messages: request.messages,
            max_tokens: 1,
            system: request.system,
            thinking: request.thinking,
            tool_choice: request.tool_choice,
            tools: request.tools,
            ..Default::default()
        };
        let (mut chat_request, kwargs) = self.to_chat_request(&messages_request)?;
        let prompt = prepare_chat_prompt(formatter, &mut chat_request, kwargs.as_ref())
            .map_err(AnthropicFailure::invalid)?;
        let ids = self
            .core
            .tokenize(prompt, false)
            .await
            .map_err(AnthropicFailure::from_core)?;
        Ok(api::AnthropicCountTokensResponse {
            input_tokens: i64::try_from(ids.len()).unwrap_or(i64::MAX),
        })
    }
}

fn flush_user_parts(messages: &mut Vec<serde_json::Value>, parts: &mut Vec<serde_json::Value>) {
    if parts.is_empty() {
        return;
    }
    messages.push(serde_json::json!({"role": "user", "content": collapse_parts(parts)}));
    parts.clear();
}

/// A lone text part is sent as a string, as Python does.
fn collapse_parts(parts: &[serde_json::Value]) -> serde_json::Value {
    match parts {
        [part] if part["type"] == "text" => part["text"].clone(),
        _ => serde_json::Value::Array(parts.to_vec()),
    }
}

/// Python's `STOP_REASON_MAP`; anything else reports `end_turn`.
fn stop_reason(finish: Option<&OpenAIFinishReason>) -> &'static str {
    match finish {
        Some(OpenAIFinishReason::Length) => "max_tokens",
        Some(OpenAIFinishReason::ToolCalls) => "tool_use",
        Some(OpenAIFinishReason::Stop) | None => "end_turn",
        Some(other) => {
            tracing::warn!(
                ?other,
                "unmapped OpenAI finish_reason; defaulting to end_turn"
            );
            "end_turn"
        }
    }
}

/// Python's `_anthropic_usage_from_openai`: cached prompt tokens leave
/// `input_tokens` and go to `cache_read_input_tokens`.
fn usage_from_openai(
    usage: Option<&CompletionUsage>,
    include_input: bool,
    include_output: bool,
    force_zero_output: bool,
) -> api::AnthropicUsage {
    let Some(usage) = usage else {
        return api::AnthropicUsage {
            input_tokens: include_input.then_some(0),
            output_tokens: include_output.then_some(0),
            ..Default::default()
        };
    };
    let cached = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        .unwrap_or(0);
    if cached > usage.prompt_tokens {
        tracing::warn!(
            cached,
            prompt = usage.prompt_tokens,
            "cached tokens exceed prompt tokens"
        );
    }
    api::AnthropicUsage {
        input_tokens: include_input.then(|| i64::from(usage.prompt_tokens.saturating_sub(cached))),
        output_tokens: include_output.then(|| {
            if force_zero_output {
                0
            } else {
                i64::from(usage.completion_tokens)
            }
        }),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: (include_input && cached > 0).then(|| i64::from(cached)),
    }
}

fn text_block(text: impl Into<String>) -> api::AnthropicContentBlock {
    api::AnthropicContentBlock {
        block: Some(Block::Text(api::AnthropicTextBlock { text: text.into() })),
    }
}

fn thinking_block(thinking: impl Into<String>) -> api::AnthropicContentBlock {
    api::AnthropicContentBlock {
        block: Some(Block::Thinking(api::AnthropicThinkingBlock {
            thinking: thinking.into(),
            signature: None,
        })),
    }
}

fn tool_use_block(id: String, name: String, input: String) -> api::AnthropicContentBlock {
    api::AnthropicContentBlock {
        block: Some(Block::ToolUse(api::AnthropicToolUseBlock {
            id,
            name,
            input: Some(input),
        })),
    }
}

/// Tool arguments as an object; invalid JSON becomes `{}` with a warning,
/// as Python does, so a broken call is never mistaken for a real one.
fn tool_input(name: &str, arguments: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(value) if value.is_object() => value.to_string(),
        _ => {
            tracing::warn!(tool = %name, "tool emitted invalid JSON arguments; defaulting to empty input");
            "{}".into()
        }
    }
}

/// Python's `_convert_response`.
fn convert_response(response: CreateChatCompletionResponse) -> api::AnthropicMessagesResponse {
    let Some(choice) = response.choices.into_iter().next() else {
        return api::AnthropicMessagesResponse {
            id: message_id(),
            r#type: "message".into(),
            role: "assistant".into(),
            content: Some(api::AnthropicContentBlockList {
                items: vec![text_block("")],
            }),
            model: response.model,
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: Some(api::AnthropicUsage {
                input_tokens: Some(0),
                output_tokens: Some(0),
                ..Default::default()
            }),
        };
    };
    let mut content = Vec::new();
    if let Some(reasoning) = choice.message.reasoning_content.filter(|r| !r.is_empty()) {
        content.push(thinking_block(reasoning));
    }
    match choice.message.content {
        Some(ChatCompletionMessageContent::Text(text)) if !text.is_empty() => {
            content.push(text_block(text));
        }
        Some(ChatCompletionMessageContent::Parts(parts)) => {
            let text = serde_json::to_string(&parts).unwrap_or_default();
            content.push(text_block(text));
        }
        _ => {}
    }
    for call in choice.message.tool_calls.into_iter().flatten() {
        let input = tool_input(&call.function.name, &call.function.arguments);
        content.push(tool_use_block(call.id, call.function.name, input));
    }
    if content.is_empty() {
        content.push(text_block(""));
    }
    api::AnthropicMessagesResponse {
        id: message_id(),
        r#type: "message".into(),
        role: "assistant".into(),
        content: Some(api::AnthropicContentBlockList { items: content }),
        model: response.model,
        stop_reason: Some(stop_reason(choice.finish_reason.as_ref()).into()),
        stop_sequence: None,
        usage: Some(usage_from_openai(
            response.usage.as_ref(),
            true,
            true,
            false,
        )),
    }
}

/// The SSE `event:` name of an event, which is its JSON `type`.
pub(crate) fn event_name(event: &api::AnthropicStreamEvent) -> &'static str {
    match &event.event {
        Some(Event::MessageStart(_)) => "message_start",
        Some(Event::MessageDelta(_)) => "message_delta",
        Some(Event::MessageStop(_)) => "message_stop",
        Some(Event::ContentBlockStart(_)) => "content_block_start",
        Some(Event::ContentBlockDelta(_)) => "content_block_delta",
        Some(Event::ContentBlockStop(_)) => "content_block_stop",
        Some(Event::Ping(_)) => "ping",
        Some(Event::Error(_)) | None => "error",
    }
}

fn event(event: Event) -> api::AnthropicStreamEvent {
    api::AnthropicStreamEvent { event: Some(event) }
}

fn delta_event(index: i64, delta: Delta) -> api::AnthropicStreamEvent {
    event(Event::ContentBlockDelta(
        api::AnthropicContentBlockDeltaEvent {
            index,
            delta: Some(api::AnthropicContentDelta { delta: Some(delta) }),
        },
    ))
}

/// The state of one streamed message: Python's closure variables.
struct StreamState {
    id: String,
    model: String,
    index: i64,
    open: Option<&'static str>,
    started: bool,
    had_content: bool,
    finish_reason: Option<OpenAIFinishReason>,
    final_usage: Option<api::AnthropicUsage>,
}

impl StreamState {
    fn message_start(&self, usage: Option<&CompletionUsage>) -> api::AnthropicStreamEvent {
        event(Event::MessageStart(api::AnthropicMessageStartEvent {
            message: Some(api::AnthropicMessagesResponse {
                id: self.id.clone(),
                r#type: "message".into(),
                role: "assistant".into(),
                content: Some(api::AnthropicContentBlockList { items: Vec::new() }),
                model: self.model.clone(),
                stop_reason: None,
                stop_sequence: None,
                usage: Some(usage_from_openai(usage, true, true, true)),
            }),
        }))
    }

    fn ensure_started(
        &mut self,
        usage: Option<&CompletionUsage>,
    ) -> Vec<api::AnthropicStreamEvent> {
        if self.started {
            return Vec::new();
        }
        self.started = true;
        vec![self.message_start(usage)]
    }

    fn close_block(&mut self) -> Vec<api::AnthropicStreamEvent> {
        if self.open.take().is_none() {
            return Vec::new();
        }
        let stop = event(Event::ContentBlockStop(
            api::AnthropicContentBlockStopEvent { index: self.index },
        ));
        self.index += 1;
        vec![stop]
    }

    /// Open a block of `kind`, closing the current one when it differs or
    /// when `force_new` (each tool call owns its own block).
    fn ensure_block(
        &mut self,
        kind: &'static str,
        block: api::AnthropicContentBlock,
        force_new: bool,
    ) -> Vec<api::AnthropicStreamEvent> {
        let mut events = Vec::new();
        if self.open.is_some() && (force_new || self.open != Some(kind)) {
            events.extend(self.close_block());
        }
        if self.open.is_none() {
            events.push(event(Event::ContentBlockStart(
                api::AnthropicContentBlockStartEvent {
                    index: self.index,
                    content_block: Some(block),
                },
            )));
            self.open = Some(kind);
        }
        events
    }

    /// A terminal sequence that leaves the wire valid: message_start if
    /// missing, the open block closed, then error and message_stop.
    fn flush_on_error(&mut self, kind: &str, message: &str) -> Vec<api::AnthropicStreamEvent> {
        let mut events = self.ensure_started(None);
        events.extend(self.close_block());
        events.push(event(Event::Error(api::AnthropicErrorEvent {
            error: Some(api::AnthropicError {
                r#type: kind.into(),
                message: message.into(),
            }),
        })));
        events.push(event(Event::MessageStop(api::AnthropicMessageStopEvent {})));
        events
    }

    /// The `[DONE]` handling: close, then message_delta and message_stop.
    fn finish(&mut self) -> Vec<api::AnthropicStreamEvent> {
        let mut events = self.ensure_started(None);
        if !self.had_content && self.finish_reason.is_none() {
            tracing::warn!("stream produced no content and no finish_reason; emitting api_error");
            events.push(event(Event::Error(api::AnthropicErrorEvent {
                error: Some(api::AnthropicError {
                    r#type: "api_error".into(),
                    message: "Backend produced no content".into(),
                }),
            })));
            events.push(event(Event::MessageStop(api::AnthropicMessageStopEvent {})));
            return events;
        }
        events.extend(self.close_block());
        events.push(event(Event::MessageDelta(
            api::AnthropicMessageDeltaEvent {
                delta: Some(api::AnthropicMessageEndDelta {
                    stop_reason: Some(stop_reason(self.finish_reason.as_ref()).into()),
                    stop_sequence: None,
                }),
                usage: Some(self.final_usage.take().unwrap_or(api::AnthropicUsage {
                    output_tokens: Some(0),
                    ..Default::default()
                })),
            },
        )));
        events.push(event(Event::MessageStop(api::AnthropicMessageStopEvent {})));
        events
    }

    /// One OpenAI chunk's events.
    fn chunk(
        &mut self,
        chunk: CreateChatCompletionStreamResponse,
    ) -> Vec<api::AnthropicStreamEvent> {
        let mut events = Vec::new();
        if let Some(usage) = chunk.usage.as_ref() {
            // The chat stream reports usage once, at the end, so the input
            // count is only known here; the delta carries both counts.
            self.final_usage = Some(usage_from_openai(Some(usage), true, true, false));
        }
        let Some(choice) = chunk.choices.into_iter().next() else {
            return events;
        };
        if choice.finish_reason.is_some() {
            self.finish_reason = choice.finish_reason;
        }
        let delta = choice.delta;
        let text = match delta.content {
            Some(ChatCompletionMessageContent::Text(text)) if !text.is_empty() => Some(text),
            Some(ChatCompletionMessageContent::Parts(parts)) => {
                Some(serde_json::to_string(&parts).unwrap_or_default())
            }
            _ => None,
        };
        let reasoning = delta.reasoning_content.filter(|r| !r.is_empty());
        let has_payload = reasoning.is_some()
            || delta
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())
            || text.is_some()
            || chunk.usage.is_some();
        if (has_payload || choice.finish_reason.is_some()) && !self.started {
            events.push(self.message_start(chunk.usage.as_ref()));
            self.started = true;
        }
        if !has_payload {
            return events;
        }
        if let Some(reasoning) = reasoning {
            events.extend(self.ensure_block("thinking", thinking_block(""), false));
            events.push(delta_event(
                self.index,
                Delta::ThinkingDelta(api::AnthropicThinkingDelta {
                    thinking: reasoning,
                }),
            ));
            self.had_content = true;
        }
        for call in delta.tool_calls.into_iter().flatten() {
            let Some(function) = call.function else {
                continue;
            };
            if let Some(name) = function.name.filter(|name| !name.is_empty()) {
                let id = call
                    .id
                    .filter(|id| !id.is_empty())
                    .unwrap_or_else(|| format!("toolu_{}", uuid::Uuid::new_v4().simple()));
                events.extend(self.ensure_block(
                    "tool_use",
                    tool_use_block(id, name, "{}".into()),
                    true,
                ));
                self.had_content = true;
                if let Some(arguments) = function.arguments.filter(|a| !a.is_empty()) {
                    events.push(delta_event(
                        self.index,
                        Delta::InputJsonDelta(api::AnthropicInputJsonDelta {
                            partial_json: arguments,
                        }),
                    ));
                }
            } else if let Some(arguments) = function.arguments.filter(|a| !a.is_empty()) {
                if self.open != Some("tool_use") {
                    tracing::warn!("dropping tool_call argument delta with no open tool_use block");
                    continue;
                }
                events.push(delta_event(
                    self.index,
                    Delta::InputJsonDelta(api::AnthropicInputJsonDelta {
                        partial_json: arguments,
                    }),
                ));
                self.had_content = true;
            }
        }
        if let Some(text) = text {
            events.extend(self.ensure_block("text", text_block(""), false));
            events.push(delta_event(
                self.index,
                Delta::TextDelta(api::AnthropicTextDelta { text }),
            ));
            self.had_content = true;
        }
        events
    }
}

/// The OpenAI chat stream's error frame, `{"error": {"message", "code"}}`.
fn upstream_error(payload: &serde_json::Value) -> Option<(&'static str, String)> {
    let error = payload.get("error")?.as_object()?;
    let message = error
        .get("message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("Upstream error")
        .to_owned();
    let status = error
        .get("code")
        .and_then(serde_json::Value::as_u64)
        .and_then(|code| u16::try_from(code).ok())
        .and_then(|code| StatusCode::from_u16(code).ok());
    let kind = status.map(error_type).unwrap_or("api_error");
    let message = if status.is_some_and(|s| s.is_server_error()) {
        "Internal server error".to_owned()
    } else {
        message
    };
    Some((kind, message))
}

/// Python's `_generate_anthropic_stream` over the chat stream's frames.
fn anthropic_events(
    chunks: impl Stream<Item = String> + Send + 'static,
    model: String,
) -> impl Stream<Item = api::AnthropicStreamEvent> + Send + 'static {
    async_stream::stream! {
        let mut state = StreamState {
            id: message_id(),
            model,
            index: 0,
            open: None,
            started: false,
            had_content: false,
            finish_reason: None,
            final_usage: None,
        };
        futures::pin_mut!(chunks);
        while let Some(frame) = chunks.next().await {
            if frame == "[DONE]" {
                for event in state.finish() {
                    yield event;
                }
                continue;
            }
            match serde_json::from_str::<CreateChatCompletionStreamResponse>(&frame) {
                Ok(chunk) => {
                    for event in state.chunk(chunk) {
                        yield event;
                    }
                }
                Err(_) => {
                    let payload: serde_json::Value = serde_json::from_str(&frame).unwrap_or_default();
                    let (kind, message) = upstream_error(&payload)
                        .unwrap_or(("api_error", "Stream processing error".to_owned()));
                    for event in state.flush_on_error(kind, &message) {
                        yield event;
                    }
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::http::openai::test_utils::core_handle;

    fn request(body: serde_json::Value) -> api::AnthropicMessagesRequest {
        serde_json::from_value(body).unwrap()
    }

    fn adapter(server_args: &ServerArgs) -> Anthropic<'_> {
        Anthropic {
            core: Box::leak(Box::new(core_handle())),
            chat_formatter: None,
            server_args,
        }
    }

    /// Python's conversion rules: `system` and inline system turns merge
    /// into one system message; a user turn's tool results become tool
    /// messages in wire order; a tool_use becomes an assistant tool call;
    /// custom tools become function tools and server tools are skipped.
    #[test]
    fn converts_messages_tools_and_tool_choice_like_python() {
        let server_args = ServerArgs::default();
        let anthropic = adapter(&server_args);
        let (chat, kwargs) = anthropic
            .to_chat_request(&request(serde_json::json!({
                "model": "m",
                "max_tokens": 8,
                "system": [{"type": "text", "text": "be brief"}],
                "messages": [
                    {"role": "system", "content": "inline"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "calling"},
                        {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"x": 1}}
                    ]},
                    {"role": "user", "content": [
                        {"type": "text", "text": "before"},
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"},
                        {"type": "text", "text": "after"}
                    ]}
                ],
                "tools": [
                    {"name": "f", "input_schema": {"properties": {}}},
                    {"type": "web_search_20250305", "name": "web_search"}
                ],
                "tool_choice": {"type": "tool", "name": "f"},
                "stop_sequences": ["END"],
                "temperature": 0.2
            })))
            .unwrap();
        assert!(kwargs.is_none());
        let json = serde_json::to_value(&chat).unwrap();
        let messages = json["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "be brief\ninline");
        assert_eq!(messages[1]["content"], "hi");
        assert_eq!(messages[2]["content"], "calling");
        assert_eq!(messages[2]["tool_calls"][0]["function"]["name"], "f");
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["arguments"],
            "{\"x\":1}"
        );
        assert_eq!(messages[3]["content"], "before");
        assert_eq!(messages[4]["role"], "tool");
        assert_eq!(messages[4]["tool_call_id"], "toolu_1");
        assert_eq!(messages[4]["content"], "42");
        assert_eq!(messages[5]["content"], "after");
        assert_eq!(
            json["tools"].as_array().unwrap().len(),
            1,
            "server tool skipped"
        );
        assert_eq!(json["tools"][0]["function"]["parameters"]["type"], "object");
        assert_eq!(json["tool_choice"]["function"]["name"], "f");
        assert_eq!(json["stop"], serde_json::json!(["END"]));
        assert_eq!(json["max_tokens"], 8);
    }

    /// The hard requirements Python rejects: a named tool that was skipped,
    /// `any` with only server tools, thinking without a reasoning parser,
    /// and redacted thinking history.
    #[test]
    fn rejects_what_python_rejects() {
        let server_args = ServerArgs::default();
        let anthropic = adapter(&server_args);
        let base = serde_json::json!({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]});
        let with = |extra: serde_json::Value| {
            let mut body = base.clone();
            for (key, value) in extra.as_object().unwrap() {
                body[key] = value.clone();
            }
            request(body)
        };
        let error = anthropic
            .to_chat_request(&with(serde_json::json!({
                "tools": [{"type": "bash_20250124", "name": "bash"}],
                "tool_choice": {"type": "any"}
            })))
            .unwrap_err();
        assert!(
            error.message.contains("requires at least one custom tool"),
            "{}",
            error.message
        );
        let error = anthropic
            .to_chat_request(&with(serde_json::json!({
                "thinking": {"type": "enabled", "budget_tokens": 2048}
            })))
            .unwrap_err();
        assert!(
            error.message.contains("reasoning parser"),
            "{}",
            error.message
        );
        let error = anthropic
            .to_chat_request(&with(serde_json::json!({
                "messages": [{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]}]
            })))
            .unwrap_err();
        assert!(
            error.message.contains("redacted_thinking"),
            "{}",
            error.message
        );
        assert_eq!(
            anthropic
                .to_chat_request(&with(serde_json::json!({"max_tokens": 0})))
                .unwrap_err()
                .message,
            "max_tokens must be positive"
        );
    }

    /// With a reasoning parser, `thinking` toggles the template's
    /// `enable_thinking` and prior thinking becomes `reasoning_content`.
    #[test]
    fn thinking_maps_onto_the_reasoning_parser() {
        let server_args = ServerArgs {
            reasoning_parser: Some("qwen3".into()),
            ..Default::default()
        };
        let anthropic = adapter(&server_args);
        let (chat, kwargs) = anthropic
            .to_chat_request(&request(serde_json::json!({
                "model": "m",
                "max_tokens": 8,
                "thinking": {"type": "disabled"},
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": "ok"}
                    ]},
                    {"role": "user", "content": "more"}
                ]
            })))
            .unwrap();
        assert_eq!(kwargs.unwrap()["enable_thinking"], false);
        let json = serde_json::to_value(&chat).unwrap();
        assert_eq!(json["messages"][1]["reasoning_content"], "hmm");
        assert_eq!(json["messages"][1]["content"], "ok");
    }

    /// The stream state machine over OpenAI chunks emits Python's event
    /// sequence: deferred message_start, one block per content kind, each
    /// tool call in its own block, then message_delta and message_stop.
    #[tokio::test]
    async fn stream_events_follow_the_anthropic_sequence() {
        let chunk =
            |delta: serde_json::Value, finish: Option<&str>, usage: Option<serde_json::Value>| {
                let mut value = serde_json::json!({
                    "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
                });
                if let Some(usage) = usage {
                    value["choices"] = serde_json::json!([]);
                    value["usage"] = usage;
                }
                value.to_string()
            };
        let frames = vec![
            chunk(serde_json::json!({"role": "assistant"}), None, None),
            chunk(
                serde_json::json!({"reasoning_content": "think"}),
                None,
                None,
            ),
            chunk(serde_json::json!({"content": "hel"}), None, None),
            chunk(serde_json::json!({"content": "lo"}), None, None),
            chunk(
                serde_json::json!({"tool_calls": [{"index": 0, "id": "toolu_1", "type": "function", "function": {"name": "f", "arguments": "{\"x\":"}}]}),
                None,
                None,
            ),
            chunk(
                serde_json::json!({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}),
                Some("tool_calls"),
                None,
            ),
            chunk(
                serde_json::json!({}),
                None,
                Some(
                    serde_json::json!({"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}),
                ),
            ),
            "[DONE]".to_string(),
        ];
        let events: Vec<api::AnthropicStreamEvent> =
            anthropic_events(futures::stream::iter(frames), "m".into())
                .collect()
                .await;
        let names: Vec<&str> = events.iter().map(event_name).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let json: Vec<serde_json::Value> = events
            .iter()
            .map(|event| serde_json::to_value(event).unwrap())
            .collect();
        assert_eq!(json[0]["message"]["content"], serde_json::json!([]));
        assert_eq!(json[0]["message"]["usage"]["output_tokens"], 0);
        assert_eq!(json[1]["content_block"]["type"], "thinking");
        assert_eq!(
            json[2]["delta"],
            serde_json::json!({"type": "thinking_delta", "thinking": "think"})
        );
        assert_eq!(
            json[5]["delta"],
            serde_json::json!({"type": "text_delta", "text": "hel"})
        );
        assert_eq!(
            json[8]["content_block"],
            serde_json::json!({"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}})
        );
        assert_eq!(json[8]["index"], 2);
        assert_eq!(json[9]["delta"]["partial_json"], "{\"x\":");
        assert_eq!(json[12]["delta"]["stop_reason"], "tool_use");
        assert_eq!(json[12]["usage"]["output_tokens"], 3);
        assert_eq!(json[12]["usage"]["input_tokens"], 7);
    }

    /// An OpenAI error frame mid-stream closes the open block and ends with
    /// error + message_stop, never a half-open stream.
    #[tokio::test]
    async fn stream_error_frames_close_the_wire_cleanly() {
        let frames = vec![
            serde_json::json!({
                "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": null}]
            })
            .to_string(),
            serde_json::json!({"error": {"message": "boom", "type": "InternalServerError", "code": 500}})
                .to_string(),
        ];
        let events: Vec<api::AnthropicStreamEvent> =
            anthropic_events(futures::stream::iter(frames), "m".into())
                .collect()
                .await;
        let names: Vec<&str> = events.iter().map(event_name).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "error",
                "message_stop"
            ]
        );
        let error = serde_json::to_value(&events[4]).unwrap();
        assert_eq!(error["error"]["type"], "api_error");
        assert_eq!(error["error"]["message"], "Internal server error");
    }
}
