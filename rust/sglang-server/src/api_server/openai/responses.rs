//! OpenAI Responses endpoint, response storage, and lifecycle events.

use std::{collections::HashMap, convert::Infallible, sync::Arc};

use axum::{
    Extension, Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use dynamo_parsers::{ToolChoice as DynamoToolChoice, ToolDefinition};
use dynamo_protocols::types::responses::{
    CreateResponse, EasyInputContent, FunctionCallOutput, FunctionToolCall, InputContent,
    InputItem, InputOutputMessageContent, InputParam, InputRole, Item, MessageItem, OutputItem,
    OutputMessageContent, OutputStatus, ReasoningItem, ReasoningItemContent, ReasoningTextContent,
    Response as OpenAIResponse, Status, TextResponseFormatConfiguration, Tool as ResponseTool,
    ToolChoiceOptions, ToolChoiceParam, Truncation, UpstreamInputContent,
};
use dynamo_protocols::types::{
    ChatCompletionMessageToolCall, ChatCompletionNamedToolChoice,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionTool, ChatCompletionToolChoiceOption,
    ChatCompletionToolType, CreateChatCompletionRequest, FunctionCall, FunctionName,
    FunctionObject, FunctionType, ReasoningContent, ReasoningEffort as ChatReasoningEffort,
    ResponseFormat,
};
use futures::StreamExt;
use tokio::sync::{RwLock, mpsc};

use super::super::guard::AbortGuard;
use super::chat::{SamplingDefaults, chat_sampling, prepare_chat_request};
use super::reasoning::split_reasoning_unary;
use super::response_stream::{
    chunk_response_logprobs, response_object, response_status, responses_event_stream,
    responses_usage, text_output_content, text_response_message,
};
use super::tools::{dynamo_tool_choice, parse_chat_tool_calls};
use super::{AppState, collect_output, openai_error, submit_generation, unix_seconds};
use crate::ids::Rid;
use crate::message::{ChunkEvent, EgressItem, GenerateRequest};
use crate::tokenizer_manager::AbortSource;

#[derive(Clone)]
pub(super) struct StoredResponse {
    pub(super) response: OpenAIResponse,
    pub(super) messages: Vec<ChatCompletionRequestMessage>,
    pub(super) rid: Option<Rid>,
}

pub(super) type ResponseStore = Arc<RwLock<HashMap<String, StoredResponse>>>;

pub(super) fn new_response_store() -> ResponseStore {
    Arc::new(RwLock::new(HashMap::new()))
}

pub(super) fn routes() -> Router<AppState> {
    routes_with_store(new_response_store())
}

pub(super) fn routes_with_store(store: ResponseStore) -> Router<AppState> {
    Router::new()
        .route("/v1/responses", post(responses))
        .route("/v1/responses/{response_id}", get(retrieve_response))
        .route("/v1/responses/{response_id}/cancel", post(cancel_response))
        .layer(Extension(store))
}

fn invalid_response_id(response_id: &str) -> Option<String> {
    (!response_id.starts_with("resp_")).then(|| {
        format!("Invalid 'response_id': '{response_id}'. Expected an ID that begins with 'resp'.")
    })
}

/// Build the axum SSE event for one Responses stream frame. Python
/// `_send_event` uses the payload's `type` field as the `event:` name; frames
/// without one (`[DONE]`, error payloads) stay data-only.
pub(super) fn sse_frame(data: String) -> Event {
    match serde_json::from_str::<serde_json::Value>(&data)
        .ok()
        .and_then(|value| value["type"].as_str().map(str::to_owned))
    {
        Some(name) => Event::default().event(name).data(data),
        None => Event::default().data(data),
    }
}

async fn retrieve_response(
    Extension(store): Extension<ResponseStore>,
    Path(response_id): Path<String>,
) -> Response {
    if let Some(message) = invalid_response_id(&response_id) {
        return openai_error(StatusCode::BAD_REQUEST, message);
    }
    let response = store
        .read()
        .await
        .get(&response_id)
        .map(|stored| stored.response.clone());
    match response {
        Some(response) => Json(unary_response_value(response)).into_response(),
        None => openai_error(
            StatusCode::NOT_FOUND,
            format!("Response with id '{response_id}' not found."),
        ),
    }
}

async fn cancel_response(
    State(state): State<AppState>,
    Extension(store): Extension<ResponseStore>,
    Path(response_id): Path<String>,
) -> Response {
    if let Some(message) = invalid_response_id(&response_id) {
        return openai_error(StatusCode::BAD_REQUEST, message);
    }
    let mut store = store.write().await;
    let Some(stored) = store.get_mut(&response_id) else {
        return openai_error(
            StatusCode::NOT_FOUND,
            format!("Response with id '{response_id}' not found."),
        );
    };
    if !matches!(stored.response.status, Status::Queued | Status::InProgress) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "Response is no longer cancellable.",
        );
    }
    stored.response.status = Status::Cancelled;
    let rid = stored.rid.take();
    let response = stored.response.clone();
    drop(store);
    if let Some(rid) = rid {
        let _ = state.senders.abort.send(AbortSource::Guard(rid));
    }
    Json(unary_response_value(response)).into_response()
}

fn responses_text(parts: &[InputContent]) -> Result<String, String> {
    let mut text = String::new();
    for part in parts {
        match part {
            InputContent::InputText(part) => text.push_str(&part.text),
            InputContent::InputImage(_) | InputContent::InputFile(_) => {
                return Err(
                    "image and file Responses inputs are not supported by native generation".into(),
                );
            }
        }
    }
    Ok(text)
}

fn upstream_responses_text(parts: &[UpstreamInputContent]) -> Result<String, String> {
    let mut text = String::new();
    for part in parts {
        match part {
            UpstreamInputContent::InputText(part) => text.push_str(&part.text),
            UpstreamInputContent::InputImage(_) | UpstreamInputContent::InputFile(_) => {
                return Err(
                    "image and file function outputs are not supported by native generation".into(),
                );
            }
        }
    }
    Ok(text)
}

#[allow(deprecated)]
pub(super) fn responses_chat_request(
    request: &CreateResponse,
    model: &str,
) -> Result<CreateChatCompletionRequest, String> {
    let mut messages = Vec::new();
    if let Some(instructions) = request.instructions.as_ref() {
        messages.push(ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text(instructions.clone()),
                name: None,
            },
        ));
    }

    match &request.input {
        InputParam::Text(text) => messages.push(ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(text.clone()),
                name: None,
            },
        )),
        InputParam::Items(items) => {
            for item in items {
                match item {
                    InputItem::Item(Item::Message(MessageItem::Input(message))) => {
                        let text = responses_text(&message.content)?;
                        match message.role {
                            InputRole::User => {
                                messages.push(ChatCompletionRequestMessage::User(
                                    ChatCompletionRequestUserMessage {
                                        content: ChatCompletionRequestUserMessageContent::Text(
                                            text,
                                        ),
                                        name: None,
                                    },
                                ));
                            }
                            InputRole::System | InputRole::Developer => {
                                messages.push(ChatCompletionRequestMessage::System(
                                    ChatCompletionRequestSystemMessage {
                                        content: ChatCompletionRequestSystemMessageContent::Text(
                                            text,
                                        ),
                                        name: None,
                                    },
                                ));
                            }
                        }
                    }
                    InputItem::Item(Item::Message(MessageItem::Output(message))) => {
                        let text = message
                            .content
                            .iter()
                            .map(|part| match part {
                                InputOutputMessageContent::OutputText(part) => part.text.as_str(),
                                InputOutputMessageContent::Refusal(part) => part.refusal.as_str(),
                            })
                            .collect::<String>();
                        messages.push(ChatCompletionRequestMessage::Assistant(
                            ChatCompletionRequestAssistantMessage {
                                content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                                    text,
                                )),
                                reasoning_content: None,
                                refusal: None,
                                name: None,
                                audio: None,
                                tool_calls: None,
                                function_call: None,
                            },
                        ));
                    }
                    InputItem::Item(Item::FunctionCall(call)) => {
                        let tool_call = ChatCompletionMessageToolCall {
                            id: call.call_id.clone(),
                            r#type: FunctionType::Function,
                            function: FunctionCall {
                                name: call.name.clone(),
                                arguments: call.arguments.clone(),
                            },
                        };
                        if let Some(ChatCompletionRequestMessage::Assistant(message)) =
                            messages.last_mut()
                            && message.content.is_none()
                            && let Some(tool_calls) = message.tool_calls.as_mut()
                        {
                            // Parallel Responses calls are consecutive items from one
                            // assistant turn. Keep them together so all following tool
                            // outputs refer to the same assistant message.
                            tool_calls.push(tool_call);
                        } else {
                            messages.push(ChatCompletionRequestMessage::Assistant(
                                ChatCompletionRequestAssistantMessage {
                                    content: None,
                                    reasoning_content: None,
                                    refusal: None,
                                    name: None,
                                    audio: None,
                                    tool_calls: Some(vec![tool_call]),
                                    function_call: None,
                                },
                            ));
                        }
                    }
                    InputItem::Item(Item::FunctionCallOutput(output)) => {
                        let text = match &output.output {
                            FunctionCallOutput::Text(text) => text.clone(),
                            FunctionCallOutput::Content(parts) => upstream_responses_text(parts)?,
                        };
                        messages.push(ChatCompletionRequestMessage::Tool(
                            ChatCompletionRequestToolMessage {
                                content: ChatCompletionRequestToolMessageContent::Text(text),
                                tool_call_id: output.call_id.clone(),
                            },
                        ));
                    }
                    InputItem::EasyMessage(message) => {
                        let text = match &message.content {
                            EasyInputContent::Text(text) => text.clone(),
                            EasyInputContent::ContentList(parts) => responses_text(parts)?,
                        };
                        match message.role {
                            dynamo_protocols::types::responses::Role::User => {
                                messages.push(ChatCompletionRequestMessage::User(
                                    ChatCompletionRequestUserMessage {
                                        content: ChatCompletionRequestUserMessageContent::Text(
                                            text,
                                        ),
                                        name: None,
                                    },
                                ));
                            }
                            dynamo_protocols::types::responses::Role::System
                            | dynamo_protocols::types::responses::Role::Developer => {
                                messages.push(ChatCompletionRequestMessage::System(
                                    ChatCompletionRequestSystemMessage {
                                        content: ChatCompletionRequestSystemMessageContent::Text(
                                            text,
                                        ),
                                        name: None,
                                    },
                                ));
                            }
                            dynamo_protocols::types::responses::Role::Assistant => {
                                messages.push(ChatCompletionRequestMessage::Assistant(
                                    ChatCompletionRequestAssistantMessage {
                                        content: Some(
                                            ChatCompletionRequestAssistantMessageContent::Text(
                                                text,
                                            ),
                                        ),
                                        reasoning_content: None,
                                        refusal: None,
                                        name: None,
                                        audio: None,
                                        tool_calls: None,
                                        function_call: None,
                                    },
                                ));
                            }
                        }
                    }
                    InputItem::ItemReference(_) => {
                        return Err("Responses item references are not supported".into());
                    }
                    InputItem::Item(_) => {
                        return Err("this Responses input item type is not supported".into());
                    }
                }
            }
        }
    }
    if messages.is_empty() {
        return Err("input cannot be empty".into());
    }

    let tools = request
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .map(|tool| match tool {
                    ResponseTool::Function(function) => Ok(ChatCompletionTool {
                        r#type: ChatCompletionToolType::Function,
                        function: FunctionObject {
                            name: function.name.clone(),
                            description: function.description.clone(),
                            parameters: function.parameters.clone(),
                            strict: function.strict,
                        },
                    }),
                    _ => Err("only function tools are supported by the Responses API".to_string()),
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .filter(|tools| !tools.is_empty());
    let tool_choice = match request.tool_choice.as_ref() {
        Some(ToolChoiceParam::Mode(ToolChoiceOptions::None)) => {
            Some(ChatCompletionToolChoiceOption::None)
        }
        Some(ToolChoiceParam::Mode(ToolChoiceOptions::Required)) => {
            Some(ChatCompletionToolChoiceOption::Required)
        }
        Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto)) | None => {
            Some(ChatCompletionToolChoiceOption::Auto)
        }
        Some(ToolChoiceParam::Function(function)) => Some(ChatCompletionToolChoiceOption::Named(
            ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: FunctionName {
                    name: function.name.clone(),
                },
            },
        )),
        Some(_) => return Err("this Responses tool_choice is not supported".into()),
    };
    let response_format = request.text.as_ref().and_then(|text| match &text.format {
        TextResponseFormatConfiguration::Text => None,
        TextResponseFormatConfiguration::JsonObject => Some(ResponseFormat::JsonObject),
        TextResponseFormatConfiguration::JsonSchema(schema) => Some(ResponseFormat::JsonSchema {
            json_schema: schema.clone(),
        }),
    });

    Ok(CreateChatCompletionRequest {
        messages,
        model: model.to_owned(),
        max_completion_tokens: request.max_output_tokens,
        stream: request.stream,
        temperature: request.temperature,
        top_p: request.top_p,
        response_format,
        top_logprobs: request.top_logprobs,
        logprobs: request.top_logprobs.map(|_| true),
        parallel_tool_calls: request.parallel_tool_calls,
        tools,
        tool_choice,
        reasoning_effort: request
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.clone())
            .map(ChatReasoningEffort::from),
        ..Default::default()
    })
}

fn insert_previous_response_history(
    messages: &mut Vec<ChatCompletionRequestMessage>,
    previous_messages: Vec<ChatCompletionRequestMessage>,
    has_current_instructions: bool,
) {
    // `responses_chat_request` emits the current `instructions` as the first
    // system message. Keep it first; the prior exchange belongs between that
    // instruction and the current input. Without instructions, history remains
    // at the beginning as before.
    let insert_at = usize::from(has_current_instructions);
    messages.splice(insert_at..insert_at, previous_messages);
}

async fn responses(
    State(state): State<AppState>,
    Extension(store): Extension<ResponseStore>,
    body: Result<Json<CreateResponse>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => return openai_error(StatusCode::BAD_REQUEST, rejection.body_text()),
    };
    let model = request
        .model
        .clone()
        .unwrap_or_else(|| state.server_args.served_model_name.clone());
    if model != state.server_args.served_model_name {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("The model `{model}` does not exist"),
        );
    }
    if request.max_output_tokens == Some(0) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "max_output_tokens must be positive",
        );
    }
    if request.conversation.is_some() || request.prompt.is_some() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "conversation and prompt templates are not supported",
        );
    }
    if request
        .include
        .as_ref()
        .is_some_and(|include| !include.is_empty())
    {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "include is not supported by native Responses generation",
        );
    }
    if request.max_tool_calls.is_some() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "max_tool_calls is not supported by native Responses generation",
        );
    }
    if request.prompt_cache_key.is_some() || request.prompt_cache_retention.is_some() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "prompt cache keys and retention are not supported",
        );
    }
    if matches!(request.truncation, Some(Truncation::Auto)) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "automatic Responses input truncation is not supported",
        );
    }
    if request
        .reasoning
        .as_ref()
        .is_some_and(|reasoning| reasoning.summary.is_some())
    {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "reasoning summaries are not supported",
        );
    }
    let previous_messages = if let Some(response_id) = request.previous_response_id.as_deref() {
        if let Some(message) = invalid_response_id(response_id) {
            return openai_error(StatusCode::BAD_REQUEST, message);
        }
        match store.read().await.get(response_id) {
            Some(stored) => stored.messages.clone(),
            None => {
                return openai_error(
                    StatusCode::NOT_FOUND,
                    format!("Response with id '{response_id}' not found."),
                );
            }
        }
    } else {
        Vec::new()
    };
    let mut chat_request = match responses_chat_request(&request, &model) {
        Ok(request) => request,
        Err(message) => return openai_error(StatusCode::BAD_REQUEST, message),
    };
    insert_previous_response_history(
        &mut chat_request.messages,
        previous_messages,
        request.instructions.is_some(),
    );
    let tools = chat_request.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|tool| ToolDefinition {
                name: tool.function.name.clone(),
                parameters: tool.function.parameters.clone(),
                strict: tool.function.strict,
            })
            .collect::<Vec<_>>()
    });
    let tool_choice = dynamo_tool_choice(&chat_request.tool_choice);
    let tools_enabled = tools.as_ref().is_some_and(|tools| !tools.is_empty())
        && tool_choice != DynamoToolChoice::None;
    let parser = tools_enabled
        .then(|| state.server_args.tool_call_parser.clone())
        .flatten();
    if tools_enabled && parser.is_none() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "tool calls require --tool-call-parser",
        );
    }
    let reasoning_parser = state.server_args.reasoning_parser.clone();

    let (chat_request, prompt) = match prepare_chat_request(&state, chat_request).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    let response_messages = chat_request.messages.clone();
    let stream_tool_choice = chat_request.tool_choice.clone();
    let sampling = match chat_sampling(
        &chat_request,
        SamplingDefaults::RESPONSES,
        parser.as_deref(),
        &tool_choice,
        tools.as_deref().unwrap_or_default(),
        request.parallel_tool_calls,
        &state.server_args,
    ) {
        Ok(sampling) => sampling,
        Err(message) => return openai_error(StatusCode::BAD_REQUEST, message),
    };
    let uses_tool_call_structural_tag = sampling.structural_tag.is_some();

    let stream = request.stream.unwrap_or(false);
    if request.background == Some(true) && stream {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "background Responses requests cannot stream",
        );
    }
    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let created_at = unix_seconds();
    let rid = Rid::from_client(&response_id);
    let native = GenerateRequest {
        rid: rid.clone(),
        text: Some(prompt),
        // Rendered templates own their special tokens — the pool must not
        // add another BOS/EOS (Python's `add_special_tokens=False`).
        skip_special_tokens: true,
        sampling_params: sampling,
        stream,
        return_logprob: request.top_logprobs.is_some(),
        logprob_start_len: -1,
        top_logprobs_num: request.top_logprobs.unwrap_or(0) as i64,
        return_text_in_logprobs: request.top_logprobs.map(|_| true),
        ..Default::default()
    };
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let rx = match submit_generation(&state, native, stream, &mut guard).await {
        Ok(rx) => rx,
        Err(response) => return response,
    };

    if request.background == Some(true) {
        let queued = response_object(
            &response_id,
            &model,
            &request,
            created_at,
            Status::Queued,
            vec![],
            None,
        );
        store.write().await.insert(
            response_id.clone(),
            StoredResponse {
                response: queued.clone(),
                messages: response_messages.clone(),
                rid: Some(rid.clone()),
            },
        );
        let store = store.clone();
        let task_response_id = response_id.clone();
        tokio::spawn(async move {
            {
                let mut responses = store.write().await;
                let Some(stored) = responses.get_mut(&task_response_id) else {
                    return;
                };
                if stored.response.status == Status::Cancelled {
                    return;
                }
                stored.response.status = Status::InProgress;
            }
            match collect_response_items(
                rx,
                guard,
                &rid,
                parser.as_deref(),
                reasoning_parser.as_deref(),
                tools.as_deref(),
                request.parallel_tool_calls.unwrap_or(true),
                request.top_logprobs.is_some(),
            )
            .await
            {
                Ok((output, items)) => {
                    let mut messages = response_messages;
                    append_response_output(&mut messages, &items);
                    let final_status = response_status(&output);
                    let completed = response_object(
                        &task_response_id,
                        &model,
                        &request,
                        created_at,
                        final_status,
                        items,
                        Some(responses_usage(
                            output.prompt_tokens,
                            output.completion_tokens,
                        )),
                    );
                    let mut responses = store.write().await;
                    if let Some(stored) = responses.get_mut(&task_response_id)
                        && stored.response.status != Status::Cancelled
                    {
                        *stored = StoredResponse {
                            response: completed,
                            messages,
                            rid: None,
                        };
                    }
                }
                Err(_) => {
                    let mut responses = store.write().await;
                    if let Some(stored) = responses.get_mut(&task_response_id)
                        && stored.response.status != Status::Cancelled
                    {
                        stored.response.status = Status::Failed;
                        stored.rid = None;
                    }
                }
            }
        });
        return Json(unary_response_value(queued)).into_response();
    }

    if stream {
        let event_stream = responses_event_stream(
            rx,
            guard,
            rid,
            response_id,
            created_at,
            model,
            request,
            parser,
            reasoning_parser,
            tools,
            stream_tool_choice,
            uses_tool_call_structural_tag,
            store,
            response_messages,
        )
        // Python `_send_event` frames each event as
        // `event: {type}\ndata: {payload}` — the event name is the payload's
        // `type` field, so consumers can dispatch on it instead of receiving
        // only generic `message` events. `[DONE]` and error frames carry no
        // `type` and stay data-only (as in Python).
        .map(|data| Ok::<_, Infallible>(sse_frame(data)));
        Sse::new(event_stream).into_response()
    } else {
        unary_responses(
            rx,
            guard,
            rid,
            response_id,
            created_at,
            model,
            request,
            parser,
            reasoning_parser,
            tools,
            store,
            response_messages,
        )
        .await
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn unary_responses(
    rx: mpsc::Receiver<EgressItem>,
    guard: AbortGuard,
    rid: Rid,
    response_id: String,
    created_at: u64,
    model: String,
    request: CreateResponse,
    parser: Option<String>,
    reasoning_parser: Option<String>,
    tools: Option<Vec<ToolDefinition>>,
    store: ResponseStore,
    mut messages: Vec<ChatCompletionRequestMessage>,
) -> Response {
    let (output, items) = match collect_response_items(
        rx,
        guard,
        &rid,
        parser.as_deref(),
        reasoning_parser.as_deref(),
        tools.as_deref(),
        request.parallel_tool_calls.unwrap_or(true),
        request.top_logprobs.is_some(),
    )
    .await
    {
        Ok(result) => result,
        Err((status, message)) => return openai_error(status, message),
    };
    append_response_output(&mut messages, &items);
    let status = response_status(&output);
    let response = response_object(
        &response_id,
        &model,
        &request,
        created_at,
        status,
        items,
        Some(responses_usage(
            output.prompt_tokens,
            output.completion_tokens,
        )),
    );
    if request.store.unwrap_or(true) {
        store.write().await.insert(
            response_id,
            StoredResponse {
                response: response.clone(),
                messages,
                rid: None,
            },
        );
    }
    Json(unary_response_value(response)).into_response()
}

async fn collect_response_items(
    rx: mpsc::Receiver<EgressItem>,
    mut guard: AbortGuard,
    rid: &Rid,
    parser: Option<&str>,
    reasoning_parser: Option<&str>,
    tools: Option<&[ToolDefinition]>,
    parallel_tool_calls: bool,
    want_logprobs: bool,
) -> Result<(ChunkEvent, Vec<OutputItem>), (StatusCode, String)> {
    let output = collect_output(rx, &mut guard, rid).await?;
    // Split reasoning markers out of the content first (Python splits before
    // tool-call parsing too), then parse tool calls on the clean normal text.
    let (reasoning_text, text) =
        split_reasoning_unary(reasoning_parser, &output.text, &output.token_ids);
    let (text, tool_calls) = parse_chat_tool_calls(text, parser, tools, parallel_tool_calls).await;
    let mut items = Vec::new();
    if !reasoning_text.is_empty() {
        items.push(response_reasoning_item(
            format!("rs_{}", uuid::Uuid::new_v4().simple()),
            reasoning_text,
            None,
        ));
    }
    items.extend(
        tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(response_function_call),
    );
    if !text.is_empty() || items.is_empty() {
        let logprobs = want_logprobs.then(|| chunk_response_logprobs(output.extras.as_deref()));
        items.push(text_response_message(
            format!("msg_{}", uuid::Uuid::new_v4().simple()),
            OutputStatus::Completed,
            text_output_content(text, logprobs),
        ));
    }
    Ok((output, items))
}

pub(super) fn response_reasoning_item(
    id: String,
    text: String,
    status: Option<OutputStatus>,
) -> OutputItem {
    OutputItem::Reasoning(ReasoningItem {
        id: Some(id),
        summary: vec![],
        content: Some(vec![ReasoningItemContent::ReasoningText(
            ReasoningTextContent { text },
        )]),
        encrypted_content: None,
        status,
    })
}

/// The in-progress item behind a `response.output_item.added` event. Python
/// opens the reasoning item with an empty `content: []` and fills the content
/// only in the completed item of the close event
/// (`serving_responses.py._open_reasoning_item` vs `_close_reasoning_item`).
pub(super) fn pending_reasoning_item(id: String) -> OutputItem {
    OutputItem::Reasoning(ReasoningItem {
        id: Some(id),
        summary: vec![],
        content: Some(vec![]),
        encrypted_content: None,
        status: Some(OutputStatus::InProgress),
    })
}

pub(super) fn response_function_call(call: ChatCompletionMessageToolCall) -> OutputItem {
    OutputItem::FunctionCall(FunctionToolCall {
        arguments: call.function.arguments,
        call_id: call.id,
        namespace: None,
        name: call.function.name,
        id: Some(format!("fc_{}", uuid::Uuid::new_v4().simple())),
        status: Some(OutputStatus::Completed),
    })
}

#[allow(deprecated)]
pub(super) fn append_response_output(
    messages: &mut Vec<ChatCompletionRequestMessage>,
    items: &[OutputItem],
) {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for item in items {
        match item {
            OutputItem::Message(message) => {
                for content in &message.content {
                    match content {
                        OutputMessageContent::OutputText(content) => text.push_str(&content.text),
                        OutputMessageContent::Refusal(content) => text.push_str(&content.refusal),
                    }
                }
            }
            // Reasoning items render as `reasoning_content` on the assistant
            // message (Python `_response_to_chat_messages`).
            OutputItem::Reasoning(item) => {
                if let Some(content) = item.content.as_deref() {
                    for part in content {
                        let ReasoningItemContent::ReasoningText(content) = part;
                        reasoning.push_str(&content.text);
                    }
                }
            }
            OutputItem::FunctionCall(call) => {
                tool_calls.push(ChatCompletionMessageToolCall {
                    id: call.call_id.clone(),
                    r#type: FunctionType::Function,
                    function: FunctionCall {
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    },
                });
            }
            _ => {}
        }
    }
    messages.push(ChatCompletionRequestMessage::Assistant(
        ChatCompletionRequestAssistantMessage {
            content: (!text.is_empty())
                .then_some(ChatCompletionRequestAssistantMessageContent::Text(text)),
            reasoning_content: (!reasoning.is_empty()).then_some(ReasoningContent::Text(reasoning)),
            refusal: None,
            name: None,
            audio: None,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            function_call: None,
        },
    ));
}

fn unary_response_value(response: OpenAIResponse) -> serde_json::Value {
    let mut value = serde_json::to_value(response).expect("OpenAI response must serialize");
    if let Some(usage) = value["usage"].as_object_mut() {
        usage.insert("prompt_tokens".into(), usage["input_tokens"].clone());
        usage.insert("completion_tokens".into(), usage["output_tokens"].clone());
    }
    value
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::{chunk, response_request, senders};
    use super::{
        insert_previous_response_history, new_response_store, responses_chat_request,
        responses_event_stream, sse_frame, unary_responses,
    };
    use crate::api_server::guard::AbortGuard;
    use crate::message::{ChunkExtras, EgressItem};
    use dynamo_protocols::types::responses::CreateResponse;
    use dynamo_protocols::types::{ChatCompletionRequestMessage, ChatCompletionToolChoiceOption};
    use futures::StreamExt;
    use tokio::sync::mpsc;

    fn with_logprobs(
        mut item: EgressItem,
        token_id: i32,
        token: &str,
        logprob: f32,
        alternative_id: i32,
        alternative: &str,
        alternative_logprob: f32,
    ) -> EgressItem {
        let output = match &mut item {
            EgressItem::Frame(output) | EgressItem::Done(output) => output,
            _ => unreachable!(),
        };
        output.extras = Some(Box::new(ChunkExtras {
            out_lp_val: vec![logprob],
            out_lp_idx: vec![token_id],
            out_lp_txt: vec![token.into()],
            out_top_val: vec![logprob, alternative_logprob],
            out_top_idx: vec![token_id, alternative_id],
            out_top_lens: vec![2],
            out_top_txt: vec![token.into(), alternative.into()],
            ..Default::default()
        }));
        item
    }

    #[test]
    fn structured_responses_input_reuses_chat_history_and_function_tools() {
        let request: CreateResponse = serde_json::from_value(serde_json::json!({
            "model": "model",
            "input": [
                {"role": "user", "content": "What is the weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Paris\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "sunny"
                }
            ],
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object"}
            }],
            "tool_choice": "required"
        }))
        .unwrap();
        let chat = responses_chat_request(&request, "model").unwrap();
        assert_eq!(chat.messages.len(), 3);
        assert!(matches!(
            chat.messages[0],
            ChatCompletionRequestMessage::User(_)
        ));
        assert!(matches!(
            chat.messages[1],
            ChatCompletionRequestMessage::Assistant(_)
        ));
        assert!(matches!(
            chat.messages[2],
            ChatCompletionRequestMessage::Tool(_)
        ));
        assert_eq!(chat.tools.unwrap()[0].function.name, "get_weather");
        assert!(matches!(
            chat.tool_choice,
            Some(ChatCompletionToolChoiceOption::Required)
        ));
    }

    #[test]
    fn parallel_function_calls_share_one_assistant_turn() {
        let request: CreateResponse = serde_json::from_value(serde_json::json!({
            "model": "model",
            "input": [
                {"role": "user", "content": "Compare Paris and Tokyo weather"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Paris\"}"
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Tokyo\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "sunny"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "rainy"
                }
            ]
        }))
        .unwrap();

        let chat = responses_chat_request(&request, "model").unwrap();
        assert_eq!(chat.messages.len(), 4);
        let ChatCompletionRequestMessage::Assistant(assistant) = &chat.messages[1] else {
            panic!("parallel calls must produce one assistant message");
        };
        let calls = assistant.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[1].id, "call_2");
        assert!(matches!(
            &chat.messages[2],
            ChatCompletionRequestMessage::Tool(tool) if tool.tool_call_id == "call_1"
        ));
        assert!(matches!(
            &chat.messages[3],
            ChatCompletionRequestMessage::Tool(tool) if tool.tool_call_id == "call_2"
        ));
    }

    #[test]
    fn current_instructions_precede_previous_response_history() {
        let request: CreateResponse = serde_json::from_value(serde_json::json!({
            "model": "model",
            "instructions": "Follow the new instruction",
            "input": "current input"
        }))
        .unwrap();
        let previous_request: CreateResponse = serde_json::from_value(serde_json::json!({
            "model": "model",
            "input": "previous input"
        }))
        .unwrap();
        let previous = responses_chat_request(&previous_request, "model")
            .unwrap()
            .messages;
        let mut chat = responses_chat_request(&request, "model").unwrap();

        insert_previous_response_history(
            &mut chat.messages,
            previous,
            request.instructions.is_some(),
        );

        assert_eq!(chat.messages.len(), 3);
        assert!(matches!(
            &chat.messages[0],
            ChatCompletionRequestMessage::System(message)
                if matches!(&message.content,
                    dynamo_protocols::types::ChatCompletionRequestSystemMessageContent::Text(text)
                    if text == "Follow the new instruction")
        ));
        assert!(matches!(
            &chat.messages[1],
            ChatCompletionRequestMessage::User(message)
                if matches!(&message.content,
                    dynamo_protocols::types::ChatCompletionRequestUserMessageContent::Text(text)
                    if text == "previous input")
        ));
        assert!(matches!(
            &chat.messages[2],
            ChatCompletionRequestMessage::User(message)
                if matches!(&message.content,
                    dynamo_protocols::types::ChatCompletionRequestUserMessageContent::Text(text)
                    if text == "current input")
        ));
    }

    #[tokio::test]
    async fn unary_responses_uses_standard_output_items() {
        let (tx, rx) = mpsc::channel(8);
        tx.send(chunk("r0", "Paris", true)).await.unwrap();
        let response = unary_responses(
            rx,
            AbortGuard::new_empty(senders()),
            "r0".into(),
            "resp_test".into(),
            1_234_567_890,
            "model".into(),
            response_request(false),
            None,
            None,
            None,
            new_response_store(),
            vec![],
        )
        .await;
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["object"], "response");
        assert_eq!(value["created_at"], 1_234_567_890);
        assert_eq!(value["status"], "completed");
        assert_eq!(value["output"][0]["type"], "message");
        assert_eq!(value["output"][0]["content"][0]["text"], "Paris");
        assert_eq!(value["usage"]["input_tokens"], 5);
        assert_eq!(value["usage"]["output_tokens"], 1);
        assert_eq!(value["usage"]["prompt_tokens"], 5);
        assert_eq!(value["usage"]["completion_tokens"], 1);
    }

    #[tokio::test]
    async fn unary_responses_populates_requested_logprobs() {
        let (tx, rx) = mpsc::channel(8);
        tx.send(with_logprobs(
            chunk("r0", "Paris", true),
            7,
            "Paris",
            -0.25,
            8,
            "London",
            -1.0,
        ))
        .await
        .unwrap();
        let mut request = response_request(false);
        request.top_logprobs = Some(2);
        let response = unary_responses(
            rx,
            AbortGuard::new_empty(senders()),
            "r0".into(),
            "resp_test".into(),
            1_234_567_890,
            "model".into(),
            request,
            None,
            None,
            None,
            new_response_store(),
            vec![],
        )
        .await;
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let logprob = &value["output"][0]["content"][0]["logprobs"][0];
        assert_eq!(logprob["token"], "Paris");
        assert_eq!(logprob["bytes"], serde_json::json!([80, 97, 114, 105, 115]));
        assert_eq!(logprob["logprob"], -0.25);
        assert_eq!(logprob["top_logprobs"][1]["token"], "London");
        assert_eq!(logprob["top_logprobs"][1]["logprob"], -1.0);
    }

    #[tokio::test]
    async fn streaming_responses_emits_lifecycle_and_text_deltas() {
        let (tx, rx) = mpsc::channel(8);
        tx.send(chunk("r0", "Par", false)).await.unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();
        let stream = responses_event_stream(
            rx,
            AbortGuard::new_empty(senders()),
            "r0".into(),
            "resp_test".into(),
            1_234_567_890,
            "model".into(),
            response_request(true),
            None,
            None,
            None,
            None,
            false,
            new_response_store(),
            vec![],
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        let event_types = frames[..frames.len() - 1]
            .iter()
            .map(|frame| {
                serde_json::from_str::<serde_json::Value>(frame).unwrap()["type"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect::<Vec<_>>();
        assert_eq!(event_types[0], "response.created");
        assert_eq!(event_types[1], "response.in_progress");
        assert_eq!(
            event_types
                .iter()
                .filter(|event| *event == "response.output_text.delta")
                .count(),
            2
        );
        assert_eq!(event_types.last().unwrap(), "response.completed");
        let created: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
        let completed: serde_json::Value = serde_json::from_str(&frames[frames.len() - 2]).unwrap();
        assert_eq!(created["response"]["created_at"], 1_234_567_890);
        assert_eq!(completed["response"]["created_at"], 1_234_567_890);
        assert_eq!(completed["response"]["usage"]["output_tokens"], 2);
        assert_eq!(frames.last().unwrap(), "[DONE]");
    }

    #[tokio::test]
    async fn streaming_responses_populates_delta_and_accumulated_logprobs() {
        let (tx, rx) = mpsc::channel(8);
        tx.send(with_logprobs(
            chunk("r0", "Par", false),
            7,
            "Par",
            -0.25,
            8,
            "Bar",
            -1.0,
        ))
        .await
        .unwrap();
        tx.send(with_logprobs(
            chunk("r0", "is", true),
            9,
            "is",
            -0.5,
            10,
            " was",
            -1.5,
        ))
        .await
        .unwrap();
        let mut request = response_request(true);
        request.top_logprobs = Some(2);
        let stream = responses_event_stream(
            rx,
            AbortGuard::new_empty(senders()),
            "r0".into(),
            "resp_test".into(),
            1_234_567_890,
            "model".into(),
            request,
            None,
            None,
            None,
            None,
            false,
            new_response_store(),
            vec![],
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        let events = frames[..frames.len() - 1]
            .iter()
            .map(|frame| serde_json::from_str::<serde_json::Value>(frame).unwrap())
            .collect::<Vec<_>>();
        let deltas = events
            .iter()
            .filter(|event| event["type"] == "response.output_text.delta")
            .collect::<Vec<_>>();
        assert_eq!(deltas[0]["logprobs"][0]["token"], "Par");
        assert_eq!(deltas[0]["logprobs"][0]["top_logprobs"][1]["token"], "Bar");
        assert_eq!(deltas[1]["logprobs"][0]["token"], "is");

        let done = events
            .iter()
            .find(|event| event["type"] == "response.output_text.done")
            .unwrap();
        assert_eq!(done["logprobs"].as_array().unwrap().len(), 2);
        assert_eq!(done["logprobs"][1]["token"], "is");

        let completed = events
            .iter()
            .find(|event| event["type"] == "response.completed")
            .unwrap();
        let logprobs = &completed["response"]["output"][0]["content"][0]["logprobs"];
        assert_eq!(logprobs.as_array().unwrap().len(), 2);
        assert_eq!(logprobs[0]["bytes"], serde_json::json!([80, 97, 114]));
        assert_eq!(logprobs[1]["top_logprobs"][1]["token"], " was");
        assert_eq!(frames.last().unwrap(), "[DONE]");
    }

    /// Python `_send_event` frames each event as `event: {type}\ndata: {payload}`.
    /// Assert the actual SSE wire bytes: lifecycle frames carry their event name;
    /// `[DONE]` and error frames stay data-only.
    #[tokio::test]
    async fn responses_sse_frames_carry_event_names() {
        use axum::response::IntoResponse;
        use std::convert::Infallible;

        let payload =
            r#"{"type":"response.created","sequence_number":0,"response":{}}"#.to_string();
        let stream = futures::stream::iter(vec![
            Ok::<_, Infallible>(sse_frame(payload)),
            Ok::<_, Infallible>(sse_frame("[DONE]".into())),
            Ok::<_, Infallible>(sse_frame(r#"{"error":{"message":"boom"}}"#.into())),
        ]);
        let response = axum::response::Sse::new(stream).into_response();
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        let mut frames = text.split("\n\n");
        let first = frames.next().unwrap();
        assert!(
            first.contains("event: response.created"),
            "missing event name in {text:?}"
        );
        assert!(
            first.contains("response.created"),
            "missing payload in {text:?}"
        );
        assert!(!frames.next().unwrap().contains("event:"));
        assert!(!frames.next().unwrap().contains("event:"));
    }

    #[tokio::test]
    async fn streaming_responses_with_tools_emits_normal_text_deltas() {
        let (tx, rx) = mpsc::channel(8);
        tx.send(chunk("r0", "Par", false)).await.unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();
        let stream = responses_event_stream(
            rx,
            AbortGuard::new_empty(senders()),
            "r0".into(),
            "resp_test".into(),
            1_234_567_890,
            "model".into(),
            response_request(true),
            Some("llama3".into()),
            None,
            None,
            None,
            false,
            new_response_store(),
            vec![],
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        let deltas = frames
            .iter()
            .filter_map(|frame| serde_json::from_str::<serde_json::Value>(frame).ok())
            .filter(|frame| frame["type"] == "response.output_text.delta")
            .filter_map(|frame| frame["delta"].as_str().map(str::to_owned))
            .collect::<Vec<_>>();
        assert_eq!(deltas, ["Par", "is"]);
    }

    #[tokio::test]
    async fn streaming_responses_emits_function_call_events() {
        let (tx, rx) = mpsc::channel(8);
        tx.send(chunk(
            "r0",
            r#"<|python_tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
            true,
        ))
        .await
        .unwrap();
        let stream = responses_event_stream(
            rx,
            AbortGuard::new_empty(senders()),
            "r0".into(),
            "resp_test".into(),
            1_234_567_890,
            "model".into(),
            response_request(true),
            Some("llama3".into()),
            None,
            None,
            None,
            false,
            new_response_store(),
            vec![],
        );
        futures::pin_mut!(stream);
        let frames = stream.collect::<Vec<_>>().await;
        let event_types = frames
            .iter()
            .filter_map(|frame| serde_json::from_str::<serde_json::Value>(frame).ok())
            .filter_map(|event| event["type"].as_str().map(str::to_owned))
            .collect::<Vec<_>>();
        assert!(event_types.contains(&"response.output_item.added".into()));
        assert!(event_types.contains(&"response.function_call_arguments.delta".into()));
        assert!(event_types.contains(&"response.function_call_arguments.done".into()));
        assert!(event_types.contains(&"response.output_item.done".into()));
        assert!(event_types.contains(&"response.completed".into()));
    }

    /// `--reasoning-parser` splits the think block into a reasoning item
    /// before tool-call parsing sees the text, and the stored conversation
    /// carries it as `reasoning_content`.
    #[tokio::test]
    async fn unary_responses_splits_reasoning_before_tool_parsing() {
        let (tx, rx) = mpsc::channel(8);
        tx.send(chunk(
            "r0",
            r#"<think>check the weather</think><|python_tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
            true,
        ))
        .await
        .unwrap();
        let response = unary_responses(
            rx,
            AbortGuard::new_empty(senders()),
            "r0".into(),
            "resp_test".into(),
            1_234_567_890,
            "model".into(),
            response_request(false),
            Some("llama3".into()),
            Some("deepseek-r1".into()),
            None,
            new_response_store(),
            vec![],
        )
        .await;
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let output = value["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["type"], "reasoning_text");
        assert_eq!(output[0]["content"][0]["text"], "check the weather");
        assert_eq!(output[1]["type"], "function_call");
        assert_eq!(output[1]["name"], "get_weather");
        assert_eq!(output[1]["arguments"], r#"{"city":"Paris"}"#);
        assert_eq!(
            output.len(),
            2,
            "think markers must not leak into a message"
        );
    }

    #[tokio::test]
    async fn unary_responses_emits_reasoning_item_without_tools() {
        let (tx, rx) = mpsc::channel(8);
        tx.send(chunk("r0", "<think>think hard</think>Paris", true))
            .await
            .unwrap();
        let response = unary_responses(
            rx,
            AbortGuard::new_empty(senders()),
            "r0".into(),
            "resp_test".into(),
            1_234_567_890,
            "model".into(),
            response_request(false),
            None,
            Some("deepseek-r1".into()),
            None,
            new_response_store(),
            vec![],
        )
        .await;
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let output = value["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["text"], "think hard");
        assert_eq!(output[1]["type"], "message");
        assert_eq!(output[1]["content"][0]["text"], "Paris");
    }

    /// Streaming with `--reasoning-parser`: the split chunks become a
    /// reasoning item (added → deltas → done) opened before the message item,
    /// and the completed snapshot carries both items.
    #[tokio::test]
    async fn streaming_responses_emits_reasoning_item_and_clean_text() {
        let (tx, rx) = mpsc::channel(8);
        tx.send(chunk("r0", "<think>rea", false)).await.unwrap();
        tx.send(chunk("r0", "son</think>Par", false)).await.unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();
        let stream = responses_event_stream(
            rx,
            AbortGuard::new_empty(senders()),
            "r0".into(),
            "resp_test".into(),
            1_234_567_890,
            "model".into(),
            response_request(true),
            None,
            Some("deepseek-r1".into()),
            None,
            None,
            false,
            new_response_store(),
            vec![],
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        let events = frames
            .iter()
            .filter_map(|frame| serde_json::from_str::<serde_json::Value>(frame).ok())
            .collect::<Vec<_>>();
        let types = events
            .iter()
            .map(|event| event["type"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        // Deltas split into reasoning_text.delta then output_text.delta, with
        // the reasoning item added before the message item.
        let added_index = types
            .iter()
            .position(|event| event == "response.output_item.added")
            .unwrap();
        assert_eq!(events[added_index]["item"]["type"], "reasoning");
        assert_eq!(events[added_index]["item"]["status"], "in_progress");
        // Python opens the reasoning item with empty content/summary and fills
        // them only in the completed close event.
        assert_eq!(
            events[added_index]["item"]["content"],
            serde_json::json!([])
        );
        assert_eq!(
            events[added_index]["item"]["summary"],
            serde_json::json!([])
        );
        let reasoning_deltas = events
            .iter()
            .filter(|event| event["type"] == "response.reasoning_text.delta")
            .map(|event| event["delta"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(reasoning_deltas, ["rea", "son"]);
        let text_deltas = events
            .iter()
            .filter(|event| event["type"] == "response.output_text.delta")
            .map(|event| event["delta"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(text_deltas, ["Par", "is"]);
        assert!(
            events
                .iter()
                .flat_map(|event| event["delta"].as_str())
                .all(|delta| !delta.contains("<think>")),
            "markers must never surface as deltas"
        );
        // The reasoning item closes before the message item opens.
        let reasoning_done = types
            .iter()
            .position(|event| event == "response.reasoning_text.done")
            .unwrap();
        assert_eq!(events[reasoning_done]["text"], "reason");
        let message_added = types
            .iter()
            .rposition(|event| event == "response.output_item.added")
            .unwrap();
        assert!(reasoning_done < message_added);
        // Completed snapshot: reasoning item first, then the clean message.
        let completed = events
            .iter()
            .find(|event| event["type"] == "response.completed")
            .unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["text"], "reason");
        assert_eq!(output[0]["status"], "completed");
        assert_eq!(output[1]["type"], "message");
        assert_eq!(output[1]["content"][0]["text"], "Paris");
        assert_eq!(output.len(), 2);
    }

    /// MiniMax M3's implicit-tool-start recovery buffers the answer text until
    /// a boundary establishes the mode; with no `<mm:think>` opener the whole
    /// message body is released by the terminal tail flush — both columns,
    /// not just the reasoning half.
    #[tokio::test]
    async fn streaming_responses_releases_normal_tail_at_completion() {
        let (tx, rx) = mpsc::channel(8);
        tx.send(chunk("r0", "The answer is", false)).await.unwrap();
        tx.send(chunk("r0", " 42", true)).await.unwrap();
        let stream = responses_event_stream(
            rx,
            AbortGuard::new_empty(senders()),
            "r0".into(),
            "resp_test".into(),
            1_234_567_890,
            "model".into(),
            response_request(true),
            None,
            Some("minimax_m3".into()),
            None,
            None,
            false,
            new_response_store(),
            vec![],
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        let events = frames
            .iter()
            .filter_map(|frame| serde_json::from_str::<serde_json::Value>(frame).ok())
            .collect::<Vec<_>>();
        let text_deltas = events
            .iter()
            .filter(|event| event["type"] == "response.output_text.delta")
            .map(|event| event["delta"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(text_deltas, ["The answer is 42"]);
        let completed = events
            .iter()
            .find(|event| event["type"] == "response.completed")
            .unwrap();
        assert_eq!(
            completed["response"]["output"][0]["content"][0]["text"],
            "The answer is 42"
        );
        assert_eq!(completed["response"]["output"][0]["type"], "message");
    }

    /// The unary response is stored with the assistant message carrying
    /// `reasoning_content`, and feeding those stored messages into the next
    /// turn (what a `previous_response_id` chain does) keeps the reasoning —
    /// Python renders reasoning items as `{role: assistant, reasoning_content}`
    /// when rebuilding conversation state.
    #[tokio::test]
    async fn unary_responses_stores_reasoning_for_the_next_turn() {
        use dynamo_protocols::types::ChatCompletionRequestMessage;

        let store = new_response_store();
        let (tx, rx) = mpsc::channel(8);
        tx.send(chunk("r0", "<think>think hard</think>Paris", true))
            .await
            .unwrap();
        let response = unary_responses(
            rx,
            AbortGuard::new_empty(senders()),
            "r0".into(),
            "resp_test".into(),
            1_234_567_890,
            "model".into(),
            response_request(false),
            None,
            Some("deepseek-r1".into()),
            None,
            store.clone(),
            vec![],
        )
        .await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let stored = store
            .read()
            .await
            .get("resp_test")
            .expect("the response must be stored")
            .clone();
        let ChatCompletionRequestMessage::Assistant(first) = &stored.messages[0] else {
            panic!("expected an assistant message, got {:?}", stored.messages);
        };
        assert_eq!(
            first.reasoning_content,
            Some(dynamo_protocols::types::ReasoningContent::Text(
                "think hard".into()
            ))
        );

        // Follow-up turn seeded with the previous turn's stored messages.
        let (tx, rx) = mpsc::channel(8);
        tx.send(chunk("r0", "done", true)).await.unwrap();
        let response = unary_responses(
            rx,
            AbortGuard::new_empty(senders()),
            "r0".into(),
            "resp_test2".into(),
            1_234_567_890,
            "model".into(),
            response_request(false),
            None,
            Some("deepseek-r1".into()),
            None,
            store.clone(),
            stored.messages,
        )
        .await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let stored = store
            .read()
            .await
            .get("resp_test2")
            .expect("the follow-up response must be stored")
            .clone();
        assert_eq!(stored.messages.len(), 2);
        let ChatCompletionRequestMessage::Assistant(chained) = &stored.messages[0] else {
            panic!("expected the chained assistant message");
        };
        assert_eq!(
            chained.reasoning_content,
            Some(dynamo_protocols::types::ReasoningContent::Text(
                "think hard".into()
            ))
        );
    }
}
