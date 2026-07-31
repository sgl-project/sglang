//! OpenAI Responses endpoint, response storage, and lifecycle events.

use std::convert::Infallible;

use axum::{
    Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
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
    OutputMessageContent, OutputStatus, Response as OpenAIResponse, Status,
    TextResponseFormatConfiguration, Tool as ResponseTool, ToolChoiceOptions, ToolChoiceParam,
    Truncation, UpstreamInputContent,
};
use dynamo_protocols::types::{
    ChatCompletionMessageToolCall, ChatCompletionNamedToolChoice,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionTool, ChatCompletionToolChoiceOption,
    ChatCompletionToolType, CreateChatCompletionRequest, FunctionCall, FunctionName,
    FunctionObject, FunctionType, ReasoningEffort as ChatReasoningEffort, ResponseFormat,
};
use futures::StreamExt;
use tokio::sync::mpsc;

use super::super::guard::AbortGuard;
use super::chat::{SamplingDefaults, chat_sampling_params, prepare_chat_request};
use super::response_stream::{
    response_object, response_status, responses_event_stream, responses_usage,
    text_response_message,
};
use super::tools::{apply_tool_constraint, parse_chat_tool_calls};
use super::{
    AppState, ResponseStore, StoredResponse, authorize, collect_output, openai_error,
    submit_generation, unix_seconds,
};
use crate::ids::Rid;
use crate::message::{ChunkEvent, EgressItem, GenerateRequest};
use crate::tokenizer_manager::AbortSource;

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/responses", post(responses))
        .route("/v1/responses/{response_id}", get(retrieve_response))
        .route("/v1/responses/{response_id}/cancel", post(cancel_response))
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
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(response_id): Path<String>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    if let Some(message) = invalid_response_id(&response_id) {
        return openai_error(StatusCode::BAD_REQUEST, message);
    }
    let response = state
        .response_store
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
    headers: HeaderMap,
    Path(response_id): Path<String>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    if let Some(message) = invalid_response_id(&response_id) {
        return openai_error(StatusCode::BAD_REQUEST, message);
    }
    let mut store = state.response_store.write().await;
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
                        messages.push(ChatCompletionRequestMessage::Assistant(
                            ChatCompletionRequestAssistantMessage {
                                content: None,
                                reasoning_content: None,
                                refusal: None,
                                name: None,
                                audio: None,
                                tool_calls: Some(vec![ChatCompletionMessageToolCall {
                                    id: call.call_id.clone(),
                                    r#type: FunctionType::Function,
                                    function: FunctionCall {
                                        name: call.name.clone(),
                                        arguments: call.arguments.clone(),
                                    },
                                }]),
                                function_call: None,
                            },
                        ));
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
        logprobs: request.top_logprobs.map(|value| value > 0),
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

async fn responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<CreateResponse>, JsonRejection>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
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
        match state.response_store.read().await.get(response_id) {
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
    if !previous_messages.is_empty() {
        chat_request.messages.splice(0..0, previous_messages);
    }
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
    let tool_choice = match &chat_request.tool_choice {
        Some(ChatCompletionToolChoiceOption::None) => DynamoToolChoice::None,
        Some(ChatCompletionToolChoiceOption::Required) => DynamoToolChoice::Required,
        Some(ChatCompletionToolChoiceOption::Named(choice)) => {
            DynamoToolChoice::Named(choice.function.name.clone())
        }
        _ => DynamoToolChoice::Auto,
    };
    if tool_choice == DynamoToolChoice::Required
        && tools.as_ref().is_none_or(|tools| tools.is_empty())
    {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "tool_choice is \"required\" but tools is empty",
        );
    }
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

    let (Some(formatter), Some(tokenizer)) =
        (state.chat_formatter.clone(), state.chat_tokenizer.clone())
    else {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "this model has no usable chat template",
        );
    };
    let (chat_request, input_ids) =
        match prepare_chat_request(chat_request, formatter, tokenizer).await {
            Ok(prepared) => prepared,
            Err(message) => return openai_error(StatusCode::BAD_REQUEST, message),
        };
    let response_messages = chat_request.messages.clone();
    let stream_tool_choice = chat_request.tool_choice.clone();
    let mut sampling = match chat_sampling_params(
        &chat_request,
        &SamplingDefaults::RESPONSES
            .with_model_defaults(&state.server_args.model_config.default_sampling_params),
    ) {
        Ok(sampling) => sampling,
        Err(message) => return openai_error(StatusCode::BAD_REQUEST, message),
    };
    if let Some(parser) = parser.as_deref()
        && let Err(message) = apply_tool_constraint(
            &mut sampling,
            parser,
            &tool_choice,
            tools.as_deref().unwrap_or_default(),
            request.parallel_tool_calls,
        )
    {
        return openai_error(StatusCode::BAD_REQUEST, message);
    }
    if let Err(error) = sampling.normalize(
        state.server_args.skip_tokenizer_init,
        state
            .server_args
            .model_config
            .vocab_size
            .unwrap_or(u64::MAX),
    ) {
        return openai_error(StatusCode::BAD_REQUEST, error.to_string());
    }
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
        input_ids: Some(input_ids),
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
        state.response_store.write().await.insert(
            response_id.clone(),
            StoredResponse {
                response: queued.clone(),
                messages: response_messages.clone(),
                rid: Some(rid.clone()),
            },
        );
        let store = state.response_store.clone();
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
                tools.as_deref(),
                request.parallel_tool_calls.unwrap_or(true),
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
            tools,
            stream_tool_choice,
            uses_tool_call_structural_tag,
            state.response_store,
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
            tools,
            state.response_store,
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
    tools: Option<Vec<ToolDefinition>>,
    store: ResponseStore,
    mut messages: Vec<ChatCompletionRequestMessage>,
) -> Response {
    let (output, items) = match collect_response_items(
        rx,
        guard,
        &rid,
        parser.as_deref(),
        tools.as_deref(),
        request.parallel_tool_calls.unwrap_or(true),
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
    tools: Option<&[ToolDefinition]>,
    parallel_tool_calls: bool,
) -> Result<(ChunkEvent, Vec<OutputItem>), (StatusCode, String)> {
    let output = collect_output(rx, &mut guard, rid).await?;
    let (text, tool_calls) =
        parse_chat_tool_calls(output.text.clone(), parser, tools, parallel_tool_calls).await;
    let mut items = tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(response_function_call)
        .collect::<Vec<_>>();
    if !text.is_empty() || items.is_empty() {
        items.push(text_response_message(
            format!("msg_{}", uuid::Uuid::new_v4().simple()),
            text,
            OutputStatus::Completed,
        ));
    }
    Ok((output, items))
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
            reasoning_content: None,
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
