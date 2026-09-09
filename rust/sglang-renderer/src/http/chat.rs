//! HTTP chat completion adapter and SGLang JSON encoding.

use super::{
    ChatCompletionRequest, OpenAIHttpFrontend,
    error::{json_rejection_response, response_error},
    response::sse_response,
};
use crate::openai::{
    chat::{chat_event_stream, prepare_request, unary_chat},
    submission::submit_generate_requests,
};
use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    response::{IntoResponse, Response},
    routing::post,
};
use dynamo_protocols::types::{
    ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionMessageContent,
    ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    ChatCompletionStreamResponseDeltaFunctionCall, CompletionUsage,
    CreateChatCompletionStreamResponse, FinishReason as OpenAIFinishReason, Role,
    ServiceTier as ChatServiceTier,
};
use serde::Serialize;
use std::sync::Arc;

pub(super) fn routes() -> Router<Arc<OpenAIHttpFrontend>> {
    Router::new().route("/v1/chat/completions", post(chat_completions))
}

async fn chat_completions(
    State(state): State<Arc<OpenAIHttpFrontend>>,
    body: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(error) => return json_rejection_response(error),
    };
    let stream = request.stream.unwrap_or(false);
    let (chat, context) = match prepare_request(&state.renderer, request).await {
        Ok(prepared) => prepared,
        Err(error) => return response_error(error, false),
    };
    let streams = match submit_generate_requests(&state.generate_client, chat.requests).await {
        Ok(streams) => streams,
        Err(error) => return response_error(error, stream),
    };
    let submitted = streams.into_iter().enumerate().collect();
    if stream {
        sse_response(
            chat_event_stream(submitted, chat.response_processor, context),
            serialize_chat_stream_response,
        )
    } else {
        match unary_chat(
            submitted,
            chat.response_processor,
            context.response_id,
            context.model,
            context.created,
            context.want_logprobs,
            context.service_tier,
        )
        .await
        {
            Ok(response) => Json(response).into_response(),
            Err(error) => response_error(error, false),
        }
    }
}

fn serialize_chat_stream_response(response: CreateChatCompletionStreamResponse) -> String {
    serde_json::to_string(&ChatStreamResponseWire::from(&response))
        .expect("OpenAI response must serialize")
}

/// The Dynamo response type omits an absent `reasoning_content`. SGLang's
/// streaming contract emits it explicitly as `null`, so use a borrowed wire
/// view instead of building and patching a `serde_json::Value` tree.
#[derive(Serialize)]
struct ChatStreamResponseWire<'a> {
    id: &'a str,
    choices: Vec<ChatChoiceStreamWire<'a>>,
    created: u32,
    model: &'a str,
    service_tier: &'a Option<ChatServiceTier>,
    system_fingerprint: &'a Option<String>,
    object: &'a str,
    usage: &'a Option<CompletionUsage>,
}

impl<'a> From<&'a CreateChatCompletionStreamResponse> for ChatStreamResponseWire<'a> {
    fn from(response: &'a CreateChatCompletionStreamResponse) -> Self {
        Self {
            id: &response.id,
            choices: response
                .choices
                .iter()
                .map(ChatChoiceStreamWire::from)
                .collect(),
            created: response.created,
            model: &response.model,
            service_tier: &response.service_tier,
            system_fingerprint: &response.system_fingerprint,
            object: &response.object,
            usage: &response.usage,
        }
    }
}

#[derive(Serialize)]
struct ChatChoiceStreamWire<'a> {
    index: u32,
    delta: ChatDeltaWire<'a>,
    finish_reason: &'a Option<OpenAIFinishReason>,
    logprobs: &'a Option<ChatChoiceLogprobs>,
}

impl<'a> From<&'a ChatChoiceStream> for ChatChoiceStreamWire<'a> {
    fn from(choice: &'a ChatChoiceStream) -> Self {
        Self {
            index: choice.index,
            delta: ChatDeltaWire::from(&choice.delta),
            finish_reason: &choice.finish_reason,
            logprobs: &choice.logprobs,
        }
    }
}

#[derive(Serialize)]
struct ChatDeltaWire<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a ChatCompletionMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<&'a ChatCompletionStreamResponseDeltaFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<&'a Vec<ChatCompletionMessageToolCallChunk>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'a Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refusal: Option<&'a String>,
    reasoning_content: Option<&'a str>,
}

impl<'a> From<&'a ChatCompletionStreamResponseDelta> for ChatDeltaWire<'a> {
    fn from(delta: &'a ChatCompletionStreamResponseDelta) -> Self {
        Self {
            content: delta.content.as_ref(),
            function_call: delta.function_call.as_ref(),
            tool_calls: delta.tool_calls.as_ref(),
            role: delta.role.as_ref(),
            refusal: delta.refusal.as_ref(),
            reasoning_content: delta.reasoning_content.as_deref(),
        }
    }
}
