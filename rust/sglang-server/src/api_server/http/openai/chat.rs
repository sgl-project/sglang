//! OpenAI Chat Completions endpoint and chat-template preparation.

use std::sync::Arc;

use dynamo_parsers::{ToolChoice as DynamoToolChoice, ToolDefinition};
use dynamo_protocols::types::CreateChatCompletionRequest;
use futures::StreamExt;
use http::StatusCode;

use super::super::response::sse_encode;
use super::super::response::{HttpResponse, json_typed_response, read_json};
use super::{AppState, contains_media, openai_error, submit_generation, unix_seconds_u32};
use crate::api_server::core::guard::AbortGuard;
use crate::api_server::core::openai::chat::{
    SamplingDefaults, chat_event_stream, chat_sampling, chat_sse_payload, prepare_chat_request,
    unary_chat,
};
use crate::api_server::core::openai::tools::dynamo_tool_choice;
use crate::message::ids::Rid;
use crate::message::request::GenerateRequest;

pub(in crate::api_server) async fn chat_completions<B: http_body::Body>(
    state: Arc<AppState>,
    req: http::Request<B>,
) -> HttpResponse {
    let request = match read_json::<CreateChatCompletionRequest, _>(req).await {
        Ok(request) => request,
        Err(rejection) => {
            return openai_error(StatusCode::BAD_REQUEST, rejection.body_text, false);
        }
    };
    if request.model != state.server_args.served_model_name {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("The model `{}` does not exist", request.model),
            false,
        );
    }
    if request.messages.is_empty() {
        return openai_error(StatusCode::BAD_REQUEST, "messages cannot be empty", false);
    }
    if serde_json::to_value(&request.messages).is_ok_and(|messages| contains_media(&messages)) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "image, audio, video, and file message content is not supported",
            false,
        );
    }
    if request.n == Some(0) {
        return openai_error(StatusCode::BAD_REQUEST, "n must be at least 1", false);
    }
    #[allow(deprecated)]
    let max_tokens = request.max_completion_tokens.or(request.max_tokens);
    if max_tokens == Some(0) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "max_completion_tokens must be positive",
            false,
        );
    }
    if request.modalities.as_ref().is_some_and(|modalities| {
        serde_json::to_value(modalities).is_ok_and(|value| value.to_string().contains("\"audio\""))
    }) || request.audio.is_some()
        || request.prediction.is_some()
        || request.web_search_options.is_some()
        || request.mm_processor_kwargs.is_some()
    {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "audio, prediction, web search, and multimodal inputs are not supported",
            false,
        );
    }
    #[allow(deprecated)]
    if request.function_call.is_some() || request.functions.is_some() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "deprecated function_call/functions are not supported; use tools and tool_choice",
            false,
        );
    }

    let tool_choice = dynamo_tool_choice(&request.tool_choice);
    let tools_enabled = request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        && tool_choice != DynamoToolChoice::None;
    let parser = tools_enabled
        .then(|| state.server_args.tool_call_parser.clone())
        .flatten();
    if tools_enabled && parser.is_none() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "tool calls require --tool-call-parser",
            false,
        );
    }
    // Python gates the split on `request.separate_reasoning` (default true);
    // the Dynamo request type has no such field, so it is always on when the
    // server was launched with `--reasoning-parser`.
    let reasoning_parser = state.server_args.reasoning_parser.clone();
    let tools = request.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|tool| ToolDefinition {
                name: tool.function.name.clone(),
                parameters: tool.function.parameters.clone(),
                strict: tool.function.strict,
            })
            .collect::<Vec<_>>()
    });
    let tools_slice = tools.as_deref().unwrap_or_default();

    let (request, prompt) = match prepare_chat_request(&state, request).await {
        Ok(prepared) => prepared,
        Err(e) => return openai_error(e.http_status(), e.message, false),
    };

    let sampling = match chat_sampling(
        &request,
        SamplingDefaults::CHAT,
        parser.as_deref(),
        &tool_choice,
        tools_slice,
        request.parallel_tool_calls,
        &state.server_args,
    ) {
        Ok(sampling) => sampling,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, message, false);
        }
    };

    let stream = request.stream.unwrap_or(false);
    let n = request.n.unwrap_or(1) as usize;
    let want_logprobs = request.logprobs.unwrap_or(false);
    let parallel_tool_calls = request.parallel_tool_calls.unwrap_or(true);
    let stream_tool_choice = request.tool_choice.clone();
    let uses_tool_call_structural_tag = sampling.structural_tag.is_some();
    let service_tier = request.service_tier;
    let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = unix_seconds_u32();
    let model = request.model;
    let include_usage = request
        .stream_options
        .is_some_and(|options| options.include_usage)
        || state.server_args.stream_response_default_include_usage;
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut submitted = Vec::with_capacity(n);

    let mut prompt = Some(prompt);
    for index in 0..n {
        let rid = Rid::from_client(&format!("{response_id}-{index}"));
        let choice_prompt = if index + 1 == n {
            prompt.take().expect("last chat choice owns the prompt")
        } else {
            prompt
                .as_ref()
                .expect("chat prompt exists until the last choice")
                .clone()
        };
        let native = GenerateRequest {
            rid: rid.clone(),
            text: Some(choice_prompt),
            // Rendered templates own their special tokens — the pool must not
            // add another BOS/EOS (Python's `add_special_tokens=False`).
            skip_special_tokens: true,
            sampling_params: sampling.clone(),
            stream,
            return_logprob: want_logprobs,
            logprob_start_len: -1,
            top_logprobs_num: request.top_logprobs.unwrap_or(0) as i64,
            return_text_in_logprobs: want_logprobs.then_some(true),
            ..Default::default()
        };
        let rx = match submit_generation(&state, native, &mut guard).await {
            Ok(rx) => rx,
            Err(e) => return openai_error(e.http_status(), e.message, stream),
        };
        submitted.push((index, rid, rx));
    }

    if stream {
        let event_stream = chat_event_stream(
            submitted,
            guard,
            response_id,
            model,
            created,
            want_logprobs,
            include_usage,
            parser,
            reasoning_parser,
            tools,
            stream_tool_choice,
            uses_tool_call_structural_tag,
            parallel_tool_calls,
            service_tier,
        );
        sse_encode(event_stream.map(chat_sse_payload))
    } else {
        match unary_chat(
            submitted,
            guard,
            response_id,
            model,
            created,
            want_logprobs,
            parser,
            reasoning_parser,
            tools,
            parallel_tool_calls,
            service_tier,
        )
        .await
        {
            Ok(response) => json_typed_response(StatusCode::OK, &response),
            Err(e) => openai_error(e.http_status(), e.message, false),
        }
    }
}
