//! SGLang-compatible prompt tokenization routes.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    response::{IntoResponse, Response},
    routing::post,
};
use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionTool, ChatCompletionToolChoiceOption,
    ReasoningEffort,
};
use futures::future::try_join_all;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{ChatRequest, RendererService};

use super::error::{error_payload, renderer_status};

pub(super) fn routes(renderer: Arc<RendererService>) -> Router<()> {
    Router::new()
        .route("/tokenize", post(tokenize))
        .route("/v1/tokenize", post(tokenize))
        .with_state(renderer)
}

async fn tokenize(
    State(renderer): State<Arc<RendererService>>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, Response> {
    let object = body
        .as_object()
        .ok_or_else(|| bad_request("request body must be a JSON object"))?;
    let has_prompt = object.get("prompt").is_some_and(|value| !value.is_null());
    let has_messages = object.get("messages").is_some_and(|value| !value.is_null());
    if has_prompt == has_messages {
        return Err(exactly_one_input());
    }
    let request = serde_json::from_value::<TokenizeRequest>(body)
        .map_err(|error| bad_request(error.to_string()))?;
    let (tokens, count) = match request {
        TokenizeRequest::Prompt(request) => {
            if request.messages.is_some() {
                return Err(exactly_one_input());
            }
            let add_special_tokens = request.add_special_tokens;
            let prompt = request.prompt;
            match prompt {
                TokenizePrompt::One(text) => {
                    let tokens = renderer
                        .tokenize_prompt(text, add_special_tokens)
                        .await
                        .map_err(renderer_error)?;
                    (json!(tokens), json!(tokens.len()))
                }
                TokenizePrompt::Many(texts) => {
                    let tokens = try_join_all(
                        texts
                            .into_iter()
                            .map(|text| renderer.tokenize_prompt(text, add_special_tokens)),
                    )
                    .await
                    .map_err(renderer_error)?;
                    let count = tokens.iter().map(Vec::len).collect::<Vec<_>>();
                    (json!(tokens), json!(count))
                }
            }
        }
        TokenizeRequest::Chat(request) => {
            if request.prompt.is_some() {
                return Err(exactly_one_input());
            }
            let request = request
                .into_chat(&renderer.config().served_model_name)
                .map_err(renderer_error)?;
            let tokens = renderer
                .tokenize_chat(request)
                .await
                .map_err(renderer_error)?;
            (json!(tokens), json!(tokens.len()))
        }
    };
    Ok(Json(json!({
        "tokens": tokens,
        "count": count,
        "max_model_len": renderer.config().limits.context_len,
    })))
}

#[derive(Deserialize)]
#[serde(untagged)]
enum TokenizeRequest {
    Prompt(TokenizePromptRequest),
    Chat(TokenizeChatRequest),
}

#[derive(Deserialize)]
struct TokenizePromptRequest {
    prompt: TokenizePrompt,
    #[serde(default)]
    messages: Option<Value>,
    #[serde(default = "default_true")]
    add_special_tokens: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum TokenizePrompt {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct TokenizeChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatCompletionRequestMessage>,
    #[serde(default)]
    prompt: Option<Value>,
    #[serde(default)]
    tools: Option<Vec<ChatCompletionTool>>,
    #[serde(default)]
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    #[serde(default)]
    reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    continue_final_message: bool,
    #[serde(default)]
    chat_template_kwargs: Option<std::collections::HashMap<String, Value>>,
}

impl TokenizeChatRequest {
    fn into_chat(self, served_model: &str) -> Result<ChatRequest, crate::RendererError> {
        let model = self.model.unwrap_or_else(|| served_model.to_owned());
        if model != served_model {
            return Err(format!("The model `{model}` does not exist").into());
        }
        Ok(ChatRequest {
            rid: "tokenize".into(),
            model,
            messages: self.messages,
            tools: self.tools,
            tool_choice: self.tool_choice,
            response_format: None,
            reasoning_effort: self.reasoning_effort,
            continue_final_message: self.continue_final_message,
            chat_template_args: self.chat_template_kwargs,
            sampling_params: Default::default(),
            choice_count: 1,
            stream: false,
            return_logprob: false,
            top_logprobs_num: 0,
            parallel_tool_calls: true,
            metadata: crate::GenerateRequestMetadata::default(),
        })
    }
}

const fn default_true() -> bool {
    true
}

fn exactly_one_input() -> Response {
    bad_request("Exactly one of 'prompt' or 'messages' must be provided.")
}

fn renderer_error(error: crate::RendererError) -> Response {
    let status = renderer_status(&error);
    (status, Json(error_payload(status, error.to_string()))).into_response()
}

fn bad_request(message: impl Into<String>) -> Response {
    let status = axum::http::StatusCode::BAD_REQUEST;
    (status, Json(error_payload(status, message))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    use crate::{
        PooledTokenizer, RendererConfig, RendererError, RendererLimits, SamplingDefaults,
        TextTokenizer,
    };

    struct PrefixTokenizer;

    impl TextTokenizer for PrefixTokenizer {
        fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, RendererError> {
            Ok(add_special_tokens
                .then_some(1)
                .into_iter()
                .chain(text.split_whitespace().map(|_| 7))
                .chain(add_special_tokens.then_some(2))
                .collect())
        }
    }

    fn app() -> Router<()> {
        let config = RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: ".".into(),
            revision: None,
            model_path: String::new(),
            chat_template: Some("chatml".into()),
            tool_call_parser: None,
            reasoning_parser: None,
            default_chat_template_kwargs: Default::default(),
            stream_response_default_include_usage: false,
            skip_tokenizer_init: false,
            vocab_size: 100,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                skip_tokenizer_init: false,
                vocab_size: 100,
                context_len: 64,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        };
        routes(Arc::new(RendererService::new(
            config,
            Arc::new(PooledTokenizer::new(Arc::new(PrefixTokenizer), 2, 2)),
        )))
    }

    async fn post(body: Value) -> (StatusCode, Value) {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/tokenize")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body =
            serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap())
                .unwrap();
        (status, body)
    }

    #[tokio::test]
    async fn prompt_tokenization_preserves_batch_shape_and_special_token_choice() {
        let (status, body) = post(json!({
            "prompt": ["one two", ""],
            "add_special_tokens": false
        }))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["tokens"], json!([[7, 7], []]));
        assert_eq!(body["count"], json!([2, 0]));

        let (_, body) = post(json!({"prompt": "one"})).await;
        assert_eq!(body["tokens"], json!([1, 7, 2]));
    }

    #[tokio::test]
    async fn chat_tokenization_applies_the_template_without_generation_limits() {
        let (status, body) = post(json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 10_000
        }))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            body["tokens"]
                .as_array()
                .is_some_and(|tokens| !tokens.is_empty())
        );
        assert_ne!(body["tokens"][0], json!(1));
        assert_ne!(
            body["tokens"][body["tokens"].as_array().unwrap().len() - 1],
            json!(2)
        );
        assert_eq!(
            body["count"],
            json!(body["tokens"].as_array().unwrap().len())
        );
    }

    #[tokio::test]
    async fn chat_tokenization_continues_the_final_assistant_message() {
        let (_, regular) = post(json!({
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "partial answer"}
            ]
        }))
        .await;
        let (status, continued) = post(json!({
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "partial answer"}
            ],
            "continue_final_message": true,
            "chat_template_kwargs": {
                "continue_final_message": false,
                "add_generation_prompt": true
            }
        }))
        .await;

        assert_eq!(status, StatusCode::OK);
        assert!(continued["count"].as_u64().unwrap() < regular["count"].as_u64().unwrap());
    }

    #[test]
    fn chat_tokenization_lowers_tokenize_specific_options() {
        let request: TokenizeChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "assistant", "content": "partial"}],
            "reasoning_effort": "high",
            "continue_final_message": true,
            "chat_template_kwargs": {"marker": true}
        }))
        .unwrap();

        let chat = request.into_chat("model").unwrap();

        assert!(chat.continue_final_message);
        assert_eq!(
            chat.chat_template_args
                .as_ref()
                .and_then(|args| args.get("marker")),
            Some(&json!(true))
        );
        assert_eq!(
            serde_json::to_value(chat.reasoning_effort).unwrap(),
            json!("high")
        );
    }
}
