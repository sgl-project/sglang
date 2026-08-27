//! Optional standalone HTTP frontend built on the reusable renderer service.

use std::{collections::HashMap, sync::Arc};

use axum::Router;
use dynamo_protocols::types::{
    ChatCompletionAudio, ChatCompletionFunctionCall, ChatCompletionFunctions,
    ChatCompletionRequestMessage, ChatCompletionStreamOptions, ChatCompletionTool,
    ChatCompletionToolChoiceOption, CreateChatCompletionRequest as DynamoChatCompletionRequest,
    CreateCompletionRequest as DynamoCompletionRequest, PredictionContent, Prompt, ReasoningEffort,
    ResponseFormat, ServiceTier, Stop, WebSearchOptions,
};
use serde::Deserialize;
use serde_json::Value;

use crate::{GenerateRequestMetadata, SamplingParamsOverrides};

mod error;
mod generate_client;
mod openai;
mod render;
mod runtime;
mod submission;
mod tokenize;

/// SGLang's chat-completions HTTP contract. Dynamo is only the temporary
/// lowering representation for shared OpenAI fields.
#[derive(Deserialize)]
pub(crate) struct ChatCompletionRequest {
    pub messages: Vec<ChatCompletionRequestMessage>,
    pub model: String,
    #[serde(default)]
    pub mm_processor_kwargs: Option<Value>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub n: Option<u8>,
    #[serde(default)]
    pub modalities: Option<Value>,
    #[serde(default)]
    pub prediction: Option<PredictionContent>,
    #[serde(default)]
    pub audio: Option<ChatCompletionAudio>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub service_tier: Option<ServiceTier>,
    #[serde(default)]
    pub stop: Option<Stop>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub tools: Option<Vec<ChatCompletionTool>>,
    #[serde(default)]
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub function_call: Option<ChatCompletionFunctionCall>,
    #[serde(default)]
    pub functions: Option<Vec<ChatCompletionFunctions>>,
    #[serde(default)]
    pub web_search_options: Option<WebSearchOptions>,
    #[serde(default)]
    pub chat_template_kwargs: Option<std::collections::HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub continue_final_message: bool,
    #[serde(flatten)]
    pub sampling_overrides: SamplingParamsOverrides,
    #[serde(flatten)]
    pub extensions: RequestExtensions,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, Value>,
}

/// SGLang's legacy-completions HTTP contract.
#[derive(Deserialize)]
pub(crate) struct CompletionRequest {
    pub model: String,
    pub prompt: Prompt,
    #[serde(default)]
    pub prompt_embeds: Option<String>,
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<u8>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(default)]
    pub logprobs: Option<u8>,
    #[serde(default)]
    pub echo: Option<bool>,
    #[serde(default)]
    pub stop: Option<Stop>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub best_of: Option<u8>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(flatten)]
    pub sampling_overrides: SamplingParamsOverrides,
    #[serde(flatten)]
    pub extensions: RequestExtensions,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, Value>,
}

pub(crate) struct ChatRequestParts {
    pub request: DynamoChatCompletionRequest,
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    pub continue_final_message: bool,
    pub sampling_overrides: SamplingParamsOverrides,
    pub extensions: RequestExtensions,
}

impl ChatCompletionRequest {
    #[allow(deprecated)]
    pub fn into_parts(self) -> Result<ChatRequestParts, String> {
        reject_unsupported_fields(&self.unsupported_fields)?;
        let modalities = self
            .modalities
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| format!("invalid modalities: {error}"))?;
        Ok(ChatRequestParts {
            request: DynamoChatCompletionRequest {
                messages: self.messages,
                model: self.model,
                mm_processor_kwargs: self.mm_processor_kwargs,
                store: self.store,
                reasoning_effort: self.reasoning_effort,
                metadata: self.metadata,
                frequency_penalty: self.frequency_penalty,
                logit_bias: self.logit_bias,
                logprobs: self.logprobs,
                top_logprobs: self.top_logprobs,
                max_tokens: self.max_tokens,
                max_completion_tokens: self.max_completion_tokens,
                n: self.n,
                modalities,
                prediction: self.prediction,
                audio: self.audio,
                presence_penalty: self.presence_penalty,
                response_format: self.response_format,
                seed: self.seed,
                service_tier: self.service_tier,
                stop: self.stop,
                stream: self.stream,
                stream_options: self.stream_options,
                temperature: self.temperature,
                top_p: self.top_p,
                tools: self.tools,
                tool_choice: self.tool_choice,
                parallel_tool_calls: self.parallel_tool_calls,
                user: self.user,
                function_call: self.function_call,
                functions: self.functions,
                web_search_options: self.web_search_options,
            },
            chat_template_kwargs: self.chat_template_kwargs,
            continue_final_message: self.continue_final_message,
            sampling_overrides: self.sampling_overrides,
            extensions: self.extensions,
        })
    }
}

impl CompletionRequest {
    pub fn into_parts(
        self,
    ) -> Result<
        (
            DynamoCompletionRequest,
            SamplingParamsOverrides,
            RequestExtensions,
        ),
        String,
    > {
        reject_unsupported_fields(&self.unsupported_fields)?;
        Ok((
            DynamoCompletionRequest {
                model: self.model,
                prompt: self.prompt,
                prompt_embeds: self.prompt_embeds,
                suffix: self.suffix,
                max_tokens: self.max_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                n: self.n,
                stream: self.stream,
                stream_options: self.stream_options,
                logprobs: self.logprobs,
                echo: self.echo,
                stop: self.stop,
                presence_penalty: self.presence_penalty,
                frequency_penalty: self.frequency_penalty,
                best_of: self.best_of,
                logit_bias: self.logit_bias,
                user: self.user,
                seed: self.seed,
            },
            self.sampling_overrides,
            self.extensions,
        ))
    }
}

fn reject_unsupported_fields(fields: &HashMap<String, Value>) -> Result<(), String> {
    if fields.is_empty() {
        return Ok(());
    }
    let mut names = fields.keys().cloned().collect::<Vec<_>>();
    names.sort_unstable();
    Err(format!(
        "unsupported request field{}: {}",
        if names.len() == 1 { "" } else { "s" },
        names.join(", ")
    ))
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct RequestExtensions {
    #[serde(default)]
    pub rid: Option<String>,
    #[serde(default)]
    pub cache_salt: Option<String>,
    #[serde(default)]
    pub extra_key: Option<String>,
    #[serde(default)]
    pub priority: Option<i64>,
    #[serde(default)]
    pub bootstrap_host: Option<String>,
    #[serde(default)]
    pub bootstrap_port: Option<i64>,
    #[serde(default)]
    pub bootstrap_room: Option<i64>,
    #[serde(default)]
    pub routed_dp_rank: Option<i64>,
    #[serde(default)]
    pub disagg_prefill_dp_rank: Option<i64>,
    #[serde(default)]
    pub data_parallel_rank: Option<i64>,
    #[serde(default)]
    pub session_id: Option<serde_json::Value>,
    #[serde(default)]
    pub session_params: Option<serde_json::Value>,
    #[serde(default)]
    pub lora_path: Option<serde_json::Value>,
    #[serde(default)]
    pub custom_logit_processor: Option<serde_json::Value>,
    #[serde(default)]
    pub image_data: Option<serde_json::Value>,
    #[serde(default)]
    pub video_data: Option<serde_json::Value>,
    #[serde(default)]
    pub audio_data: Option<serde_json::Value>,
    #[serde(default)]
    pub mm_hashes: Option<serde_json::Value>,
}

impl RequestExtensions {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("session_id", &self.session_id),
            ("session_params", &self.session_params),
            ("lora_path", &self.lora_path),
            ("custom_logit_processor", &self.custom_logit_processor),
            ("image_data", &self.image_data),
            ("video_data", &self.video_data),
            ("audio_data", &self.audio_data),
            ("mm_hashes", &self.mm_hashes),
        ] {
            if value.is_some() {
                return Err(format!(
                    "{name} is not supported by the text-only Rust frontend"
                ));
            }
        }
        Ok(())
    }

    pub fn metadata(self, model: String) -> GenerateRequestMetadata {
        GenerateRequestMetadata {
            model: Some(model),
            cache_salt: self.cache_salt,
            extra_key: self.extra_key,
            priority: self.priority,
            bootstrap_host: self.bootstrap_host,
            bootstrap_port: self.bootstrap_port,
            bootstrap_room: self.bootstrap_room,
            routed_dp_rank: self.routed_dp_rank.or(self.data_parallel_rank),
            disagg_prefill_dp_rank: self.disagg_prefill_dp_rank,
        }
    }
}

pub(crate) use generate_client::HttpGenerateClient;
pub use runtime::{RendererRuntimeConfig, serve};
#[cfg(test)]
mod test_utils;

pub(crate) struct OpenAIHttpFrontend {
    pub(crate) renderer: Arc<crate::RendererService>,
    pub(crate) generate_client: HttpGenerateClient,
}

impl OpenAIHttpFrontend {
    pub(crate) fn new(
        renderer: Arc<crate::RendererService>,
        generate_client: HttpGenerateClient,
    ) -> Self {
        Self {
            renderer,
            generate_client,
        }
    }
}

pub(crate) fn inference_routes(frontend: OpenAIHttpFrontend) -> Router<()> {
    let renderer = frontend.renderer.clone();
    Router::new()
        .merge(openai::routes())
        .with_state(Arc::new(frontend))
        .merge(tokenize::routes(renderer))
}

pub(crate) fn standalone_routes(frontend: OpenAIHttpFrontend) -> Router<()> {
    let renderer = frontend.renderer.clone();
    inference_routes(frontend).merge(render::routes(renderer))
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::{Arc, Mutex};

    use axum::{
        Json, Router,
        body::{Body, to_bytes},
        extract::State,
        http::{Request, StatusCode},
        response::sse::{Event, Sse},
        routing::post,
    };
    use futures::future::BoxFuture;
    use tower::ServiceExt;

    use super::{ChatCompletionRequest, HttpGenerateClient, OpenAIHttpFrontend, standalone_routes};
    use crate::{
        RendererConfig, RendererError, RendererLimits, RendererService, SamplingDefaults,
        TextRequest, TokenIdsRequest, TokenizationBackend,
    };

    #[test]
    fn chat_request_preserves_template_controls() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": false},
            "continue_final_message": true,
            "top_k": 17,
            "min_p": 0.2,
            "min_tokens": 3,
            "stop_regex": "END[0-9]",
            "ignore_eos": true,
            "skip_special_tokens": false
        }))
        .unwrap();

        let parts = request.into_parts().unwrap();

        assert_eq!(parts.request.model, "model");
        assert_eq!(
            parts
                .chat_template_kwargs
                .as_ref()
                .and_then(|args| args.get("enable_thinking")),
            Some(&serde_json::Value::Bool(false))
        );
        assert!(parts.continue_final_message);
        assert_eq!(parts.sampling_overrides.top_k, Some(17));
        assert_eq!(parts.sampling_overrides.min_p, Some(0.2));
        assert_eq!(parts.sampling_overrides.min_tokens, Some(3));
        assert_eq!(parts.sampling_overrides.ignore_eos, Some(true));
        assert_eq!(parts.sampling_overrides.skip_special_tokens, Some(false));
    }

    #[test]
    fn unsupported_sglang_fields_are_rejected_instead_of_ignored() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "input_ids": [1, 2, 3],
            "task": "domain"
        }))
        .unwrap();

        let error = match request.into_parts() {
            Err(error) => error,
            Ok(_) => panic!("unsupported fields must be rejected"),
        };

        assert_eq!(error, "unsupported request fields: input_ids, task");
    }

    struct WordTokenizer;

    impl TokenizationBackend for WordTokenizer {
        fn tokenize(
            &self,
            request: TextRequest,
        ) -> BoxFuture<'static, Result<TokenIdsRequest, RendererError>> {
            Box::pin(async move {
                Ok(TokenIdsRequest {
                    rid: request.rid,
                    input_ids: request
                        .prompt
                        .as_str()
                        .split_whitespace()
                        .map(|_| 7)
                        .collect(),
                    options: request.options,
                    metadata: request.metadata,
                })
            })
        }
    }

    #[derive(Clone)]
    struct EngineState {
        requests: Arc<Mutex<Vec<serde_json::Value>>>,
    }

    async fn generate(
        State(state): State<EngineState>,
        Json(body): Json<serde_json::Value>,
    ) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
        state.requests.lock().unwrap().push(body);
        let frame = serde_json::json!({
            "output_ids": [104],
            "meta_info": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "finish_reason": {"type": "stop", "matched": null}
            }
        })
        .to_string();
        Sse::new(futures::stream::iter([
            Ok(Event::default().data(frame)),
            Ok(Event::default().data("[DONE]")),
        ]))
    }

    fn renderer_config() -> RendererConfig {
        RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: ".".into(),
            revision: None,
            model_path: String::new(),
            chat_template: Some("chatml".into()),
            tool_call_parser: None,
            reasoning_parser: None,
            stream_response_default_include_usage: false,
            skip_tokenizer_init: false,
            vocab_size: 128,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                skip_tokenizer_init: false,
                vocab_size: 128,
                context_len: 128,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        }
    }

    fn tiny_tokenizer() -> dynamo_tokenizers::Tokenizer {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../experimental/sgl-router/tests/fixtures/tiny_tokenizer.json");
        dynamo_tokenizers::Tokenizer::from_file_with_options(
            path.to_str().unwrap(),
            dynamo_tokenizers::TokenizerOptions {
                add_special_tokens: false,
            },
        )
        .unwrap()
    }

    async fn post_request(
        app: Router<()>,
        uri: &str,
        body: &serde_json::Value,
    ) -> axum::response::Response {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn inference_and_render_share_chat_preparation() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let engine = tokio::spawn(
            axum::serve(
                listener,
                Router::new()
                    .route("/generate", post(generate))
                    .with_state(EngineState {
                        requests: captured.clone(),
                    }),
            )
            .into_future(),
        );
        let renderer = Arc::new(RendererService::new(
            renderer_config(),
            Arc::new(WordTokenizer),
        ));
        let client =
            HttpGenerateClient::new(format!("http://{address}"), tiny_tokenizer()).unwrap();
        let app = standalone_routes(OpenAIHttpFrontend::new(renderer, client));
        let body = serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello world"}],
            "rid": "chatcmpl-parity",
            "max_tokens": 8,
            "temperature": 0.4,
            "top_k": 17,
            "min_p": 0.2,
            "min_tokens": 3,
            "stop_regex": "END[0-9]",
            "ignore_eos": true,
            "skip_special_tokens": false,
            "chat_template_kwargs": {"enable_thinking": false},
            "cache_salt": "tenant-a",
            "extra_key": "interactive",
            "priority": 7,
            "bootstrap_host": "prefill",
            "bootstrap_port": 8998,
            "bootstrap_room": 42,
            "routed_dp_rank": 2,
            "disagg_prefill_dp_rank": 1
        });

        let render_response = post_request(app.clone(), "/v1/chat/completions/render", &body).await;
        assert_eq!(render_response.status(), StatusCode::OK);
        let mut rendered: serde_json::Value = serde_json::from_slice(
            &to_bytes(render_response.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        let inference_response = post_request(app, "/v1/chat/completions", &body).await;
        assert_eq!(inference_response.status(), StatusCode::OK);
        let engine_request = captured.lock().unwrap().pop().unwrap();
        engine.abort();
        assert!(engine_request.get("text").is_none());

        rendered["stream"] = serde_json::Value::Bool(true);
        rendered["return_text_in_logprobs"] = serde_json::Value::Bool(false);
        rendered["sampling_params"]["stop"] = serde_json::json!([]);
        rendered["incremental_streaming_output"] = serde_json::Value::Bool(true);
        assert_eq!(engine_request, rendered);
    }
}
