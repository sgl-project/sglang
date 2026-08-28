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

use crate::{GenerateRequestMetadata, OneOrMany, OneOrManyItem, SamplingParamsOverrides};

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
    pub rid: Option<OneOrMany<String>>,
    #[serde(default)]
    pub cache_salt: Option<OneOrMany<String>>,
    #[serde(default)]
    pub extra_key: Option<OneOrMany<String>>,
    #[serde(default)]
    pub priority: Option<i64>,
    #[serde(default)]
    pub bootstrap_host: Option<OneOrMany<String>>,
    #[serde(default)]
    pub bootstrap_port: Option<OneOrMany<Option<i64>>>,
    #[serde(default)]
    pub bootstrap_room: Option<OneOrMany<i64>>,
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

#[derive(Debug)]
pub(crate) struct ExpandedRequestContext {
    pub request_id: String,
    pub metadata: GenerateRequestMetadata,
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

    pub fn response_id(&self, prefix: &str) -> String {
        match self.rid.as_ref() {
            Some(OneOrMany::One(rid)) => rid.clone(),
            Some(OneOrMany::Many(rids)) => rids
                .first()
                .cloned()
                .unwrap_or_else(|| generated_response_id(prefix)),
            None => generated_response_id(prefix),
        }
    }

    pub fn expand(
        self,
        model: String,
        prompt_count: usize,
        choice_count: usize,
        response_id: &str,
    ) -> Result<Vec<ExpandedRequestContext>, String> {
        let list_rids = matches!(&self.rid, Some(OneOrMany::Many(_)));
        let rids = expand_per_prompt("rid", self.rid, prompt_count)?;
        if list_rids {
            let mut seen = std::collections::HashSet::new();
            for rid in rids.iter().flatten() {
                if !seen.insert(rid) {
                    return Err(format!("duplicate request ID in rid: {rid}"));
                }
            }
        }
        let cache_salts = expand_per_prompt("cache_salt", self.cache_salt, prompt_count)?;
        let extra_keys = expand_per_prompt("extra_key", self.extra_key, prompt_count)?;
        let bootstrap_hosts =
            expand_per_prompt("bootstrap_host", self.bootstrap_host, prompt_count)?;
        let bootstrap_ports =
            expand_per_prompt("bootstrap_port", self.bootstrap_port, prompt_count)?;
        let bootstrap_rooms = match self.bootstrap_room {
            Some(OneOrMany::One(base)) => (0..prompt_count)
                .map(|prompt_index| {
                    let offset = i64::try_from(prompt_index)
                        .map_err(|_| "bootstrap_room prompt index exceeds i64".to_owned())?;
                    base.checked_add(offset)
                        .map(Some)
                        .ok_or_else(|| "bootstrap_room overflows i64".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?,
            value => expand_per_prompt("bootstrap_room", value, prompt_count)?,
        };
        let routed_dp_rank = self.routed_dp_rank.or(self.data_parallel_rank);
        let total = prompt_count
            .checked_mul(choice_count)
            .ok_or_else(|| "prompt count times n overflows usize".to_owned())?;
        let mut contexts = Vec::with_capacity(total);
        for prompt_index in 0..prompt_count {
            for sample_index in 0..choice_count {
                let index = prompt_index * choice_count + sample_index;
                let request_id = match (&rids[prompt_index], list_rids) {
                    (Some(rid), true) if choice_count == 1 => rid.clone(),
                    (Some(rid), true) => format!("{rid}-{sample_index}"),
                    _ => format!("{response_id}-{index}"),
                };
                contexts.push(ExpandedRequestContext {
                    request_id,
                    metadata: GenerateRequestMetadata {
                        model: Some(model.clone()),
                        cache_salt: cache_salts[prompt_index]
                            .clone()
                            .filter(|value| !value.is_empty()),
                        extra_key: extra_keys[prompt_index]
                            .clone()
                            .filter(|value| !value.is_empty()),
                        priority: self.priority,
                        bootstrap_host: bootstrap_hosts[prompt_index].clone(),
                        bootstrap_port: bootstrap_ports[prompt_index].flatten(),
                        bootstrap_room: bootstrap_rooms[prompt_index],
                        routed_dp_rank,
                        disagg_prefill_dp_rank: self.disagg_prefill_dp_rank,
                    },
                });
            }
        }
        Ok(contexts)
    }
}

fn expand_per_prompt<T: Clone + OneOrManyItem>(
    name: &str,
    value: Option<OneOrMany<T>>,
    prompt_count: usize,
) -> Result<Vec<Option<T>>, String> {
    match value {
        None => Ok(vec![None; prompt_count]),
        Some(OneOrMany::One(value)) => Ok(vec![Some(value); prompt_count]),
        Some(OneOrMany::Many(values)) if values.len() == prompt_count => {
            Ok(values.into_iter().map(Some).collect())
        }
        Some(OneOrMany::Many(values)) => Err(format!(
            "the length of {name} must equal the prompt batch size ({prompt_count}), got {}",
            values.len()
        )),
    }
}

fn generated_response_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
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

    use super::{
        ChatCompletionRequest, CompletionRequest, HttpGenerateClient, OpenAIHttpFrontend,
        RequestExtensions, standalone_routes,
    };
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

    #[test]
    fn batched_request_metadata_expands_in_prompt_major_order() {
        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "prompt": ["one", "two"],
            "n": 2,
            "rid": ["prompt-a", "prompt-b"],
            "cache_salt": ["tenant-a", "tenant-b"],
            "extra_key": ["", "batch"],
            "bootstrap_host": ["prefill-a", "prefill-b"],
            "bootstrap_port": [8998, null],
            "bootstrap_room": [41, 52],
            "priority": 7,
            "routed_dp_rank": 2
        }))
        .unwrap();
        let (_, _, extensions) = request.into_parts().unwrap();
        let response_id = extensions.response_id("cmpl");

        let contexts = extensions
            .expand("model".into(), 2, 2, &response_id)
            .unwrap();

        assert_eq!(response_id, "prompt-a");
        assert_eq!(
            contexts
                .iter()
                .map(|context| context.request_id.as_str())
                .collect::<Vec<_>>(),
            ["prompt-a-0", "prompt-a-1", "prompt-b-0", "prompt-b-1"]
        );
        assert_eq!(contexts[0].metadata.cache_salt.as_deref(), Some("tenant-a"));
        assert_eq!(contexts[1].metadata.extra_key, None);
        assert_eq!(contexts[2].metadata.extra_key.as_deref(), Some("batch"));
        assert_eq!(contexts[0].metadata.bootstrap_port, Some(8998));
        assert_eq!(contexts[2].metadata.bootstrap_port, None);
        assert_eq!(contexts[1].metadata.bootstrap_room, Some(41));
        assert_eq!(contexts[3].metadata.bootstrap_room, Some(52));
        assert_eq!(contexts[3].metadata.routed_dp_rank, Some(2));
    }

    #[test]
    fn batch_metadata_validates_lengths_duplicates_and_scalar_rooms() {
        let extensions: RequestExtensions = serde_json::from_value(serde_json::json!({
            "rid": ["duplicate", "duplicate"],
            "cache_salt": ["only-one"]
        }))
        .unwrap();
        let error = extensions
            .expand("model".into(), 2, 1, "cmpl-test")
            .unwrap_err();
        assert!(error.contains("duplicate request ID"));

        let extensions: RequestExtensions = serde_json::from_value(serde_json::json!({
            "cache_salt": ["only-one"]
        }))
        .unwrap();
        let error = extensions
            .expand("model".into(), 2, 1, "cmpl-test")
            .unwrap_err();
        assert!(error.contains("prompt batch size (2)"));

        let extensions: RequestExtensions = serde_json::from_value(serde_json::json!({
            "bootstrap_room": 90
        }))
        .unwrap();
        let contexts = extensions
            .expand("model".into(), 2, 2, "cmpl-test")
            .unwrap();
        assert_eq!(
            contexts
                .iter()
                .map(|context| context.metadata.bootstrap_room)
                .collect::<Vec<_>>(),
            [Some(90), Some(90), Some(91), Some(91)]
        );
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
            default_chat_template_kwargs: Default::default(),
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
    async fn inference_and_render_share_request_preparation() {
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

        let inference_response = post_request(app.clone(), "/v1/chat/completions", &body).await;
        assert_eq!(inference_response.status(), StatusCode::OK);
        let engine_request = captured.lock().unwrap().pop().unwrap();
        assert!(engine_request.get("text").is_none());

        rendered["stream"] = serde_json::Value::Bool(true);
        rendered["return_text_in_logprobs"] = serde_json::Value::Bool(false);
        rendered["sampling_params"]["stop"] = serde_json::json!([]);
        rendered["incremental_streaming_output"] = serde_json::Value::Bool(true);
        assert_eq!(engine_request, rendered);

        let batch = serde_json::json!({
            "model": "model",
            "prompt": ["one", "two"],
            "n": 2,
            "rid": ["prompt-a", "prompt-b"],
            "cache_salt": ["tenant-a", "tenant-b"],
            "extra_key": ["interactive", "batch"],
            "bootstrap_host": ["prefill-a", "prefill-b"],
            "bootstrap_port": [8998, null],
            "bootstrap_room": [41, 52]
        });
        let render_response = post_request(app.clone(), "/v1/completions/render", &batch).await;
        assert_eq!(render_response.status(), StatusCode::OK);
        let rendered: serde_json::Value = serde_json::from_slice(
            &to_bytes(render_response.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(rendered[0]["rid"], "prompt-a-0");
        assert_eq!(rendered[1]["rid"], "prompt-a-1");
        assert_eq!(rendered[2]["rid"], "prompt-b-0");
        assert_eq!(rendered[3]["rid"], "prompt-b-1");
        assert_eq!(rendered[3]["cache_salt"], "tenant-b");
        assert_eq!(rendered[3]["bootstrap_room"], 52);

        let inference_response = post_request(app, "/v1/completions", &batch).await;
        assert_eq!(inference_response.status(), StatusCode::OK);
        let mut engine_requests = std::mem::take(&mut *captured.lock().unwrap());
        engine.abort();
        engine_requests.sort_by(|left, right| left["rid"].as_str().cmp(&right["rid"].as_str()));
        assert_eq!(engine_requests.len(), 4);
        assert_eq!(engine_requests[0]["rid"], "prompt-a-0");
        assert_eq!(engine_requests[1]["rid"], "prompt-a-1");
        assert_eq!(engine_requests[2]["rid"], "prompt-b-0");
        assert_eq!(engine_requests[3]["rid"], "prompt-b-1");
        assert_eq!(engine_requests[2]["cache_salt"], "tenant-b");
        assert_eq!(engine_requests[2]["bootstrap_host"], "prefill-b");
        assert_eq!(
            engine_requests[2]["bootstrap_port"],
            serde_json::Value::Null
        );
        assert_eq!(engine_requests[3]["bootstrap_room"], 52);
    }
}
