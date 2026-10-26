//! Ollama adapter shared by the HTTP routes and the gRPC service: it turns an
//! `api.v1` Ollama request into one native generate call and renders the
//! events back as `api.v1` Ollama frames. Python's `entrypoints/ollama/serving.py`
//! is the reference for every translation here.

use std::pin::Pin;
use std::time::Instant;

use futures::Stream;
use sglang_api_types::api::v1 as api;

use super::core::{CoreError, CoreEvent, CoreOutput};
use super::handlers::{CoreCall, CoreHandle};
use super::http::openai::{ChatFormatter, render_chat_prompt};
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::request::GenerateRequest;
use crate::message::sampling::SamplingParams;
use crate::message::wire::string_or_list;

/// Ollama users expect long answers; Python's handler defaults
/// `max_new_tokens` to this instead of SGLang's 128.
pub(crate) const DEFAULT_MAX_NEW_TOKENS: i64 = 2048;

const DIGEST: &str = "sha256:sglang0000000000000000000000000000000000000000000000000000000000";

pub(crate) type Frames<T> = Pin<Box<dyn Stream<Item = Result<T, CoreError>> + Send + 'static>>;

/// The per-listener facts an Ollama call needs; both transports build one
/// from their own state.
pub(crate) struct Ollama<'a> {
    pub(crate) core: &'a CoreHandle,
    pub(crate) chat_formatter: Option<&'a ChatFormatter>,
    pub(crate) server_args: &'a ServerArgs,
}

/// Python's `datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"`.
pub(crate) fn timestamp() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%S%.3fZ")
        .to_string()
}

/// Python's `_convert_options_to_sampling_params`: the eight Ollama options
/// it maps and the 2048-token default.
pub(crate) fn sampling_params(options: Option<&api::OllamaOptions>) -> SamplingParams {
    let mut sampling = SamplingParams::default();
    let options = options.cloned().unwrap_or_default();
    if let Some(temperature) = options.temperature {
        sampling.temperature = temperature;
    }
    if let Some(top_p) = options.top_p {
        sampling.top_p = top_p;
    }
    if let Some(top_k) = options.top_k {
        sampling.top_k = top_k;
    }
    sampling.max_new_tokens = Some(options.num_predict.unwrap_or(DEFAULT_MAX_NEW_TOKENS));
    if let Some(stop) = options.stop {
        sampling.stop = string_or_list(stop);
    }
    if let Some(presence_penalty) = options.presence_penalty {
        sampling.presence_penalty = presence_penalty;
    }
    if let Some(frequency_penalty) = options.frequency_penalty {
        sampling.frequency_penalty = frequency_penalty;
    }
    sampling.sampling_seed = options.seed;
    sampling
}

/// The token counts of a finished call, on the frame that carries `done`.
struct Counts {
    total_duration: i64,
    prompt_eval_count: i64,
    eval_count: i64,
}

fn nanos(started: Instant) -> i64 {
    i64::try_from(started.elapsed().as_nanos()).unwrap_or(i64::MAX)
}

impl Ollama<'_> {
    /// `/api/chat`: render the messages with the served chat template, as
    /// Python's `tokenizer.apply_chat_template(add_generation_prompt=True)`.
    pub(crate) async fn chat(
        &self,
        request: api::OllamaChatRequest,
    ) -> Result<Frames<api::OllamaChatResponse>, CoreError> {
        let Some(formatter) = self.chat_formatter else {
            return Err(CoreError::InvalidArgument(
                "this model has no usable chat template".into(),
            ));
        };
        let model = self.core.served_model_name().await;
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|message| serde_json::json!({"role": message.role, "content": message.content}))
            .collect();
        let chat_request =
            serde_json::from_value(serde_json::json!({"model": model, "messages": messages}))
                .map_err(|error| {
                    CoreError::InvalidArgument(format!("invalid messages: {error}"))
                })?;
        let prompt = render_chat_prompt(formatter, &chat_request).map_err(|error| {
            CoreError::InvalidArgument(format!("chat template render failed: {error}"))
        })?;
        let stream = request.stream_or_default();
        let call = self
            .submit(
                prompt,
                true,
                sampling_params(request.options.as_ref()),
                stream,
            )
            .await?;
        Ok(frames(call, stream, move |text, done, counts| {
            api::OllamaChatResponse {
                model: model.clone(),
                created_at: timestamp(),
                message: Some(api::OllamaMessage {
                    role: "assistant".into(),
                    content: text,
                    images: None,
                }),
                done,
                done_reason: done.then(|| "stop".into()),
                total_duration: counts.as_ref().map(|c| c.total_duration),
                load_duration: None,
                prompt_eval_count: counts.as_ref().map(|c| c.prompt_eval_count),
                prompt_eval_duration: None,
                eval_count: counts.as_ref().map(|c| c.eval_count),
                eval_duration: None,
            }
        }))
    }

    /// `/api/generate`: the raw prompt with `system` prepended, as Python
    /// does. An empty prompt answers a `done` frame without generating (the
    /// Ollama CLI probes with one on startup).
    pub(crate) async fn generate(
        &self,
        request: api::OllamaGenerateRequest,
    ) -> Result<Frames<api::OllamaGenerateResponse>, CoreError> {
        let model = self.core.served_model_name().await;
        let prompt = match &request.system {
            Some(system) => format!("{system}\n\n{}", request.prompt),
            None => request.prompt.clone(),
        };
        let frame =
            move |text: String, done: bool, counts: Option<Counts>| api::OllamaGenerateResponse {
                model: model.clone(),
                created_at: timestamp(),
                response: text,
                done,
                done_reason: done.then(|| "stop".into()),
                context: None,
                total_duration: counts.as_ref().map(|c| c.total_duration),
                load_duration: None,
                prompt_eval_count: counts.as_ref().map(|c| c.prompt_eval_count),
                prompt_eval_duration: None,
                eval_count: counts.as_ref().map(|c| c.eval_count),
                eval_duration: None,
            };
        if prompt.trim().is_empty() {
            let empty = frame(String::new(), true, None);
            return Ok(Box::pin(futures::stream::once(async move { Ok(empty) })));
        }
        let stream = request.stream_or_default();
        let call = self
            .submit(
                prompt,
                false,
                sampling_params(request.options.as_ref()),
                stream,
            )
            .await?;
        Ok(frames(call, stream, frame))
    }

    /// `/api/tags`: one card for the served model, with Python's placeholder
    /// size and digest.
    pub(crate) async fn tags(&self) -> api::OllamaTagsResponse {
        let model = self.core.served_model_name().await;
        api::OllamaTagsResponse {
            models: vec![api::OllamaModelInfo {
                name: model.clone(),
                model: model.clone(),
                modified_at: timestamp(),
                size: 0,
                digest: DIGEST.into(),
                details: Some(api::OllamaModelDetails {
                    parent_model: None,
                    format: "sglang".into(),
                    family: family(&model).into(),
                    families: Vec::new(),
                    parameter_size: "unknown".into(),
                    quantization_level: None,
                }),
            }],
        }
    }

    /// `/api/show`: Python's synthesized model card around the context length.
    pub(crate) fn show(&self, request: api::OllamaShowRequest) -> api::OllamaShowResponse {
        let model = request.model;
        let mut family = family(&model).to_owned();
        for suffix in ["-Instruct", "-Chat", "-Base"] {
            if let Some(stripped) = family.strip_suffix(suffix) {
                family = stripped.to_owned();
                break;
            }
        }
        let context_len = self.server_args.model_config.context_len;
        let model_info = serde_json::json!({
            "general.architecture": family,
            "general.name": model,
            "general.parameter_count": 0,
            format!("{family}.context_length"): context_len,
            format!("{family}.block_count"): 0,
            format!("{family}.embedding_length"): 0,
            format!("{family}.attention.head_count"): 0,
        });
        api::OllamaShowResponse {
            license: String::new(),
            modelfile: format!("FROM {model}\nPARAMETER num_ctx {context_len}\n"),
            parameters: format!("num_ctx {context_len}"),
            template: String::new(),
            modified_at: timestamp(),
            details: Some(api::OllamaModelDetails {
                parent_model: Some(String::new()),
                format: "sglang".into(),
                family: family.clone(),
                families: vec![family],
                parameter_size: "unknown".into(),
                quantization_level: Some(String::new()),
            }),
            model_info: Some(model_info.to_string()),
            capabilities: vec!["completion".into()],
        }
    }

    async fn submit(
        &self,
        prompt: String,
        rendered_template: bool,
        mut sampling_params: SamplingParams,
        stream: bool,
    ) -> Result<CoreCall, CoreError> {
        sampling_params
            .normalize(
                self.server_args.skip_tokenizer_init,
                self.server_args.model_config.vocab_size,
            )
            .map_err(|error| CoreError::InvalidArgument(error.to_string()))?;
        let request = GenerateRequest {
            rid: Rid::new(),
            text: Some(prompt),
            // A rendered template owns its special tokens (Python tokenizes it
            // with `apply_chat_template`, no extra BOS); a raw prompt gets them.
            skip_special_tokens: rendered_template,
            sampling_params,
            stream,
            ..Default::default()
        };
        self.core.generate(request).await
    }
}

fn family(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

/// Streamed: one frame per delta, then the `done` frame with the counts.
/// Unary: the one `done` frame with the whole text. Runtime deltas are
/// already incremental, so each frame's text is the new text, as Python
/// computes by slicing the cumulative text.
fn frames<T: Send + 'static>(
    mut call: CoreCall,
    stream: bool,
    frame: impl Fn(String, bool, Option<Counts>) -> T + Send + 'static,
) -> Frames<T> {
    let started = Instant::now();
    Box::pin(async_stream::stream! {
        let mut output = CoreOutput::default();
        loop {
            match call.recv().await {
                Some(CoreEvent::Delta(delta)) => {
                    output.append_delta(&delta);
                    if stream {
                        yield Ok(frame(delta.text, false, None));
                    }
                }
                Some(CoreEvent::Finished(delta)) => {
                    output.append_delta(&delta);
                    if stream && !delta.text.is_empty() {
                        yield Ok(frame(delta.text, false, None));
                    }
                    let counts = Counts {
                        total_duration: nanos(started),
                        prompt_eval_count: i64::from(output.prompt_tokens),
                        eval_count: i64::try_from(output.completion_tokens).unwrap_or(i64::MAX),
                    };
                    let text = if stream { String::new() } else { output.text };
                    yield Ok(frame(text, true, Some(counts)));
                    return;
                }
                Some(CoreEvent::Failed(error)) => {
                    yield Err(error);
                    return;
                }
                None => {
                    yield Err(CoreError::ResponseTruncated);
                    return;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::types::OneOrMany;

    /// Options map by Python's table, `num_predict` becomes `max_new_tokens`,
    /// and the 2048 default applies only when the client sent none.
    #[test]
    fn options_map_to_sampling_params_with_the_ollama_default() {
        let sampling = sampling_params(None);
        assert_eq!(sampling.max_new_tokens, Some(DEFAULT_MAX_NEW_TOKENS));
        let options: api::OllamaOptions = serde_json::from_value(serde_json::json!({
            "num_predict": 7, "temperature": 0.5, "top_k": 3, "stop": ["x"], "seed": 9,
            "num_ctx": 4096
        }))
        .unwrap();
        let sampling = sampling_params(Some(&options));
        assert_eq!(sampling.max_new_tokens, Some(7));
        assert_eq!(sampling.temperature, 0.5);
        assert_eq!(sampling.top_k, 3);
        assert_eq!(sampling.stop, Some(OneOrMany::Many(vec!["x".into()])));
        assert_eq!(sampling.sampling_seed, Some(9));
        assert!(
            serde_json::from_value::<api::OllamaOptions>(serde_json::json!({"temperature": "hot"}))
                .is_err()
        );
    }

    /// `/api/show` strips the variant suffix for the family and keys the
    /// model info by it; `/api/tags` is not async-free so it is covered by
    /// the transports' tests.
    #[test]
    fn show_synthesizes_the_card_from_the_name_and_context_length() {
        let core = crate::api_server::http::openai::test_utils::core_handle();
        let server_args = ServerArgs {
            model_config: crate::message::config::ModelConfig {
                context_len: 4096,
                ..Default::default()
            },
            ..Default::default()
        };
        let ollama = Ollama {
            core: &core,
            chat_formatter: None,
            server_args: &server_args,
        };
        let card = ollama.show(api::OllamaShowRequest {
            model: "org/Llama-3-Instruct".into(),
        });
        let json = serde_json::to_value(&card).unwrap();
        assert_eq!(json["details"]["family"], "Llama-3");
        assert_eq!(json["details"]["families"], serde_json::json!(["Llama-3"]));
        assert_eq!(json["model_info"]["general.name"], "org/Llama-3-Instruct");
        assert_eq!(json["model_info"]["Llama-3.context_length"], 4096);
        assert_eq!(json["capabilities"], serde_json::json!(["completion"]));
        assert_eq!(json["parameters"], "num_ctx 4096");
    }
}
