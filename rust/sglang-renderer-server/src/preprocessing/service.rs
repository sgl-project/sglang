//! Reusable SGLang request preprocessing.

use std::sync::Arc;

use futures::future::try_join_all;
use sglang_processor::{
    ChatConfig, ChatPreprocessor, ChatRequest, ChatResponseProcessor, TextTokenizer, ToolConstraint,
};

use super::tokenizer::{
    PooledTokenizer, check_total_tokens, validate_request_id, validate_text_request,
    validate_token_ids_request,
};
use crate::{
    GenerateRequest, GenerateRequestIdentity, GenerateRequestMetadata, GenerationOptions,
    OneOrMany, RendererConfig, RendererError, SamplingParams, TextRequest, TextRequestGroup,
    TokenIdsRequest,
};

/// A chat request plus the generation settings submitted with each choice.
#[derive(Debug, Clone)]
pub struct ChatGenerateRequest {
    pub rid: String,
    pub chat: ChatRequest,
    pub sampling_params: SamplingParams,
    pub choice_count: usize,
    pub stream: bool,
    pub return_logprob: bool,
    pub top_logprobs_num: i64,
    pub metadata: GenerateRequestMetadata,
}

/// Chat-to-text result plus the state needed to interpret generated output.
pub(crate) struct LoweredChat {
    pub text_requests: Vec<TextRequestGroup>,
    pub response_processor: ChatResponseProcessor,
}

/// Shared preprocessing used by inference and render-only frontends.
pub struct RendererService {
    config: RendererConfig,
    chat_preprocessor: ChatPreprocessor,
    tokenizer: PooledTokenizer,
}

/// Prepared token-only chat requests plus the state needed to interpret their
/// generated output.
pub struct PreparedChat {
    pub requests: Vec<GenerateRequest>,
    pub response_processor: ChatResponseProcessor,
}

impl RendererService {
    pub fn with_tokenizer(
        config: RendererConfig,
        tokenizer: Arc<dyn TextTokenizer>,
        worker_count: usize,
        queue_capacity: usize,
    ) -> Self {
        let chat_preprocessor = ChatPreprocessor::load(&chat_config(&config));
        Self {
            config,
            chat_preprocessor,
            tokenizer: PooledTokenizer::new(tokenizer, worker_count, queue_capacity),
        }
    }

    pub fn config(&self) -> &RendererConfig {
        &self.config
    }

    pub(crate) fn preprocess_chat(
        &self,
        request: ChatGenerateRequest,
    ) -> Result<LoweredChat, RendererError> {
        let ChatGenerateRequest {
            rid,
            chat,
            mut sampling_params,
            choice_count,
            stream,
            return_logprob,
            top_logprobs_num,
            metadata,
        } = request;
        if choice_count == 0 {
            return Err("choice_count must be at least 1".into());
        }
        let rendered = self.chat_preprocessor.preprocess(chat)?;
        merge_template_stops(&mut sampling_params, rendered.template_stops.clone());
        if rendered.tool_calls_enabled {
            sampling_params.skip_special_tokens = false;
        }
        match &rendered.tool_constraint {
            Some(ToolConstraint::StructuralTag(tag)) => {
                sampling_params.structural_tag = Some(tag.clone());
            }
            Some(ToolConstraint::JsonSchema(schema)) => {
                sampling_params.json_schema = Some(schema.clone());
            }
            None => {}
        }
        let response_processor =
            rendered.response_processor(choice_count, sampling_params.structural_tag.is_some());

        let options = GenerationOptions {
            sampling_params,
            require_reasoning: rendered.require_reasoning,
            stream,
            return_logprob,
            logprob_start_len: -1,
            top_logprobs_num,
            return_text_in_logprobs: return_logprob.then_some(true),
            ..Default::default()
        };
        let requests = (0..choice_count)
            .map(|index| GenerateRequestIdentity {
                rid: format!("{rid}-{index}"),
                metadata: metadata.clone(),
            })
            .collect();
        Ok(LoweredChat {
            text_requests: vec![TextRequestGroup {
                prompt: rendered.prompt,
                add_special_tokens: false,
                options,
                requests,
            }],
            response_processor,
        })
    }

    pub async fn prepare_chat(
        &self,
        request: ChatGenerateRequest,
    ) -> Result<PreparedChat, RendererError> {
        let lowered = self.preprocess_chat(request)?;
        Ok(PreparedChat {
            requests: self
                .prepare_text_request_groups(lowered.text_requests)
                .await?,
            response_processor: lowered.response_processor,
        })
    }

    pub async fn prepare_text_request_groups(
        &self,
        groups: Vec<TextRequestGroup>,
    ) -> Result<Vec<GenerateRequest>, RendererError> {
        let groups = try_join_all(
            groups
                .into_iter()
                .map(|group| async move { self.prepare_text_request_group(group).await }),
        )
        .await?;
        Ok(groups.into_iter().flatten().collect())
    }

    pub async fn tokenize_prompt(
        &self,
        text: String,
        add_special_tokens: bool,
    ) -> Result<crate::TokenIds, RendererError> {
        let request = TextRequest::text("tokenize", text, add_special_tokens, Default::default());
        Ok(self.tokenizer.tokenize(request).await?.input_ids)
    }

    pub async fn tokenize_chat(&self, chat: ChatRequest) -> Result<crate::TokenIds, RendererError> {
        let prompt = self.chat_preprocessor.render_prompt(chat)?;
        let request = TextRequest::rendered("tokenize", prompt, false, Default::default());
        Ok(self.tokenizer.tokenize(request).await?.input_ids)
    }

    pub fn prepare_token_ids_requests(
        &self,
        requests: Vec<TokenIdsRequest>,
    ) -> Result<Vec<GenerateRequest>, RendererError> {
        requests
            .into_iter()
            .map(|request| self.prepare_token_ids(request).map(GenerateRequest::from))
            .collect()
    }

    async fn prepare_text(
        &self,
        mut request: TextRequest,
    ) -> Result<TokenIdsRequest, RendererError> {
        validate_text_request(&request, &self.config.limits)?;
        request
            .options
            .sampling_params
            .normalize(self.config.limits.vocab_size)?;
        let mut request = self.tokenizer.tokenize(request).await?;
        check_total_tokens(&mut request, &self.config.limits)?;
        Ok(request)
    }

    async fn prepare_text_request_group(
        &self,
        group: TextRequestGroup,
    ) -> Result<Vec<GenerateRequest>, RendererError> {
        let TextRequestGroup {
            prompt,
            add_special_tokens,
            options,
            requests,
        } = group;
        for request in &requests {
            validate_request_id(&request.rid)?;
        }
        let mut requests = requests.into_iter();
        let first = requests
            .next()
            .ok_or_else(|| RendererError::from("text request group must contain a request"))?;
        let tokenized = self
            .prepare_text(TextRequest {
                rid: first.rid,
                prompt,
                add_special_tokens,
                options,
                metadata: first.metadata,
            })
            .await?;
        let additional = requests
            .map(|request| {
                GenerateRequest::from(TokenIdsRequest {
                    rid: request.rid,
                    input_ids: tokenized.input_ids.clone(),
                    options: tokenized.options.clone(),
                    metadata: request.metadata,
                })
            })
            .collect::<Vec<_>>();
        let mut prepared = Vec::with_capacity(1 + additional.len());
        prepared.push(tokenized.into());
        prepared.extend(additional);
        Ok(prepared)
    }

    fn prepare_token_ids(
        &self,
        mut request: TokenIdsRequest,
    ) -> Result<TokenIdsRequest, RendererError> {
        validate_token_ids_request(&request, &self.config.limits)?;
        request
            .options
            .sampling_params
            .normalize(self.config.limits.vocab_size)?;
        check_total_tokens(&mut request, &self.config.limits)?;
        Ok(request)
    }
}

fn chat_config(config: &RendererConfig) -> ChatConfig {
    ChatConfig {
        tokenizer_path: config.tokenizer_path.clone(),
        model_path: config.model_path.clone(),
        revision: config.revision.clone(),
        chat_template: config.chat_template.clone(),
        tool_call_parser: config.tool_call_parser.clone(),
        reasoning_parser: config.reasoning_parser.clone(),
        default_chat_template_kwargs: config.default_chat_template_kwargs.clone(),
    }
}

fn merge_template_stops(sampling: &mut SamplingParams, template_stops: Option<Vec<String>>) {
    let Some(mut stops) = template_stops else {
        return;
    };
    if let Some(request_stops) = sampling.stop.take() {
        match request_stops {
            OneOrMany::One(stop) => stops.push(stop),
            OneOrMany::Many(request_stops) => stops.extend(request_stops),
        }
    }
    sampling.stop = Some(OneOrMany::Many(stops));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RendererLimits, SamplingDefaults};
    use sglang_processor::ProcessorError;
    use sglang_processor::dynamo_protocols::types::ChatCompletionRequestMessage;
    use sglang_processor::dynamo_renderer::RenderedPrompt;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn model_config(model_path: String) -> RendererConfig {
        RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: model_path.clone(),
            revision: None,
            model_path,
            chat_template: None,
            tool_call_parser: None,
            reasoning_parser: None,
            default_chat_template_kwargs: Default::default(),
            stream_response_default_include_usage: false,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                vocab_size: 128,
                context_len: 128,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        }
    }

    struct UnexpectedTokenizer;

    impl TextTokenizer for UnexpectedTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_special_tokens: bool,
        ) -> Result<crate::TokenIds, ProcessorError> {
            panic!("token-ID input must not enter the tokenizer")
        }
    }

    struct CountingTokenizer {
        calls: Arc<AtomicUsize>,
    }

    impl TextTokenizer for CountingTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> Result<crate::TokenIds, ProcessorError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(text.split_whitespace().map(|_| 7).collect())
        }
    }

    #[test]
    fn text_choices_tokenize_once_per_prompt() {
        futures::executor::block_on(async {
            let calls = Arc::new(AtomicUsize::new(0));
            let service = RendererService::with_tokenizer(
                model_config(String::new()),
                Arc::new(CountingTokenizer {
                    calls: calls.clone(),
                }),
                2,
                4,
            );
            let group = |prompt: &str, ids: &[&str]| TextRequestGroup {
                prompt: RenderedPrompt::text(prompt.to_owned()),
                add_special_tokens: true,
                options: GenerationOptions {
                    sampling_params: SamplingParams {
                        max_new_tokens: Some(4),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                requests: ids
                    .iter()
                    .map(|rid| GenerateRequestIdentity {
                        rid: (*rid).to_owned(),
                        metadata: GenerateRequestMetadata::default(),
                    })
                    .collect(),
            };

            let prepared = service
                .prepare_text_request_groups(vec![
                    group("one two", &["a-0", "a-1", "a-2"]),
                    group("three", &["b-0", "b-1"]),
                ])
                .await
                .unwrap();

            assert_eq!(calls.load(Ordering::Relaxed), 2);
            assert_eq!(
                prepared
                    .iter()
                    .map(|request| request.rid.as_str())
                    .collect::<Vec<_>>(),
                ["a-0", "a-1", "a-2", "b-0", "b-1"]
            );
            assert_eq!(prepared[0].input_ids, [7, 7]);
            assert_eq!(prepared[2].input_ids, [7, 7]);
            assert_eq!(prepared[3].input_ids, [7]);
        });
    }

    #[test]
    fn chat_lowering_carries_rendered_prompt_and_template_stops() {
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
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                vocab_size: 128,
                context_len: 128,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        };
        let messages: Vec<ChatCompletionRequestMessage> =
            serde_json::from_value(serde_json::json!([
                {"role": "user", "content": "hello"}
            ]))
            .unwrap();
        let request = ChatGenerateRequest {
            rid: "chatcmpl-test".into(),
            chat: ChatRequest {
                model: "model".into(),
                messages,
                tools: None,
                tool_choice: None,
                response_format: None,
                reasoning_effort: None,
                continue_final_message: false,
                chat_template_args: Some(std::collections::HashMap::from([(
                    "enable_thinking".to_owned(),
                    serde_json::Value::Bool(false),
                )])),
                parallel_tool_calls: true,
            },
            sampling_params: SamplingParams {
                stop: Some(OneOrMany::One("client-stop".into())),
                ..Default::default()
            },
            choice_count: 1,
            stream: false,
            return_logprob: false,
            top_logprobs_num: 0,
            metadata: GenerateRequestMetadata::default(),
        };

        let service = RendererService::with_tokenizer(config, Arc::new(UnexpectedTokenizer), 1, 1);
        let chat = service.preprocess_chat(request).unwrap();
        let text_request = &chat.text_requests[0];

        assert!(text_request.prompt.as_str().contains("<|im_start|>user"));
        assert!(matches!(
            text_request.options.sampling_params.stop.as_ref(),
            Some(OneOrMany::Many(stops))
                if stops.iter().map(String::as_str).collect::<Vec<_>>()
                    == ["<|endoftext|>", "<|im_end|>", "client-stop"]
        ));
    }

    #[test]
    fn token_id_completion_bypasses_tokenization_and_builds_generate_request() {
        let config = RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: String::new(),
            revision: None,
            model_path: String::new(),
            chat_template: None,
            tool_call_parser: None,
            reasoning_parser: None,
            default_chat_template_kwargs: Default::default(),
            stream_response_default_include_usage: false,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                vocab_size: 128,
                context_len: 5,
                num_reserved_tokens: 0,
                allow_auto_truncate: true,
                enable_return_hidden_states: false,
            },
        };
        let service = RendererService::with_tokenizer(config, Arc::new(UnexpectedTokenizer), 1, 1);
        let request = TokenIdsRequest::new(
            "cmpl-test-0",
            vec![11, 12, 13],
            GenerationOptions {
                sampling_params: SamplingParams {
                    max_new_tokens: Some(4),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let requests = service.prepare_token_ids_requests(vec![request]).unwrap();

        assert_eq!(requests[0].input_ids, vec![11, 12, 13]);
        assert_eq!(requests[0].sampling_params.max_new_tokens, Some(2));
    }

    #[derive(Clone, Default)]
    struct RecordingTokenizer {
        prompts: Arc<Mutex<Vec<(String, bool)>>>,
    }

    impl TextTokenizer for RecordingTokenizer {
        fn encode(
            &self,
            text: &str,
            add_special_tokens: bool,
        ) -> Result<crate::TokenIds, ProcessorError> {
            self.prompts
                .lock()
                .unwrap()
                .push((text.to_owned(), add_special_tokens));
            Ok(vec![7])
        }
    }

    #[test]
    fn completion_and_chat_share_the_text_preparation_boundary() {
        let tokenizer = RecordingTokenizer::default();
        let prompts = tokenizer.prompts.clone();
        let mut config = model_config(".".into());
        config.model_path = String::new();
        config.chat_template = Some("chatml".into());
        let renderer = RendererService::with_tokenizer(config, Arc::new(tokenizer), 1, 8);

        let completion = TextRequest::text(
            "completion-0",
            "plain completion",
            true,
            GenerationOptions {
                sampling_params: SamplingParams {
                    max_new_tokens: Some(1),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        futures::executor::block_on(renderer.prepare_text_request_groups(vec![completion.into()]))
            .unwrap();

        let messages: Vec<ChatCompletionRequestMessage> =
            serde_json::from_value(serde_json::json!([
                {"role": "user", "content": "hello"}
            ]))
            .unwrap();
        let chat = ChatGenerateRequest {
            rid: "chat".into(),
            chat: ChatRequest {
                model: "model".into(),
                messages,
                tools: None,
                tool_choice: None,
                response_format: None,
                reasoning_effort: None,
                continue_final_message: false,
                chat_template_args: None,
                parallel_tool_calls: true,
            },
            sampling_params: SamplingParams {
                max_new_tokens: Some(1),
                ..Default::default()
            },
            choice_count: 1,
            stream: false,
            return_logprob: false,
            top_logprobs_num: 0,
            metadata: GenerateRequestMetadata::default(),
        };
        futures::executor::block_on(renderer.prepare_chat(chat)).unwrap();

        let prompts = prompts.lock().unwrap();
        assert!(prompts.iter().any(|(text, add_special_tokens)| {
            text == "plain completion" && *add_special_tokens
        }));
        assert!(
            prompts
                .iter()
                .any(|(text, add_special_tokens)| text.contains("hello") && !add_special_tokens)
        );
    }
}
