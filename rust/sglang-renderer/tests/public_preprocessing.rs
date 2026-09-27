use std::sync::{Arc, Mutex};

use dynamo_protocols::types::ChatCompletionRequestMessage;
use sglang_renderer::{
    ChatRequest, GenerateRequestMetadata, GenerationOptions, RendererConfig, RendererError,
    RendererLimits, RendererService, SamplingDefaults, SamplingParams, TextRequest, TextTokenizer,
};

#[derive(Clone, Default)]
struct RecordingTokenizer {
    prompts: Arc<Mutex<Vec<(String, bool)>>>,
}

impl TextTokenizer for RecordingTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, RendererError> {
        self.prompts
            .lock()
            .unwrap()
            .push((text.to_owned(), add_special_tokens));
        Ok(vec![7])
    }
}

fn config() -> RendererConfig {
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

#[test]
fn completion_and_chat_share_the_public_text_preparation_boundary() {
    let tokenizer = RecordingTokenizer::default();
    let prompts = tokenizer.prompts.clone();
    let renderer = RendererService::with_tokenizer(config(), Arc::new(tokenizer), 1, 8);

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
    futures::executor::block_on(renderer.prepare_text_requests(vec![completion])).unwrap();

    let messages: Vec<ChatCompletionRequestMessage> = serde_json::from_value(serde_json::json!([
        {"role": "user", "content": "hello"}
    ]))
    .unwrap();
    let chat = ChatRequest {
        rid: "chat".into(),
        model: "model".into(),
        messages,
        tools: None,
        tool_choice: None,
        response_format: None,
        reasoning_effort: None,
        continue_final_message: false,
        chat_template_args: None,
        sampling_params: SamplingParams {
            max_new_tokens: Some(1),
            ..Default::default()
        },
        choice_count: 1,
        stream: false,
        return_logprob: false,
        top_logprobs_num: 0,
        parallel_tool_calls: true,
        metadata: GenerateRequestMetadata::default(),
    };
    futures::executor::block_on(renderer.prepare_chat(chat)).unwrap();

    let prompts = prompts.lock().unwrap();
    assert!(
        prompts
            .iter()
            .any(|(text, add_special_tokens)| text == "plain completion" && *add_special_tokens)
    );
    assert!(
        prompts
            .iter()
            .any(|(text, add_special_tokens)| text.contains("hello") && !add_special_tokens)
    );
}
