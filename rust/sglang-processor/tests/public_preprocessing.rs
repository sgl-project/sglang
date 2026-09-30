use std::sync::{Arc, Mutex};

use sglang_processor::dynamo_protocols::types::ChatCompletionRequestMessage;
use sglang_processor::{ChatConfig, ChatPreprocessor, ChatRequest, ProcessorError, TextTokenizer};

#[derive(Clone, Default)]
struct RecordingTokenizer {
    prompts: Arc<Mutex<Vec<(String, bool)>>>,
}

impl TextTokenizer for RecordingTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, ProcessorError> {
        self.prompts
            .lock()
            .unwrap()
            .push((text.to_owned(), add_special_tokens));
        Ok(vec![7])
    }
}

fn chat_request() -> ChatRequest {
    let messages: Vec<ChatCompletionRequestMessage> = serde_json::from_value(serde_json::json!([
        {"role": "user", "content": "hello"}
    ]))
    .unwrap();
    ChatRequest {
        model: "model".into(),
        messages,
        tools: None,
        tool_choice: None,
        response_format: None,
        reasoning_effort: None,
        continue_final_message: false,
        chat_template_args: None,
        parallel_tool_calls: true,
    }
}

#[test]
fn rendered_chat_is_tokenized_through_the_public_tokenizer_boundary() {
    let preprocessor = ChatPreprocessor::load(&ChatConfig {
        tokenizer_path: ".".into(),
        chat_template: Some("chatml".into()),
        ..Default::default()
    });
    let chat = preprocessor.preprocess(chat_request()).unwrap();
    assert_eq!(
        chat.template_stops.as_deref(),
        Some(["<|endoftext|>".to_owned(), "<|im_end|>".to_owned()].as_slice())
    );
    assert!(chat.tool_constraint.is_none());
    assert!(!chat.tool_calls_enabled);

    let tokenizer = RecordingTokenizer::default();
    let input_ids = tokenizer.encode(chat.prompt.as_str(), false).unwrap();

    assert_eq!(input_ids, [7]);
    let prompts = tokenizer.prompts.lock().unwrap();
    assert!(prompts.iter().any(
        |(text, add_special_tokens)| text.contains("<|im_start|>user\nhello")
            && !add_special_tokens
    ));
}
