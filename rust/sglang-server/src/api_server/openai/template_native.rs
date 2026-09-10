//! SGLang request normalization for Dynamo's native prompt formatters.

use std::collections::HashMap;

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::{OAIChatLikeRequest, PromptFormatter};
use minijinja::Value as TemplateValue;
use serde_json::{Value, json};

use super::template::TemplateError;
use crate::utils::environ::env_bool;

pub(super) fn render(
    formatter: &PromptFormatter,
    request: &CreateChatCompletionRequest,
    thinking: Option<bool>,
) -> Result<String, TemplateError> {
    let mut messages: Vec<Value> = request
        .messages
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<_, _>>()
        .map_err(render_error)?;
    for message in &mut messages {
        // Python flattens text parts with spaces before invoking encoding_dsv4.
        message["content"] = match &message["content"] {
            Value::Array(parts) => Value::String(
                parts
                    .iter()
                    .filter_map(|part| part["text"].as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
            Value::Null => json!(""),
            content => content.clone(),
        };
    }
    if messages
        .first()
        .is_none_or(|message| message["role"] != "system")
    {
        messages.insert(0, json!({"role": "system", "content": ""}));
    }
    if let Some(tools) = &request.tools {
        // Match Python Tool.model_dump(): field order and defaults affect prompt bytes.
        messages[0]["tools"] = Value::Array(
            tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "function": {
                            "description": tool.function.description,
                            "name": tool.function.name,
                            "parameters": tool.function.parameters,
                            "strict": tool.function.strict.unwrap_or(false),
                        }
                    })
                })
                .collect(),
        );
    }

    let effort = serde_json::to_value(&request.reasoning_effort).map_err(render_error)?;
    let env_effort = std::env::var("SGLANG_DSV4_REASONING_EFFORT").ok();
    // Python's official V4 profile defaults to low and ignores unsupported aliases.
    let effort = effort
        .as_str()
        .or(env_effort.as_deref())
        .filter(|effort| matches!(*effort, "low" | "high" | "max"))
        .unwrap_or("low");
    let request = NativeRequest {
        model: &request.model,
        messages: TemplateValue::from_serialize(messages),
        args: HashMap::from([
            (
                "thinking".into(),
                json!(thinking.unwrap_or_else(|| env_bool("SGLANG_DEFAULT_THINKING", false))),
            ),
            ("reasoning_effort".into(), json!(effort)),
        ]),
    };
    let PromptFormatter::OAI(formatter) = formatter;
    let prompt = formatter.render_prompt(&request).map_err(render_error)?;
    // Segment-aware formats need a different tokenizer interface; never flatten
    // their special-token trust boundaries into an ordinary text request.
    if prompt.segments().is_some() {
        return Err(render_error(
            "this native formatter requires segmented tokenization",
        ));
    }
    Ok(prompt.into_text())
}

struct NativeRequest<'a> {
    model: &'a str,
    messages: TemplateValue,
    args: HashMap<String, Value>,
}

impl OAIChatLikeRequest for NativeRequest<'_> {
    fn model(&self) -> String {
        self.model.to_owned()
    }
    fn messages(&self) -> TemplateValue {
        self.messages.clone()
    }
    fn should_add_generation_prompt(&self) -> bool {
        true
    }
    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        Some(&self.args)
    }
}

fn render_error(error: impl std::fmt::Display) -> TemplateError {
    TemplateError::Renderer {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::openai::load_chat_support;
    use crate::api_server::openai::template::ChatFormatter;
    use crate::message::config::ServerArgs;

    /// The CPU Python test generates expectations through its real serving
    /// formatter/tokenizer, then invokes this test with those temporary fixtures.
    #[test]
    #[ignore = "run test/registered/unit/entrypoints/test_deepseek_v4_rust.py"]
    fn python_rust_chat_tokenization_parity() {
        use std::sync::Arc;

        use crate::message::request::{GenerateRequest, Request, RequestKind};
        use crate::message::response::ResponseSink;
        use crate::runtime::Runnable;
        use crate::tokenizer_manager::tokenizer::{
            DynamoTokenizer, TokenizerWorker, load_tokenizer,
        };
        use crate::tokenizer_manager::wiring::TmEvent;
        use crate::utils::fsm::RequestState;

        let fixture: Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("SGLANG_TEST_CHAT_PARITY_FIXTURE").unwrap())
                .unwrap(),
        )
        .unwrap();
        let path = fixture["tokenizer_path"].as_str().unwrap();
        let mut args = ServerArgs {
            model_path: path.into(),
            tokenizer_path: path.into(),
            ..Default::default()
        };
        args.model_config.model_type = Some(fixture["model_type"].as_str().unwrap().into());
        let formatter = load_chat_support(&args).unwrap();
        assert!(matches!(formatter, ChatFormatter::Native(_)));
        let tokenizer = load_tokenizer(Some(path), None, false).unwrap().unwrap();
        let (req_tx, req_rx) = flume::unbounded();
        let (tm_tx, tm_rx) = flume::unbounded();
        let (sink_tx, _sink_rx) = tokio::sync::mpsc::channel(1);
        let cases = fixture["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        for case in cases {
            let request = serde_json::from_value(case["request"].clone()).unwrap();
            let thinking = case["request"]["chat_template_kwargs"]["thinking"].as_bool();
            let prompt = formatter.render(&request, thinking).unwrap();
            assert_eq!(prompt, case["prompt"].as_str().unwrap(), "{}", case["name"]);
            req_tx
                .send(Request {
                    rid: case["name"].as_str().unwrap().into(),
                    state: RequestState::Tokenizing,
                    sink: ResponseSink::Local(sink_tx.clone()),
                    kind: RequestKind::Generate(Box::new(GenerateRequest {
                        text: Some(prompt),
                        skip_special_tokens: true,
                        ..Default::default()
                    })),
                })
                .unwrap();
        }
        drop(req_tx);
        TokenizerWorker::new(req_rx, tm_tx, Arc::new(DynamoTokenizer::new(tokenizer))).run();
        for case in cases {
            let TmEvent::Tokenized(request) = tm_rx.recv().unwrap() else {
                panic!("expected tokenized request");
            };
            let RequestKind::Generate(request) = request.kind else {
                panic!("expected generate request");
            };
            assert_eq!(
                json!(request.input_ids),
                case["input_ids"],
                "{}",
                case["name"]
            );
        }
    }

    #[test]
    fn native_fallback_preserves_template_priority_and_rejects_unknown_models() {
        let dir = std::env::temp_dir().join(format!("sglang-native-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = dir.join("tokenizer_config.json");
        let mut args = ServerArgs {
            model_path: dir.to_str().unwrap().into(),
            tokenizer_path: dir.to_str().unwrap().into(),
            served_model_name: "renamed-model".into(),
            ..Default::default()
        };
        args.model_config.model_type = Some("deepseek_v4".into());
        assert!(matches!(
            load_chat_support(&args),
            Some(ChatFormatter::Native(_))
        ));
        std::fs::write(&config, r#"{"chat_template":null}"#).unwrap();
        assert!(matches!(
            load_chat_support(&args),
            Some(ChatFormatter::Native(_))
        ));
        std::fs::write(&config, r#"{"chat_template":"Hello"}"#).unwrap();
        assert!(matches!(
            load_chat_support(&args),
            Some(ChatFormatter::HuggingFace(_))
        ));
        std::fs::write(&config, "{}").unwrap();
        args.chat_template = Some("chatml".into());
        assert!(matches!(
            load_chat_support(&args),
            Some(ChatFormatter::Legacy(_))
        ));
        args.chat_template = Some("/nonexistent/template.jinja".into());
        assert!(load_chat_support(&args).is_none());
        args.chat_template = None;
        args.model_config.model_type = Some("llama".into());
        assert!(load_chat_support(&args).is_none());
        // A malformed config is an error, not a missing-template fallback.
        args.model_config.model_type = Some("deepseek_v4".into());
        std::fs::write(&config, "{").unwrap();
        assert!(load_chat_support(&args).is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn official_v4_effort_and_thinking_match_python() {
        let formatter =
            dynamo_renderer::native_formatter_for(&Some("deepseek_v4".into()), "alias").unwrap();
        for (effort, preamble) in [
            (None, ""),
            (Some("low"), ""),
            (Some("medium"), ""),
            (Some("none"), ""),
            (Some("high"), "Reasoning Effort: Absolute maximum"),
            (Some("max"), "Reasoning Effort: Beyond maximum"),
        ] {
            let request = serde_json::from_value(json!({
                "model": "test", "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": effort,
            }))
            .unwrap();
            assert_eq!(
                render(&formatter, &request, Some(false)).unwrap(),
                "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
            );
            let prompt = render(&formatter, &request, Some(true)).unwrap();
            assert!(prompt.starts_with(&format!("<｜begin▁of▁sentence｜>{preamble}")));
            assert!(prompt.ends_with("<｜User｜>Hello<｜Assistant｜><think>"));
            assert_eq!(prompt.contains("Reasoning Effort:"), !preamble.is_empty());
        }
    }

    #[test]
    fn segmented_native_prompts_are_not_flattened() {
        let formatter =
            dynamo_renderer::kimi_k3_formatter_for(&Some("kimi_k3".into()), "alias", false)
                .unwrap();
        let request = serde_json::from_value(json!({
            "model": "test", "messages": [{"role": "user", "content": "Hello"}],
        }))
        .unwrap();
        assert!(
            render(&formatter, &request, Some(false))
                .unwrap_err()
                .to_string()
                .contains("segmented tokenization")
        );
    }
}
