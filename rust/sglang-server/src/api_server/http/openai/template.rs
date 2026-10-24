//! Resolve chat-template names and files to chat prompt formatters.
//
//! Hugging Face tokenizer configs contain Jinja templates. SGLang also accepts
//! legacy conversation JSON files and the names in Python's template registry.
//! Legacy definitions are rendered by a Rust implementation of Python's
//! `Conversation.get_prompt()` so there is exactly one implementation of the
//! per-style formatting logic (no Jinja translation to drift).

use std::collections::HashMap;
use std::path::PathBuf;

use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionToolChoiceOption, CreateChatCompletionRequest,
};
use dynamo_renderer::{OAIChatLikeRequest, PromptFormatter, TextInput};
use thiserror::Error;

use crate::message::types::OneOrMany;

#[cfg(test)]
pub(super) use super::template_builtins::builtin_template;
pub(super) use super::template_legacy::LegacyFormatter;
#[cfg(test)]
pub(super) use super::template_legacy::LegacySpec;
#[cfg(test)]
use super::template_loader::infer_legacy_template_from_model_path;
pub(super) use super::template_loader::load_chat_formatter;

#[path = "template_content.rs"]
mod template_content;
pub(super) use template_content::requires_content_arrays;

#[path = "template_tools.rs"]
mod template_tools;
pub(super) use template_tools::{normalize_prompt_tools, python_bool};

#[path = "template_reasoning.rs"]
mod template_reasoning;
pub(super) use template_reasoning::prepare_reasoning;

/// Extra variables for the chat template (`chat_template_kwargs`).
pub type ChatTemplateKwargs = HashMap<String, serde_json::Value>;

/// A chat prompt formatter: either the model's HuggingFace Jinja template (or
/// Dynamo's built-in encoder for models that ship none) or a legacy SGLang
/// conversation template.
#[derive(Clone)]
pub enum ChatFormatter {
    HuggingFace(PromptFormatter, bool),
    Native(PromptFormatter),
    Legacy(Box<LegacyFormatter>),
}

impl ChatFormatter {
    /// Render the request's messages to a single prompt string.
    pub(super) fn render(
        &self,
        request: &CreateChatCompletionRequest,
        kwargs: Option<&ChatTemplateKwargs>,
    ) -> Result<String, TemplateError> {
        if matches!(self, ChatFormatter::HuggingFace(..))
            && kwargs.is_some_and(|extra| extra.contains_key("_sglang_prompt_messages"))
        {
            let mut extra = kwargs.cloned().unwrap_or_default();
            extra.remove("_sglang_prompt_messages");
            return self.render_context(request, Some(&extra), None, None);
        }
        self.render_context(request, kwargs, None, None)
    }

    pub(super) fn render_chat(
        &self,
        request: &CreateChatCompletionRequest,
        kwargs: Option<&ChatTemplateKwargs>,
        prompt_tools: Option<&serde_json::Value>,
        continue_final_message: bool,
    ) -> Result<(String, Option<String>), TemplateError> {
        let selected = prompt_tools
            .and_then(|value| value.as_array())
            .map(|tools| {
                serde_json::Value::Array(
                    tools
                        .iter()
                        .filter(|tool| match &request.tool_choice {
                            Some(ChatCompletionToolChoiceOption::Named(choice)) => {
                                tool["function"]["name"].as_str()
                                    == Some(choice.function.name.as_str())
                            }
                            _ => true,
                        })
                        .cloned()
                        .collect(),
                )
            });
        let mut append_text = None;
        let mut extra = kwargs.cloned().unwrap_or_default();
        let mut messages = None;
        if let ChatFormatter::HuggingFace(_, arrays) = self {
            if let Some(effort) = &request.reasoning_effort {
                extra
                    .entry("reasoning_effort".into())
                    .or_insert_with(|| serde_json::json!(effort));
            }
            {
                let mut value =
                    serde_json::to_value(&request.messages).expect("messages serialize");
                for message in value.as_array_mut().expect("messages are an array") {
                    if let Some(calls) = message
                        .get_mut("tool_calls")
                        .and_then(serde_json::Value::as_array_mut)
                    {
                        for call in calls {
                            let arguments = &mut call["function"]["arguments"];
                            let parsed: serde_json::Value = serde_json::from_str(
                                arguments.as_str().unwrap_or(""),
                            )
                            .map_err(|error| TemplateError::Renderer {
                                message: format!(
                                    "assistant tool arguments must be a JSON object: {error}"
                                ),
                            })?;
                            if !parsed.is_object() {
                                return Err(TemplateError::Renderer {
                                    message: "assistant tool arguments must be a JSON object"
                                        .into(),
                                });
                            }
                            *arguments = parsed;
                        }
                    }
                    if message
                        .get("content")
                        .is_none_or(serde_json::Value::is_null)
                    {
                        message["content"] = serde_json::Value::String(String::new());
                    }
                    if (!*arrays || message["role"] == "tool")
                        && let Some(parts) =
                            message.get("content").and_then(serde_json::Value::as_array)
                        && parts.iter().all(|part| part["type"] == "text")
                    {
                        let text = parts
                            .iter()
                            .filter_map(|part| part["text"].as_str())
                            .collect::<Vec<_>>()
                            .join(" ");
                        message["content"] = serde_json::Value::String(text);
                    }
                }
                if let Some(last) = value
                    .as_array_mut()
                    .and_then(|messages| messages.last_mut())
                    && last["role"] == "assistant"
                    && let Some(text) = last["content"].as_str()
                {
                    if continue_final_message {
                        append_text = Some(text.to_owned());
                        value.as_array_mut().unwrap().pop();
                    } else {
                        *last = serde_json::json!({"role":"user", "content":text});
                    }
                }
                if value.as_array().is_some_and(Vec::is_empty) {
                    return Err(TemplateError::Renderer {
                        message: "messages cannot be empty after continuation".into(),
                    });
                }
                extra.insert("_sglang_prompt_messages".into(), value.clone());
                messages = Some(minijinja::Value::from_serialize(value));
            }
        }
        let kwargs = if matches!(self, ChatFormatter::HuggingFace(..)) {
            Some(&extra)
        } else {
            kwargs
        };
        self.render_context(request, kwargs, messages.as_ref(), selected.as_ref())
            .or_else(|error| {
                // Python retries generic Jinja with function-only tools when
                // a template rejects the OpenAI tool wrapper (e.g. Mistral).
                if let ChatFormatter::HuggingFace(..) = self
                    && let Some(tools) = selected.as_ref().and_then(serde_json::Value::as_array)
                    && !tools.is_empty()
                {
                    let flat = serde_json::Value::Array(
                        tools
                            .iter()
                            .map(|tool| tool.get("function").unwrap_or(tool).clone())
                            .collect(),
                    );
                    self.render_context(request, kwargs, messages.as_ref(), Some(&flat))
                } else {
                    Err(error)
                }
            })
            .map(|prompt| (prompt, append_text))
    }

    fn render_context(
        &self,
        request: &CreateChatCompletionRequest,
        kwargs: Option<&ChatTemplateKwargs>,
        messages: Option<&minijinja::Value>,
        tools: Option<&serde_json::Value>,
    ) -> Result<String, TemplateError> {
        match self {
            ChatFormatter::HuggingFace(PromptFormatter::OAI(formatter), _)
            | ChatFormatter::Native(PromptFormatter::OAI(formatter)) => formatter
                .render(&TemplateRequest {
                    request,
                    kwargs,
                    messages,
                    tools,
                })
                .map_err(|error| TemplateError::Renderer {
                    message: error.to_string(),
                }),
            ChatFormatter::Legacy(formatter) => formatter.render(request),
        }
    }

    /// The template's stop strings — Python `Conversation.stop_str`
    /// (`str | list[str] | None`). Legacy/builtin templates define them (e.g.
    /// chatml's `<|im_end|>`); the HuggingFace renderer carries none, matching
    /// Python's jinja path, which keeps only the request's own stops.
    pub(super) fn stop_strs(&self) -> Option<OneOrMany<String>> {
        match self {
            ChatFormatter::HuggingFace(..) | ChatFormatter::Native(_) => None,
            ChatFormatter::Legacy(formatter) => formatter.spec.stop_str.clone(),
        }
    }
}

/// The wire request plus its `chat_template_kwargs`, which the protocol type
/// does not carry.
struct TemplateRequest<'a> {
    request: &'a CreateChatCompletionRequest,
    kwargs: Option<&'a ChatTemplateKwargs>,
    messages: Option<&'a minijinja::Value>,
    tools: Option<&'a serde_json::Value>,
}

impl OAIChatLikeRequest for TemplateRequest<'_> {
    fn model(&self) -> String {
        self.request.model()
    }
    fn messages(&self) -> minijinja::Value {
        self.messages
            .cloned()
            .unwrap_or_else(|| self.request.messages())
    }
    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        if self.messages.is_some() {
            None
        } else {
            self.request.typed_messages()
        }
    }
    fn tools(&self) -> Option<minijinja::Value> {
        match self.tools {
            Some(tools) => tools
                .as_array()
                .filter(|tools| !tools.is_empty())
                .map(minijinja::Value::from_serialize),
            None => self.request.tools(),
        }
    }
    fn tool_choice(&self) -> Option<minijinja::Value> {
        self.request.tool_choice()
    }
    fn response_format(&self) -> Option<minijinja::Value> {
        self.request.response_format()
    }
    fn reasoning_effort(&self) -> Option<minijinja::Value> {
        self.request.reasoning_effort()
    }
    fn should_add_generation_prompt(&self) -> bool {
        self.request.should_add_generation_prompt()
    }
    fn chat_template_args(&self) -> Option<&ChatTemplateKwargs> {
        self.kwargs
    }
    fn extract_text(&self) -> Option<TextInput> {
        self.request.extract_text()
    }
    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        self.request.mm_processor_kwargs()
    }
}

#[derive(Debug, Error)]
pub(super) enum TemplateError {
    #[error("failed to read {kind} `{path}`: {source}")]
    Read {
        kind: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse {kind} `{path}`: {source}")]
    Parse {
        kind: &'static str,
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("chat template `{path}` is not a built-in name or a valid file path")]
    NotFound { path: PathBuf },

    #[error("chat template `{path}` is not a file")]
    NotFile { path: PathBuf },

    #[error("tokenizer config must be a JSON object")]
    ConfigNotObject,

    #[error("invalid chat template config: {source}")]
    Config {
        #[source]
        source: serde_json::Error,
    },

    #[error("tokenizer has no chat template")]
    Missing,

    #[error("tokenizer_config.json is required for this chat template source but was not found")]
    MissingConfig,

    #[error("invalid chat template: {message}")]
    Renderer { message: String },

    #[error("legacy chat template `{path}` must be a JSON object")]
    LegacyNotObject { path: PathBuf },

    #[error("legacy chat template `{path}` requires string field `{field}`")]
    LegacyMissingField { path: PathBuf, field: String },

    #[error("unknown separator style `{style}` in `{path}`")]
    UnknownStyle { path: PathBuf, style: String },

    #[error("unknown separator style `{style}`")]
    InvalidStyle { style: String },

    #[error("sep2 is required for separator style `{style}` but is not set")]
    MissingSep2 { style: String },

    #[error("stop_str must be a single string for separator style `{style}`")]
    InvalidStopString { style: String },

    #[error("the {role} message should be a single text")]
    NonTextContent { role: &'static str },

    #[error("multimodal {role} message content is not supported by legacy templates")]
    MediaContent { role: &'static str },

    #[error("unsupported message role `{role}` in legacy chat template")]
    UnsupportedRole { role: &'static str },
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_protocols::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
        ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequest,
    };

    use super::{
        ChatFormatter, LegacyFormatter, LegacySpec, OneOrMany, TemplateError, builtin_template,
        infer_legacy_template_from_model_path, load_chat_formatter,
    };

    fn request() -> CreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"}
            ]
        }))
        .unwrap()
    }

    fn spec(style: &str) -> LegacySpec {
        LegacySpec {
            name: "test".into(),
            system_template: "{system_message}".into(),
            system_message: "sys".into(),
            roles: ("USER".into(), "ASSISTANT".into()),
            style: style.into(),
            sep: "|sep|".into(),
            sep2: Some("|sep2|".into()),
            stop_str: Some(OneOrMany::One("<stop>".into())),
            ..Default::default()
        }
    }

    fn render(style: &str) -> String {
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "World"}
            ]
        }))
        .unwrap();
        LegacyFormatter { spec: spec(style) }
            .render(&request)
            .unwrap()
    }

    #[test]
    fn built_in_chatml_does_not_require_a_file() {
        let formatter = ChatFormatter::Legacy(Box::new(LegacyFormatter {
            spec: builtin_template("chatml").unwrap(),
        }));
        let rendered = formatter.render(&request(), None).unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// Every separator style renders exactly like Python's `Conversation.get_prompt()`.
    /// Messages: system "sys", user "Hello", assistant "World", then the
    /// unconditional assistant opening.
    #[test]
    fn all_separator_styles_render_like_python() {
        let cases = [
            (
                "ADD_COLON_SINGLE",
                "sys|sep|USER: Hello|sep|ASSISTANT: World|sep|ASSISTANT:",
            ),
            (
                "ADD_COLON_TWO",
                "sys|sep|USER: Hello|sep|ASSISTANT: World|sep2|ASSISTANT:",
            ),
            (
                "ADD_COLON_SPACE_SINGLE",
                "sys|sep|USER: Hello|sep|ASSISTANT: World|sep|ASSISTANT: ",
            ),
            (
                "ADD_NEW_LINE_SINGLE",
                "sys|sep|USER\nHello|sep|ASSISTANT\nWorld|sep|ASSISTANT\n",
            ),
            (
                "QWEN2_VL_EMBED",
                "sys|sep|USER\nHello|sep|ASSISTANT\nWorld|sep|ASSISTANT\n<stop>",
            ),
            (
                "NO_COLON_SINGLE",
                "sysUSERHello|sep|ASSISTANTWorld|sep|ASSISTANT",
            ),
            (
                "NO_COLON_TWO",
                "sysUSERHello|sep|ASSISTANTWorld|sep2|ASSISTANT",
            ),
            ("RWKV", "sysUSER: Hello\n\nASSISTANT: World\n\nASSISTANT:"),
            (
                "LLAMA4",
                "sys<|header_start|>USER<|header_end|>\n\nHello<|eot|><|header_start|>ASSISTANT<|header_end|>\n\nWorld<|eot|><|header_start|>ASSISTANT<|header_end|>\n\n",
            ),
            (
                "LLAMA3",
                "sys<|start_header_id|>USER<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>ASSISTANT<|end_header_id|>\n\nWorld<|eot_id|><|start_header_id|>ASSISTANT<|end_header_id|>\n\n",
            ),
            ("LLAMA2", "sysHello ASSISTANT World|sep2|USER"),
            (
                "CHATGLM",
                "sys|sep|[Round 0]|sep|USER：Hello|sep|ASSISTANT：World|sep|[Round 1]|sep|ASSISTANT：",
            ),
            (
                "CHATML",
                "sys|sep|\nUSER\nHello|sep|\nASSISTANT\nWorld|sep|\nASSISTANT\n",
            ),
            ("CHATGLM3", "sysUSER\nHelloASSISTANT\nWorldASSISTANT"),
            (
                "CHATINTERN",
                "sys<s>USER:Hello|sep|\nASSISTANT:World|sep2|\n<s>ASSISTANT:",
            ),
            (
                "DOLLY",
                "sysUSER:\nHello|sep|ASSISTANT:\nWorld|sep2|\n\nASSISTANT:\n",
            ),
            (
                "PHOENIX",
                "sysUSER: <s>Hello</s>ASSISTANT: <s>World</s>ASSISTANT: <s>",
            ),
            (
                "ROBIN",
                "sys|sep|USER:\nHello|sep|ASSISTANT:\nWorld|sep|ASSISTANT:\n",
            ),
            (
                "FALCON_CHAT",
                "sys|sep|USER: Hello|sep|ASSISTANT: World|sep|ASSISTANT:",
            ),
            (
                "METAMATH",
                "sys|sep|USER:\nHello|sep|ASSISTANT: |sep2|WorldASSISTANT:\n",
            ),
            (
                "DEEPSEEK_CHAT",
                "sysUSER: Hello|sep|ASSISTANT: World|sep2|ASSISTANT:",
            ),
            (
                "DeepSeekVL2",
                "sys|sep|USER: Hello|sep|ASSISTANT: World|sep2|ASSISTANT:",
            ),
            ("GEMMA3", "sysHello|sep|ASSISTANTWorld|sep|ASSISTANT"),
            ("MPT", "sys|sep|USERHello|sep|ASSISTANTWorld|sep|ASSISTANT"),
            (
                "QWEN2_AUDIO",
                "sys|sep|USER\nHello|sep|ASSISTANT\nWorld|sep|ASSISTANT\n",
            ),
            (
                "PADDLE_OCR",
                "sysUSER: Hello\nASSISTANT: World|sep|ASSISTANT: ",
            ),
            (
                "UNLIMITED_OCR",
                "sys|sep|USERHello|sep|ASSISTANTWorld|sep2|ASSISTANT",
            ),
        ];
        for (style, expected) in cases {
            assert_eq!(render(style), expected, "style {style}");
        }
    }

    /// LLAMA2's no-system path starts with `[INST] ` and tags messages by
    /// index parity — the opening at an even index takes the user tag.
    #[test]
    fn llama2_without_system_starts_with_inst() {
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "World"}
            ]
        }))
        .unwrap();
        let mut spec = spec("LLAMA2");
        spec.system_message = String::new();
        let rendered = LegacyFormatter { spec }.render(&request).unwrap();
        assert_eq!(rendered, "[INST] Hello ASSISTANT World|sep2|USER");
    }

    /// FALCON_CHAT without a system message starts with the first user message.
    #[test]
    fn falcon_chat_without_system_starts_with_user() {
        let mut spec = spec("FALCON_CHAT");
        spec.system_message = String::new();
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "World"}
            ]
        }))
        .unwrap();
        let rendered = LegacyFormatter { spec }.render(&request).unwrap();
        assert_eq!(rendered, "USER: Hello|sep|ASSISTANT: World|sep|ASSISTANT:");
    }

    /// QWEN2_AUDIO indexes each audio-token occurrence (Python
    /// `audio_token.format(idx=counter)`).
    #[test]
    fn qwen2_audio_indexes_audio_tokens() {
        let mut spec = spec("QWEN2_AUDIO");
        spec.audio_token = "Audio {idx}: <|audio_bos|><|AUDIO|><|audio_eos|>\n".into();
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": "x Audio {idx}: <|audio_bos|><|AUDIO|><|audio_eos|>\n y Audio {idx}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
            }]
        }))
        .unwrap();
        let rendered = LegacyFormatter { spec }.render(&request).unwrap();
        assert_eq!(
            rendered,
            "sys|sep|USER\nx Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n y Audio 2: <|audio_bos|><|AUDIO|><|audio_eos|>\n|sep|ASSISTANT\n"
        );
    }

    /// PADDLE_OCR drops the newline after an image token in user messages.
    #[test]
    fn paddle_ocr_normalizes_image_token_newline() {
        let mut spec = spec("PADDLE_OCR");
        spec.image_token = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>".into();
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>\nWhat is this?"
            }]
        }))
        .unwrap();
        let rendered = LegacyFormatter { spec }.render(&request).unwrap();
        assert_eq!(
            rendered,
            "sysUSER: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>What is this?\nASSISTANT: "
        );
    }

    /// stop_str preserves Python's `str | list[str] | None` typing.
    #[test]
    fn stop_str_keeps_python_type_semantics() {
        let request = serde_json::from_value::<CreateChatCompletionRequest>(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        // QWEN2_VL_EMBED requires a single string stop (Python raises TypeError
        // on a list / None) — error deliberately.
        let mut spec = spec("QWEN2_VL_EMBED");
        spec.stop_str = Some(OneOrMany::Many(vec!["a".into(), "b".into()]));
        let error = LegacyFormatter { spec }.render(&request).unwrap_err();
        assert!(error.to_string().contains("stop_str"));
        // The built-in gme-qwen2-vl registers a single-string stop.
        let spec = builtin_template("gme-qwen2-vl").unwrap();
        assert!(matches!(spec.stop_str, Some(OneOrMany::One(_))));
        // gemma-it registers a list.
        let spec = builtin_template("gemma-it").unwrap();
        assert!(matches!(spec.stop_str, Some(OneOrMany::Many(_))));
        // An explicit `"stop_str": null` in a legacy JSON file maps to `None`
        // (Python accepts a present null), while a missing key stays an error.
        let base = std::env::temp_dir().join(format!(
            "sglang-openai-template-stopnull-{}-test.json",
            std::process::id()
        ));
        std::fs::write(
            &base,
            r#"{
                "name": "test",
                "system": "System",
                "user": "USER",
                "assistant": "ASSISTANT",
                "sep_style": "ADD_COLON_SINGLE",
                "stop_str": null
            }"#,
        )
        .unwrap();
        let formatter = load_chat_formatter(
            Some(base.to_str().unwrap()),
            None,
            None,
            Some(base.to_str().unwrap()),
        )
        .unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert!(formatter.spec.stop_str.is_none());
        let _ = std::fs::remove_file(base);
    }

    #[test]
    fn json_legacy_template_is_rendered() {
        let base = std::env::temp_dir().join(format!(
            "sglang-openai-template-base-{}-test.json",
            std::process::id()
        ));
        let legacy = base.with_file_name("sglang-openai-template-legacy.json");
        std::fs::write(&base, r#"{"chat_template":"unused"}"#).unwrap();
        std::fs::write(
            &legacy,
            r#"{
                "name": "test-legacy",
                "system": "System",
                "system_message": "default",
                "user": "USER",
                "assistant": "ASSISTANT",
                "sep_style": "ADD_COLON_SINGLE",
                "sep": "\n",
                "stop_str": "<stop>"
            }"#,
        )
        .unwrap();

        let formatter = load_chat_formatter(
            Some(base.to_str().unwrap()),
            None,
            None,
            Some(legacy.to_str().unwrap()),
        )
        .unwrap();
        let rendered = formatter.render(&request(), None).unwrap();
        assert_eq!(rendered, "System\nBe concise.\nUSER: Hello\nASSISTANT:");

        let _ = std::fs::remove_file(base);
        let _ = std::fs::remove_file(legacy);
    }

    /// Content extraction matches `generate_chat_conv`: system/assistant arrays
    /// must be a single text part; user arrays concatenate text parts; tool
    /// roles are rejected.
    #[test]
    fn content_extraction_matches_python() {
        let formatter = LegacyFormatter {
            spec: spec("CHATML"),
        };
        // User array content concatenates text parts.
        let request = serde_json::from_value::<CreateChatCompletionRequest>(serde_json::json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]
            }]
        }))
        .unwrap();
        let rendered = formatter.render(&request).unwrap();
        assert!(rendered.contains("USER\nHello world|sep|"));

        // System array with exactly one text part is fine.
        let request = serde_json::from_value::<CreateChatCompletionRequest>(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
                {"role": "user", "content": "hi"}
            ]
        }))
        .unwrap();
        let rendered = formatter.render(&request).unwrap();
        assert!(!rendered.starts_with("sys|sep|")); // overridden by "Be brief."
        assert!(rendered.contains("Be brief."));

        // System array with two parts is rejected.
        let request = serde_json::from_value::<CreateChatCompletionRequest>(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                {"role": "user", "content": "hi"}
            ]
        }))
        .unwrap();
        let error = formatter.render(&request).unwrap_err();
        assert!(error.to_string().contains("system message"));

        // Tool messages are rejected like Python's "Unknown role".
        let request = serde_json::from_value::<CreateChatCompletionRequest>(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "tool", "tool_call_id": "c1", "content": "42"}
            ]
        }))
        .unwrap();
        let error = formatter.render(&request).unwrap_err();
        assert!(error.to_string().contains("tool"));
    }

    /// The typed message structs are usable directly (no serde path needed).
    #[test]
    fn typed_messages_render_identically() {
        let request = CreateChatCompletionRequest {
            messages: vec![
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text("sys".into()),
                    name: None,
                    tools: None,
                }),
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Array(vec![
                        ChatCompletionRequestUserMessageContentPart::Text(
                            ChatCompletionRequestMessageContentPartText {
                                text: "Hello".into(),
                                prompt_cache_breakpoint: None,
                            },
                        ),
                    ]),
                    name: None,
                }),
            ],
            ..Default::default()
        };
        let rendered = LegacyFormatter {
            spec: spec("ADD_COLON_SINGLE"),
        }
        .render(&request)
        .unwrap();
        assert_eq!(rendered, "sys|sep|USER: Hello|sep|ASSISTANT:");
    }

    /// A built-in `--chat-template` name resolves without any tokenizer config.
    #[test]
    fn builtin_argument_works_without_tokenizer_config() {
        let formatter = load_chat_formatter(None, None, None, Some("chatml")).unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert_eq!(formatter.spec.name, "chatml");
    }

    /// Python `load_chat_template`: without `--chat-template`, the model path
    /// infers a legacy template before the HF fallback — so a legacy model
    /// with no `chat_template` in its config still gets one, and even a config
    /// that HAS one loses to the inference.
    #[test]
    fn model_path_inference_precedes_tokenizer_config() {
        let base = std::env::temp_dir().join(format!(
            "sglang-openai-template-infer-{}-test.json",
            std::process::id()
        ));
        std::fs::write(
            &base,
            r#"{"tokenizer_class":"LlamaTokenizer","chat_template":"{{messages}}"}"#,
        )
        .unwrap();

        // Path matcher: vicuna/llava-v1.5-style paths.
        let formatter = load_chat_formatter(
            Some(base.to_str().unwrap()),
            Some("models/vicuna-7b-v1.5"),
            None,
            None,
        )
        .unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert_eq!(formatter.spec.name, "vicuna_v1.1");
        // No config at all + path matcher.
        let formatter = load_chat_formatter(None, Some("deepseek-vl2-7b"), None, None).unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert_eq!(formatter.spec.name, "deepseek-vl2");

        // Model-type matcher: reads `<model_path>/config.json`.
        let model_dir = std::env::temp_dir().join(format!(
            "sglang-openai-template-infer-model-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"model_type":"phi4mm","architectures":["Phi4MMForCausalLM"]}"#,
        )
        .unwrap();
        let formatter =
            load_chat_formatter(None, Some(model_dir.to_str().unwrap()), None, None).unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert_eq!(formatter.spec.name, "phi-4-mm");

        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_dir_all(model_dir);
    }

    /// Every name the model-path matchers can produce must exist in the
    /// built-in table (parity guard for `MODEL_TYPE_TO_TEMPLATE`).
    #[test]
    fn inferred_template_names_resolve_to_builtins() {
        for model_path in [
            "points-7b-chat",
            "moss-vl",
            "moss2-vl",
            "internvl-2.5",
            "janus-pro",
            "vicuna-7b",
            "llava-v1.5-7b",
            "deepseek-vl2-small",
            "llava-v1.6-34b",
            "minicpm-v-2.6",
            "minicpm-o-4.5",
            "phi-4-multimodal",
            "deepseek-ocr",
            "unlimited-ocr",
            "paddleocr-vl",
            "whisper",
        ] {
            let spec = infer_legacy_template_from_model_path(model_path)
                .unwrap_or_else(|| panic!("no inference for {model_path}"));
            let _ = spec;
        }
    }

    /// MiniCPM 4.6+ must NOT fall back to the legacy template; with no config
    /// and nothing else to try, that surfaces as the missing-config error.
    #[test]
    fn minicpm_4_6_skips_legacy_inference() {
        assert!(infer_legacy_template_from_model_path("minicpm-v-4.6").is_none());
        assert!(matches!(
            load_chat_formatter(None, Some("minicpm-v-4.6"), None, None),
            Err(TemplateError::MissingConfig)
        ));
    }

    fn temp_config(contents: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("sglang-template-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = dir.join("tokenizer_config.json");
        std::fs::write(&config, contents).unwrap();
        config
    }

    #[test]
    fn chat_template_kwargs_reach_the_template() {
        let config = temp_config(r#"{"chat_template": "thinking={{ thinking }}"}"#);
        let formatter =
            load_chat_formatter(Some(config.to_str().unwrap()), None, None, None).unwrap();
        let kwargs = HashMap::from([("thinking".into(), serde_json::json!(true))]);
        assert_eq!(formatter.render(&request(), None).unwrap(), "thinking=");
        assert_eq!(
            formatter.render(&request(), Some(&kwargs)).unwrap(),
            "thinking=True"
        );
    }

    #[test]
    fn malformed_hf_config_does_not_fall_back_to_native_formatter() {
        for contents in [
            "[]",
            "null",
            r#"{"chat_template":7}"#,
            r#"{"chat_template":{}}"#,
            r#"{"chat_template":[]}"#,
        ] {
            let config = temp_config(contents);
            let result = load_chat_formatter(
                Some(config.to_str().unwrap()),
                None,
                Some("deepseek_v4"),
                None,
            );
            assert!(
                result.is_err(),
                "malformed config silently selected native: {contents}"
            );
            std::fs::remove_dir_all(config.parent().unwrap()).unwrap();
        }
    }

    #[test]
    fn missing_template_falls_back_to_native_formatter() {
        let config = temp_config("{}");
        let config = config.to_str().unwrap();
        let load = |config, model_type, arg| {
            load_chat_formatter(config, Some("/models/x"), model_type, arg)
        };

        let formatter = load(Some(config), Some("deepseek_v4"), None).unwrap();
        assert!(matches!(formatter, ChatFormatter::Native(_)));
        let kwargs = HashMap::from([("thinking".into(), serde_json::json!(false))]);
        assert_eq!(
            formatter.render(&request(), Some(&kwargs)).unwrap(),
            "<｜begin▁of▁sentence｜>Be concise.<｜User｜>Hello<｜Assistant｜></think>"
        );
        assert!(matches!(
            load(None, Some("deepseek_v4"), None),
            Ok(ChatFormatter::Native(_))
        ));

        // A template, `--chat-template`, or an unknown architecture wins.
        let templated = temp_config(r#"{"chat_template": "Hi"}"#);
        let formatter = load(Some(templated.to_str().unwrap()), Some("deepseek_v4"), None).unwrap();
        assert_eq!(formatter.render(&request(), None).unwrap(), "Hi");
        assert!(matches!(
            load(Some(config), Some("deepseek_v4"), Some("chatml")),
            Ok(ChatFormatter::Legacy(_))
        ));
        assert!(matches!(
            load(Some(config), Some("llama"), None),
            Err(TemplateError::Missing)
        ));
        assert!(matches!(
            load(None, None, None),
            Err(TemplateError::MissingConfig)
        ));
    }
}
