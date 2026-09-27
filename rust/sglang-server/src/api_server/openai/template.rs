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
use dynamo_renderer::{OAIChatLikeRequest, PromptFormatter, TextInput, may_be_fix_tool_schema};
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

#[path = "template_tools.rs"]
mod template_tools;
pub(super) use template_tools::{normalize_prompt_tools, python_bool};

/// Extra variables for the chat template (`chat_template_kwargs`).
pub type ChatTemplateKwargs = HashMap<String, serde_json::Value>;

/// A chat prompt formatter, retaining whether it uses a Jinja template,
/// a native encoder, or a legacy SGLang conversation template.
#[derive(Clone)]
pub enum ChatFormatter {
    HuggingFace(PromptFormatter),
    Native(PromptFormatter),
    Legacy(Box<LegacyFormatter>),
}

impl ChatFormatter {
    /// Render with Python's prompt-visible tools, retaining the raw typed request.
    pub(super) fn render_with_tools(
        &self,
        request: &CreateChatCompletionRequest,
        kwargs: Option<&ChatTemplateKwargs>,
        prompt_tools: Option<&serde_json::Value>,
        prompt_messages: Option<&minijinja::Value>,
    ) -> Result<String, TemplateError> {
        match self {
            ChatFormatter::HuggingFace(PromptFormatter::OAI(formatter))
            | ChatFormatter::Native(PromptFormatter::OAI(formatter)) => formatter
                .render(&TemplateRequest {
                    request,
                    kwargs,
                    filter_named_tools: matches!(self, ChatFormatter::HuggingFace(_)),
                    prompt_tools,
                    prompt_messages,
                })
                .map_err(|error| TemplateError::Renderer {
                    message: error.to_string(),
                }),
            ChatFormatter::Legacy(formatter) => formatter.render(request),
        }
    }

    /// Render the request's messages to a single prompt string.
    #[cfg(test)]
    pub(super) fn render(
        &self,
        request: &CreateChatCompletionRequest,
        kwargs: Option<&ChatTemplateKwargs>,
    ) -> Result<String, TemplateError> {
        self.render_with_tools(request, kwargs, None, None)
    }

    /// The template's stop strings — Python `Conversation.stop_str`
    /// (`str | list[str] | None`). Legacy/builtin templates define them (e.g.
    /// chatml's `<|im_end|>`); the HuggingFace renderer carries none, matching
    /// Python's jinja path, which keeps only the request's own stops.
    pub(super) fn stop_strs(&self) -> Option<OneOrMany<String>> {
        match self {
            ChatFormatter::HuggingFace(_) | ChatFormatter::Native(_) => None,
            ChatFormatter::Legacy(formatter) => formatter.spec.stop_str.clone(),
        }
    }

    /// Whether this Jinja formatter requires structured OpenAI content parts.
    /// Native and legacy renderers stay outside the Jinja normalization path.
    pub(super) fn requires_content_arrays(&self) -> bool {
        match self {
            ChatFormatter::HuggingFace(formatter) => formatter.requires_content_arrays(),
            ChatFormatter::Native(_) | ChatFormatter::Legacy(_) => false,
        }
    }
}

/// The wire request plus its `chat_template_kwargs`, which the protocol type
/// does not carry.
struct TemplateRequest<'a> {
    request: &'a CreateChatCompletionRequest,
    kwargs: Option<&'a ChatTemplateKwargs>,
    filter_named_tools: bool,
    prompt_tools: Option<&'a serde_json::Value>,
    prompt_messages: Option<&'a minijinja::Value>,
}

impl OAIChatLikeRequest for TemplateRequest<'_> {
    fn model(&self) -> String {
        self.request.model()
    }
    fn messages(&self) -> minijinja::Value {
        self.prompt_messages
            .cloned()
            .unwrap_or_else(|| self.request.messages())
    }
    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        self.request.typed_messages()
    }
    fn tools(&self) -> Option<minijinja::Value> {
        if self.filter_named_tools
            && let Some(prompt_tools) = self.prompt_tools
        {
            let tools = prompt_tools.as_array()?;
            let selected = tools
                .iter()
                .filter(|tool| match &self.request.tool_choice {
                    Some(ChatCompletionToolChoiceOption::Named(choice)) => {
                        tool["function"]["name"].as_str() == Some(choice.function.name.as_str())
                    }
                    _ => true,
                })
                .collect::<Vec<_>>();
            return (!selected.is_empty()).then(|| minijinja::Value::from_serialize(selected));
        }
        if self.filter_named_tools
            && let Some(ChatCompletionToolChoiceOption::Named(choice)) = &self.request.tool_choice
            && let Some(tools) = &self.request.tools
        {
            // Python limits the Jinja prompt to the named tool. Keep the full
            // request available to validation, constraints, and output parsing.
            let selected = tools
                .iter()
                .filter(|tool| tool.function.name == choice.function.name)
                .collect::<Vec<_>>();
            return may_be_fix_tool_schema(serde_json::to_value(selected).unwrap());
        }
        self.request.tools()
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
                }),
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Array(vec![
                        ChatCompletionRequestUserMessageContentPart::Text(
                            ChatCompletionRequestMessageContentPartText {
                                text: "Hello".into(),
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

    fn tools_request(choice: Option<serde_json::Value>) -> CreateChatCompletionRequest {
        let mut value = serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [
                {"type": "function", "function": {
                    "name": "get_weather", "description": "first", "strict": false,
                    "parameters": {"properties": {
                        "z": {"type": "string"}, "a": {"type": "integer"}
                    }, "type": "object"}
                }},
                {"type": "function", "function": {
                    "name": "get_time", "description": "unselected",
                    "parameters": {"type": "object", "properties": {}}
                }},
                {"type": "function", "function": {
                    "name": "get_weather", "description": "second", "strict": true,
                    "parameters": {"type": "object", "properties": {
                        "b": {"type": "boolean"}
                    }}
                }}
            ]
        });
        if let Some(choice) = choice {
            value["tool_choice"] = choice;
        }
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn named_tool_prompt_keeps_all_matches_and_their_schema_order() {
        let config = temp_config(
            &serde_json::json!({
                "chat_template": "{% for tool in tools or [] %}{{ tool.function.name }}={{ tool.function.description }};{% for key, value in tool.function.parameters.properties.items() %}{{ key }}={{ value.type }};{% endfor %}{{ 'strict' if tool.function.strict else 'loose' }}\n{% endfor %}"
            })
            .to_string(),
        );
        let formatter =
            load_chat_formatter(Some(config.to_str().unwrap()), None, None, None).unwrap();
        let request = tools_request(Some(serde_json::json!({
            "type": "function", "function": {"name": "get_weather"}
        })));
        let before = serde_json::to_value(&request).unwrap();
        let tools = super::normalize_prompt_tools(&before["tools"]).unwrap();
        for prompt_tools in [None, tools.as_ref()] {
            assert_eq!(
                formatter
                    .render_with_tools(&request, None, prompt_tools, None)
                    .unwrap(),
                "get_weather=first;z=string;a=integer;loose\nget_weather=second;b=boolean;strict\n"
            );
        }
        assert_eq!(serde_json::to_value(&request).unwrap(), before);
    }

    #[test]
    fn unnamed_tool_prompt_controls_keep_existing_behavior() {
        let config = temp_config(
            r#"{"chat_template": "{% for tool in tools or [] %}{{ tool.function.name }}|{% endfor %}"}"#,
        );
        let formatter =
            load_chat_formatter(Some(config.to_str().unwrap()), None, None, None).unwrap();
        for choice in [None, Some(serde_json::json!("auto"))] {
            assert_eq!(
                formatter.render(&tools_request(choice), None).unwrap(),
                "get_weather|get_time|get_weather|"
            );
        }
        assert_eq!(
            formatter
                .render(&tools_request(Some(serde_json::json!("none"))), None)
                .unwrap(),
            ""
        );
    }

    #[test]
    fn python_tool_tojson_preserves_declared_and_nested_order() {
        let config = temp_config(r#"{"chat_template":"{{ tools[0] | tojson }}"}"#);
        let formatter =
            load_chat_formatter(Some(config.to_str().unwrap()), None, None, None).unwrap();
        let raw: serde_json::Value = serde_json::from_str(
            r#"[{"type":"function","function":{"parameters":{"properties":{"z":{"type":"string"},"a":{"enum":["x","y"]}}},"name":"weather"}}]"#,
        )
        .unwrap();
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model":"test", "messages":[{"role":"user","content":"hi"}], "tools":raw
        }))
        .unwrap();
        let tools = super::normalize_prompt_tools(&raw).unwrap();
        assert_eq!(
            formatter
                .render_with_tools(&request, None, tools.as_ref(), None)
                .unwrap(),
            r#"{"type": "function", "function": {"description": null, "name": "weather", "parameters": {"properties": {"z": {"type": "string"}, "a": {"enum": ["x", "y"]}}}, "strict": false}, "defer_loading": null}"#
        );
    }

    #[test]
    fn python_no_tools_view_is_none_for_null_empty_and_choice_none() {
        let config =
            temp_config(r#"{"chat_template":"{{ 'absent' if tools is none else 'present' }}"}"#);
        let formatter =
            load_chat_formatter(Some(config.to_str().unwrap()), None, None, None).unwrap();
        for raw in [serde_json::Value::Null, serde_json::json!([])] {
            assert_eq!(
                formatter
                    .render_with_tools(&request(), None, Some(&raw), None)
                    .unwrap(),
                "absent"
            );
        }
        let request = tools_request(Some(serde_json::json!("none")));
        let tools =
            super::normalize_prompt_tools(&serde_json::to_value(&request.tools).unwrap()).unwrap();
        assert_eq!(
            formatter
                .render_with_tools(&request, None, tools.as_ref(), None)
                .unwrap(),
            "absent"
        );
    }

    #[test]
    fn named_tool_filter_does_not_change_native_or_legacy_rendering() {
        let request = tools_request(Some(serde_json::json!({
            "type": "function", "function": {"name": "get_weather"}
        })));
        let native = load_chat_formatter(None, None, Some("deepseek_v4"), None).unwrap();
        let rendered = native
            .render_with_tools(&request, None, Some(&serde_json::Value::Null), None)
            .unwrap();
        assert!(rendered.contains("get_weather"));
        assert!(rendered.contains("get_time"));

        let legacy = load_chat_formatter(None, None, None, Some("chatml")).unwrap();
        assert_eq!(
            legacy.render(&request, None).unwrap(),
            legacy
                .render(&tools_request(Some(serde_json::json!("auto"))), None)
                .unwrap()
        );
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
        assert!(formatter.stop_strs().is_none());
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
        assert!(matches!(formatter, ChatFormatter::HuggingFace(_)));
        assert!(formatter.stop_strs().is_none());
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
