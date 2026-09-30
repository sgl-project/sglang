use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequest,
};

use super::thinking::{ThinkingPolicy, detect_thinking_policy};
use super::{
    ChatFormatter, ChatFormatterOptions, LegacyFormatter, LegacySpec, OneOrMany, TemplateError,
    builtin_template, infer_legacy_template_from_model_path, load_chat_formatter,
    select_chat_formatter,
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

#[test]
fn thinking_toggle_detection_matches_template_defaults() {
    assert_eq!(
        detect_thinking_policy(
            "{% if enable_thinking is not defined %}{% set enable_thinking = true %}{% endif %}"
        ),
        ThinkingPolicy::TemplateToggle {
            key: "enable_thinking",
            default_enabled: true,
        }
    );
    assert_eq!(
        detect_thinking_policy(
            "{% if not thinking is defined %}{% set thinking = false %}{% endif %}"
        ),
        ThinkingPolicy::TemplateToggle {
            key: "thinking",
            default_enabled: false,
        }
    );
    assert_eq!(
        detect_thinking_policy("{{ messages }}"),
        ThinkingPolicy::Unknown
    );
}

#[test]
fn thinking_default_filters_match_python_semantics() {
    let cases = [
        ("{{ enable_thinking | d(true) }}", true),
        ("{{ enable_thinking|default(false) }}", false),
        ("{{ enable_thinking | default(true, false) }}", true),
        ("{{ enable_thinking | default(false, true) }}", false),
        (
            "{{ enable_thinking | default(false, boolean=true) }}",
            false,
        ),
        (
            "{% if enable_thinking | default(false) %}x{% endif %}",
            false,
        ),
    ];
    for (template, default_enabled) in cases {
        assert_eq!(
            detect_thinking_policy(template),
            ThinkingPolicy::TemplateToggle {
                key: "enable_thinking",
                default_enabled,
            },
            "{template}"
        );
    }
}

#[test]
fn thinking_detection_ignores_non_executable_toggle_text() {
    for template in [
        "{# {{ enable_thinking | default(false) }} #}{{ enable_thinking | default(true) }}",
        "{% raw %}{{ enable_thinking | default(false) }}{% endraw %}{{ enable_thinking | default(true) }}",
        "{{ 'enable_thinking | default(false)' }}{{ enable_thinking | default(true) }}",
    ] {
        assert_eq!(
            detect_thinking_policy(template),
            ThinkingPolicy::TemplateToggle {
                key: "enable_thinking",
                default_enabled: true,
            },
            "{template}"
        );
    }
    assert_eq!(
        detect_thinking_policy("{{ enable_thinking | default(true, true) }}"),
        ThinkingPolicy::Unknown
    );
    assert_eq!(
        detect_thinking_policy("{% if enable_thinking %}think{% endif %}"),
        ThinkingPolicy::Unknown
    );
}

#[test]
fn channel_templates_are_always_on() {
    assert_eq!(
        detect_thinking_policy("<|start|>assistant<|channel|>analysis<|message|>"),
        ThinkingPolicy::Always
    );
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
    let rendered = formatter.render(&request()).unwrap();
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
fn json_legacy_template_is_rendered_natively() {
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
    let rendered = formatter.render(&request()).unwrap();
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

    // Model-type matcher.
    let formatter = load_chat_formatter(None, Some("models/opaque"), Some("phi4mm"), None).unwrap();
    let ChatFormatter::Legacy(formatter) = &formatter else {
        panic!("expected a legacy formatter");
    };
    assert_eq!(formatter.spec.name, "phi-4-mm");

    // `select_chat_formatter` reads the model type from `<model_path>/config.json`.
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
    let (formatter, error) = select_chat_formatter(&ChatFormatterOptions {
        tokenizer_path: model_dir.to_str().unwrap().into(),
        model_path: model_dir.to_str().unwrap().into(),
        ..Default::default()
    });
    assert!(error.is_none(), "{error:?}");
    let Some(ChatFormatter::Legacy(formatter)) = &formatter else {
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
        let spec = infer_legacy_template_from_model_path(model_path, None)
            .unwrap_or_else(|| panic!("no inference for {model_path}"));
        let _ = spec;
    }
}

/// MiniCPM 4.6+ must NOT fall back to the legacy template; with no config
/// and nothing else to try, that surfaces as the missing-config error.
#[test]
fn minicpm_4_6_skips_legacy_inference() {
    assert!(infer_legacy_template_from_model_path("minicpm-v-4.6", None).is_none());
    assert!(matches!(
        load_chat_formatter(None, Some("minicpm-v-4.6"), None, None),
        Err(TemplateError::MissingConfig)
    ));
}
