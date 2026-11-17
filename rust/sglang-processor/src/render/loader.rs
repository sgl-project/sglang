//! Resolve the `--chat-template` argument and the tokenizer config to a formatter.

use std::path::Path;
use std::sync::Arc;

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::{
    ChatTemplate, ContextMixins, OAIChatLikeRequest, OAIPromptFormatter, PromptContextMixin,
    PromptFormatter,
};
use serde_json::{Value, json};

use super::legacy::{
    LegacyFormatter, builtin_template, infer_legacy_template_from_model_path, parse_legacy_template,
};
use super::thinking::ThinkingTemplates;
use super::{
    ChatFormatter, TemplateError, is_glm53, is_minimax_m3, is_qwen38, is_step3p5,
    preserve_messages, prints_arguments_as_xml, takes_content_parts,
};

pub fn load_chat_formatter(
    config_file: Option<&str>,
    model_path: Option<&str>,
    model_type: Option<&str>,
    chat_template_arg: Option<&str>,
) -> Result<ChatFormatter, TemplateError> {
    // Python resolves registry names before looking at the filesystem — and
    // before touching the tokenizer config, so a built-in name works even when
    // `tokenizer_config.json` is absent.
    if let Some(argument) = chat_template_arg
        && let Some(spec) = builtin_template(argument)
    {
        return Ok(ChatFormatter::Legacy(Box::new(LegacyFormatter { spec })));
    }

    // Python `load_chat_template` (no `--chat-template`): infer a legacy
    // template from the model path before falling back to the HF template, so
    // a legacy model whose config has no `chat_template` still gets one.
    if chat_template_arg.is_none()
        && let Some(model_path) = model_path
        && let Some(spec) = infer_legacy_template_from_model_path(model_path, model_type)
    {
        tracing::info!(%model_path, "inferred legacy chat template from model path");
        return Ok(ChatFormatter::Legacy(Box::new(LegacyFormatter { spec })));
    }

    // Every remaining source builds the HF renderer around the tokenizer
    // config (the template itself, or the argument injected into it).
    let Some(config_file) = config_file else {
        return Err(TemplateError::MissingConfig);
    };

    let config_path = Path::new(config_file);
    let config_text = read_to_string(config_path, "tokenizer config")?;
    let mut config = parse_json(&config_text, config_path, "tokenizer config")?;

    let Some(argument) = chat_template_arg else {
        return ChatFormatter::from_tokenizer_config_with_type(&config, model_type);
    };

    let path = Path::new(argument);
    if !path.exists() {
        return Err(TemplateError::NotFound {
            path: path.to_path_buf(),
        });
    }
    if !path.is_file() {
        return Err(TemplateError::NotFile {
            path: path.to_path_buf(),
        });
    }

    if path.extension().and_then(|extension| extension.to_str()) == Some("jinja") {
        let template = read_to_string(path, "chat template")?;
        set_chat_template(
            &mut config,
            Value::String(template.trim_matches('\n').replace("\\n", "\n")),
        )?;
        return ChatFormatter::from_tokenizer_config_with_type(&config, model_type);
    }

    let template_text = read_to_string(path, "chat template")?;
    let template = parse_json(&template_text, path, "chat template")?;

    // HF-style JSON files may carry chat_template directly. Legacy SGLang
    // files carry Conversation fields and are translated below.
    if template.is_string() {
        set_chat_template(&mut config, template)?;
        ChatFormatter::from_tokenizer_config_with_type(&config, model_type)
    } else if let Some(chat_template) = template.get("chat_template") {
        set_chat_template(&mut config, chat_template.clone())?;
        ChatFormatter::from_tokenizer_config_with_type(&config, model_type)
    } else {
        Ok(ChatFormatter::Legacy(Box::new(LegacyFormatter {
            spec: parse_legacy_template(&template, path)?,
        })))
    }
}

fn read_to_string(path: &Path, kind: &'static str) -> Result<String, TemplateError> {
    std::fs::read_to_string(path).map_err(|source| TemplateError::Read {
        kind,
        path: path.to_path_buf(),
        source,
    })
}

fn parse_json(text: &str, path: &Path, kind: &'static str) -> Result<Value, TemplateError> {
    serde_json::from_str(text).map_err(|source| TemplateError::Parse {
        kind,
        path: path.to_path_buf(),
        source,
    })
}

fn set_chat_template(config: &mut Value, chat_template: Value) -> Result<(), TemplateError> {
    let Some(config) = config.as_object_mut() else {
        return Err(TemplateError::ConfigNotObject);
    };
    config.insert("chat_template".to_string(), chat_template);
    Ok(())
}

impl ChatFormatter {
    /// Build the HuggingFace formatter from parsed `tokenizer_config.json`
    /// contents, which must carry a `chat_template`; GLM-5.3's, MiniMax-M3's,
    /// Qwen3.8's and StepFun's templates build [`ChatFormatter::Glm53`],
    /// [`ChatFormatter::MiniMaxM3`], [`ChatFormatter::Qwen38`] and
    /// [`ChatFormatter::Step3p5`].
    pub fn from_tokenizer_config(config: &Value) -> Result<Self, TemplateError> {
        Self::from_tokenizer_config_with_type(config, None)
    }

    /// [`Self::from_tokenizer_config`], with the checkpoint's `config.json`
    /// `model_type` for the models Python keys on it. MiniMax-M3 handling is
    /// gated on `model_type == "minimax_m3_vl"` (`serving_chat.py`), not the
    /// template text: a `minimax_m3_vl` checkpoint served with a foreign
    /// `--chat-template` still gets M3's rules, and a template that merely
    /// contains M3's tags does not. `None` falls back to template detection
    /// for callers without the checkpoint config.
    pub fn from_tokenizer_config_with_type(
        config: &Value,
        model_type: Option<&str>,
    ) -> Result<Self, TemplateError> {
        let glm53 = config["chat_template"].as_str().is_some_and(is_glm53);
        let minimax_m3 = model_type.map_or_else(
            || config["chat_template"].as_str().is_some_and(is_minimax_m3),
            |model_type| model_type == "minimax_m3_vl",
        );
        let qwen38 = config["chat_template"].as_str().filter(|t| is_qwen38(t));
        let step3p5 = config["chat_template"].as_str().is_some_and(is_step3p5);
        let thinking = ThinkingTemplates::from_config(config);
        let mut render_config = config.clone();
        let message_context = if glm53 {
            String::new()
        } else if minimax_m3 {
            preserve_messages(&mut render_config, "__sglang_minimax_m3_messages")
        } else if qwen38.is_none() && step3p5 {
            preserve_messages(&mut render_config, "__sglang_step3p5_messages")
        } else {
            String::new()
        };
        let template: ChatTemplate = serde_json::from_value::<ChatTemplate>(render_config)
            .map_err(|source| TemplateError::Config { source })?;
        if template.chat_template.is_none() {
            return Err(TemplateError::Missing);
        }
        let PromptFormatter::OAI(formatter) = PromptFormatter::from_parts(
            template,
            ContextMixins::new(&[PromptContextMixin::OaiChat]),
            true,
        )
        .map_err(|error| TemplateError::Renderer {
            message: error.to_string(),
        })?;
        let formatter = PromptFormatter::OAI(Arc::new(WholePrompt(formatter)));
        Ok(if glm53 {
            Self::Glm53(formatter)
        } else if minimax_m3 {
            Self::MiniMaxM3 {
                formatter,
                message_context,
                xml_arguments: config["chat_template"]
                    .as_str()
                    .is_some_and(prints_arguments_as_xml),
                content_parts: config["chat_template"]
                    .as_str()
                    .and_then(super::content_format::takes_content_parts),
            }
        } else if let Some(template) = qwen38 {
            Self::Qwen38 {
                formatter,
                thinking,
                content_parts: takes_content_parts(template),
            }
        } else if step3p5 {
            Self::Step3p5 {
                formatter,
                thinking,
                message_context,
                content_parts: config["chat_template"]
                    .as_str()
                    .and_then(super::content_format::takes_content_parts),
            }
        } else {
            let developer_errors = DeveloperErrors::probe(config, &formatter);
            Self::HuggingFace {
                formatter,
                thinking,
                developer_errors,
            }
        })
    }
}

/// A Jinja template's prompt as one string. SGLang tokenizes the rendered
/// prompt whole, so a control-token spelling in request text encodes as that
/// control token; Dynamo's own prompt sets request text apart, and always does
/// for Harmony templates.
struct WholePrompt(Arc<dyn OAIPromptFormatter>);

impl OAIPromptFormatter for WholePrompt {
    fn supports_add_generation_prompt(&self) -> bool {
        self.0.supports_add_generation_prompt()
    }

    fn supports_native_tools(&self, request: &dyn OAIChatLikeRequest) -> bool {
        self.0.supports_native_tools(request)
    }

    fn render(&self, request: &dyn OAIChatLikeRequest) -> anyhow::Result<String> {
        self.0.render(request)
    }

    fn media_message_order(&self, request: &dyn OAIChatLikeRequest) -> Option<Vec<usize>> {
        self.0.media_message_order(request)
    }

    fn renders_continuation_natively(&self) -> bool {
        self.0.renders_continuation_natively()
    }
}

/// The errors a template raises on a leading developer turn, without and with
/// request tools, where Dynamo would render that turn as a system turn instead.
/// Python's template raises them. Dynamo probes the template once without tools
/// and once with one, and picks between the two by the request's tools.
#[derive(Clone, Default)]
pub struct DeveloperErrors {
    without_tools: Option<String>,
    with_tools: Option<String>,
}

impl DeveloperErrors {
    fn probe(config: &Value, formatter: &PromptFormatter) -> Self {
        let Some(probe) = developer_probe(config) else {
            return Self::default();
        };
        let error = |tools: Option<Value>| {
            let request = |role: &str| {
                let mut request = json!({
                    "model": "probe",
                    "messages": [{"role": role, "content": "s"}, {"role": "user", "content": "u"}]
                });
                if let Some(tools) = &tools {
                    request["tools"] = tools.clone();
                }
                serde_json::from_value::<CreateChatCompletionRequest>(request).ok()
            };
            super::render_oai(formatter, &request("system")?).ok()?;
            let Err(TemplateError::Renderer { message }) =
                super::render_oai(&probe, &request("system")?)
            else {
                return None;
            };
            // Dynamo renders the developer turn only when it folds it into a system turn.
            super::render_oai(formatter, &request("developer")?).ok()?;
            Some(message)
        };
        Self {
            without_tools: error(None),
            // Dynamo's tool probe.
            with_tools: error(Some(json!([{
                "type": "function",
                "function": {
                    "name": "probe",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]))),
        }
    }

    /// The error for this request, by the tools Dynamo renders it with.
    pub(super) fn for_request(&self, request: &dyn OAIChatLikeRequest) -> Option<&str> {
        if !super::has_developer_turn(request) {
            return None;
        }
        let excluded = request
            .tool_choice()
            .is_some_and(|choice| choice.as_str() == Some("none"));
        let has_tools = !excluded
            && request
                .tools()
                .and_then(|tools| tools.len())
                .is_some_and(|len| len > 0);
        if has_tools {
            &self.with_tools
        } else {
            &self.without_tools
        }
        .as_deref()
    }
}

/// The template behind a prefix that renames the leading system turn to a
/// developer turn. Dynamo rewrites a developer turn before the template sees
/// it, so a probe sends a system turn and the prefix renames it past that
/// rewrite, keeping Dynamo's template syntax and content handling.
fn developer_probe(config: &Value) -> Option<PromptFormatter> {
    let source = config["chat_template"].as_str()?;
    let mut config = config.clone();
    config["chat_template"] = Value::String(format!(
        "{{%- set sglang_probe = namespace(messages=[]) -%}}\
         {{%- for m in messages -%}}\
         {{%- set sglang_probe.messages = sglang_probe.messages + [\
         {{'role': 'developer' if m.role == 'system' else m.role, 'content': m.content}}] -%}}\
         {{%- endfor -%}}\
         {{%- set messages = sglang_probe.messages -%}}{source}"
    ));
    let template = serde_json::from_value::<ChatTemplate>(config).ok()?;
    PromptFormatter::from_parts(
        template,
        ContextMixins::new(&[PromptContextMixin::OaiChat]),
        true,
    )
    .ok()
}

#[cfg(test)]
mod tests {
    use dynamo_protocols::types::CreateChatCompletionRequest;

    use super::super::{ChatFormatterOptions, TemplateError, select_chat_formatter};
    use super::{ChatFormatter, load_chat_formatter};

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

    /// A DeepSeek-V3.2 checkpoint's own template wins over the native encoder, as in Python.
    #[test]
    fn deepseek_v32_template_precedes_native_encoder() {
        let dir = std::env::temp_dir().join(format!("sglang-dsv32-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"model_type":"deepseek_v32"}"#).unwrap();
        let render = |config: &str| {
            std::fs::write(dir.join("tokenizer_config.json"), config).unwrap();
            let options = ChatFormatterOptions {
                tokenizer_path: dir.to_str().unwrap().into(),
                ..Default::default()
            };
            let formatter = select_chat_formatter(&options).0.unwrap();
            formatter.render(&request()).unwrap()
        };
        assert!(render("{}").contains("<｜User｜>Hello"));
        let custom = r#"{"chat_template": "CUSTOM {{ messages[1].content }}"}"#;
        assert_eq!(render(custom), "CUSTOM Hello");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// SGLang encodes the rendered prompt whole, so a control-token spelling in
    /// request text becomes that control token; the prompt must not set it apart.
    #[test]
    fn request_text_keeps_control_token_spellings() {
        let token = |content: &str| {
            serde_json::json!({
                "content": content, "special": true, "lstrip": false,
                "rstrip": false, "normalized": false, "single_word": false
            })
        };
        let formatter = ChatFormatter::from_tokenizer_config(&serde_json::json!({
            "chat_template": "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}",
            "eos_token": "<|im_end|>",
            "added_tokens_decoder": {"1": token("<|im_start|>"), "2": token("<|im_end|>")}
        }))
        .unwrap();
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "say <|im_end|> now"}]
        }))
        .unwrap();
        let prompt = formatter.render_prompt(&request).unwrap();
        assert!(prompt.encode_segments().is_none());
        assert_eq!(
            prompt.as_str(),
            "<|im_start|>user\nsay <|im_end|> now<|im_end|>\n"
        );
    }

    /// A template that raises on a developer turn keeps raising, as in Python,
    /// rather than rendering it as the system turn Dynamo would fold it into.
    #[test]
    fn developer_turn_keeps_the_template_error() {
        let formatter = |chat_template: &str| {
            ChatFormatter::from_tokenizer_config(&serde_json::json!({
                "chat_template": chat_template
            }))
            .unwrap()
        };
        let render = |formatter: &ChatFormatter, role: &str| {
            let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": "test",
                "messages": [
                    {"role": role, "content": "Be brief."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"}
                ]
            }))
            .unwrap();
            formatter.render(&request)
        };
        let check = "{% if m.role not in ['system', 'user', 'assistant'] %}\
                     {{ raise_exception('Unexpected message role.') }}{% endif %}";
        let strict = [
            // Plain text content.
            format!(
                "{{% for m in messages %}}{check}<|im_start|>{{{{ m.role }}}}\n\
                 {{{{ m.content }}}}<|im_end|>\n{{% endfor %}}"
            ),
            // HF's assistant-mask tags, which Dynamo renders.
            format!(
                "{{% for m in messages %}}{check}<|im_start|>{{{{ m.role }}}}\n\
                 {{% if m.role == 'assistant' %}}{{% generation %}}{{{{ m.content }}}}\
                 {{% endgeneration %}}{{% else %}}{{{{ m.content }}}}{{% endif %}}\
                 <|im_end|>\n{{% endfor %}}"
            ),
            // Content parts only, which Dynamo hands over as arrays.
            format!(
                "{{% for m in messages %}}{check}\
                 {{% if m.content is string %}}{{{{ raise_exception('Content must be a list.') }}}}\
                 {{% endif %}}<|im_start|>{{{{ m.role }}}}\n\
                 {{% for part in m.content %}}{{{{ part.text }}}}{{% endfor %}}<|im_end|>\n\
                 {{% endfor %}}"
            ),
        ];
        for template in &strict {
            let strict = formatter(template);
            assert!(render(&strict, "system").is_ok(), "{template}");
            match render(&strict, "developer") {
                Err(TemplateError::Renderer { message }) => {
                    assert!(message.contains("Unexpected message role."), "{message}")
                }
                other => panic!("{template}: {other:?}"),
            }
        }

        let lenient = formatter(
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}",
        );
        assert!(
            render(&lenient, "developer")
                .unwrap()
                .contains("<|im_start|>developer\n")
        );
    }

    /// A Harmony template's request text encodes as the whole prompt does in
    /// Python, a control-token spelling as that control token; Dynamo sets
    /// Harmony request text apart whatever the template config says.
    #[cfg(feature = "tokenizer")]
    #[test]
    fn harmony_request_text_encodes_like_the_whole_prompt() {
        use dynamo_tokenizers::EncodeSegment;

        use crate::tokenizer::{DynamoTokenizer, TextTokenizer};

        let special = |id: u32, content: &str| {
            serde_json::json!({
                "id": id, "content": content, "special": true, "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false
            })
        };
        let markers = [(1, "<|start|>"), (2, "<|message|>"), (3, "<|end|>")];
        let dir = std::env::temp_dir().join(format!("sglang-harmony-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("tokenizer.json");
        let tokenizer = serde_json::json!({
            "added_tokens": markers.iter().map(|(id, content)| special(*id, content)).collect::<Vec<_>>(),
            "pre_tokenizer": {"type": "Whitespace"},
            "model": {"type": "WordLevel", "unk_token": "[UNK]", "vocab": {
                "[UNK]": 0, "<|start|>": 1, "<|message|>": 2, "<|end|>": 3, "user": 4,
                "assistant": 5, "a": 6, "b": 7, "<|": 8, "end": 9, "|>": 10
            }},
        });
        std::fs::write(&file, tokenizer.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
        let tokenizer = DynamoTokenizer::load(file.to_str().unwrap(), None).unwrap();
        std::fs::remove_dir_all(&dir).ok();

        let formatter = ChatFormatter::from_tokenizer_config(&serde_json::json!({
            "chat_template": "{% for m in messages %}<|start|>{{ m.role }}<|message|>\
                              {{ m.content }}<|end|>{% endfor %}<|start|>assistant",
            "added_tokens_decoder": markers
                .iter()
                .map(|(id, content)| (id.to_string(), special(*id, content)))
                .collect::<serde_json::Map<_, _>>()
        }))
        .unwrap();
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "a <|end|> b"}]
        }))
        .unwrap();
        let prompt = formatter.render_prompt(&request).unwrap();
        let segments = prompt
            .encode_segments()
            .unwrap_or_else(|| vec![EncodeSegment::new(prompt.as_str(), true)]);
        let ids = tokenizer.encode_chat(&segments, "").unwrap();
        assert_eq!(ids, tokenizer.encode(prompt.as_str(), false).unwrap());
        assert_eq!(ids, [1, 4, 2, 6, 3, 7, 3, 1, 5]);
    }

    /// A template that raises on a developer turn only with tools, or only without,
    /// raises as Python's does on each kind of request.
    #[test]
    fn developer_turn_error_follows_the_request_tools() {
        let render = |raise_with_tools: bool, tools: bool, tool_choice: Option<&str>| {
            let condition = if raise_with_tools {
                "tools"
            } else {
                "not tools"
            };
            let formatter = ChatFormatter::from_tokenizer_config(&serde_json::json!({
                "chat_template": format!(
                    "{{% for m in messages %}}{{% if m.role == 'developer' and {condition} %}}\
                     {{{{ raise_exception('Unexpected message role.') }}}}{{% endif %}}\
                     <|im_start|>{{{{ m.role }}}}\n{{{{ m.content }}}}<|im_end|>\n{{% endfor %}}"
                )
            }))
            .unwrap();
            let mut body = serde_json::json!({
                "model": "test",
                "messages": [
                    {"role": "developer", "content": "Be brief."},
                    {"role": "user", "content": "Hi"}
                ]
            });
            if tools {
                body["tools"] = serde_json::json!([{
                    "type": "function",
                    "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}
                }]);
            }
            if let Some(choice) = tool_choice {
                body["tool_choice"] = choice.into();
            }
            let request: CreateChatCompletionRequest = serde_json::from_value(body).unwrap();
            formatter.render(&request)
        };
        let raises = |result: Result<String, TemplateError>| match result {
            Err(TemplateError::Renderer { message }) => {
                message.contains("Unexpected message role.")
            }
            Ok(prompt) => {
                assert!(prompt.starts_with("<|im_start|>developer\n"), "{prompt}");
                false
            }
            Err(error) => panic!("{error:?}"),
        };
        assert!(raises(render(true, true, None)));
        assert!(!raises(render(true, false, None)));
        assert!(!raises(render(true, true, Some("none"))));
        assert!(raises(render(false, false, None)));
        assert!(!raises(render(false, true, None)));
        assert!(raises(render(false, true, Some("none"))));
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
        let formatter =
            load_chat_formatter(None, Some("models/opaque"), Some("phi4mm"), None).unwrap();
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
        // Python guesses the legacy template before reading the checkpoint's.
        std::fs::write(model_dir.join("chat_template.jinja"), "{{ messages }}").unwrap();
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

    /// MiniMax-M3 keys on `model_type`, not the template text (Python's
    /// `is_minimax_m3`): a `minimax_m3_vl` checkpoint with a template that lacks
    /// M3's tags still gets M3's rules, and a template carrying M3's tags under
    /// another `model_type` does not. With no `model_type` (no `config.json`)
    /// the template is the only signal, so it falls back to detection.
    #[test]
    fn minimax_m3_keys_on_model_type() {
        let foreign = r#"{"chat_template": "{{ messages }}"}"#;
        let m3_template = r#"{"chat_template": "<mm:think>{{ messages }}"}"#;

        // model_type wins over a tagless template.
        let formatter = ChatFormatter::from_tokenizer_config_with_type(
            &serde_json::from_str(foreign).unwrap(),
            Some("minimax_m3_vl"),
        )
        .unwrap();
        assert!(matches!(formatter, ChatFormatter::MiniMaxM3 { .. }));

        // The same tagless template without the model_type is not M3.
        let formatter =
            ChatFormatter::from_tokenizer_config(&serde_json::from_str(foreign).unwrap()).unwrap();
        assert!(matches!(formatter, ChatFormatter::HuggingFace { .. }));

        // M3's tags under another model_type are not M3.
        let formatter = ChatFormatter::from_tokenizer_config_with_type(
            &serde_json::from_str(m3_template).unwrap(),
            Some("qwen3"),
        )
        .unwrap();
        assert!(matches!(formatter, ChatFormatter::HuggingFace { .. }));

        // No model_type: the template is the fallback signal.
        let formatter =
            ChatFormatter::from_tokenizer_config(&serde_json::from_str(m3_template).unwrap())
                .unwrap();
        assert!(matches!(formatter, ChatFormatter::MiniMaxM3 { .. }));
    }

    /// A `--chat-template` override keeps the checkpoint's `model_type`: a
    /// `minimax_m3_vl` checkpoint served with a foreign template still gets M3's rules.
    #[test]
    fn chat_template_override_keeps_minimax_m3_model_type() {
        let base = std::env::temp_dir().join(format!(
            "sglang-openai-template-m3-{}-test.json",
            std::process::id()
        ));
        let override_path = base.with_extension("jinja");
        std::fs::write(&base, r#"{"chat_template": "{{ messages }}"}"#).unwrap();
        std::fs::write(&override_path, "{{ messages }}").unwrap();
        let load = |model_type| {
            load_chat_formatter(
                Some(base.to_str().unwrap()),
                None,
                model_type,
                Some(override_path.to_str().unwrap()),
            )
            .unwrap()
        };
        assert!(matches!(
            load(Some("minimax_m3_vl")),
            ChatFormatter::MiniMaxM3 { .. }
        ));
        assert!(matches!(load(None), ChatFormatter::HuggingFace { .. }));

        let _ = std::fs::remove_file(base);
        let _ = std::fs::remove_file(override_path);
    }
}
