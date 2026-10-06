//! Resolve the `--chat-template` argument and the tokenizer config to a formatter.

use std::path::Path;

use dynamo_renderer::{ChatTemplate, ContextMixins, PromptContextMixin, PromptFormatter};
use serde_json::Value;

use super::legacy::{
    LegacyFormatter, builtin_template, infer_legacy_template_from_model_path, parse_legacy_template,
};
use super::thinking::ThinkingTemplates;
use super::{ChatFormatter, TemplateError};

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
        return ChatFormatter::from_tokenizer_config(&config);
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
        return ChatFormatter::from_tokenizer_config(&config);
    }

    let template_text = read_to_string(path, "chat template")?;
    let template = parse_json(&template_text, path, "chat template")?;

    // HF-style JSON files may carry chat_template directly. Legacy SGLang
    // files carry Conversation fields and are translated below.
    if template.is_string() {
        set_chat_template(&mut config, template)?;
        ChatFormatter::from_tokenizer_config(&config)
    } else if let Some(chat_template) = template.get("chat_template") {
        set_chat_template(&mut config, chat_template.clone())?;
        ChatFormatter::from_tokenizer_config(&config)
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
    /// contents, which must carry a `chat_template`.
    pub fn from_tokenizer_config(config: &Value) -> Result<Self, TemplateError> {
        let thinking = ThinkingTemplates::from_config(config);
        let template: ChatTemplate = serde_json::from_value(config.clone())
            .map_err(|source| TemplateError::Config { source })?;
        if template.chat_template.is_none() {
            return Err(TemplateError::Missing);
        }
        let formatter = PromptFormatter::from_parts(
            template,
            ContextMixins::new(&[PromptContextMixin::OaiChat]),
            true,
        )
        .map_err(|error| TemplateError::Renderer {
            message: error.to_string(),
        })?;
        Ok(Self::HuggingFace {
            formatter,
            thinking,
        })
    }
}

#[cfg(test)]
mod tests {
    use dynamo_protocols::types::CreateChatCompletionRequest;

    use super::super::{ChatFormatterOptions, select_chat_formatter};
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
}
