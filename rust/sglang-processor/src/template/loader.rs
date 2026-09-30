//! Chat-template loading and legacy model-path inference.

use std::path::Path;

use dynamo_renderer::{ChatTemplate, ContextMixins, PromptContextMixin, PromptFormatter};
use serde_json::Value;

use super::builtins::builtin_template;
use super::legacy::{LegacyFormatter, LegacySpec};
use super::{ChatFormatter, OneOrMany, TemplateError, ThinkingTemplates};

const SUPPORTED_STYLES: &[&str] = &[
    "ADD_COLON_SINGLE",
    "ADD_COLON_TWO",
    "ADD_COLON_SPACE_SINGLE",
    "NO_COLON_SINGLE",
    "NO_COLON_TWO",
    "ADD_NEW_LINE_SINGLE",
    "LLAMA2",
    "LLAMA3",
    "LLAMA4",
    "CHATGLM",
    "CHATML",
    "CHATINTERN",
    "DOLLY",
    "RWKV",
    "PHOENIX",
    "ROBIN",
    "FALCON_CHAT",
    "CHATGLM3",
    "DEEPSEEK_CHAT",
    "METAMATH",
    "DeepSeekVL2",
    "QWEN2_VL_EMBED",
    "QWEN2_AUDIO",
    "GEMMA3",
    "MPT",
    "PADDLE_OCR",
    "UNLIMITED_OCR",
];

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

/// Port of Python `get_conv_template_by_model_path` (conversation.py
/// `matching_function_registry`, run in registration order): infer a legacy
/// built-in template from the model path, optionally consulting the model's
/// `config.json` `model_type`. `None` when nothing matches — the HF template
/// is the fallback then, as in Python.
fn infer_legacy_template_from_model_path(
    model_path: &str,
    model_type: Option<&str>,
) -> Option<LegacySpec> {
    let lower = model_path.to_lowercase();
    // Regexes without regex: every Python pattern here is a plain substring or
    // a `prefix.*suffix` pair, both on a lowercased path.
    let contains = |needle: &str| lower.contains(needle);
    let precedes = |prefix: &str, suffix: &str| {
        lower
            .find(prefix)
            .is_some_and(|start| lower[start + prefix.len()..].contains(suffix))
    };

    if lower
        .split(|c: char| !c.is_alphanumeric())
        .any(|word| word == "points")
    {
        return builtin_template("points-v15-chat");
    }
    if precedes("moss", "vl") {
        return builtin_template("moss-vl");
    }
    if contains("internvl") {
        return builtin_template("internvl-2-5");
    }
    if contains("janus") {
        return builtin_template("janus-pro");
    }
    if contains("vicuna") || contains("llava-v1.5") || contains("llava-next-video-7b") {
        return builtin_template("vicuna_v1.1");
    }
    if precedes("deepseek", "vl2") {
        return builtin_template("deepseek-vl2");
    }
    if contains("llava-v1.6-34b")
        || contains("llava-v1.6-yi-34b")
        || contains("llava-next-video-34b")
        || contains("llava-onevision-qwen2")
    {
        return builtin_template("chatml-llava");
    }
    // MiniCPM: 4.6+ uses its own template and must not fall back to the
    // legacy conv template.
    if contains("minicpm-v-4.6")
        || contains("minicpm-v-4_6")
        || contains("minicpm-o-4.6")
        || contains("minicpm-o-4_6")
    {
        return None;
    }
    if contains("minicpm-v") {
        return builtin_template("minicpmv");
    }
    if contains("minicpm-o") {
        return builtin_template("minicpmo");
    }
    if contains("phi-4-multimodal") {
        return builtin_template("phi-4-mm");
    }
    if contains("deepseek-ocr") {
        return builtin_template("deepseek-ocr");
    }
    if contains("unlimited") {
        return builtin_template("unlimited-ocr");
    }
    if contains("paddleocr") {
        return builtin_template("paddle-ocr");
    }
    if contains("whisper") {
        return builtin_template("whisper");
    }

    // Python `MODEL_TYPE_TO_TEMPLATE`; minicpmv4_6 is deliberately absent.
    let name = match model_type? {
        "moss_vl" => "moss-vl",
        "internvl_chat" => "internvl-2-5",
        "multi_modality" => "janus-pro",
        "deepseek_vl_v2" => "deepseek-vl2",
        "minicpmv" => "minicpmv",
        "minicpmo" => "minicpmo",
        "phi4mm" => "phi-4-mm",
        "deepseek-ocr" => "deepseek-ocr",
        "unlimited-ocr" => "unlimited-ocr",
        "paddleocr_vl" => "paddle-ocr",
        _ => return None,
    };
    builtin_template(name)
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

/// Port of Python `_load_json_chat_template`: fields mirror `Conversation`
/// exactly (missing `sep2`/`image_token`/`audio_token` stay at Python defaults).
fn parse_legacy_template(value: &Value, path: &Path) -> Result<LegacySpec, TemplateError> {
    let object = value
        .as_object()
        .ok_or_else(|| TemplateError::LegacyNotObject {
            path: path.to_path_buf(),
        })?;

    let required_string = |name: &str| -> Result<String, TemplateError> {
        object
            .get(name)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .ok_or_else(|| TemplateError::LegacyMissingField {
                path: path.to_path_buf(),
                field: name.to_string(),
            })
    };

    let style = required_string("sep_style")?;
    if !SUPPORTED_STYLES.contains(&style.as_str()) {
        return Err(TemplateError::UnknownStyle {
            path: path.to_path_buf(),
            style,
        });
    }

    // Python `Conversation.stop_str: str | list[str] | None` — the key is
    // required (`template["stop_str"]` raises KeyError when missing), but an
    // explicit `null` value means `None`.
    let stop_str = match object.get("stop_str") {
        Some(Value::String(value)) => Some(OneOrMany::One(value.clone())),
        Some(Value::Array(values)) => {
            let strings = values
                .iter()
                .map(Value::as_str)
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| TemplateError::LegacyMissingField {
                    path: path.to_path_buf(),
                    field: "stop_str".to_string(),
                })?;
            Some(OneOrMany::Many(
                strings.into_iter().map(str::to_owned).collect(),
            ))
        }
        Some(Value::Null) => None,
        Some(_) => {
            return Err(TemplateError::LegacyMissingField {
                path: path.to_path_buf(),
                field: "stop_str".to_string(),
            });
        }
        None => {
            return Err(TemplateError::LegacyMissingField {
                path: path.to_path_buf(),
                field: "stop_str".to_string(),
            });
        }
    };

    Ok(LegacySpec {
        name: required_string("name")?,
        // Python: `system_template=template["system"] + "\n{system_message}"`.
        system_template: format!("{}\n{{system_message}}", required_string("system")?),
        system_message: object
            .get("system_message")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        roles: (required_string("user")?, required_string("assistant")?),
        style,
        sep: object
            .get("sep")
            .and_then(Value::as_str)
            .unwrap_or("\n")
            .to_string(),
        sep2: None,
        stop_str,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use dynamo_protocols::types::CreateChatCompletionRequest;

    use super::super::{ChatFormatter, ChatFormatterOptions, TemplateError, select_chat_formatter};
    use super::{infer_legacy_template_from_model_path, load_chat_formatter};

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
}
