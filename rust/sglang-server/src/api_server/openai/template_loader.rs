//! Chat-template loading and model-path inference.

use std::path::Path;

use dynamo_renderer::{ChatTemplate, ContextMixins, PromptContextMixin, PromptFormatter};
use serde_json::Value;

use crate::message::types::OneOrMany;

use super::template::{ChatFormatter, TemplateError};
use super::template_builtins::builtin_template;
use super::template_legacy::{LegacyFormatter, LegacySpec};

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

pub(super) fn load_chat_formatter(
    config_file: Option<&str>,
    model_path: Option<&str>,
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
        && let Some(spec) = infer_legacy_template_from_model_path(model_path)
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
        return formatter_from_config(&config);
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
        return formatter_from_config(&config);
    }

    let template_text = read_to_string(path, "chat template")?;
    let template = parse_json(&template_text, path, "chat template")?;

    // HF-style JSON files may carry chat_template directly. Legacy SGLang
    // files carry Conversation fields and are translated below.
    if let Some(chat_template) = template.get("chat_template") {
        set_chat_template(&mut config, chat_template.clone())?;
        formatter_from_config(&config)
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
pub(super) fn infer_legacy_template_from_model_path(model_path: &str) -> Option<LegacySpec> {
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

    // Model-type matchers read `<model_path>/config.json` (local dirs only —
    // Python's `get_model_type` cannot resolve HF repo ids either).
    let model_type = read_model_type(model_path)?;
    // Python `MODEL_TYPE_TO_TEMPLATE`; minicpmv4_6 is deliberately absent.
    let name = match model_type.as_str() {
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

/// Python `get_model_type`: the `model_type` field of the model's `config.json`.
fn read_model_type(model_path: &str) -> Option<String> {
    let config_path = Path::new(model_path).join("config.json");
    if !config_path.is_file() {
        return None;
    }
    let config: Value = parse_json(
        &read_to_string(&config_path, "model config").ok()?,
        &config_path,
        "model config",
    )
    .ok()?;
    config.get("model_type")?.as_str().map(str::to_owned)
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

fn formatter_from_config(config: &Value) -> Result<ChatFormatter, TemplateError> {
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
    Ok(ChatFormatter::HuggingFace(formatter))
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
