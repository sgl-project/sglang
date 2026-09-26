//! Chat-template loading and model-path inference.

use std::path::Path;

use dynamo_renderer::{
    ChatTemplate, ContextMixins, PromptContextMixin, PromptFormatter, native_formatter_for,
};
use serde_json::Value;

use sglang_renderer::{
    LegacyFormatter, builtin_template, infer_legacy_template_from_model_path, parse_legacy_template,
};

use super::template::{ChatFormatter, TemplateError};

pub(super) fn load_chat_formatter(
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
        && let Some(spec) = infer_legacy_template_from_model_path(model_path)
    {
        tracing::info!(%model_path, "inferred legacy chat template from model path");
        return Ok(ChatFormatter::Legacy(Box::new(LegacyFormatter { spec })));
    }

    // Models that ship no template (Python `resolve_chat_encoding_spec`) use
    // Dynamo's built-in encoder for their architecture.
    let native_formatter = || match chat_template_arg {
        None => native_formatter_for(
            &model_type.map(str::to_lowercase),
            &model_path.unwrap_or_default().to_lowercase(),
        )
        .map(ChatFormatter::HuggingFace),
        Some(_) => None,
    };

    // Every remaining source builds the HF renderer around the tokenizer
    // config (the template itself, or the argument injected into it).
    let Some(config_file) = config_file else {
        return native_formatter().ok_or(TemplateError::MissingConfig);
    };

    let config_path = Path::new(config_file);
    let config_text = read_to_string(config_path, "tokenizer config")?;
    let mut config = parse_json(&config_text, config_path, "tokenizer config")?;

    let Some(argument) = chat_template_arg else {
        return match formatter_from_config(&config) {
            Err(TemplateError::Missing) => native_formatter().ok_or(TemplateError::Missing),
            result => result,
        };
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
