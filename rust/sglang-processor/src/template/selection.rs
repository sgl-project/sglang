//! Chat formatter selection from a model's template files and identity.

use std::sync::Arc;

use dynamo_renderer::deepseek::v4::DeepSeekV4Formatter;
use dynamo_renderer::deepseek::v32::DeepSeekV32Formatter;
use dynamo_renderer::{PromptFormatter, kimi_k3_formatter_for, native_formatter_for};

use super::{ChatFormatter, DeepSeekV4Profile, ThinkingTemplates, load_chat_formatter};
use crate::model_files::{resolve_chat_template_file, resolve_model_file};

const DSV4_REASONING_EFFORT_ENV: &str = "SGLANG_DSV4_REASONING_EFFORT";

/// Model and template sources used to select a chat formatter.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatFormatterOptions {
    pub tokenizer_path: String,
    pub model_path: String,
    pub revision: Option<String>,
    pub chat_template: Option<String>,
}

#[derive(Debug, Default)]
struct ModelIdentity {
    model_type: Option<String>,
    architectures: Vec<String>,
    dsv4_reasoning_effort_profile: Option<String>,
}

/// Select the model's chat formatter. Loading errors are returned as a message
/// so hosts can defer them until a chat request needs the template.
pub fn select_chat_formatter(
    config: &ChatFormatterOptions,
) -> (Option<ChatFormatter>, Option<String>) {
    if config.tokenizer_path.is_empty() {
        return (None, None);
    }
    let tokenizer_config_file = resolve_model_file(
        &config.tokenizer_path,
        config.revision.as_deref(),
        "tokenizer_config.json",
    );
    let model_source = if config.model_path.is_empty() {
        config.tokenizer_path.as_str()
    } else {
        config.model_path.as_str()
    };
    let model_config_file =
        resolve_model_file(model_source, config.revision.as_deref(), "config.json");
    let identity = match model_config_file.as_deref().map(load_model_identity) {
        Some(Err(error)) => return (None, Some(error)),
        Some(Ok(identity)) => identity,
        None => ModelIdentity::default(),
    };
    let model_type_lower = identity.model_type.as_deref().map(str::to_ascii_lowercase);
    let display_name_lower = model_source.to_ascii_lowercase();
    if config.chat_template.is_none() {
        if identity.is_deepseek_v4() {
            let profile = match resolve_dsv4_profile(
                identity.dsv4_reasoning_effort_profile.as_deref(),
                model_source,
                config.revision.as_deref(),
            ) {
                Ok(profile) => profile,
                Err(error) => return (None, Some(error)),
            };
            return (
                Some(ChatFormatter::DeepSeekV4 {
                    formatter: PromptFormatter::OAI(Arc::new(DeepSeekV4Formatter::new_chat())),
                    profile,
                    environment_effort: std::env::var(DSV4_REASONING_EFFORT_ENV).ok(),
                }),
                None,
            );
        }
        if identity.is_deepseek_v32() {
            return (
                Some(ChatFormatter::HuggingFace {
                    formatter: PromptFormatter::OAI(Arc::new(DeepSeekV32Formatter::new_chat())),
                    thinking: ThinkingTemplates::native(false, false),
                }),
                None,
            );
        }
        if let Some(formatter) = kimi_k3_formatter_for(&model_type_lower, &display_name_lower, true)
        {
            return (
                Some(ChatFormatter::HuggingFace {
                    formatter,
                    thinking: ThinkingTemplates::native(true, true),
                }),
                None,
            );
        }
        if model_type_lower.as_deref() == Some("inkling_mm_model")
            && let Some(formatter) = native_formatter_for(&model_type_lower, &display_name_lower)
        {
            return (
                Some(ChatFormatter::HuggingFace {
                    formatter,
                    thinking: ThinkingTemplates::always(),
                }),
                None,
            );
        }
        if let Some(formatter) = native_formatter_for(&model_type_lower, &display_name_lower) {
            return (
                Some(ChatFormatter::HuggingFace {
                    formatter,
                    // The remaining display-name fallback formatters are
                    // constructed with their thinking mode enabled.
                    thinking: ThinkingTemplates::native(true, false),
                }),
                None,
            );
        }
    }
    let discovered_template = config
        .chat_template
        .is_none()
        .then(|| resolve_chat_template_file(&config.tokenizer_path, config.revision.as_deref()))
        .flatten();
    let template_source = config
        .chat_template
        .as_deref()
        .or(discovered_template.as_deref());
    match load_chat_formatter(
        tokenizer_config_file.as_deref(),
        (!config.model_path.is_empty()).then_some(config.model_path.as_str()),
        identity.model_type.as_deref(),
        template_source,
    ) {
        Ok(mut formatter) => {
            if identity.is_kimi_k25()
                && let ChatFormatter::HuggingFace {
                    formatter: inner,
                    thinking,
                } = formatter
            {
                formatter = ChatFormatter::KimiK25 {
                    formatter: inner,
                    thinking,
                };
            }
            tracing::info!(
                config = ?tokenizer_config_file.as_deref().unwrap_or("<built-in / inferred>"),
                template = ?template_source,
                "loaded OpenAI chat template"
            );
            (Some(formatter), None)
        }
        Err(error) => {
            tracing::warn!(%error, "OpenAI chat completions disabled");
            (
                None,
                Some(format!("this model has no usable chat template: {error}")),
            )
        }
    }
}

impl ModelIdentity {
    fn is_deepseek_v4(&self) -> bool {
        self.model_type.as_deref() == Some("deepseek_v4")
            || self
                .architectures
                .iter()
                .any(|architecture| architecture.starts_with("DeepseekV4"))
    }

    fn is_deepseek_v32(&self) -> bool {
        matches!(
            self.model_type.as_deref(),
            Some("deepseek_v32" | "deepseek_v3_2")
        ) || self
            .architectures
            .iter()
            .any(|architecture| architecture == "DeepseekV32ForCausalLM")
    }

    fn is_kimi_k25(&self) -> bool {
        self.model_type.as_deref() == Some("kimi_k25")
            || self
                .architectures
                .iter()
                .any(|architecture| architecture == "KimiK25ForConditionalGeneration")
    }
}

fn load_model_identity(config_file: &str) -> Result<ModelIdentity, String> {
    let Ok(config) = std::fs::read_to_string(config_file) else {
        return Ok(ModelIdentity::default());
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&config) else {
        return Ok(ModelIdentity::default());
    };
    let model_type = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let architectures = config
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .map(|architectures| {
            architectures
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let dsv4_reasoning_effort_profile = match config.get("dsv4_reasoning_effort_profile") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(profile)) => Some(profile.clone()),
        Some(profile) => {
            return Err(format!(
                "invalid dsv4_reasoning_effort_profile: {profile}; expected \"preview\" or \"official\""
            ));
        }
    };
    Ok(ModelIdentity {
        model_type,
        architectures,
        dsv4_reasoning_effort_profile,
    })
}

pub(super) fn resolve_dsv4_profile(
    profile: Option<&str>,
    model_source: &str,
    revision: Option<&str>,
) -> Result<DeepSeekV4Profile, String> {
    if let Some(profile) = profile {
        return match profile {
            "preview" => Ok(DeepSeekV4Profile::Preview),
            "official" => Ok(DeepSeekV4Profile::Official),
            _ => Err(format!(
                "invalid dsv4_reasoning_effort_profile: {profile:?}; expected \"preview\" or \"official\""
            )),
        };
    }
    let Some(encoder) = resolve_model_file(model_source, revision, "encoding/encoding_dsv4.py")
    else {
        return Ok(DeepSeekV4Profile::Preview);
    };
    let Ok(metadata) = std::fs::metadata(&encoder) else {
        return Ok(DeepSeekV4Profile::Preview);
    };
    if metadata.len() > 1 << 20 {
        return Ok(DeepSeekV4Profile::Preview);
    }
    let Ok(source) = std::fs::read_to_string(encoder) else {
        return Ok(DeepSeekV4Profile::Preview);
    };
    let default = top_level_python_assignment(&source, "DEFAULT_REASONING_EFFORT")
        .and_then(python_string_literal);
    let prompt_keys = top_level_python_assignment(&source, "REASONING_EFFORT_PROMPTS")
        .and_then(python_dict_keys)
        .unwrap_or_default();
    if default.as_deref() == Some("low")
        && ["low", "high", "max"]
            .iter()
            .all(|key| prompt_keys.iter().any(|candidate| candidate == key))
    {
        Ok(DeepSeekV4Profile::Official)
    } else {
        Ok(DeepSeekV4Profile::Preview)
    }
}

fn top_level_python_assignment<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if !trimmed.starts_with(char::is_whitespace)
            && let Some((target, _)) = trimmed.split_once('=')
            && target
                .split(':')
                .next()
                .is_some_and(|target| target.trim() == name)
        {
            let equals = line.find('=')?;
            return Some(&source[offset + equals + 1..]);
        }
        offset += line.len();
    }
    None
}

fn python_string_literal(source: &str) -> Option<String> {
    let source = source.trim_start();
    let quote = source.chars().next()?;
    if !matches!(quote, '\'' | '"') {
        return None;
    }
    let mut escaped = false;
    let mut value = String::new();
    for character in source[quote.len_utf8()..].chars() {
        if escaped {
            value.push(character);
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character == quote {
            return Some(value);
        } else {
            value.push(character);
        }
    }
    None
}

fn python_dict_keys(source: &str) -> Option<Vec<String>> {
    let source = source.trim_start();
    if !source.starts_with('{') {
        return None;
    }
    let mut keys = Vec::new();
    let mut depth = 0usize;
    let mut index = 0usize;
    let bytes = source.as_bytes();
    while index < bytes.len() {
        match bytes[index] {
            b'{' | b'[' | b'(' => {
                depth += 1;
                index += 1;
            }
            b'}' | b']' | b')' => {
                depth = depth.checked_sub(1)?;
                index += 1;
                if depth == 0 {
                    return Some(keys);
                }
            }
            quote @ (b'\'' | b'"') => {
                let start = index + 1;
                index = start;
                let mut escaped = false;
                while index < bytes.len() {
                    if escaped {
                        escaped = false;
                    } else if bytes[index] == b'\\' {
                        escaped = true;
                    } else if bytes[index] == quote {
                        break;
                    }
                    index += 1;
                }
                if index == bytes.len() {
                    return None;
                }
                let value = std::str::from_utf8(&bytes[start..index]).ok()?;
                index += 1;
                if depth == 1 {
                    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
                        index += 1;
                    }
                    if bytes.get(index) == Some(&b':') {
                        keys.push(value.to_owned());
                    }
                }
            }
            b'#' => {
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            _ => index += 1,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        ChatFormatterOptions, DeepSeekV4Profile, resolve_dsv4_profile, select_chat_formatter,
    };
    use crate::OneOrMany;

    #[test]
    fn built_in_chatml_preserves_stop_strings() {
        let (formatter, error) = select_chat_formatter(&ChatFormatterOptions {
            tokenizer_path: ".".into(),
            chat_template: Some("chatml".into()),
            ..Default::default()
        });
        assert!(error.is_none());
        let Some(OneOrMany::Many(stops)) = formatter.unwrap().stop_strs() else {
            panic!("chatml declares multiple stop strings");
        };
        assert_eq!(stops, ["<|endoftext|>", "<|im_end|>"]);
    }

    #[test]
    fn deepseek_v4_profile_resolution_uses_override_then_checkpoint_source() {
        let directory = std::env::temp_dir().join(format!(
            "sglang-processor-deepseek-v4-profile-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(directory.join("encoding")).unwrap();
        std::fs::write(
            directory.join("encoding/encoding_dsv4.py"),
            "DEFAULT_REASONING_EFFORT: str = 'low'\n\
REASONING_EFFORT_PROMPTS = {'low': '', 'high': 'absolute', 'max': 'beyond'}",
        )
        .unwrap();
        let source = directory.to_string_lossy();
        assert_eq!(
            resolve_dsv4_profile(None, &source, None).unwrap(),
            DeepSeekV4Profile::Official
        );
        assert_eq!(
            resolve_dsv4_profile(Some("preview"), &source, None).unwrap(),
            DeepSeekV4Profile::Preview
        );
        assert!(resolve_dsv4_profile(Some("future"), &source, None).is_err());

        std::fs::write(
            directory.join("encoding/encoding_dsv4.py"),
            r#"DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORT_PROMPTS = {"low": "", "high": "absolute", "max": "Beyond maximum"}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_dsv4_profile(None, &source, None).unwrap(),
            DeepSeekV4Profile::Preview
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
