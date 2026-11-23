//! Chat formatter selection from a model's template files and identity.

use std::sync::Arc;

use dynamo_renderer::deepseek::v32::DeepSeekV32Formatter;
use dynamo_renderer::{PromptFormatter, kimi_k3_formatter_for, native_formatter_for};

use super::legacy::LegacyFormatter;
use super::loader::{DeveloperErrors, load_template_file_or_config, resolve_builtin_template};
use super::{ChatFormatter, TemplateError, ThinkingTemplates};
use crate::model_files::{resolve_chat_template_file, resolve_model_file, resolve_tokenizer_file};

/// Model and template sources used to select a chat formatter.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatFormatterOptions {
    pub tokenizer_path: String,
    pub model_path: String,
    pub revision: Option<String>,
    pub chat_template: Option<String>,
}

#[derive(Debug, Default, serde::Serialize)]
pub(super) struct ModelIdentity {
    model_type: Option<String>,
    architectures: Vec<String>,
}

impl ChatFormatterOptions {
    pub(super) fn model_source(&self) -> &str {
        if self.model_path.is_empty() {
            &self.tokenizer_path
        } else {
            &self.model_path
        }
    }

    pub(super) fn identity(&self) -> ModelIdentity {
        resolve_model_file(self.model_source(), self.revision.as_deref(), "config.json")
            .as_deref()
            .map(load_model_identity)
            .unwrap_or_default()
    }
}

/// Select the model's chat formatter. Loading errors are returned as a message
/// so hosts can defer them until a chat request needs the template.
pub fn select_chat_formatter(
    config: &ChatFormatterOptions,
) -> (Option<ChatFormatter>, Option<String>) {
    if config.tokenizer_path.is_empty() {
        return (None, None);
    }
    let selected = resolve_formatter(config, &config.identity());
    let loaded = match selected {
        SelectedFormatter::DeepSeekRecipe { v41 } => {
            // Prompt ids need the checkpoint tokenizer; without it only text renders.
            let tokenizer =
                resolve_tokenizer_file(&config.tokenizer_path, config.revision.as_deref())
                    .and_then(|file| std::fs::read_to_string(file).ok());
            ChatFormatter::deepseek_v4(v41, tokenizer.as_deref())
        }
        SelectedFormatter::Native(_, formatter) => Ok(formatter),
        SelectedFormatter::Builtin(formatter) => Ok(ChatFormatter::Legacy(formatter)),
        SelectedFormatter::Template { loaded, .. } => {
            loaded.map_err(|error| format!("this model has no usable chat template: {error}"))
        }
    };
    match loaded {
        Ok(formatter) => (Some(formatter), None),
        Err(error) => (None, Some(error)),
    }
}

/// Resolution chooses the formatter and its fingerprint inputs together. Only
/// the rendering host constructs the DeepSeek recipe's checkpoint tokenizer.
pub(super) enum SelectedFormatter {
    DeepSeekRecipe {
        v41: bool,
    },
    Native(Option<&'static str>, ChatFormatter),
    Builtin(Box<LegacyFormatter>),
    Template {
        path: Option<String>,
        loaded: Result<ChatFormatter, TemplateError>,
    },
}

pub(super) fn resolve_formatter(
    config: &ChatFormatterOptions,
    identity: &ModelIdentity,
) -> SelectedFormatter {
    let tokenizer_config_file = resolve_model_file(
        &config.tokenizer_path,
        config.revision.as_deref(),
        "tokenizer_config.json",
    );
    let model_type_lower = identity.model_type.as_deref().map(str::to_ascii_lowercase);
    let display_name_lower = config.model_source().to_ascii_lowercase();
    if config.chat_template.is_none() {
        if identity.is_deepseek_v41() || identity.is_deepseek_v4() {
            return SelectedFormatter::DeepSeekRecipe {
                v41: identity.is_deepseek_v41(),
            };
        }
        if kimi_k3_formatter_for(&model_type_lower, &display_name_lower, true).is_some() {
            return SelectedFormatter::Native(Some("kimi_k3"), ChatFormatter::KimiK3);
        }
        if model_type_lower.as_deref() == Some("inkling_mm_model")
            && let Some(formatter) = native_formatter_for(&model_type_lower, &display_name_lower)
        {
            return SelectedFormatter::Native(
                Some("inkling"),
                ChatFormatter::HuggingFace {
                    formatter,
                    thinking: ThinkingTemplates::always(),
                    developer_errors: DeveloperErrors::default(),
                },
            );
        }
    }
    // Python's `load_chat_template` guesses a legacy template from the model
    // path before it reads the checkpoint's own template.
    if let Some(spec) = resolve_builtin_template(
        (!config.model_path.is_empty()).then_some(config.model_path.as_str()),
        identity.model_type.as_deref(),
        config.chat_template.as_deref(),
    ) {
        return SelectedFormatter::Builtin(Box::new(LegacyFormatter { spec }));
    }
    let path = config
        .chat_template
        .clone()
        .or_else(|| resolve_chat_template_file(&config.tokenizer_path, config.revision.as_deref()));
    let loaded = load_template_file_or_config(
        tokenizer_config_file.as_deref(),
        identity.model_type.as_deref(),
        path.as_deref(),
    );
    // Python's `resolve_chat_encoding_spec`: a model's own encoder only when it
    // ships no template.
    if matches!(
        loaded,
        Err(TemplateError::Missing | TemplateError::MissingConfig)
    ) && config.chat_template.is_none()
        && let Some(formatter) = native_fallback(identity, &model_type_lower, &display_name_lower)
    {
        return formatter;
    }
    let loaded = loaded.map(|mut formatter| {
        if identity.is_kimi_k25()
            && let ChatFormatter::HuggingFace {
                formatter: inner,
                thinking,
                ..
            } = formatter
        {
            formatter = ChatFormatter::KimiK25 {
                formatter: inner,
                thinking,
            };
        }
        tracing::info!(
            config = ?tokenizer_config_file.as_deref().unwrap_or("<built-in / inferred>"),
            template = ?path,
            "loaded OpenAI chat template"
        );
        formatter
    });
    SelectedFormatter::Template { path, loaded }
}

fn native_fallback(
    identity: &ModelIdentity,
    model_type_lower: &Option<String>,
    display_name_lower: &str,
) -> Option<SelectedFormatter> {
    if identity.is_deepseek_v32() {
        return Some(SelectedFormatter::Native(
            Some("deepseek_v32_chat"),
            ChatFormatter::HuggingFace {
                formatter: PromptFormatter::OAI(Arc::new(DeepSeekV32Formatter::new_chat())),
                thinking: ThinkingTemplates::native(false, false),
                developer_errors: DeveloperErrors::default(),
            },
        ));
    }
    // Let Dynamo decide whether and how to render, using the original inputs.
    // Its erased formatter has no family accessor. The label below is only
    // for the digest; an unrecognized family can still serve chat requests.
    let formatter = native_formatter_for(model_type_lower, display_name_lower)?;
    let family = match model_type_lower.as_deref() {
        Some("deepseek_v41") => Some("deepseek_v41"),
        Some("deepseek_v4") => Some("deepseek_v4"),
        Some("deepseek_v32" | "deepseek_v3_2") => Some("deepseek_v32"),
        Some(_) => None,
        None => deepseek_fallback_type(display_name_lower),
    };
    Some(SelectedFormatter::Native(
        family,
        ChatFormatter::HuggingFace {
            formatter,
            // The remaining display-name fallback formatters are constructed with
            // their thinking mode enabled.
            thinking: ThinkingTemplates::native(true, false),
            developer_errors: DeveloperErrors::default(),
        },
    ))
}

/// Dynamo's name-only fallback. V4 names must start with the family and end
/// the version at a separator, while V3.2 also matches qualified names.
fn deepseek_fallback_type(name: &str) -> Option<&'static str> {
    if let Some(rest) = name.strip_prefix("deepseek") {
        let version = rest.strip_prefix(['-', '_', '.']).unwrap_or(rest);
        for (prefix, family) in [("v4.1", "deepseek_v41"), ("v4", "deepseek_v4")] {
            if version
                .strip_prefix(prefix)
                .is_some_and(|suffix| suffix.is_empty() || suffix.starts_with(['-', '_', '.']))
            {
                return Some(family);
            }
        }
    }
    // Dynamo has already excluded experimental checkpoints.
    (name.contains("deepseek") && name.contains("v3.2")).then_some("deepseek_v32")
}

impl ModelIdentity {
    /// `chat_encoding.is_deepseek_v41_arch`; V4.1 may keep the V4 architecture name.
    fn is_deepseek_v41(&self) -> bool {
        self.model_type.as_deref() == Some("deepseek_v41")
            || self
                .architectures
                .first()
                .is_some_and(|architecture| architecture.contains("DeepseekV41"))
    }

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

fn load_model_identity(config_file: &str) -> ModelIdentity {
    let Ok(config) = std::fs::read_to_string(config_file) else {
        return ModelIdentity::default();
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&config) else {
        return ModelIdentity::default();
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
    ModelIdentity {
        model_type,
        architectures,
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatFormatterOptions, select_chat_formatter};
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
}
