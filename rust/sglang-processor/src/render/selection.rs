//! Chat formatter selection from a model's template files and identity.

use std::sync::Arc;

use dynamo_renderer::deepseek::v4::DeepSeekV4Formatter;
use dynamo_renderer::deepseek::v32::DeepSeekV32Formatter;
use dynamo_renderer::{PromptFormatter, kimi_k3_formatter_for, native_formatter_for};

use super::models::resolve_dsv4_profile;
use super::{ChatFormatter, ThinkingTemplates, load_chat_formatter};
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
