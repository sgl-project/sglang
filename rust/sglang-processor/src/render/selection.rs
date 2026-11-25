//! Chat formatter selection from a model's template files and identity.

use dynamo_renderer::{kimi_k3_formatter_for, native_formatter_for};

use super::loader::load_chat_formatter_in_dir;
use super::{ChatFormatter, TemplateError, ThinkingTemplates};
use crate::model_files::{model_directory, resolve_model_file, resolve_tokenizer_file};

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
}

/// Select the model's chat formatter. Loading errors are returned as a message
/// so hosts can defer them until a chat request needs the template.
pub fn select_chat_formatter(
    config: &ChatFormatterOptions,
) -> (Option<ChatFormatter>, Option<String>) {
    if config.tokenizer_path.is_empty() {
        return (None, None);
    }
    let tokenizer_dir = model_directory(&config.tokenizer_path, config.revision.as_deref());
    let tokenizer_config = tokenizer_dir
        .as_ref()
        .map(|dir| dir.join("tokenizer_config.json"))
        .filter(|path| path.is_file());
    let model_source = if config.model_path.is_empty() {
        config.tokenizer_path.as_str()
    } else {
        config.model_path.as_str()
    };
    let model_config_file =
        resolve_model_file(model_source, config.revision.as_deref(), "config.json");
    let identity = model_config_file
        .as_deref()
        .map(load_model_identity)
        .unwrap_or_default();
    let model_type_lower = identity.model_type.as_deref().map(str::to_ascii_lowercase);
    let display_name_lower = model_source.to_ascii_lowercase();
    if config.chat_template.is_none() {
        if identity.is_deepseek_v41() || identity.is_deepseek_v4() {
            return deepseek_v4_formatter(config, identity.is_deepseek_v41());
        }
        if kimi_k3_formatter_for(&model_type_lower, &display_name_lower, true).is_some() {
            return (Some(ChatFormatter::KimiK3), None);
        }
        if model_type_lower.as_deref() == Some("inkling_mm_model")
            && let Some(formatter) = native_formatter_for(&model_type_lower, &display_name_lower)
        {
            return (
                Some(ChatFormatter::Native {
                    formatter,
                    thinking: ThinkingTemplates::always(),
                }),
                None,
            );
        }
    }
    let loaded = load_chat_formatter_in_dir(
        tokenizer_config.as_deref(),
        tokenizer_dir.as_deref(),
        (!config.model_path.is_empty()).then_some(config.model_path.as_str()),
        identity.model_type.as_deref(),
        config.chat_template.as_deref(),
    );
    // Python's `resolve_chat_encoding_spec`: a model's own encoder only when it
    // ships no template.
    if matches!(
        loaded,
        Err(TemplateError::Missing | TemplateError::MissingConfig)
    ) && config.chat_template.is_none()
        && let Some(formatter) = native_fallback(&identity, &model_type_lower, &display_name_lower)
    {
        return (Some(formatter), None);
    }
    match loaded {
        Ok(mut formatter) => {
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
                config = ?tokenizer_config,
                template = ?config.chat_template,
                "loaded OpenAI chat template"
            );
            (Some(formatter), None)
        }
        Err(error) => (
            None,
            Some(format!("this model has no usable chat template: {error}")),
        ),
    }
}

impl ChatFormatterOptions {
    /// Mirror the worker's tool-parser override when selecting its prompt encoder.
    pub fn select_for_tool_parser(
        &self,
        parser: Option<&str>,
    ) -> (Option<ChatFormatter>, Option<String>) {
        let selected = select_chat_formatter(self);
        // A selected conversation template bypasses the worker's custom encoders.
        if matches!(selected.0, Some(ChatFormatter::Legacy(_)))
            || (self.chat_template.is_some() && selected.1.is_some())
        {
            return selected;
        }
        match parser {
            Some("deepseekv32") => (Some(ChatFormatter::DeepSeekV32), None),
            Some("kimi_k3") => (Some(ChatFormatter::KimiK3), None),
            Some("deepseekv4") if !matches!(selected.0, Some(ChatFormatter::DeepSeekV4(_))) => {
                deepseek_v4_formatter(self, false)
            }
            Some("deepseekv41") if !matches!(selected.0, Some(ChatFormatter::DeepSeekV41(_))) => {
                deepseek_v4_formatter(self, true)
            }
            _ => selected,
        }
    }
}

fn deepseek_v4_formatter(
    config: &ChatFormatterOptions,
    v41: bool,
) -> (Option<ChatFormatter>, Option<String>) {
    let tokenizer = resolve_tokenizer_file(&config.tokenizer_path, config.revision.as_deref())
        .and_then(|file| std::fs::read_to_string(file).ok());
    match ChatFormatter::deepseek_v4(v41, tokenizer.as_deref()) {
        Ok(formatter) => (Some(formatter), None),
        Err(error) => (None, Some(error)),
    }
}

fn native_fallback(
    identity: &ModelIdentity,
    model_type_lower: &Option<String>,
    display_name_lower: &str,
) -> Option<ChatFormatter> {
    if identity.is_deepseek_v32() {
        return Some(ChatFormatter::DeepSeekV32);
    }
    Some(ChatFormatter::Native {
        formatter: native_formatter_for(model_type_lower, display_name_lower)?,
        // The remaining display-name fallback formatters are constructed with
        // their thinking mode enabled.
        thinking: ThinkingTemplates::native(true, false),
    })
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
