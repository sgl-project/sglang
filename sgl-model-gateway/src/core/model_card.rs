//! Model card definitions for worker model configuration.
//!
//! This module defines [`ModelCard`] which consolidates model-related configuration
//! that was previously scattered in `WorkerMetadata.labels` HashMap.
//!
//! Also defines [`ProviderType`] for vendor-specific API transformations.
//!
//! Inspired by Dynamo's ModelDeploymentCard but simplified for router needs.

use serde::{Deserialize, Serialize};

use super::model_type::{Endpoint, ModelType};

/// Provider type for external API transformations.
///
/// Different providers have different API formats and requirements.
/// This enum identifies which vendor's API format to use for transformations.
///
/// Note: `None` (when used as `Option<ProviderType>`) means native/passthrough -
/// no transformation needed. This is the case for local SGLang backends.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// OpenAI API - strip SGLang-specific fields
    #[serde(alias = "openai")]
    OpenAI,
    /// xAI/Grok - special handling for input items
    #[serde(alias = "xai", alias = "grok")]
    XAI,
    /// Anthropic Claude - different API format
    #[serde(alias = "anthropic", alias = "claude")]
    Anthropic,
    /// Google Gemini - special logprobs handling
    #[serde(alias = "gemini", alias = "google")]
    Gemini,
    /// Custom provider with string identifier
    #[serde(untagged)]
    Custom(String),
}

impl ProviderType {
    /// Get provider name as string
    pub fn as_str(&self) -> &str {
        match self {
            Self::OpenAI => "openai",
            Self::XAI => "xai",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Detect provider from model name (heuristic fallback).
    /// Returns `None` for models that don't match known external providers
    /// (i.e., models served by local/native backends).
    pub fn from_model_name(model: &str) -> Option<Self> {
        let model_lower = model.to_lowercase();
        if model_lower.starts_with("grok") {
            Some(Self::XAI)
        } else if model_lower.starts_with("gemini") {
            Some(Self::Gemini)
        } else if model_lower.starts_with("claude") {
            Some(Self::Anthropic)
        } else if model_lower.starts_with("gpt")
            || model_lower.starts_with("o1")
            || model_lower.starts_with("o3")
        {
            Some(Self::OpenAI)
        } else {
            None // Native/local model, no provider transformation needed
        }
    }
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Model card containing model configuration and capabilities.
///
/// Consolidates fields previously scattered in `WorkerMetadata.labels`:
/// - `model_id` -> `id`
/// - `tokenizer_path` -> `tokenizer_path`
/// - `chat_template` -> `chat_template`
/// - `reasoning_parser` -> `reasoning_parser`
/// - `tool_parser` -> `tool_parser`
///
/// # Example
///
/// ```
/// use sgl_model_gateway::core::{ModelCard, ModelType, ProviderType};
///
/// let card = ModelCard::new("meta-llama/Llama-3.1-8B-Instruct")
///     .with_display_name("Llama 3.1 8B Instruct")
///     .with_alias("llama-3.1-8b")
///     .with_model_type(ModelType::VISION_LLM)
///     .with_context_length(128_000)
///     .with_tokenizer_path("meta-llama/Llama-3.1-8B-Instruct");
///
/// assert!(card.matches("llama-3.1-8b"));
/// assert!(card.model_type.supports_vision());
/// assert!(card.provider.is_none()); // Local model, no external provider
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    // === Identity ===
    /// Primary model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
    /// Previously: labels.get("model_id")
    pub id: String,

    /// Optional display name (e.g., "Llama 3.1 8B Instruct")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,

    /// Alternative names/aliases for this model
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,

    // === Capabilities ===
    /// Supported endpoint types (bitflags)
    #[serde(default = "default_model_type")]
    pub model_type: ModelType,

    /// HuggingFace model type string (e.g., "llama", "qwen2", "gpt-oss")
    /// This is different from `model_type` which is capability bitflags.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_model_type: Option<String>,

    /// Model architectures from HuggingFace config (e.g., ["LlamaForCausalLM"])
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub architectures: Vec<String>,

    /// Provider hint for API transformations.
    /// `None` means native/passthrough (no transformation needed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderType>,

    /// Maximum context length in tokens
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,

    // === Tokenization & Parsing (previously in labels) ===
    /// Path to tokenizer (e.g., HuggingFace model ID or local path)
    /// Previously: labels.get("tokenizer_path")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_path: Option<String>,

    /// Chat template (Jinja2 template string or path)
    /// Previously: labels.get("chat_template")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,

    /// Reasoning parser type (e.g., "deepseek", "qwen")
    /// Previously: labels.get("reasoning_parser")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,

    /// Tool/function calling parser type (e.g., "llama", "mistral")
    /// Previously: labels.get("tool_parser")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_parser: Option<String>,

    /// User-defined metadata (for fields not covered above)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

fn default_model_type() -> ModelType {
    ModelType::LLM
}

impl ModelCard {
    /// Create a new model card with minimal configuration.
    ///
    /// Defaults to `ModelType::LLM` and no provider (native/passthrough).
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: None,
            aliases: Vec::new(),
            model_type: ModelType::LLM,
            hf_model_type: None,
            architectures: Vec::new(),
            provider: None,
            context_length: None,
            tokenizer_path: None,
            chat_template: None,
            reasoning_parser: None,
            tool_parser: None,
            metadata: None,
        }
    }

    // === Builder-style methods ===

    /// Set the display name
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Add a single alias
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }

    /// Add multiple aliases
    pub fn with_aliases(mut self, aliases: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.aliases.extend(aliases.into_iter().map(|a| a.into()));
        self
    }

    /// Set the model type (capabilities)
    pub fn with_model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = model_type;
        self
    }

    /// Set the HuggingFace model type string
    pub fn with_hf_model_type(mut self, hf_model_type: impl Into<String>) -> Self {
        self.hf_model_type = Some(hf_model_type.into());
        self
    }

    /// Set the model architectures
    pub fn with_architectures(mut self, architectures: Vec<String>) -> Self {
        self.architectures = architectures;
        self
    }

    /// Set the provider type (for external API transformations)
    pub fn with_provider(mut self, provider: ProviderType) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the context length
    pub fn with_context_length(mut self, length: u32) -> Self {
        self.context_length = Some(length);
        self
    }

    /// Set the tokenizer path
    pub fn with_tokenizer_path(mut self, path: impl Into<String>) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }

    /// Set the chat template
    pub fn with_chat_template(mut self, template: impl Into<String>) -> Self {
        self.chat_template = Some(template.into());
        self
    }

    /// Set the reasoning parser type
    pub fn with_reasoning_parser(mut self, parser: impl Into<String>) -> Self {
        self.reasoning_parser = Some(parser.into());
        self
    }

    /// Set the tool parser type
    pub fn with_tool_parser(mut self, parser: impl Into<String>) -> Self {
        self.tool_parser = Some(parser.into());
        self
    }

    /// Set custom metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    // === Query methods ===

    /// Check if this model matches the given ID (including aliases)
    pub fn matches(&self, model_id: &str) -> bool {
        self.id == model_id || self.aliases.iter().any(|a| a == model_id)
    }

    /// Check if this model supports a given endpoint
    pub fn supports_endpoint(&self, endpoint: Endpoint) -> bool {
        self.model_type.supports_endpoint(endpoint)
    }

    /// Get the display name or fall back to ID
    pub fn name(&self) -> &str {
        self.display_name.as_deref().unwrap_or(&self.id)
    }

    /// Check if this is a native/local model (no external provider)
    #[inline]
    pub fn is_native(&self) -> bool {
        self.provider.is_none()
    }

    /// Check if this model uses an external provider
    #[inline]
    pub fn has_external_provider(&self) -> bool {
        self.provider.is_some()
    }

    /// Check if this is an LLM (supports chat)
    #[inline]
    pub fn is_llm(&self) -> bool {
        self.model_type.is_llm()
    }

    /// Check if this is an embedding model
    #[inline]
    pub fn is_embedding_model(&self) -> bool {
        self.model_type.is_embedding_model()
    }

    /// Check if this model supports vision/multimodal
    #[inline]
    pub fn supports_vision(&self) -> bool {
        self.model_type.supports_vision()
    }

    /// Check if this model supports tools/function calling
    #[inline]
    pub fn supports_tools(&self) -> bool {
        self.model_type.supports_tools()
    }

    /// Check if this model supports reasoning
    #[inline]
    pub fn supports_reasoning(&self) -> bool {
        self.model_type.supports_reasoning()
    }
}

impl Default for ModelCard {
    fn default() -> Self {
        Self::new("default")
    }
}

impl std::fmt::Display for ModelCard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
