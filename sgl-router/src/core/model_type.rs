//! Model type definitions using bitflags for endpoint support.
//!
//! This module defines [`ModelType`] using bitflags to represent which endpoints
//! a model can support. This allows combining capabilities like
//! `ModelType::CHAT | ModelType::COMPLETIONS`.
//!
//! Inspired by Dynamo's model_type.rs implementation.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

bitflags! {
    #[derive(Copy, Debug, Default, Clone, Eq, PartialEq, Hash)]
    pub struct ModelType: u16 {
        /// OpenAI Chat Completions API (/v1/chat/completions)
        const CHAT        = 1 << 0;
        /// OpenAI Completions API - legacy (/v1/completions)
        const COMPLETIONS = 1 << 1;
        /// OpenAI Responses API (/v1/responses)
        const RESPONSES   = 1 << 2;
        /// Embeddings API (/v1/embeddings)
        const EMBEDDINGS  = 1 << 3;
        /// Rerank API (/v1/rerank)
        const RERANK      = 1 << 4;
        /// SGLang Generate API (/generate)
        const GENERATE    = 1 << 5;
        /// Vision/multimodal support (images in input)
        const VISION      = 1 << 6;
        /// Tool/function calling support
        const TOOLS       = 1 << 7;
        /// Reasoning/thinking support (e.g., o1, DeepSeek-R1)
        const REASONING   = 1 << 8;

        // === Convenience combinations ===
        // Note: Within bitflags! macro, we must use .bits() for combining flags

        /// Standard LLM: chat + completions + responses + tools
        const LLM = Self::CHAT.bits() | Self::COMPLETIONS.bits()
                  | Self::RESPONSES.bits() | Self::TOOLS.bits();

        /// Vision-capable LLM: LLM + vision
        const VISION_LLM = Self::LLM.bits() | Self::VISION.bits();

        /// Reasoning LLM: LLM + reasoning (e.g., o1, o3, DeepSeek-R1)
        const REASONING_LLM = Self::LLM.bits() | Self::REASONING.bits();

        /// Full-featured LLM: all text generation capabilities
        const FULL_LLM = Self::VISION_LLM.bits() | Self::REASONING.bits();

        /// Embedding model only
        const EMBED_MODEL = Self::EMBEDDINGS.bits();

        /// Reranker model only
        const RERANK_MODEL = Self::RERANK.bits();
    }
}

/// Mapping of individual capability flags to their names.
/// Used by `as_capability_names()` for a data-driven approach.
const CAPABILITY_NAMES: &[(ModelType, &str)] = &[
    (ModelType::CHAT, "chat"),
    (ModelType::COMPLETIONS, "completions"),
    (ModelType::RESPONSES, "responses"),
    (ModelType::EMBEDDINGS, "embeddings"),
    (ModelType::RERANK, "rerank"),
    (ModelType::GENERATE, "generate"),
    (ModelType::VISION, "vision"),
    (ModelType::TOOLS, "tools"),
    (ModelType::REASONING, "reasoning"),
];

impl ModelType {
    /// Check if this model type supports the chat completions endpoint
    #[inline]
    pub fn supports_chat(&self) -> bool {
        self.contains(Self::CHAT)
    }

    /// Check if this model type supports the legacy completions endpoint
    #[inline]
    pub fn supports_completions(&self) -> bool {
        self.contains(Self::COMPLETIONS)
    }

    /// Check if this model type supports the responses endpoint
    #[inline]
    pub fn supports_responses(&self) -> bool {
        self.contains(Self::RESPONSES)
    }

    /// Check if this model type supports the embeddings endpoint
    #[inline]
    pub fn supports_embeddings(&self) -> bool {
        self.contains(Self::EMBEDDINGS)
    }

    /// Check if this model type supports the rerank endpoint
    #[inline]
    pub fn supports_rerank(&self) -> bool {
        self.contains(Self::RERANK)
    }

    /// Check if this model type supports the generate endpoint
    #[inline]
    pub fn supports_generate(&self) -> bool {
        self.contains(Self::GENERATE)
    }

    /// Check if this model type supports vision/multimodal input
    #[inline]
    pub fn supports_vision(&self) -> bool {
        self.contains(Self::VISION)
    }

    /// Check if this model type supports tool/function calling
    #[inline]
    pub fn supports_tools(&self) -> bool {
        self.contains(Self::TOOLS)
    }

    /// Check if this model type supports reasoning/thinking
    #[inline]
    pub fn supports_reasoning(&self) -> bool {
        self.contains(Self::REASONING)
    }

    /// Check if this model type supports a given endpoint
    pub fn supports_endpoint(&self, endpoint: Endpoint) -> bool {
        match endpoint {
            Endpoint::Chat => self.supports_chat(),
            Endpoint::Completions => self.supports_completions(),
            Endpoint::Responses => self.supports_responses(),
            Endpoint::Embeddings => self.supports_embeddings(),
            Endpoint::Rerank => self.supports_rerank(),
            Endpoint::Generate => self.supports_generate(),
            Endpoint::Models => true, // Models endpoint is always supported
        }
    }

    /// Convert to a list of supported capability names
    pub fn as_capability_names(&self) -> Vec<&'static str> {
        let mut result = Vec::with_capacity(CAPABILITY_NAMES.len());
        for &(flag, name) in CAPABILITY_NAMES {
            if self.contains(flag) {
                result.push(name);
            }
        }
        result
    }

    /// Check if this is an LLM (supports at least chat)
    #[inline]
    pub fn is_llm(&self) -> bool {
        self.supports_chat()
    }

    /// Check if this is an embedding model
    #[inline]
    pub fn is_embedding_model(&self) -> bool {
        self.supports_embeddings() && !self.supports_chat()
    }

    /// Check if this is a reranker model
    #[inline]
    pub fn is_reranker(&self) -> bool {
        self.supports_rerank() && !self.supports_chat()
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names = self.as_capability_names();
        if names.is_empty() {
            write!(f, "none")
        } else {
            write!(f, "{}", names.join(","))
        }
    }
}

// Custom Serialize/Deserialize for ModelType to handle bitflags properly
impl Serialize for ModelType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize as the underlying u16 bits
        serializer.serialize_u16(self.bits())
    }
}

impl<'de> Deserialize<'de> for ModelType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bits = u16::deserialize(deserializer)?;
        ModelType::from_bits(bits)
            .ok_or_else(|| serde::de::Error::custom(format!("invalid ModelType bits: {}", bits)))
    }
}

/// Endpoint types for routing decisions.
///
/// This enum represents the different API endpoints that can be routed to workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Endpoint {
    /// Chat completions endpoint (/v1/chat/completions)
    Chat,
    /// Legacy completions endpoint (/v1/completions)
    Completions,
    /// Responses endpoint (/v1/responses)
    Responses,
    /// Embeddings endpoint (/v1/embeddings)
    Embeddings,
    /// Rerank endpoint (/v1/rerank)
    Rerank,
    /// SGLang generate endpoint (/generate)
    Generate,
    /// Models listing endpoint (/v1/models)
    Models,
}

impl Endpoint {
    /// Get the URL path for this endpoint
    pub fn path(&self) -> &'static str {
        match self {
            Endpoint::Chat => "/v1/chat/completions",
            Endpoint::Completions => "/v1/completions",
            Endpoint::Responses => "/v1/responses",
            Endpoint::Embeddings => "/v1/embeddings",
            Endpoint::Rerank => "/v1/rerank",
            Endpoint::Generate => "/generate",
            Endpoint::Models => "/v1/models",
        }
    }

    /// Parse an endpoint from a URL path
    pub fn from_path(path: &str) -> Option<Self> {
        // Normalize: strip trailing slash and match
        let path = path.trim_end_matches('/');
        match path {
            "/v1/chat/completions" => Some(Endpoint::Chat),
            "/v1/completions" => Some(Endpoint::Completions),
            "/v1/responses" => Some(Endpoint::Responses),
            "/v1/embeddings" => Some(Endpoint::Embeddings),
            "/v1/rerank" => Some(Endpoint::Rerank),
            "/generate" => Some(Endpoint::Generate),
            "/v1/models" => Some(Endpoint::Models),
            _ => None,
        }
    }

    /// Get the required ModelType flag for this endpoint
    pub fn required_capability(&self) -> Option<ModelType> {
        match self {
            Endpoint::Chat => Some(ModelType::CHAT),
            Endpoint::Completions => Some(ModelType::COMPLETIONS),
            Endpoint::Responses => Some(ModelType::RESPONSES),
            Endpoint::Embeddings => Some(ModelType::EMBEDDINGS),
            Endpoint::Rerank => Some(ModelType::RERANK),
            Endpoint::Generate => Some(ModelType::GENERATE),
            Endpoint::Models => None, // No specific capability required
        }
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Endpoint::Chat => write!(f, "chat"),
            Endpoint::Completions => write!(f, "completions"),
            Endpoint::Responses => write!(f, "responses"),
            Endpoint::Embeddings => write!(f, "embeddings"),
            Endpoint::Rerank => write!(f, "rerank"),
            Endpoint::Generate => write!(f, "generate"),
            Endpoint::Models => write!(f, "models"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_individual_flags() {
        let chat_only = ModelType::CHAT;
        assert!(chat_only.supports_chat());
        assert!(!chat_only.supports_completions());
        assert!(!chat_only.supports_embeddings());
    }

    #[test]
    fn test_model_type_combinations() {
        let llm = ModelType::LLM;
        assert!(llm.supports_chat());
        assert!(llm.supports_completions());
        assert!(llm.supports_responses());
        assert!(llm.supports_tools());
        assert!(!llm.supports_embeddings());
        assert!(!llm.supports_vision());
    }

    #[test]
    fn test_model_type_custom_combination() {
        let custom = ModelType::CHAT | ModelType::EMBEDDINGS;
        assert!(custom.supports_chat());
        assert!(custom.supports_embeddings());
        assert!(!custom.supports_completions());
        assert!(!custom.supports_tools());
    }

    #[test]
    fn test_model_type_vision_llm() {
        let vision = ModelType::VISION_LLM;
        assert!(vision.supports_chat());
        assert!(vision.supports_completions());
        assert!(vision.supports_responses());
        assert!(vision.supports_tools());
        assert!(vision.supports_vision());
        assert!(!vision.supports_embeddings());
        assert!(!vision.supports_reasoning());
    }

    #[test]
    fn test_model_type_reasoning_llm() {
        let reasoning = ModelType::REASONING_LLM;
        assert!(reasoning.supports_chat());
        assert!(reasoning.supports_reasoning());
        assert!(!reasoning.supports_vision());
    }

    #[test]
    fn test_model_type_full_llm() {
        let full = ModelType::FULL_LLM;
        assert!(full.supports_chat());
        assert!(full.supports_completions());
        assert!(full.supports_responses());
        assert!(full.supports_tools());
        assert!(full.supports_vision());
        assert!(full.supports_reasoning());
        assert!(!full.supports_embeddings());
    }

    #[test]
    fn test_model_type_supports_endpoint() {
        let llm = ModelType::LLM;
        assert!(llm.supports_endpoint(Endpoint::Chat));
        assert!(llm.supports_endpoint(Endpoint::Completions));
        assert!(llm.supports_endpoint(Endpoint::Responses));
        assert!(llm.supports_endpoint(Endpoint::Models)); // Always true
        assert!(!llm.supports_endpoint(Endpoint::Embeddings));
        assert!(!llm.supports_endpoint(Endpoint::Rerank));
    }

    #[test]
    fn test_model_type_as_capability_names() {
        let llm = ModelType::LLM;
        let names = llm.as_capability_names();
        assert!(names.contains(&"chat"));
        assert!(names.contains(&"completions"));
        assert!(names.contains(&"responses"));
        assert!(names.contains(&"tools"));
        assert!(!names.contains(&"embeddings"));
    }

    #[test]
    fn test_model_type_display() {
        let llm = ModelType::LLM;
        let display = llm.to_string();
        assert!(display.contains("chat"));
        assert!(display.contains("completions"));
    }

    #[test]
    fn test_model_type_is_llm() {
        assert!(ModelType::LLM.is_llm());
        assert!(ModelType::CHAT.is_llm());
        assert!(!ModelType::EMBEDDINGS.is_llm());
        assert!(!ModelType::RERANK.is_llm());
    }

    #[test]
    fn test_model_type_is_embedding_model() {
        assert!(ModelType::EMBED_MODEL.is_embedding_model());
        assert!(ModelType::EMBEDDINGS.is_embedding_model());
        // LLM with embeddings is not an "embedding model"
        let llm_with_embed = ModelType::LLM | ModelType::EMBEDDINGS;
        assert!(!llm_with_embed.is_embedding_model());
    }

    #[test]
    fn test_model_type_is_reranker() {
        assert!(ModelType::RERANK_MODEL.is_reranker());
        assert!(ModelType::RERANK.is_reranker());
        // LLM with rerank is not a "reranker"
        let llm_with_rerank = ModelType::LLM | ModelType::RERANK;
        assert!(!llm_with_rerank.is_reranker());
    }

    #[test]
    fn test_model_type_default() {
        let default = ModelType::default();
        assert!(default.is_empty());
        assert!(!default.supports_chat());
    }

    #[test]
    fn test_model_type_serialization() {
        let llm = ModelType::LLM;
        let json = serde_json::to_string(&llm).unwrap();
        let deserialized: ModelType = serde_json::from_str(&json).unwrap();
        assert_eq!(llm, deserialized);
    }

    #[test]
    fn test_endpoint_path() {
        assert_eq!(Endpoint::Chat.path(), "/v1/chat/completions");
        assert_eq!(Endpoint::Embeddings.path(), "/v1/embeddings");
        assert_eq!(Endpoint::Generate.path(), "/generate");
    }

    #[test]
    fn test_endpoint_from_path() {
        assert_eq!(
            Endpoint::from_path("/v1/chat/completions"),
            Some(Endpoint::Chat)
        );
        assert_eq!(
            Endpoint::from_path("/v1/chat/completions/"),
            Some(Endpoint::Chat)
        );
        assert_eq!(
            Endpoint::from_path("/v1/embeddings"),
            Some(Endpoint::Embeddings)
        );
        assert_eq!(Endpoint::from_path("/unknown"), None);
    }

    #[test]
    fn test_endpoint_required_capability() {
        assert_eq!(Endpoint::Chat.required_capability(), Some(ModelType::CHAT));
        assert_eq!(
            Endpoint::Embeddings.required_capability(),
            Some(ModelType::EMBEDDINGS)
        );
        assert_eq!(Endpoint::Models.required_capability(), None);
    }

    #[test]
    fn test_endpoint_display() {
        assert_eq!(Endpoint::Chat.to_string(), "chat");
        assert_eq!(Endpoint::Embeddings.to_string(), "embeddings");
    }

    #[test]
    fn test_endpoint_serialization() {
        let endpoint = Endpoint::Chat;
        let json = serde_json::to_string(&endpoint).unwrap();
        assert_eq!(json, "\"chat\"");
        let deserialized: Endpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(endpoint, deserialized);
    }
}
