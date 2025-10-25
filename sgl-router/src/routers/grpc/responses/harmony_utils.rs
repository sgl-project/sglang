//! Harmony model detection and utilities
//!
//! Provides helper functions for detecting GPT-OSS (Harmony) models
//! and determining when to use Harmony-specific processing.

/// Check if a model uses Harmony framework
///
/// Harmony models are OpenAI's open-weight gpt-oss models that were specifically
/// trained on the harmony response format.
///
/// # Arguments
/// * `model_name` - The model identifier (e.g., "gpt-oss", "openai/gpt-oss-20b")
///
/// # Returns
/// `true` if the model uses Harmony, `false` otherwise
pub fn is_harmony_model(model_name: &str) -> bool {
    let model_lower = model_name.to_lowercase();
    model_lower.contains("gpt-oss")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmony_model_detection() {
        // Harmony models (gpt-oss only)
        assert!(is_harmony_model("gpt-oss"));
        assert!(is_harmony_model("gpt-oss-20b"));
        assert!(is_harmony_model("gpt-oss-120b"));
        assert!(is_harmony_model("GPT-OSS")); // Case insensitive
        assert!(is_harmony_model("openai/gpt-oss-20b"));
        assert!(is_harmony_model("/models/openai/gpt-oss-20b"));

        // Non-Harmony models
        assert!(!is_harmony_model("llama-3"));
        assert!(!is_harmony_model("claude-3"));
        assert!(!is_harmony_model("gpt-4"));
        assert!(!is_harmony_model("gpt-4o")); // Not a harmony model
        assert!(!is_harmony_model("gpt-4.5"));
        assert!(!is_harmony_model("gpt-5"));
        assert!(!is_harmony_model("gpt-3.5"));
        assert!(!is_harmony_model("mistral"));
    }
}
