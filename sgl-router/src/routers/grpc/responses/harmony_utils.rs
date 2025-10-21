//! Harmony model detection and utilities
//!
//! Provides helper functions for detecting GPT-OSS (Harmony) models
//! and determining when to use Harmony-specific processing.

/// Check if a model uses Harmony framework
///
/// Harmony models include:
/// - gpt-oss (explicit Harmony models)
/// - gpt-4o, gpt-4.5, gpt-5 (OpenAI models with Harmony support)
///
/// # Arguments
/// * `model_name` - The model identifier (e.g., "gpt-oss", "gpt-4o")
///
/// # Returns
/// `true` if the model uses Harmony, `false` otherwise
pub fn is_harmony_model(model_name: &str) -> bool {
    let model_lower = model_name.to_lowercase();

    model_lower.contains("gpt-oss")
        || model_lower.contains("gpt-4o")
        || model_lower.contains("gpt-4.5")
        || model_lower.contains("gpt-5")
        || model_lower.contains("gpt5")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmony_model_detection() {
        // Harmony models
        assert!(is_harmony_model("gpt-oss"));
        assert!(is_harmony_model("gpt-oss-v2"));
        assert!(is_harmony_model("GPT-OSS")); // Case insensitive
        assert!(is_harmony_model("gpt-4o"));
        assert!(is_harmony_model("gpt-4.5"));
        assert!(is_harmony_model("gpt-5"));
        assert!(is_harmony_model("gpt5"));

        // Non-Harmony models
        assert!(!is_harmony_model("llama-3"));
        assert!(!is_harmony_model("claude-3"));
        assert!(!is_harmony_model("gpt-4"));
        assert!(!is_harmony_model("gpt-3.5"));
        assert!(!is_harmony_model("mistral"));
    }
}
