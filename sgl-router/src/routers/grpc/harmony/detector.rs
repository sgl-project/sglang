//! Harmony model detection

/// Harmony model detector
///
/// Detects if a model name indicates support for Harmony encoding/parsing.
pub struct HarmonyDetector;

impl HarmonyDetector {
    /// Check if a model name indicates Harmony support
    ///
    /// Returns true if the model name matches known Harmony-capable models.
    ///
    /// # Examples
    ///
    /// ```
    /// use sglang_router_rs::routers::grpc::harmony::HarmonyDetector;
    ///
    /// assert!(HarmonyDetector::is_harmony_model("gpt-oss-4o"));
    /// assert!(HarmonyDetector::is_harmony_model("gpt-4o"));
    /// assert!(HarmonyDetector::is_harmony_model("gpt-4.5-turbo"));
    /// assert!(HarmonyDetector::is_harmony_model("gpt-5-preview"));
    /// assert!(!HarmonyDetector::is_harmony_model("gpt-4-turbo"));
    /// assert!(!HarmonyDetector::is_harmony_model("gpt-3.5-turbo"));
    /// ```
    pub fn is_harmony_model(model_name: &str) -> bool {
        // Convert to lowercase for case-insensitive matching
        let model_lower = model_name.to_lowercase();

        // Check for Harmony-capable model patterns
        model_lower.contains("gpt-oss")
            || model_lower.contains("gpt-4o")
            || model_lower.contains("gpt-4.5")
            || model_lower.starts_with("gpt-5")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_oss_models() {
        assert!(HarmonyDetector::is_harmony_model("gpt-oss"));
        assert!(HarmonyDetector::is_harmony_model("gpt-oss-4o"));
        assert!(HarmonyDetector::is_harmony_model("gpt-oss-preview"));
        assert!(HarmonyDetector::is_harmony_model("custom/gpt-oss-model"));
    }

    #[test]
    fn test_gpt_4o_models() {
        assert!(HarmonyDetector::is_harmony_model("gpt-4o"));
        assert!(HarmonyDetector::is_harmony_model("gpt-4o-mini"));
        assert!(HarmonyDetector::is_harmony_model("gpt-4o-2024-05-13"));
        assert!(HarmonyDetector::is_harmony_model("openai/gpt-4o"));
    }

    #[test]
    fn test_gpt_45_models() {
        assert!(HarmonyDetector::is_harmony_model("gpt-4.5"));
        assert!(HarmonyDetector::is_harmony_model("gpt-4.5-turbo"));
        assert!(HarmonyDetector::is_harmony_model("gpt-4.5-preview"));
    }

    #[test]
    fn test_gpt_5_models() {
        assert!(HarmonyDetector::is_harmony_model("gpt-5"));
        assert!(HarmonyDetector::is_harmony_model("gpt-5-preview"));
        assert!(HarmonyDetector::is_harmony_model("gpt-5-turbo"));
        assert!(HarmonyDetector::is_harmony_model("gpt-50-preview")); // edge case
    }

    #[test]
    fn test_non_harmony_models() {
        assert!(!HarmonyDetector::is_harmony_model("gpt-4"));
        assert!(!HarmonyDetector::is_harmony_model("gpt-4-turbo"));
        assert!(!HarmonyDetector::is_harmony_model("gpt-4-0613"));
        assert!(!HarmonyDetector::is_harmony_model("gpt-3.5-turbo"));
        assert!(!HarmonyDetector::is_harmony_model("claude-3-opus"));
        assert!(!HarmonyDetector::is_harmony_model("llama-2-70b"));
        assert!(!HarmonyDetector::is_harmony_model(""));
    }

    #[test]
    fn test_case_insensitive() {
        assert!(HarmonyDetector::is_harmony_model("GPT-OSS"));
        assert!(HarmonyDetector::is_harmony_model("GPT-4O"));
        assert!(HarmonyDetector::is_harmony_model("GPT-4.5"));
        assert!(HarmonyDetector::is_harmony_model("GPT-5"));
        assert!(HarmonyDetector::is_harmony_model("gpt-OSS"));
        assert!(HarmonyDetector::is_harmony_model("gpt-4O"));
    }

    #[test]
    fn test_model_prefixes() {
        assert!(HarmonyDetector::is_harmony_model("openai/gpt-4o"));
        assert!(HarmonyDetector::is_harmony_model("azure/gpt-4.5-turbo"));
        assert!(HarmonyDetector::is_harmony_model("custom-deployment/gpt-5"));
        assert!(!HarmonyDetector::is_harmony_model("openai/gpt-4-turbo"));
    }

    #[test]
    fn test_edge_cases() {
        assert!(!HarmonyDetector::is_harmony_model("gpt"));
        assert!(!HarmonyDetector::is_harmony_model("gpt-"));
        assert!(!HarmonyDetector::is_harmony_model("gpt-4")); // not 4o or 4.5
        assert!(HarmonyDetector::is_harmony_model("gpt-4o-"));
        assert!(!HarmonyDetector::is_harmony_model("my-gpt-4-model"));
    }
}
