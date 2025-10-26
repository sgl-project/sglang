//! Harmony model detection

/// Harmony model detector
///
/// Detects if a model name indicates support for Harmony encoding/parsing.
pub struct HarmonyDetector;

impl HarmonyDetector {
    pub fn is_harmony_model(model_name: &str) -> bool {
        // Convert to lowercase for case-insensitive matching
        let model_lower = model_name.to_lowercase();

        // Check for Harmony-capable model patterns
        model_lower.contains("gpt-oss")
            || model_lower.contains("gpt-4o")
            || model_lower.contains("gpt-4.5")
            || model_lower.contains("gpt-5")
    }
}
