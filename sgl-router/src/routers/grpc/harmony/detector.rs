//! Harmony model detection

/// Harmony model detector
///
/// Detects if a model name indicates support for Harmony encoding/parsing.
pub struct HarmonyDetector;

impl HarmonyDetector {
    pub fn is_harmony_model(model_name: &str) -> bool {
        // Case-insensitive substring search without heap allocation
        // More efficient than to_lowercase() which allocates a new String
        model_name
            .as_bytes()
            .windows(7) // "gpt-oss".len()
            .any(|window| window.eq_ignore_ascii_case(b"gpt-oss"))
    }
}
