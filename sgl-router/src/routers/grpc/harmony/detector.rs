//! Harmony model detection

use crate::core::Worker;

/// Harmony model detector
///
/// Detects if a model name indicates support for Harmony encoding/parsing.
pub struct HarmonyDetector;

impl HarmonyDetector {
    /// Check if a worker is a Harmony/GPT-OSS model.
    ///
    /// Detection priority:
    /// 1. Check if any model card has architectures containing "GptOssForCausalLM"
    /// 2. Check if any model card has hf_model_type equal to "gpt_oss"
    /// 3. Check if model_id contains "gpt-oss" substring (case-insensitive)
    pub fn is_harmony_worker(worker: &dyn Worker) -> bool {
        for model_card in worker.models() {
            // 1. Check architectures for GptOssForCausalLM
            if model_card
                .architectures
                .iter()
                .any(|arch| arch == "GptOssForCausalLM")
            {
                return true;
            }

            // 2. Check hf_model_type for gpt_oss
            if let Some(ref model_type) = model_card.hf_model_type {
                if model_type == "gpt_oss" {
                    return true;
                }
            }

            // 3. Check model id for gpt-oss substring
            if Self::is_harmony_model(&model_card.id) {
                return true;
            }
        }

        // Fallback: check worker's model_id directly
        Self::is_harmony_model(worker.model_id())
    }

    /// Check if a model name contains "gpt-oss" (case-insensitive).
    pub fn is_harmony_model(model_name: &str) -> bool {
        // Case-insensitive substring search without heap allocation
        // More efficient than to_lowercase() which allocates a new String
        model_name
            .as_bytes()
            .windows(7) // "gpt-oss".len()
            .any(|window| window.eq_ignore_ascii_case(b"gpt-oss"))
    }
}
