//! Harmony model detection

use crate::core::{Worker, WorkerRegistry};

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

    /// Check if any worker for the given model is a Harmony/GPT-OSS worker.
    ///
    /// This method looks up workers from the registry by model name and checks
    /// if any of them are Harmony workers based on their metadata (architectures,
    /// hf_model_type).
    ///
    /// Falls back to string-based detection if no workers are registered for
    /// the model (e.g., during startup before workers are discovered).
    pub fn is_harmony_model_in_registry(registry: &WorkerRegistry, model_name: &str) -> bool {
        // Get workers for this model
        let workers = registry.get_by_model_fast(model_name);

        if workers.is_empty() {
            // No workers found - fall back to string-based detection
            return Self::is_harmony_model(model_name);
        }

        // Check if any worker is a Harmony worker
        workers
            .iter()
            .any(|worker| Self::is_harmony_worker(worker.as_ref()))
    }
}
