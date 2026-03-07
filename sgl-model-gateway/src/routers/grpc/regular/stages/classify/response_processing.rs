//! Response processing stage for classify requests.
//!
//! Key responsibilities:
//! 1. Extract embedding (logits) from EmbedComplete response
//! 2. Apply softmax to convert logits to probabilities
//! 3. Find predicted class (argmax)
//! 4. Map class index to label (from id2label or generic LABEL_N)
//! 5. Build ClassifyResponse

use std::collections::HashMap;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::{
    protocols::{
        classify::{ClassifyData, ClassifyResponse},
        common::UsageInfo,
    },
    routers::{
        error,
        grpc::{
            common::stages::PipelineStage,
            context::{ExecutionResult, FinalResponse, RequestContext, WorkerSelection},
        },
    },
};

/// Response processing stage for classify requests.
///
/// Takes the logits from the embedding response and converts them to
/// classification results with probabilities and labels.
///
/// The stage is stateless - id2label mapping is obtained from the
/// selected worker's model card at runtime.
pub(crate) struct ClassifyResponseProcessingStage;

impl ClassifyResponseProcessingStage {
    /// Create a new classify response processing stage.
    pub fn new() -> Self {
        Self
    }

    /// Apply softmax to logits to get probability distribution.
    ///
    /// Uses the numerically stable formula: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
    fn softmax(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return vec![];
        }

        // Find max for numerical stability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) for each element
        let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();

        // Sum of exponentials
        let sum: f32 = exp_vals.iter().sum();

        // Normalize to get probabilities
        if sum == 0.0 {
            // Avoid division by zero - return uniform distribution
            let n = exp_vals.len();
            return vec![1.0 / n as f32; n];
        }

        exp_vals.iter().map(|&x| x / sum).collect()
    }

    /// Find the index of the maximum value (argmax).
    fn argmax(probs: &[f32]) -> u32 {
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }

    /// Get label for a class index.
    ///
    /// Returns the label from id2label if available, otherwise returns generic "LABEL_N".
    fn get_label(id2label: &HashMap<u32, String>, class_idx: u32) -> String {
        id2label
            .get(&class_idx)
            .cloned()
            .unwrap_or_else(|| format!("LABEL_{}", class_idx))
    }

    /// Extract id2label mapping from the selected worker's model card.
    fn get_id2label_from_context(ctx: &RequestContext) -> HashMap<u32, String> {
        // Get the selected worker
        let worker = match ctx.state.workers.as_ref() {
            Some(WorkerSelection::Single { worker }) => worker,
            Some(WorkerSelection::Dual { prefill, .. }) => prefill, // Use prefill worker for model info
            None => return HashMap::new(),
        };

        // Get id2label from the first model card
        worker
            .metadata()
            .models
            .first()
            .map(|model| model.id2label.clone())
            .unwrap_or_default()
    }
}

impl Default for ClassifyResponseProcessingStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for ClassifyResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Extract execution result
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "ClassifyResponseProcessingStage::execute",
                "Execution result missing"
            );
            error::internal_error("execution_result_missing", "Execution result missing")
        })?;

        // Expect Embedding result variant (classify uses embed backend)
        let proto_response = if let ExecutionResult::Embedding { response } = execution_result {
            response
        } else {
            error!(
                function = "ClassifyResponseProcessingStage::execute",
                "Invalid execution result: expected Embedding"
            );
            return Err(error::internal_error(
                "invalid_execution_result",
                "Expected Embedding result for classify",
            ));
        };

        // Get logits from embedding response
        let logits = proto_response.embedding();

        if logits.is_empty() {
            error!(
                function = "ClassifyResponseProcessingStage::execute",
                "Empty logits received from scheduler"
            );
            return Err(error::internal_error(
                "empty_logits",
                "Empty logits received from scheduler",
            ));
        }

        // Get id2label from the worker's model card
        let id2label = Self::get_id2label_from_context(ctx);

        // Apply softmax to get probabilities
        let probs = Self::softmax(logits);

        // Get predicted class (argmax)
        let predicted_class = Self::argmax(&probs);

        // Get label for predicted class
        let label = Self::get_label(&id2label, predicted_class);

        // Build classify data
        let classify_data = ClassifyData {
            index: 0,
            label,
            probs: probs.clone(),
            num_classes: probs.len() as u32,
        };

        // Get dispatch metadata
        let dispatch = ctx.state.dispatch.as_ref().ok_or_else(|| {
            error!(
                function = "ClassifyResponseProcessingStage::execute",
                "Dispatch metadata missing"
            );
            error::internal_error("dispatch_missing", "Dispatch metadata missing")
        })?;

        // Build usage info
        let prompt_tokens = proto_response.prompt_tokens().max(0) as u32;
        let usage = UsageInfo {
            prompt_tokens,
            total_tokens: prompt_tokens,
            completion_tokens: 0,
            prompt_tokens_details: None,
            reasoning_tokens: None,
        };

        // Build response
        let response = ClassifyResponse::new(
            dispatch.request_id.clone(),
            dispatch.model.clone(),
            dispatch.created,
            vec![classify_data],
            usage,
        );

        // Store in context for pipeline to extract
        ctx.state.response.final_response = Some(FinalResponse::Classify(response));

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ClassifyResponseProcessing"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = ClassifyResponseProcessingStage::softmax(&logits);

        // Probabilities should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Highest logit should have highest probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let probs = ClassifyResponseProcessingStage::softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_softmax_single() {
        let probs = ClassifyResponseProcessingStage::softmax(&[5.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without max subtraction
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = ClassifyResponseProcessingStage::softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(ClassifyResponseProcessingStage::argmax(&[0.1, 0.7, 0.2]), 1);
        assert_eq!(
            ClassifyResponseProcessingStage::argmax(&[0.9, 0.05, 0.05]),
            0
        );
        assert_eq!(ClassifyResponseProcessingStage::argmax(&[0.1, 0.1, 0.8]), 2);
    }

    #[test]
    fn test_get_label_with_mapping() {
        let mut id2label = HashMap::new();
        id2label.insert(0, "negative".to_string());
        id2label.insert(1, "positive".to_string());

        assert_eq!(
            ClassifyResponseProcessingStage::get_label(&id2label, 0),
            "negative"
        );
        assert_eq!(
            ClassifyResponseProcessingStage::get_label(&id2label, 1),
            "positive"
        );
        assert_eq!(
            ClassifyResponseProcessingStage::get_label(&id2label, 2),
            "LABEL_2"
        ); // Fallback for unknown
    }

    #[test]
    fn test_get_label_without_mapping() {
        let id2label = HashMap::new();
        assert_eq!(
            ClassifyResponseProcessingStage::get_label(&id2label, 0),
            "LABEL_0"
        );
        assert_eq!(
            ClassifyResponseProcessingStage::get_label(&id2label, 5),
            "LABEL_5"
        );
    }
}
