//! Typed scheduler-to-model execution contract.
//!
//! This is the boundary the Rust scheduler will call once model loading and
//! accelerator execution are migrated. It intentionally uses token IDs and
//! typed sampling fields rather than wire-format maps or JSON values.

use std::collections::HashSet;

use crate::{EngineConfig, RuntimeError};

/// Sampling parameters needed by one generation execution request.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingConfig {
    /// Maximum number of tokens to generate.
    pub max_new_tokens: u32,
    /// Softmax temperature; zero requests greedy decoding.
    pub temperature: f32,
    /// Nucleus sampling threshold in `(0, 1]`.
    pub top_p: f32,
    /// Top-k cutoff; zero disables top-k filtering.
    pub top_k: u32,
}

impl SamplingConfig {
    /// Validate basic sampling invariants before model execution.
    pub fn validate(&self) -> Result<(), ModelContractError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(ModelContractError::InvalidSampling("temperature"));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(ModelContractError::InvalidSampling("top_p"));
        }
        Ok(())
    }
}

/// One prompt admitted for model execution.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionTask {
    /// Stable scheduler request ID.
    pub request_id: String,
    /// Tokenized prompt. Empty prompts are invalid at this boundary.
    pub prompt_token_ids: Vec<u32>,
    /// Resolved sampling parameters.
    pub sampling: SamplingConfig,
}

/// Full generation result from a model batch.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionResult {
    /// Request ID corresponding to one task in the submitted batch.
    pub request_id: String,
    /// Generated token IDs in order.
    pub output_token_ids: Vec<u32>,
    /// Optional generated-token log probabilities, aligned with output IDs.
    pub output_logprobs: Option<Vec<f32>>,
    /// Whether generation ended before reaching `max_new_tokens`.
    pub finished: bool,
}

/// Model loading and batched generation interface owned by the scheduler.
pub trait ModelExecutor: Send {
    /// Load model resources and return when the executor is ready.
    fn load(&mut self, config: &EngineConfig) -> Result<(), RuntimeError>;

    /// Execute a batch and return exactly one result for each task.
    fn execute_batch(
        &mut self,
        tasks: &[ExecutionTask],
    ) -> Result<Vec<ExecutionResult>, RuntimeError>;

    /// Release model, device, and worker resources.
    fn shutdown(&mut self) -> Result<(), RuntimeError>;
}

/// Validate request and result invariants at the model boundary.
pub fn validate_batch(
    tasks: &[ExecutionTask],
    results: &[ExecutionResult],
) -> Result<(), ModelContractError> {
    let mut task_ids = HashSet::with_capacity(tasks.len());
    for task in tasks {
        if task.request_id.is_empty() || !task_ids.insert(task.request_id.as_str()) {
            return Err(ModelContractError::InvalidRequestId);
        }
        if task.prompt_token_ids.is_empty() {
            return Err(ModelContractError::EmptyPrompt(task.request_id.clone()));
        }
        task.sampling.validate()?;
    }

    let mut result_ids = HashSet::with_capacity(results.len());
    for result in results {
        if !task_ids.contains(result.request_id.as_str()) {
            return Err(ModelContractError::UnexpectedResult(
                result.request_id.clone(),
            ));
        }
        if !result_ids.insert(result.request_id.as_str()) {
            return Err(ModelContractError::DuplicateResult(
                result.request_id.clone(),
            ));
        }
        let task = tasks
            .iter()
            .find(|task| task.request_id == result.request_id)
            .expect("membership validated above");
        if result.output_token_ids.len() > task.sampling.max_new_tokens as usize {
            return Err(ModelContractError::TooManyOutputTokens {
                request_id: result.request_id.clone(),
                actual: result.output_token_ids.len(),
                maximum: task.sampling.max_new_tokens,
            });
        }
        if result
            .output_logprobs
            .as_ref()
            .is_some_and(|values| values.len() != result.output_token_ids.len())
        {
            return Err(ModelContractError::LogprobLengthMismatch(
                result.request_id.clone(),
            ));
        }
    }
    if result_ids.len() != task_ids.len() {
        return Err(ModelContractError::MissingResults {
            expected: task_ids.len(),
            actual: result_ids.len(),
        });
    }
    Ok(())
}

/// Invalid scheduler/model contract data.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ModelContractError {
    #[error("invalid sampling parameter: {0}")]
    InvalidSampling(&'static str),
    #[error("request ID must be non-empty and unique within a batch")]
    InvalidRequestId,
    #[error("request {0} has an empty tokenized prompt")]
    EmptyPrompt(String),
    #[error("model returned a result for unknown request {0}")]
    UnexpectedResult(String),
    #[error("model returned duplicate results for request {0}")]
    DuplicateResult(String),
    #[error("model omitted results: expected {expected}, received {actual}")]
    MissingResults { expected: usize, actual: usize },
    #[error("request {request_id} returned {actual} tokens, exceeding maximum {maximum}")]
    TooManyOutputTokens {
        request_id: String,
        actual: usize,
        maximum: u32,
    },
    #[error("request {0} has logprobs that do not align with output tokens")]
    LogprobLengthMismatch(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: &str) -> ExecutionTask {
        ExecutionTask {
            request_id: id.into(),
            prompt_token_ids: vec![1, 2],
            sampling: SamplingConfig {
                max_new_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
            },
        }
    }

    fn result(id: &str) -> ExecutionResult {
        ExecutionResult {
            request_id: id.into(),
            output_token_ids: vec![3, 4],
            output_logprobs: Some(vec![-0.1, -0.2]),
            finished: true,
        }
    }

    #[test]
    fn validates_complete_batch_and_result_order_is_irrelevant() {
        validate_batch(&[task("a"), task("b")], &[result("b"), result("a")]).unwrap();
    }

    #[test]
    fn rejects_duplicate_and_missing_results() {
        assert_eq!(
            validate_batch(&[task("a")], &[result("a"), result("a")]),
            Err(ModelContractError::DuplicateResult("a".into()))
        );
        assert_eq!(
            validate_batch(&[task("a"), task("b")], &[result("a")]),
            Err(ModelContractError::MissingResults {
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn rejects_unexpected_output_and_misaligned_logprobs() {
        assert_eq!(
            validate_batch(&[task("a")], &[result("other")]),
            Err(ModelContractError::UnexpectedResult("other".into()))
        );
        let mut bad = result("a");
        bad.output_logprobs = Some(vec![-0.1]);
        assert_eq!(
            validate_batch(&[task("a")], &[bad]),
            Err(ModelContractError::LogprobLengthMismatch("a".into()))
        );
    }

    #[test]
    fn rejects_invalid_prompt_sampling_and_output_length() {
        let mut empty = task("a");
        empty.prompt_token_ids.clear();
        assert!(matches!(
            validate_batch(&[empty], &[result("a")]),
            Err(ModelContractError::EmptyPrompt(_))
        ));

        let mut invalid = task("a");
        invalid.sampling.top_p = 0.0;
        assert_eq!(
            validate_batch(&[invalid], &[result("a")]),
            Err(ModelContractError::InvalidSampling("top_p"))
        );

        let mut too_long = result("a");
        too_long.output_token_ids = vec![1, 2, 3, 4, 5];
        too_long.output_logprobs = None;
        assert!(matches!(
            validate_batch(&[task("a")], &[too_long]),
            Err(ModelContractError::TooManyOutputTokens { .. })
        ));
    }
}
