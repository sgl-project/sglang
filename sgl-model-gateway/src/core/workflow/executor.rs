//! Step executor trait and implementations

use async_trait::async_trait;

use super::types::{StepResult, WorkflowContext, WorkflowError, WorkflowResult};

/// Trait for executing individual workflow steps
#[async_trait]
pub trait StepExecutor: Send + Sync {
    /// Execute the step with the given context
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult>;

    /// Check if an error is retry-able
    ///
    /// Override this method to customize which errors should trigger retries.
    /// By default, all errors are considered retry-able.
    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }

    /// Called when the step succeeds
    ///
    /// This hook allows steps to perform cleanup or additional actions
    /// after successful execution.
    async fn on_success(&self, _context: &WorkflowContext) -> WorkflowResult<()> {
        Ok(())
    }

    /// Called when the step fails after all retries
    ///
    /// This hook allows steps to perform cleanup or compensation logic
    /// when the step cannot complete successfully.
    async fn on_failure(
        &self,
        _context: &WorkflowContext,
        _error: &WorkflowError,
    ) -> WorkflowResult<()> {
        Ok(())
    }
}

/// Simple function-based step executor
pub struct FunctionStep<F>
where
    F: Fn(
            &mut WorkflowContext,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = WorkflowResult<StepResult>> + Send + '_>,
        > + Send
        + Sync,
{
    func: F,
}

impl<F> FunctionStep<F>
where
    F: Fn(
            &mut WorkflowContext,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = WorkflowResult<StepResult>> + Send + '_>,
        > + Send
        + Sync,
{
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

#[async_trait]
impl<F> StepExecutor for FunctionStep<F>
where
    F: Fn(
            &mut WorkflowContext,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = WorkflowResult<StepResult>> + Send + '_>,
        > + Send
        + Sync,
{
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        (self.func)(context).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::workflow::types::WorkflowInstanceId;

    struct TestStep {
        should_succeed: bool,
    }

    #[async_trait]
    impl StepExecutor for TestStep {
        async fn execute(&self, _context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
            if self.should_succeed {
                Ok(StepResult::Success)
            } else {
                Err(WorkflowError::StepFailed {
                    step_id: crate::core::workflow::types::StepId::new("test"),
                    message: "test error".to_string(),
                })
            }
        }
    }

    #[tokio::test]
    async fn test_step_executor_success() {
        let step = TestStep {
            should_succeed: true,
        };
        let mut context = WorkflowContext::new(WorkflowInstanceId::new());

        let result = step.execute(&mut context).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), StepResult::Success);
    }

    #[tokio::test]
    async fn test_step_executor_failure() {
        let step = TestStep {
            should_succeed: false,
        };
        let mut context = WorkflowContext::new(WorkflowInstanceId::new());

        let result = step.execute(&mut context).await;
        assert!(result.is_err());
    }
}
