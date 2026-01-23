//! Step executor trait and implementations

use async_trait::async_trait;

use super::types::{StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult};

/// Trait for executing individual workflow steps
#[async_trait]
pub trait StepExecutor<D: WorkflowData>: Send + Sync {
    /// Execute the step with the given context
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult>;

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
    async fn on_success(&self, _context: &WorkflowContext<D>) -> WorkflowResult<()> {
        Ok(())
    }

    /// Called when the step fails after all retries
    ///
    /// This hook allows steps to perform cleanup or compensation logic
    /// when the step cannot complete successfully.
    async fn on_failure(
        &self,
        _context: &WorkflowContext<D>,
        _error: &WorkflowError,
    ) -> WorkflowResult<()> {
        Ok(())
    }
}

/// Simple function-based step executor
pub struct FunctionStep<D, F>
where
    D: WorkflowData,
    F: Fn(
            &mut WorkflowContext<D>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = WorkflowResult<StepResult>> + Send + '_>,
        > + Send
        + Sync,
{
    func: F,
    _phantom: std::marker::PhantomData<D>,
}

impl<D, F> FunctionStep<D, F>
where
    D: WorkflowData,
    F: Fn(
            &mut WorkflowContext<D>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = WorkflowResult<StepResult>> + Send + '_>,
        > + Send
        + Sync,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<D, F> StepExecutor<D> for FunctionStep<D, F>
where
    D: WorkflowData,
    F: Fn(
            &mut WorkflowContext<D>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = WorkflowResult<StepResult>> + Send + '_>,
        > + Send
        + Sync,
{
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        (self.func)(context).await
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::workflow::types::WorkflowInstanceId;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestData {
        value: i32,
    }

    impl WorkflowData for TestData {
        fn workflow_type() -> &'static str {
            "test"
        }
    }

    struct TestStep {
        should_succeed: bool,
    }

    #[async_trait]
    impl StepExecutor<TestData> for TestStep {
        async fn execute(
            &self,
            _context: &mut WorkflowContext<TestData>,
        ) -> WorkflowResult<StepResult> {
            if self.should_succeed {
                Ok(StepResult::Success)
            } else {
                Err(WorkflowError::StepFailed {
                    step_id: crate::workflow::types::StepId::new("test"),
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
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), TestData { value: 42 });

        let result = step.execute(&mut context).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), StepResult::Success);
    }

    #[tokio::test]
    async fn test_step_executor_failure() {
        let step = TestStep {
            should_succeed: false,
        };
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), TestData { value: 42 });

        let result = step.execute(&mut context).await;
        assert!(result.is_err());
    }
}
