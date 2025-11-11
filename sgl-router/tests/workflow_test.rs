//! Integration tests for workflow engine

use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};

use sglang_router_rs::core::workflow::*;
use tokio::time::sleep;

// Test step that counts invocations
struct CountingStep {
    counter: Arc<AtomicU32>,
    should_succeed_after: u32,
}

#[async_trait::async_trait]
impl StepExecutor for CountingStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let count = self.counter.fetch_add(1, Ordering::SeqCst) + 1;

        // Store count in context
        context.set("execution_count", count);

        if count >= self.should_succeed_after {
            Ok(StepResult::Success)
        } else {
            Err(WorkflowError::StepFailed {
                step_id: StepId::new("counting_step"),
                message: format!("Not ready yet, attempt {}", count),
            })
        }
    }
}

// Test step that always succeeds
struct AlwaysSucceedStep;

#[async_trait::async_trait]
impl StepExecutor for AlwaysSucceedStep {
    async fn execute(&self, _context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        Ok(StepResult::Success)
    }
}

#[tokio::test]
async fn test_simple_workflow_execution() {
    let engine = WorkflowEngine::new();

    // Subscribe to events for logging
    engine
        .event_bus()
        .subscribe(Arc::new(LoggingSubscriber))
        .await;

    // Create a simple workflow
    let workflow = WorkflowDefinition::new("test_workflow", "Simple Test Workflow")
        .add_step(StepDefinition::new(
            "step1",
            "First Step",
            Arc::new(AlwaysSucceedStep),
        ))
        .add_step(StepDefinition::new(
            "step2",
            "Second Step",
            Arc::new(AlwaysSucceedStep),
        ));

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow);

    // Start workflow
    let instance_id = engine
        .start_workflow(workflow_id, WorkflowContext::new(WorkflowInstanceId::new()))
        .await
        .unwrap();

    // Wait for completion
    sleep(Duration::from_millis(100)).await;

    // Check status
    let state = engine.get_status(instance_id).unwrap();
    assert_eq!(state.status, WorkflowStatus::Completed);
    assert_eq!(state.step_states.len(), 2);
}

#[tokio::test]
async fn test_workflow_with_retry() {
    let engine = WorkflowEngine::new();
    engine
        .event_bus()
        .subscribe(Arc::new(LoggingSubscriber))
        .await;

    let counter = Arc::new(AtomicU32::new(0));

    // Create workflow with retry logic
    let workflow = WorkflowDefinition::new("retry_workflow", "Workflow with Retry").add_step(
        StepDefinition::new(
            "retry_step",
            "Step that retries",
            Arc::new(CountingStep {
                counter: Arc::clone(&counter),
                should_succeed_after: 3,
            }),
        )
        .with_retry(RetryPolicy {
            max_attempts: 5,
            backoff: BackoffStrategy::Fixed(Duration::from_millis(10)),
        })
        .with_timeout(Duration::from_secs(5)),
    );

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow);

    // Start workflow
    let instance_id = engine
        .start_workflow(workflow_id, WorkflowContext::new(WorkflowInstanceId::new()))
        .await
        .unwrap();

    // Wait for completion
    sleep(Duration::from_millis(500)).await;

    // Check that step was retried and eventually succeeded
    let state = engine.get_status(instance_id).unwrap();
    assert_eq!(state.status, WorkflowStatus::Completed);

    let step_state = state.step_states.get(&StepId::new("retry_step")).unwrap();
    assert_eq!(step_state.status, StepStatus::Succeeded);
    assert_eq!(step_state.attempt, 3); // Should have taken 3 attempts

    // Verify counter
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn test_workflow_failure_after_max_retries() {
    let engine = WorkflowEngine::new();
    engine
        .event_bus()
        .subscribe(Arc::new(LoggingSubscriber))
        .await;

    let counter = Arc::new(AtomicU32::new(0));

    // Create workflow that will fail
    let workflow = WorkflowDefinition::new("failing_workflow", "Workflow that Fails").add_step(
        StepDefinition::new(
            "failing_step",
            "Step that always fails",
            Arc::new(CountingStep {
                counter: Arc::clone(&counter),
                should_succeed_after: 10, // Will never succeed within max_attempts
            }),
        )
        .with_retry(RetryPolicy {
            max_attempts: 3,
            backoff: BackoffStrategy::Fixed(Duration::from_millis(10)),
        })
        .with_failure_action(FailureAction::FailWorkflow),
    );

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow);

    // Start workflow
    let instance_id = engine
        .start_workflow(workflow_id, WorkflowContext::new(WorkflowInstanceId::new()))
        .await
        .unwrap();

    // Wait for completion
    sleep(Duration::from_millis(500)).await;

    // Check that workflow failed
    let state = engine.get_status(instance_id).unwrap();
    assert_eq!(state.status, WorkflowStatus::Failed);

    let step_state = state.step_states.get(&StepId::new("failing_step")).unwrap();
    assert_eq!(step_state.status, StepStatus::Failed);
    assert_eq!(step_state.attempt, 3); // Should have tried 3 times

    // Verify counter
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn test_workflow_continue_on_failure() {
    let engine = WorkflowEngine::new();
    engine
        .event_bus()
        .subscribe(Arc::new(LoggingSubscriber))
        .await;

    let counter = Arc::new(AtomicU32::new(0));

    // Create workflow where first step fails but workflow continues
    let workflow = WorkflowDefinition::new("continue_workflow", "Continue on Failure")
        .add_step(
            StepDefinition::new(
                "failing_step",
                "Step that fails",
                Arc::new(CountingStep {
                    counter: Arc::clone(&counter),
                    should_succeed_after: 10,
                }),
            )
            .with_retry(RetryPolicy {
                max_attempts: 2,
                backoff: BackoffStrategy::Fixed(Duration::from_millis(10)),
            })
            .with_failure_action(FailureAction::ContinueNextStep),
        )
        .add_step(StepDefinition::new(
            "success_step",
            "Step that succeeds",
            Arc::new(AlwaysSucceedStep),
        ));

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow);

    // Start workflow
    let instance_id = engine
        .start_workflow(workflow_id, WorkflowContext::new(WorkflowInstanceId::new()))
        .await
        .unwrap();

    // Wait for completion
    sleep(Duration::from_millis(500)).await;

    // Workflow should complete despite first step failing
    let state = engine.get_status(instance_id).unwrap();
    assert_eq!(state.status, WorkflowStatus::Completed);

    // First step should be skipped
    let step1_state = state.step_states.get(&StepId::new("failing_step")).unwrap();
    assert_eq!(step1_state.status, StepStatus::Skipped);

    // Second step should succeed
    let step2_state = state.step_states.get(&StepId::new("success_step")).unwrap();
    assert_eq!(step2_state.status, StepStatus::Succeeded);
}

#[tokio::test]
async fn test_workflow_context_sharing() {
    let engine = WorkflowEngine::new();

    struct ContextWriterStep {
        key: String,
        value: String,
    }

    #[async_trait::async_trait]
    impl StepExecutor for ContextWriterStep {
        async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
            context.set(self.key.clone(), self.value.clone());
            Ok(StepResult::Success)
        }
    }

    struct ContextReaderStep {
        key: String,
        expected_value: String,
    }

    #[async_trait::async_trait]
    impl StepExecutor for ContextReaderStep {
        async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
            let value: Arc<String> = context
                .get(&self.key)
                .ok_or_else(|| WorkflowError::ContextValueNotFound(self.key.clone()))?;

            if *value == self.expected_value {
                Ok(StepResult::Success)
            } else {
                Err(WorkflowError::StepFailed {
                    step_id: StepId::new("reader"),
                    message: format!("Expected {}, got {}", self.expected_value, value),
                })
            }
        }
    }

    let workflow = WorkflowDefinition::new("context_workflow", "Context Sharing Test")
        .add_step(StepDefinition::new(
            "writer",
            "Write to context",
            Arc::new(ContextWriterStep {
                key: "test_key".to_string(),
                value: "test_value".to_string(),
            }),
        ))
        .add_step(StepDefinition::new(
            "reader",
            "Read from context",
            Arc::new(ContextReaderStep {
                key: "test_key".to_string(),
                expected_value: "test_value".to_string(),
            }),
        ));

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow);

    let instance_id = engine
        .start_workflow(workflow_id, WorkflowContext::new(WorkflowInstanceId::new()))
        .await
        .unwrap();

    sleep(Duration::from_millis(100)).await;

    let state = engine.get_status(instance_id).unwrap();
    assert_eq!(state.status, WorkflowStatus::Completed);
}
