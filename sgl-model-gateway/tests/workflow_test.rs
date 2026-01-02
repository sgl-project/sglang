//! Integration tests for workflow engine

use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};

use sgl_model_gateway::workflow::*;
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
    engine.register_workflow(workflow).unwrap();

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
    engine.register_workflow(workflow).unwrap();

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
    engine.register_workflow(workflow).unwrap();

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
    engine.register_workflow(workflow).unwrap();

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
    engine.register_workflow(workflow).unwrap();

    let instance_id = engine
        .start_workflow(workflow_id, WorkflowContext::new(WorkflowInstanceId::new()))
        .await
        .unwrap();

    sleep(Duration::from_millis(100)).await;

    let state = engine.get_status(instance_id).unwrap();
    assert_eq!(state.status, WorkflowStatus::Completed);
}

// ============================================================================
// DAG / Parallel Execution Tests
// ============================================================================

// Step that records when it starts and ends (for testing parallel execution)
struct TimingStep {
    step_name: String,
    duration_ms: u64,
    start_times: Arc<parking_lot::RwLock<Vec<(String, std::time::Instant)>>>,
    end_times: Arc<parking_lot::RwLock<Vec<(String, std::time::Instant)>>>,
}

#[async_trait::async_trait]
impl StepExecutor for TimingStep {
    async fn execute(&self, _context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let start = std::time::Instant::now();
        self.start_times
            .write()
            .push((self.step_name.clone(), start));

        sleep(Duration::from_millis(self.duration_ms)).await;

        let end = std::time::Instant::now();
        self.end_times.write().push((self.step_name.clone(), end));

        Ok(StepResult::Success)
    }
}

#[tokio::test]
async fn test_parallel_execution_no_dependencies() {
    // Steps without dependencies should run in parallel
    let engine = WorkflowEngine::new();

    let start_times: Arc<parking_lot::RwLock<Vec<(String, std::time::Instant)>>> =
        Arc::new(parking_lot::RwLock::new(Vec::new()));
    let end_times: Arc<parking_lot::RwLock<Vec<(String, std::time::Instant)>>> =
        Arc::new(parking_lot::RwLock::new(Vec::new()));

    // Three steps, each taking 100ms, no dependencies
    // If parallel: ~100ms total
    // If sequential: ~300ms total
    let workflow = WorkflowDefinition::new("parallel_workflow", "Parallel Test")
        .add_step(StepDefinition::new(
            "step_a",
            "Step A",
            Arc::new(TimingStep {
                step_name: "step_a".to_string(),
                duration_ms: 100,
                start_times: Arc::clone(&start_times),
                end_times: Arc::clone(&end_times),
            }),
        ))
        .add_step(StepDefinition::new(
            "step_b",
            "Step B",
            Arc::new(TimingStep {
                step_name: "step_b".to_string(),
                duration_ms: 100,
                start_times: Arc::clone(&start_times),
                end_times: Arc::clone(&end_times),
            }),
        ))
        .add_step(StepDefinition::new(
            "step_c",
            "Step C",
            Arc::new(TimingStep {
                step_name: "step_c".to_string(),
                duration_ms: 100,
                start_times: Arc::clone(&start_times),
                end_times: Arc::clone(&end_times),
            }),
        ));

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow).unwrap();

    let overall_start = std::time::Instant::now();
    let instance_id = engine
        .start_workflow(workflow_id, WorkflowContext::new(WorkflowInstanceId::new()))
        .await
        .unwrap();

    // Wait for completion - give enough time for async scheduling
    for _ in 0..50 {
        sleep(Duration::from_millis(50)).await;
        let state = engine.get_status(instance_id).unwrap();
        if state.status != WorkflowStatus::Running {
            break;
        }
    }

    let overall_duration = overall_start.elapsed();

    let state = engine.get_status(instance_id).unwrap();
    assert_eq!(state.status, WorkflowStatus::Completed);

    // Check that all steps completed
    assert_eq!(end_times.read().len(), 3);

    // Verify parallel execution: all steps should start around the same time
    let starts = start_times.read();
    let first_start = starts.iter().map(|(_, t)| t).min().unwrap();
    let last_start = starts.iter().map(|(_, t)| t).max().unwrap();

    // All starts should be within 100ms of each other (allowing for scheduling variance)
    let start_spread = last_start.duration_since(*first_start);
    assert!(
        start_spread < Duration::from_millis(100),
        "Steps did not start in parallel, spread: {:?}",
        start_spread
    );

    // Total duration should be less than sequential (300ms) - use generous threshold
    assert!(
        overall_duration < Duration::from_millis(500),
        "Parallel execution took too long: {:?}",
        overall_duration
    );
}

#[tokio::test]
async fn test_dag_with_dependencies() {
    // DAG: A and B run in parallel, C waits for both
    //   A ──┐
    //       ├──> C
    //   B ──┘
    let engine = WorkflowEngine::new();

    let start_times: Arc<parking_lot::RwLock<Vec<(String, std::time::Instant)>>> =
        Arc::new(parking_lot::RwLock::new(Vec::new()));
    let end_times: Arc<parking_lot::RwLock<Vec<(String, std::time::Instant)>>> =
        Arc::new(parking_lot::RwLock::new(Vec::new()));

    let workflow = WorkflowDefinition::new("dag_workflow", "DAG Test")
        .add_step(StepDefinition::new(
            "step_a",
            "Step A",
            Arc::new(TimingStep {
                step_name: "step_a".to_string(),
                duration_ms: 50,
                start_times: Arc::clone(&start_times),
                end_times: Arc::clone(&end_times),
            }),
        ))
        .add_step(StepDefinition::new(
            "step_b",
            "Step B",
            Arc::new(TimingStep {
                step_name: "step_b".to_string(),
                duration_ms: 100,
                start_times: Arc::clone(&start_times),
                end_times: Arc::clone(&end_times),
            }),
        ))
        .add_step(
            StepDefinition::new(
                "step_c",
                "Step C",
                Arc::new(TimingStep {
                    step_name: "step_c".to_string(),
                    duration_ms: 50,
                    start_times: Arc::clone(&start_times),
                    end_times: Arc::clone(&end_times),
                }),
            )
            .depends_on(&["step_a", "step_b"]),
        );

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow).unwrap();

    let instance_id = engine
        .start_workflow(workflow_id, WorkflowContext::new(WorkflowInstanceId::new()))
        .await
        .unwrap();

    // Poll until workflow completes (or timeout)
    for _ in 0..50 {
        sleep(Duration::from_millis(50)).await;
        let state = engine.get_status(instance_id).unwrap();
        if state.status != WorkflowStatus::Running {
            break;
        }
    }

    let state = engine.get_status(instance_id).unwrap();
    assert_eq!(state.status, WorkflowStatus::Completed);

    // Verify step C started after both A and B finished
    let starts = start_times.read();
    let ends = end_times.read();

    let c_start = starts.iter().find(|(n, _)| n == "step_c").unwrap().1;
    let a_end = ends.iter().find(|(n, _)| n == "step_a").unwrap().1;
    let b_end = ends.iter().find(|(n, _)| n == "step_b").unwrap().1;

    assert!(c_start >= a_end, "Step C started before Step A finished");
    assert!(c_start >= b_end, "Step C started before Step B finished");
}

#[tokio::test]
async fn test_dag_dependency_failure_blocks_dependents() {
    // If step A fails with FailWorkflow, step B (depends on A) should not run
    let engine = WorkflowEngine::new();

    let b_executed = Arc::new(AtomicU32::new(0));

    struct FailingStep;

    #[async_trait::async_trait]
    impl StepExecutor for FailingStep {
        async fn execute(&self, _context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
            Err(WorkflowError::StepFailed {
                step_id: StepId::new("failing"),
                message: "Intentional failure".to_string(),
            })
        }

        fn is_retryable(&self, _error: &WorkflowError) -> bool {
            false // Disable retries for this test
        }
    }

    struct TrackingStep {
        counter: Arc<AtomicU32>,
    }

    #[async_trait::async_trait]
    impl StepExecutor for TrackingStep {
        async fn execute(&self, _context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(StepResult::Success)
        }
    }

    let workflow = WorkflowDefinition::new("blocked_workflow", "Blocked Test")
        .add_step(
            StepDefinition::new("step_a", "Step A", Arc::new(FailingStep))
                .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "step_b",
                "Step B",
                Arc::new(TrackingStep {
                    counter: Arc::clone(&b_executed),
                }),
            )
            .depends_on(&["step_a"]),
        );

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow).unwrap();

    let instance_id = engine
        .start_workflow(workflow_id, WorkflowContext::new(WorkflowInstanceId::new()))
        .await
        .unwrap();

    // Poll until workflow completes (or timeout)
    for _ in 0..50 {
        sleep(Duration::from_millis(50)).await;
        let state = engine.get_status(instance_id).unwrap();
        if state.status != WorkflowStatus::Running {
            break;
        }
    }

    let state = engine.get_status(instance_id).unwrap();
    assert_eq!(state.status, WorkflowStatus::Failed);

    // Step B should not have executed
    assert_eq!(b_executed.load(Ordering::SeqCst), 0);
}

#[test]
fn test_dag_validation_cycle_detection() {
    // Create a workflow with a cycle: A -> B -> C -> A
    let mut workflow = WorkflowDefinition::new("cyclic_workflow", "Cyclic Test")
        .add_step(
            StepDefinition::new("step_a", "Step A", Arc::new(AlwaysSucceedStep))
                .depends_on(&["step_c"]),
        )
        .add_step(
            StepDefinition::new("step_b", "Step B", Arc::new(AlwaysSucceedStep))
                .depends_on(&["step_a"]),
        )
        .add_step(
            StepDefinition::new("step_c", "Step C", Arc::new(AlwaysSucceedStep))
                .depends_on(&["step_b"]),
        );

    let result = workflow.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Cycle detected"));
}

#[test]
fn test_dag_validation_missing_dependency() {
    // Create a workflow with a missing dependency
    let mut workflow = WorkflowDefinition::new("missing_dep_workflow", "Missing Dep Test")
        .add_step(StepDefinition::new(
            "step_a",
            "Step A",
            Arc::new(AlwaysSucceedStep),
        ))
        .add_step(
            StepDefinition::new("step_b", "Step B", Arc::new(AlwaysSucceedStep))
                .depends_on(&["nonexistent_step"]),
        );

    let result = workflow.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("non-existent step"));
}

#[test]
fn test_dag_validation_valid_workflow() {
    // Create a valid DAG workflow
    let mut workflow = WorkflowDefinition::new("valid_workflow", "Valid Test")
        .add_step(StepDefinition::new(
            "step_a",
            "Step A",
            Arc::new(AlwaysSucceedStep),
        ))
        .add_step(StepDefinition::new(
            "step_b",
            "Step B",
            Arc::new(AlwaysSucceedStep),
        ))
        .add_step(
            StepDefinition::new("step_c", "Step C", Arc::new(AlwaysSucceedStep))
                .depends_on(&["step_a", "step_b"]),
        )
        .add_step(
            StepDefinition::new("step_d", "Step D", Arc::new(AlwaysSucceedStep))
                .depends_on(&["step_c"]),
        );

    let result = workflow.validate();
    assert!(result.is_ok());
}
