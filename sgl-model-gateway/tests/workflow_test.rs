//! Integration tests for workflow engine

use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};

use serde::{Deserialize, Serialize};
use smg::workflow::*;
use tokio::time::sleep;

/// Test workflow data type for integration tests.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct TestWorkflowData {
    /// Execution count for tracking step invocations
    pub execution_count: u32,
    /// Test key for context sharing tests
    pub test_key: Option<String>,
}

impl WorkflowData for TestWorkflowData {
    fn workflow_type() -> &'static str {
        "test_workflow"
    }
}

// Test step that counts invocations
struct CountingStep {
    counter: Arc<AtomicU32>,
    should_succeed_after: u32,
}

#[async_trait::async_trait]
impl StepExecutor<TestWorkflowData> for CountingStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<TestWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let count = self.counter.fetch_add(1, Ordering::SeqCst) + 1;

        // Store count in context
        context.data.execution_count = count;

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
impl StepExecutor<TestWorkflowData> for AlwaysSucceedStep {
    async fn execute(
        &self,
        _context: &mut WorkflowContext<TestWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        Ok(StepResult::Success)
    }
}

struct AlwaysFailStep;

#[async_trait::async_trait]
impl StepExecutor<TestWorkflowData> for AlwaysFailStep {
    async fn execute(
        &self,
        _context: &mut WorkflowContext<TestWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        Ok(StepResult::Failure)
    }
}

#[tokio::test]
async fn test_simple_workflow_execution() {
    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

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
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    // Wait for completion
    sleep(Duration::from_millis(100)).await;

    // Check status
    let state = engine.get_status(instance_id).await.unwrap();
    assert_eq!(state.status, WorkflowStatus::Completed);
    assert_eq!(state.step_states.len(), 2);
}

#[tokio::test]
async fn test_workflow_with_retry() {
    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();
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
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    // Wait for completion
    sleep(Duration::from_millis(500)).await;

    // Check that step was retried and eventually succeeded
    let state = engine.get_status(instance_id).await.unwrap();
    assert_eq!(state.status, WorkflowStatus::Completed);

    let step_state = state.step_states.get(&StepId::new("retry_step")).unwrap();
    assert_eq!(step_state.status, StepStatus::Succeeded);
    assert_eq!(step_state.attempt, 3); // Should have taken 3 attempts

    // Verify counter
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn test_workflow_failure_after_max_retries() {
    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();
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
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    // Wait for completion
    sleep(Duration::from_millis(500)).await;

    // Check that workflow failed
    let state = engine.get_status(instance_id).await.unwrap();
    assert_eq!(state.status, WorkflowStatus::Failed);

    let step_state = state.step_states.get(&StepId::new("failing_step")).unwrap();
    assert_eq!(step_state.status, StepStatus::Failed);
    assert_eq!(step_state.attempt, 3); // Should have tried 3 times

    // Verify counter
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn test_workflow_continue_on_failure() {
    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();
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
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    // Wait for completion
    sleep(Duration::from_millis(500)).await;

    // Workflow should complete despite first step failing
    let state = engine.get_status(instance_id).await.unwrap();
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
    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

    struct ContextWriterStep {
        value: String,
    }

    #[async_trait::async_trait]
    impl StepExecutor<TestWorkflowData> for ContextWriterStep {
        async fn execute(
            &self,
            context: &mut WorkflowContext<TestWorkflowData>,
        ) -> WorkflowResult<StepResult> {
            context.data.test_key = Some(self.value.clone());
            Ok(StepResult::Success)
        }
    }

    struct ContextReaderStep {
        expected_value: String,
    }

    #[async_trait::async_trait]
    impl StepExecutor<TestWorkflowData> for ContextReaderStep {
        async fn execute(
            &self,
            context: &mut WorkflowContext<TestWorkflowData>,
        ) -> WorkflowResult<StepResult> {
            let value = context
                .data
                .test_key
                .as_ref()
                .ok_or_else(|| WorkflowError::ContextValueNotFound("test_key".to_string()))?;

            if value == &self.expected_value {
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
                value: "test_value".to_string(),
            }),
        ))
        .add_step(StepDefinition::new(
            "reader",
            "Read from context",
            Arc::new(ContextReaderStep {
                expected_value: "test_value".to_string(),
            }),
        ));

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow).unwrap();

    let instance_id = engine
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    sleep(Duration::from_millis(100)).await;

    let state = engine.get_status(instance_id).await.unwrap();
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
impl StepExecutor<TestWorkflowData> for TimingStep {
    async fn execute(
        &self,
        _context: &mut WorkflowContext<TestWorkflowData>,
    ) -> WorkflowResult<StepResult> {
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
    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

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
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    // Wait for completion - give enough time for async scheduling
    for _ in 0..50 {
        sleep(Duration::from_millis(50)).await;
        let state = engine.get_status(instance_id).await.unwrap();
        if state.status != WorkflowStatus::Running {
            break;
        }
    }

    let overall_duration = overall_start.elapsed();

    let state = engine.get_status(instance_id).await.unwrap();
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
    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

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
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    // Poll until workflow completes (or timeout)
    for _ in 0..50 {
        sleep(Duration::from_millis(50)).await;
        let state = engine.get_status(instance_id).await.unwrap();
        if state.status != WorkflowStatus::Running {
            break;
        }
    }

    let state = engine.get_status(instance_id).await.unwrap();
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
    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

    let b_executed = Arc::new(AtomicU32::new(0));

    struct FailingStep;

    #[async_trait::async_trait]
    impl StepExecutor<TestWorkflowData> for FailingStep {
        async fn execute(
            &self,
            _context: &mut WorkflowContext<TestWorkflowData>,
        ) -> WorkflowResult<StepResult> {
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
    impl StepExecutor<TestWorkflowData> for TrackingStep {
        async fn execute(
            &self,
            _context: &mut WorkflowContext<TestWorkflowData>,
        ) -> WorkflowResult<StepResult> {
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
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    // Poll until workflow completes (or timeout)
    for _ in 0..50 {
        sleep(Duration::from_millis(50)).await;
        let state = engine.get_status(instance_id).await.unwrap();
        if state.status != WorkflowStatus::Running {
            break;
        }
    }

    let state = engine.get_status(instance_id).await.unwrap();
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
    assert!(matches!(
        result.unwrap_err(),
        ValidationError::CycleDetected(_)
    ));
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
    assert!(matches!(
        result.unwrap_err(),
        ValidationError::MissingDependency { .. }
    ));
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

// ============================================================================
// Scheduled/Delayed Steps Tests (#24)
// ============================================================================

#[tokio::test]
async fn test_step_delay() {
    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

    // Create a workflow with a 100ms delay
    let workflow = WorkflowDefinition::new("delay_workflow", "Delay Test").add_step(
        StepDefinition::new("delayed_step", "Delayed Step", Arc::new(AlwaysSucceedStep))
            .with_delay(Duration::from_millis(100)),
    );

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow).unwrap();

    let start = std::time::Instant::now();
    let instance_id = engine
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    // Wait for completion
    engine
        .wait_for_completion(instance_id, "test", Duration::from_secs(5))
        .await
        .unwrap();

    let duration = start.elapsed();

    // Verify delay was applied (should take at least 100ms)
    // Note: wait_for_completion cleans up state, so we verify via timing
    assert!(
        duration >= Duration::from_millis(100),
        "Step delay not applied, duration: {:?}",
        duration
    );
}

#[tokio::test]
async fn test_step_scheduled_at() {
    use chrono::Utc;

    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

    // Schedule step to run 100ms in the future
    let scheduled_time = Utc::now() + chrono::Duration::milliseconds(100);

    let workflow = WorkflowDefinition::new("scheduled_workflow", "Scheduled Test").add_step(
        StepDefinition::new(
            "scheduled_step",
            "Scheduled Step",
            Arc::new(AlwaysSucceedStep),
        )
        .scheduled_at(scheduled_time),
    );

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow).unwrap();

    let start = std::time::Instant::now();
    let instance_id = engine
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    // Wait for completion
    engine
        .wait_for_completion(instance_id, "test", Duration::from_secs(5))
        .await
        .unwrap();

    let duration = start.elapsed();

    // Verify scheduled time was respected (should take at least 100ms)
    // Note: wait_for_completion cleans up state, so we verify via timing
    assert!(
        duration >= Duration::from_millis(100),
        "Scheduled time not respected, duration: {:?}",
        duration
    );
}

// ============================================================================
// Conditional Branching Tests (#25)
// ============================================================================

#[tokio::test]
async fn test_run_if_true() {
    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

    let executed = Arc::new(AtomicU32::new(0));
    let executed_clone = Arc::clone(&executed);

    struct TrackingStep {
        counter: Arc<AtomicU32>,
    }

    #[async_trait::async_trait]
    impl StepExecutor<TestWorkflowData> for TrackingStep {
        async fn execute(
            &self,
            _context: &mut WorkflowContext<TestWorkflowData>,
        ) -> WorkflowResult<StepResult> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(StepResult::Success)
        }
    }

    // Step with run_if that always returns true
    let workflow = WorkflowDefinition::new("run_if_true_workflow", "Run If True Test").add_step(
        StepDefinition::new(
            "conditional_step",
            "Conditional Step",
            Arc::new(TrackingStep { counter: executed }),
        )
        .run_if(|_ctx| true),
    );

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow).unwrap();

    let instance_id = engine
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    engine
        .wait_for_completion(instance_id, "test", Duration::from_secs(5))
        .await
        .unwrap();

    // Step should have executed (condition was true)
    // Note: wait_for_completion cleans up state, so we verify via counter
    assert_eq!(executed_clone.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_run_if_false() {
    use tokio::time::sleep;

    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

    let executed = Arc::new(AtomicU32::new(0));
    let executed_clone = Arc::clone(&executed);

    struct TrackingStep {
        counter: Arc<AtomicU32>,
    }

    #[async_trait::async_trait]
    impl StepExecutor<TestWorkflowData> for TrackingStep {
        async fn execute(
            &self,
            _context: &mut WorkflowContext<TestWorkflowData>,
        ) -> WorkflowResult<StepResult> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(StepResult::Success)
        }
    }

    // Step with run_if that always returns false
    let workflow = WorkflowDefinition::new("run_if_false_workflow", "Run If False Test").add_step(
        StepDefinition::new(
            "conditional_step",
            "Conditional Step",
            Arc::new(TrackingStep { counter: executed }),
        )
        .run_if(|_ctx| false),
    );

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow).unwrap();

    let instance_id = engine
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    // Use polling to check status (don't use wait_for_completion which cleans up state)
    let mut state = engine.get_status(instance_id).await.unwrap();
    for _ in 0..50 {
        if state.status != WorkflowStatus::Running && state.status != WorkflowStatus::Pending {
            break;
        }
        sleep(Duration::from_millis(50)).await;
        state = engine.get_status(instance_id).await.unwrap();
    }

    assert_eq!(state.status, WorkflowStatus::Completed);

    // Step should NOT have executed (skipped due to run_if)
    assert_eq!(executed_clone.load(Ordering::SeqCst), 0);

    // Verify step was marked as skipped
    let step_state = state
        .step_states
        .get(&StepId::new("conditional_step"))
        .unwrap();
    assert_eq!(step_state.status, StepStatus::Skipped);
}

#[tokio::test]
async fn test_run_if_context_based() {
    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

    // Step that sets test_key in context
    struct SetKeyStep;

    #[async_trait::async_trait]
    impl StepExecutor<TestWorkflowData> for SetKeyStep {
        async fn execute(
            &self,
            context: &mut WorkflowContext<TestWorkflowData>,
        ) -> WorkflowResult<StepResult> {
            context.data.test_key = Some("execute_next".to_string());
            Ok(StepResult::Success)
        }
    }

    let executed = Arc::new(AtomicU32::new(0));
    let executed_clone = Arc::clone(&executed);

    struct TrackingStep {
        counter: Arc<AtomicU32>,
    }

    #[async_trait::async_trait]
    impl StepExecutor<TestWorkflowData> for TrackingStep {
        async fn execute(
            &self,
            _context: &mut WorkflowContext<TestWorkflowData>,
        ) -> WorkflowResult<StepResult> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(StepResult::Success)
        }
    }

    // Workflow where second step only runs if first step sets the right key
    let workflow = WorkflowDefinition::new("context_run_if_workflow", "Context Run If Test")
        .add_step(StepDefinition::new(
            "set_key_step",
            "Set Key",
            Arc::new(SetKeyStep),
        ))
        .add_step(
            StepDefinition::new(
                "conditional_step",
                "Conditional Step",
                Arc::new(TrackingStep { counter: executed }),
            )
            .depends_on(&["set_key_step"])
            .run_if(|ctx| ctx.data.test_key.as_deref() == Some("execute_next")),
        );

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow).unwrap();

    let instance_id = engine
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    engine
        .wait_for_completion(instance_id, "test", Duration::from_secs(5))
        .await
        .unwrap();

    // Step should have executed because context had the right value
    // Note: wait_for_completion cleans up state, so we verify via counter
    assert_eq!(executed_clone.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_depends_on_any() {
    // DAG: A and B run in parallel, C waits for ANY (not both)
    //   A ──┐
    //       ├──> C (any_of)
    //   B ──┘
    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

    let start_times: Arc<parking_lot::RwLock<Vec<(String, std::time::Instant)>>> =
        Arc::new(parking_lot::RwLock::new(Vec::new()));
    let end_times: Arc<parking_lot::RwLock<Vec<(String, std::time::Instant)>>> =
        Arc::new(parking_lot::RwLock::new(Vec::new()));

    // A takes 50ms, B takes 200ms
    // C should start after A finishes (not wait for B)
    let workflow = WorkflowDefinition::new("depends_on_any_workflow", "Depends On Any Test")
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
                duration_ms: 200,
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
            .depends_on_any(&["step_a", "step_b"]),
        );

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow).unwrap();

    let instance_id = engine
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    engine
        .wait_for_completion(instance_id, "test", Duration::from_secs(5))
        .await
        .unwrap();

    // Verify step C started after A finished but before B finished
    // Note: wait_for_completion cleans up state, so we verify via timing
    let starts = start_times.read();
    let ends = end_times.read();

    let c_start = starts.iter().find(|(n, _)| n == "step_c").unwrap().1;
    let a_end = ends.iter().find(|(n, _)| n == "step_a").unwrap().1;
    let b_end = ends.iter().find(|(n, _)| n == "step_b").unwrap().1;

    assert!(
        c_start >= a_end,
        "Step C should start after Step A finishes"
    );
    assert!(
        c_start < b_end,
        "Step C should start before Step B finishes (any_of semantics)"
    );
}

#[tokio::test]
async fn test_depends_on_any_combined_with_depends_on() {
    // DAG: C requires ALL of [A] AND ANY of [B, D]
    // A takes 50ms, B takes 100ms, D takes 200ms
    // C should start after A AND (B or D) complete
    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

    let start_times: Arc<parking_lot::RwLock<Vec<(String, std::time::Instant)>>> =
        Arc::new(parking_lot::RwLock::new(Vec::new()));
    let end_times: Arc<parking_lot::RwLock<Vec<(String, std::time::Instant)>>> =
        Arc::new(parking_lot::RwLock::new(Vec::new()));

    let workflow = WorkflowDefinition::new("combined_deps_workflow", "Combined Dependencies Test")
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
        .add_step(StepDefinition::new(
            "step_d",
            "Step D",
            Arc::new(TimingStep {
                step_name: "step_d".to_string(),
                duration_ms: 200,
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
            .depends_on(&["step_a"]) // Must wait for A
            .depends_on_any(&["step_b", "step_d"]), // AND any of B or D
        );

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow).unwrap();

    let instance_id = engine
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    engine
        .wait_for_completion(instance_id, "test", Duration::from_secs(5))
        .await
        .unwrap();

    // Verify step C started after both A AND B finished
    // (B finishes at 100ms, which is after A at 50ms)
    // Note: wait_for_completion cleans up state, so we verify via timing
    let starts = start_times.read();
    let ends = end_times.read();

    let c_start = starts.iter().find(|(n, _)| n == "step_c").unwrap().1;
    let a_end = ends.iter().find(|(n, _)| n == "step_a").unwrap().1;
    let b_end = ends.iter().find(|(n, _)| n == "step_b").unwrap().1;
    let d_end = ends.iter().find(|(n, _)| n == "step_d").unwrap().1;

    assert!(
        c_start >= a_end,
        "Step C should start after Step A (depends_on)"
    );
    assert!(
        c_start >= b_end || c_start >= d_end,
        "Step C should start after at least one of B or D (depends_on_any)"
    );
    // Since B finishes first (100ms) and A finishes before B, C should start around 100ms
    assert!(
        c_start < d_end,
        "Step C should start before D finishes (any_of semantics)"
    );
}

#[test]
fn test_dag_validation_depends_on_any_missing() {
    // Create a workflow with a missing depends_on_any dependency
    let mut workflow = WorkflowDefinition::new("missing_any_dep_workflow", "Missing Any Dep Test")
        .add_step(StepDefinition::new(
            "step_a",
            "Step A",
            Arc::new(AlwaysSucceedStep),
        ))
        .add_step(
            StepDefinition::new("step_b", "Step B", Arc::new(AlwaysSucceedStep))
                .depends_on_any(&["nonexistent_step"]),
        );

    let result = workflow.validate();
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ValidationError::MissingDependency { .. }
    ));
}

#[tokio::test]
async fn test_depends_on_any_all_fail() {
    // When ALL depends_on_any dependencies fail, the step should be blocked
    // Workflow: A and B both fail, C depends_on_any([A, B])
    // Expected: C should not run, workflow should fail
    use tokio::time::sleep;

    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

    let c_executed = Arc::new(AtomicU32::new(0));
    let c_executed_clone = Arc::clone(&c_executed);

    struct TrackingStep {
        counter: Arc<AtomicU32>,
    }

    #[async_trait::async_trait]
    impl StepExecutor<TestWorkflowData> for TrackingStep {
        async fn execute(
            &self,
            _context: &mut WorkflowContext<TestWorkflowData>,
        ) -> WorkflowResult<StepResult> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(StepResult::Success)
        }
    }

    let workflow = WorkflowDefinition::new("all_any_fail_workflow", "All Any Fail Test")
        .add_step(StepDefinition::new(
            "step_a",
            "Step A (fails)",
            Arc::new(AlwaysFailStep),
        ))
        .add_step(StepDefinition::new(
            "step_b",
            "Step B (fails)",
            Arc::new(AlwaysFailStep),
        ))
        .add_step(
            StepDefinition::new(
                "step_c",
                "Step C (depends on any of A, B)",
                Arc::new(TrackingStep {
                    counter: c_executed,
                }),
            )
            .depends_on_any(&["step_a", "step_b"]),
        );

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow).unwrap();

    let instance_id = engine
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    // Use polling to check status
    let mut state = engine.get_status(instance_id).await.unwrap();
    for _ in 0..50 {
        if state.status != WorkflowStatus::Running && state.status != WorkflowStatus::Pending {
            break;
        }
        sleep(Duration::from_millis(50)).await;
        state = engine.get_status(instance_id).await.unwrap();
    }

    // Workflow should have failed (because all depends_on_any deps failed)
    assert_eq!(state.status, WorkflowStatus::Failed);

    // Step C should NOT have executed
    assert_eq!(c_executed_clone.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn test_depends_on_any_one_fails_one_succeeds() {
    // When only SOME depends_on_any dependencies fail (but at least one succeeds),
    // the step should still run
    // Workflow: A fails, B succeeds, C depends_on_any([A, B])
    // Expected: C should run (B succeeded)
    use tokio::time::sleep;

    let engine: WorkflowEngine<TestWorkflowData> = WorkflowEngine::new();

    let c_executed = Arc::new(AtomicU32::new(0));
    let c_executed_clone = Arc::clone(&c_executed);

    struct TrackingStep {
        counter: Arc<AtomicU32>,
    }

    #[async_trait::async_trait]
    impl StepExecutor<TestWorkflowData> for TrackingStep {
        async fn execute(
            &self,
            _context: &mut WorkflowContext<TestWorkflowData>,
        ) -> WorkflowResult<StepResult> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(StepResult::Success)
        }
    }

    // Note: Step A uses ContinueNextStep so its failure doesn't fail the workflow.
    // This is the correct way to model "any of" semantics where a failing path
    // shouldn't fail the entire workflow if another path succeeds.
    let workflow = WorkflowDefinition::new("one_any_fail_workflow", "One Any Fail Test")
        .add_step(
            StepDefinition::new("step_a", "Step A (fails)", Arc::new(AlwaysFailStep))
                .with_failure_action(FailureAction::ContinueNextStep),
        )
        .add_step(StepDefinition::new(
            "step_b",
            "Step B (succeeds)",
            Arc::new(AlwaysSucceedStep),
        ))
        .add_step(
            StepDefinition::new(
                "step_c",
                "Step C (depends on any of A, B)",
                Arc::new(TrackingStep {
                    counter: c_executed,
                }),
            )
            .depends_on_any(&["step_a", "step_b"]),
        );

    let workflow_id = workflow.id.clone();
    engine.register_workflow(workflow).unwrap();

    let instance_id = engine
        .start_workflow(workflow_id, TestWorkflowData::default())
        .await
        .unwrap();

    // Use polling to check status
    let mut state = engine.get_status(instance_id).await.unwrap();
    for _ in 0..50 {
        if state.status != WorkflowStatus::Running && state.status != WorkflowStatus::Pending {
            break;
        }
        sleep(Duration::from_millis(50)).await;
        state = engine.get_status(instance_id).await.unwrap();
    }

    // Workflow should have completed (B succeeded, so C could run)
    assert_eq!(state.status, WorkflowStatus::Completed);

    // Step C SHOULD have executed (because B succeeded)
    assert_eq!(c_executed_clone.load(Ordering::SeqCst), 1);
}
