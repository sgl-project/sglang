//! Workflow event system for observability and monitoring

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use super::types::{StepId, WorkflowId, WorkflowInstanceId};

/// Default timeout for subscriber event handlers
const DEFAULT_SUBSCRIBER_TIMEOUT: Duration = Duration::from_secs(30);

/// Events emitted by the workflow engine
#[derive(Debug, Clone)]
pub enum WorkflowEvent {
    WorkflowStarted {
        instance_id: WorkflowInstanceId,
        definition_id: WorkflowId,
    },
    StepStarted {
        instance_id: WorkflowInstanceId,
        step_id: StepId,
        attempt: u32,
    },
    StepSucceeded {
        instance_id: WorkflowInstanceId,
        step_id: StepId,
        duration: Duration,
    },
    StepFailed {
        instance_id: WorkflowInstanceId,
        step_id: StepId,
        error: String,
        will_retry: bool,
    },
    StepRetrying {
        instance_id: WorkflowInstanceId,
        step_id: StepId,
        attempt: u32,
        delay: Duration,
    },
    WorkflowCompleted {
        instance_id: WorkflowInstanceId,
        duration: Duration,
    },
    WorkflowFailed {
        instance_id: WorkflowInstanceId,
        failed_step: StepId,
        error: String,
    },
    WorkflowCancelled {
        instance_id: WorkflowInstanceId,
    },
}

/// Trait for subscribing to workflow events
#[async_trait]
pub trait EventSubscriber: Send + Sync {
    async fn on_event(&self, event: &WorkflowEvent);
}

/// Event bus for publishing and subscribing to workflow events
///
/// # Subscriber Isolation
///
/// Each subscriber is notified in a separate spawned task with a timeout.
/// This ensures that:
/// - A slow subscriber doesn't block other subscribers
/// - A panicking subscriber doesn't affect other subscribers
/// - Event publishing returns quickly regardless of subscriber behavior
pub struct EventBus {
    subscribers: Arc<RwLock<Vec<Arc<dyn EventSubscriber>>>>,
    /// Timeout for each subscriber's event handler
    subscriber_timeout: Duration,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(Vec::new())),
            subscriber_timeout: DEFAULT_SUBSCRIBER_TIMEOUT,
        }
    }

    /// Create an EventBus with a custom subscriber timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(Vec::new())),
            subscriber_timeout: timeout,
        }
    }

    /// Subscribe to workflow events
    pub async fn subscribe(&self, subscriber: Arc<dyn EventSubscriber>) {
        self.subscribers.write().await.push(subscriber);
    }

    /// Unsubscribe from workflow events
    ///
    /// Removes the subscriber by Arc pointer equality.
    /// Returns true if the subscriber was found and removed.
    pub async fn unsubscribe(&self, subscriber: &Arc<dyn EventSubscriber>) -> bool {
        let mut subs = self.subscribers.write().await;
        let len_before = subs.len();
        subs.retain(|s| !Arc::ptr_eq(s, subscriber));
        subs.len() < len_before
    }

    /// Publish an event to all subscribers concurrently
    ///
    /// Each subscriber is notified in a separate spawned task with a timeout.
    /// This method returns after spawning all notification tasks, without
    /// waiting for subscribers to complete (fire-and-forget).
    ///
    /// Subscriber failures (timeout or panic) are logged but don't affect
    /// other subscribers or the caller.
    pub async fn publish(&self, event: WorkflowEvent) {
        let subscribers: Vec<_> = self.subscribers.read().await.iter().cloned().collect();
        let timeout = self.subscriber_timeout;

        for (idx, subscriber) in subscribers.into_iter().enumerate() {
            let event = event.clone();
            tokio::spawn(async move {
                let result = tokio::time::timeout(timeout, subscriber.on_event(&event)).await;
                match result {
                    Ok(()) => {}
                    Err(_) => {
                        warn!(
                            subscriber_index = idx,
                            timeout_secs = timeout.as_secs(),
                            "Event subscriber timed out"
                        );
                    }
                }
            });
        }
    }

    /// Publish an event and wait for all subscribers to complete
    ///
    /// Unlike `publish`, this method waits for all subscribers to finish
    /// (or timeout). Use this when you need to ensure all subscribers
    /// have processed the event before continuing.
    pub async fn publish_and_wait(&self, event: WorkflowEvent) {
        let subscribers: Vec<_> = self.subscribers.read().await.iter().cloned().collect();
        let timeout = self.subscriber_timeout;

        let handles: Vec<_> = subscribers
            .into_iter()
            .enumerate()
            .map(|(idx, subscriber)| {
                let event = event.clone();
                tokio::spawn(async move {
                    let result = tokio::time::timeout(timeout, subscriber.on_event(&event)).await;
                    if result.is_err() {
                        warn!(
                            subscriber_index = idx,
                            timeout_secs = timeout.as_secs(),
                            "Event subscriber timed out"
                        );
                    }
                })
            })
            .collect();

        // Wait for all spawned tasks, ignoring individual failures (panics)
        for handle in handles {
            let _ = handle.await;
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for EventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventBus").finish_non_exhaustive()
    }
}

/// Logging subscriber that logs events using tracing
pub struct LoggingSubscriber;

#[async_trait]
impl EventSubscriber for LoggingSubscriber {
    async fn on_event(&self, event: &WorkflowEvent) {
        match event {
            WorkflowEvent::WorkflowStarted {
                instance_id,
                definition_id,
            } => {
                info!(
                    instance_id = %instance_id,
                    definition_id = %definition_id,
                    "Workflow started"
                );
            }
            WorkflowEvent::StepStarted {
                instance_id,
                step_id,
                attempt,
            } => {
                info!(
                    instance_id = %instance_id,
                    step_id = %step_id,
                    attempt = attempt,
                    "Step started"
                );
            }
            WorkflowEvent::StepSucceeded {
                instance_id,
                step_id,
                duration,
            } => {
                info!(
                    instance_id = %instance_id,
                    step_id = %step_id,
                    duration_ms = duration.as_millis(),
                    "Step succeeded"
                );
            }
            WorkflowEvent::StepFailed {
                instance_id,
                step_id,
                error,
                will_retry,
            } => {
                warn!(
                    instance_id = %instance_id,
                    step_id = %step_id,
                    error = error,
                    will_retry = will_retry,
                    "Step failed"
                );
            }
            WorkflowEvent::StepRetrying {
                instance_id,
                step_id,
                attempt,
                delay,
            } => {
                info!(
                    instance_id = %instance_id,
                    step_id = %step_id,
                    attempt = attempt,
                    delay_ms = delay.as_millis(),
                    "Step retrying"
                );
            }
            WorkflowEvent::WorkflowCompleted {
                instance_id,
                duration,
            } => {
                info!(
                    instance_id = %instance_id,
                    duration_ms = duration.as_millis(),
                    "Workflow completed"
                );
            }
            WorkflowEvent::WorkflowFailed {
                instance_id,
                failed_step,
                error,
            } => {
                error!(
                    instance_id = %instance_id,
                    failed_step = %failed_step,
                    error = error,
                    "Workflow failed"
                );
            }
            WorkflowEvent::WorkflowCancelled { instance_id } => {
                info!(instance_id = %instance_id, "Workflow cancelled");
            }
        }
    }
}
