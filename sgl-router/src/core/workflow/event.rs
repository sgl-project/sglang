//! Workflow event system for observability and monitoring

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use super::types::{StepId, WorkflowId, WorkflowInstanceId};

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
pub struct EventBus {
    subscribers: Arc<RwLock<Vec<Arc<dyn EventSubscriber>>>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Subscribe to workflow events
    pub async fn subscribe(&self, subscriber: Arc<dyn EventSubscriber>) {
        self.subscribers.write().await.push(subscriber);
    }

    /// Publish an event to all subscribers
    pub async fn publish(&self, event: WorkflowEvent) {
        let subscribers = self.subscribers.read().await;
        for subscriber in subscribers.iter() {
            subscriber.on_event(&event).await;
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
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
