//! Worker-slot-aware central admission for streaming capacity.
//!
//! When enabled, requests wait centrally until at least one healthy worker has
//! a free stream slot (`worker.load() < slots_per_worker`). Admission wakes when
//! a `WorkerLoadGuard` drops (stream completes/cancels/errors) or when a worker
//! is registered.
//!
//! Boundary: this gate is wired specifically to power-of-two HTTP routing. The
//! global `TokenBucket` concurrency limiter in middleware remains unchanged for
//! other policies and non-streaming endpoints.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::sync::Notify;
use tracing::{debug, warn};

use crate::core::{ConnectionMode, Worker, WorkerRegistry, WorkerType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionError {
    Timeout,
    NoWorkers,
    /// Reserved for request-cancellation while waiting; client disconnect currently
    /// cancels the wait future by drop instead of returning this variant.
    #[allow(dead_code)]
    Cancelled,
}

/// Capacity-aware admission primitive keyed off live worker load counters.
#[derive(Debug)]
pub struct CapacityGate {
    slots_per_worker: usize,
    queue_timeout: Duration,
    notify: Arc<Notify>,
}

impl CapacityGate {
    pub fn new(slots_per_worker: usize, queue_timeout: Duration) -> Self {
        Self {
            slots_per_worker,
            queue_timeout,
            notify: Arc::new(Notify::new()),
        }
    }

    pub fn slots_per_worker(&self) -> usize {
        self.slots_per_worker
    }

    pub fn notifier(&self) -> Arc<Notify> {
        Arc::clone(&self.notify)
    }

    /// Attach to a worker registry so registration wakes waiters.
    pub fn attach_registry(self: &Arc<Self>, registry: Arc<WorkerRegistry>) {
        registry.set_capacity_notify(Some(self.notifier()));
    }

    pub fn notify_capacity_changed(&self) {
        self.notify.notify_waiters();
    }

    pub fn has_capacity(&self, workers: &[Arc<dyn Worker>]) -> bool {
        workers
            .iter()
            .any(|w| w.is_available() && w.load() < self.slots_per_worker)
    }

    /// Wait until a healthy worker has a free stream slot, or until timeout.
    pub async fn wait_for_capacity(
        &self,
        workers: &[Arc<dyn Worker>],
    ) -> Result<(), AdmissionError> {
        if self.slots_per_worker == 0 {
            return Ok(());
        }

        if workers.iter().all(|w| !w.is_available()) {
            return Err(AdmissionError::NoWorkers);
        }

        if self.has_capacity(workers) {
            return Ok(());
        }

        let deadline = Instant::now() + self.queue_timeout;
        debug!(
            slots_per_worker = self.slots_per_worker,
            workers = workers.len(),
            "CapacityGate: waiting for a free worker stream slot"
        );

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                warn!("CapacityGate: timed out waiting for worker stream slot");
                return Err(AdmissionError::Timeout);
            }

            // Re-check after registering for notify to avoid lost wakeups.
            let notified = self.notify.notified();
            tokio::pin!(notified);

            if self.has_capacity(workers) {
                return Ok(());
            }

            match tokio::time::timeout(remaining, &mut notified).await {
                Ok(()) => {
                    // Fresh worker list should be supplied by caller on next
                    // attempt; still re-check the snapshot we were given.
                    if self.has_capacity(workers) {
                        return Ok(());
                    }
                }
                Err(_) => {
                    warn!("CapacityGate: timed out waiting for worker stream slot");
                    return Err(AdmissionError::Timeout);
                }
            }
        }
    }

    /// Convenience helper that refreshes the worker snapshot from the registry
    /// on each wake (needed when discovery adds workers mid-wait).
    pub async fn wait_for_capacity_refreshing(
        &self,
        registry: &WorkerRegistry,
        model_id: Option<&str>,
        enable_igw: bool,
    ) -> Result<(), AdmissionError> {
        if self.slots_per_worker == 0 {
            return Ok(());
        }

        let deadline = Instant::now() + self.queue_timeout;
        loop {
            let workers = snapshot_regular_http_workers(registry, model_id, enable_igw);
            if workers.iter().any(|w| w.is_available()) && self.has_capacity(&workers) {
                return Ok(());
            }
            if workers.iter().all(|w| !w.is_available()) && workers.is_empty() {
                // Keep waiting for discovery unless we already timed out.
            } else if workers.iter().all(|w| !w.is_available()) && !workers.is_empty() {
                // Unhealthy/circuit-open workers: still wait for recovery or discovery.
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                if snapshot_regular_http_workers(registry, model_id, enable_igw).is_empty() {
                    return Err(AdmissionError::NoWorkers);
                }
                warn!("CapacityGate: timed out waiting for worker stream slot");
                return Err(AdmissionError::Timeout);
            }

            let notified = self.notify.notified();
            tokio::pin!(notified);

            let workers = snapshot_regular_http_workers(registry, model_id, enable_igw);
            if self.has_capacity(&workers) {
                return Ok(());
            }

            match tokio::time::timeout(remaining, &mut notified).await {
                Ok(()) => continue,
                Err(_) => {
                    let workers = snapshot_regular_http_workers(registry, model_id, enable_igw);
                    if self.has_capacity(&workers) {
                        return Ok(());
                    }
                    if workers.is_empty() {
                        return Err(AdmissionError::NoWorkers);
                    }
                    warn!("CapacityGate: timed out waiting for worker stream slot");
                    return Err(AdmissionError::Timeout);
                }
            }
        }
    }
}

fn snapshot_regular_http_workers(
    registry: &WorkerRegistry,
    model_id: Option<&str>,
    enable_igw: bool,
) -> Vec<Arc<dyn Worker>> {
    let effective_model_id = if !enable_igw { None } else { model_id };
    registry.get_workers_filtered(
        effective_model_id,
        Some(WorkerType::Regular),
        Some(ConnectionMode::Http),
        None,
        false,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::time::Duration;

    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .build(),
        )
    }

    #[tokio::test]
    async fn admits_immediately_when_slot_free() {
        let gate = CapacityGate::new(2, Duration::from_secs(1));
        let w = worker("http://w1:8000");
        assert!(gate.wait_for_capacity(&[w]).await.is_ok());
    }

    #[tokio::test]
    async fn times_out_when_all_slots_full() {
        let gate = CapacityGate::new(1, Duration::from_millis(50));
        let w = worker("http://w1:8000");
        w.increment_load();
        let err = gate.wait_for_capacity(&[w]).await.unwrap_err();
        assert_eq!(err, AdmissionError::Timeout);
    }

    #[tokio::test]
    async fn wakes_when_load_released() {
        let gate = Arc::new(CapacityGate::new(1, Duration::from_secs(2)));
        let w = worker("http://w1:8000");
        w.increment_load();

        let gate2 = Arc::clone(&gate);
        let w2 = Arc::clone(&w);
        let waiter = tokio::spawn(async move { gate2.wait_for_capacity(&[w2]).await });

        tokio::time::sleep(Duration::from_millis(30)).await;
        w.decrement_load();
        gate.notify_capacity_changed();

        assert!(waiter.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn wakes_when_new_worker_registered() {
        let gate = Arc::new(CapacityGate::new(1, Duration::from_secs(2)));
        let registry = Arc::new(WorkerRegistry::new());
        gate.attach_registry(Arc::clone(&registry));

        let full = worker("http://full:8000");
        full.increment_load();
        registry.register(full);

        let gate2 = Arc::clone(&gate);
        let registry2 = Arc::clone(&registry);
        let waiter = tokio::spawn(async move {
            gate2
                .wait_for_capacity_refreshing(registry2.as_ref(), None, false)
                .await
        });

        tokio::time::sleep(Duration::from_millis(30)).await;
        registry.register(worker("http://new:8000"));

        assert!(waiter.await.unwrap().is_ok());
    }

    #[test]
    fn has_capacity_respects_slots_per_worker() {
        let gate = CapacityGate::new(2, Duration::from_secs(1));
        let w = worker("http://w1:8000");
        w.increment_load();
        assert!(gate.has_capacity(&[Arc::clone(&w)]));
        w.increment_load();
        assert!(!gate.has_capacity(&[w]));
    }
}
