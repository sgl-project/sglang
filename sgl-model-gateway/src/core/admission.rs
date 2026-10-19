use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError},
    time::Instant,
};

use crate::observability::metrics::{AdmissionDecision, Metrics};

/// A snapshot of the limiter's current admission state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionSnapshot {
    pub inflight: usize,
    pub queued: usize,
}

/// Errors returned when a request cannot be admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AdmissionError {
    #[error("admission queue is full")]
    QueueFull,
    #[error("admission queue timed out after {waited:?}")]
    QueueTimeout { waited: Duration },
    #[error("admission limiter is shutting down")]
    ShuttingDown,
}

#[derive(Debug, Default)]
struct AdmissionState {
    inflight: AtomicUsize,
    queued: AtomicUsize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QueueExit {
    Cancelled,
    Admitted,
    TimedOut,
    ShuttingDown,
}

#[derive(Debug)]
struct QueueGuard {
    _permit: OwnedSemaphorePermit,
    queued_at: Instant,
    exit: QueueExit,
    state: Arc<AdmissionState>,
}

impl QueueGuard {
    fn new(permit: OwnedSemaphorePermit, state: Arc<AdmissionState>) -> Self {
        let queued_at = Instant::now();
        state.queued.fetch_add(1, Ordering::Relaxed);
        Metrics::increment_admission_queued();
        Self {
            _permit: permit,
            queued_at,
            exit: QueueExit::Cancelled,
            state,
        }
    }

    fn mark_admitted(&mut self) {
        self.exit = QueueExit::Admitted;
    }

    fn mark_timed_out(&mut self) {
        self.exit = QueueExit::TimedOut;
    }

    fn mark_shutting_down(&mut self) {
        self.exit = QueueExit::ShuttingDown;
    }
}

impl Drop for QueueGuard {
    fn drop(&mut self) {
        self.state.queued.fetch_sub(1, Ordering::Relaxed);
        Metrics::decrement_admission_queued();
        Metrics::record_admission_queue_wait(
            Instant::now().saturating_duration_since(self.queued_at),
        );
        let decision = match self.exit {
            QueueExit::Cancelled => AdmissionDecision::Cancelled,
            QueueExit::Admitted => AdmissionDecision::Admitted,
            QueueExit::TimedOut => AdmissionDecision::QueueTimeout,
            QueueExit::ShuttingDown => AdmissionDecision::ShuttingDown,
        };
        Metrics::record_admission_decision(decision);
    }
}

/// Limits the number of requests admitted concurrently.
#[derive(Debug)]
pub struct AdmissionLimiter {
    concurrency: Arc<Semaphore>,
    queue_slots: Arc<Semaphore>,
    queue_timeout: Duration,
    shutting_down: AtomicBool,
    state: Arc<AdmissionState>,
}

impl AdmissionLimiter {
    pub fn new(concurrency: usize, queue_capacity: usize, queue_timeout: Duration) -> Self {
        Self {
            concurrency: Arc::new(Semaphore::new(concurrency)),
            queue_slots: Arc::new(Semaphore::new(queue_capacity)),
            queue_timeout,
            shutting_down: AtomicBool::new(false),
            state: Arc::new(AdmissionState::default()),
        }
    }

    /// Acquires an admission lease, waiting in the bounded queue if necessary.
    pub async fn acquire(&self) -> Result<AdmissionLease, AdmissionError> {
        if self.shutting_down.load(Ordering::Acquire) {
            Metrics::record_admission_decision(AdmissionDecision::ShuttingDown);
            return Err(AdmissionError::ShuttingDown);
        }

        match Arc::clone(&self.concurrency).try_acquire_owned() {
            Ok(permit) => {
                return Ok(AdmissionLease::new(permit, Arc::clone(&self.state), true));
            }
            Err(TryAcquireError::Closed) => {
                Metrics::record_admission_decision(AdmissionDecision::ShuttingDown);
                return Err(AdmissionError::ShuttingDown);
            }
            Err(TryAcquireError::NoPermits) => {}
        }

        let queue_permit = match Arc::clone(&self.queue_slots).try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::Closed) => {
                Metrics::record_admission_decision(AdmissionDecision::ShuttingDown);
                return Err(AdmissionError::ShuttingDown);
            }
            Err(TryAcquireError::NoPermits) => {
                if self.shutting_down.load(Ordering::Acquire) {
                    Metrics::record_admission_decision(AdmissionDecision::ShuttingDown);
                    return Err(AdmissionError::ShuttingDown);
                }
                Metrics::record_admission_decision(AdmissionDecision::QueueFull);
                return Err(AdmissionError::QueueFull);
            }
        };
        let mut queue_guard = QueueGuard::new(queue_permit, Arc::clone(&self.state));
        let deadline = queue_guard.queued_at + self.queue_timeout;
        let permit =
            match tokio::time::timeout_at(deadline, Arc::clone(&self.concurrency).acquire_owned())
                .await
            {
                Ok(Ok(permit)) => permit,
                Ok(Err(_)) => {
                    queue_guard.mark_shutting_down();
                    return Err(AdmissionError::ShuttingDown);
                }
                Err(_) => {
                    if self.shutting_down.load(Ordering::Acquire) {
                        queue_guard.mark_shutting_down();
                        return Err(AdmissionError::ShuttingDown);
                    }
                    queue_guard.mark_timed_out();
                    return Err(AdmissionError::QueueTimeout {
                        waited: Instant::now().saturating_duration_since(queue_guard.queued_at),
                    });
                }
            };
        let lease = AdmissionLease::new(permit, Arc::clone(&self.state), false);
        queue_guard.mark_admitted();
        drop(queue_guard);

        Ok(lease)
    }

    pub fn begin_shutdown(&self) {
        self.shutting_down.store(true, Ordering::Release);
        self.concurrency.close();
        self.queue_slots.close();
    }

    pub fn snapshot(&self) -> AdmissionSnapshot {
        AdmissionSnapshot {
            inflight: self.state.inflight.load(Ordering::Acquire),
            queued: self.state.queued.load(Ordering::Acquire),
        }
    }

    pub fn available_concurrency_permits(&self) -> usize {
        self.concurrency.available_permits()
    }

    pub fn available_queue_slots(&self) -> usize {
        self.queue_slots.available_permits()
    }
}

/// Owns one concurrency permit for the lifetime of an admitted request.
#[derive(Debug)]
pub struct AdmissionLease {
    _permit: OwnedSemaphorePermit,
    state: Arc<AdmissionState>,
}

impl AdmissionLease {
    fn new(
        permit: OwnedSemaphorePermit,
        state: Arc<AdmissionState>,
        record_admitted: bool,
    ) -> Self {
        state.inflight.fetch_add(1, Ordering::AcqRel);
        Metrics::increment_admission_inflight();
        if record_admitted {
            Metrics::record_admission_decision(AdmissionDecision::Admitted);
        }
        Self {
            _permit: permit,
            state,
        }
    }
}

impl Drop for AdmissionLease {
    fn drop(&mut self) {
        self.state.inflight.fetch_sub(1, Ordering::AcqRel);
        Metrics::decrement_admission_inflight();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn wait_for_queued(limiter: &AdmissionLimiter, expected: usize) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if limiter.snapshot().queued == expected {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("admission queue did not reach the expected depth");
    }

    #[tokio::test]
    async fn lease_tracks_the_full_inflight_lifecycle() {
        let limiter = AdmissionLimiter::new(2, 1, Duration::from_secs(1));

        let first = limiter.acquire().await.unwrap();
        let second = limiter.acquire().await.unwrap();
        assert_eq!(
            limiter.snapshot(),
            AdmissionSnapshot {
                inflight: 2,
                queued: 0,
            }
        );

        drop(first);
        assert_eq!(limiter.snapshot().inflight, 1);
        drop(second);
        assert_eq!(limiter.snapshot().inflight, 0);
        assert_eq!(limiter.available_concurrency_permits(), 2);
    }

    #[tokio::test]
    async fn queue_capacity_is_a_hard_waiter_limit() {
        let limiter = Arc::new(AdmissionLimiter::new(1, 1, Duration::from_secs(1)));
        let active = limiter.acquire().await.unwrap();

        let waiting_limiter = Arc::clone(&limiter);
        let waiter = tokio::spawn(async move { waiting_limiter.acquire().await });
        wait_for_queued(&limiter, 1).await;

        assert!(matches!(
            limiter.acquire().await,
            Err(AdmissionError::QueueFull)
        ));

        drop(active);
        let admitted = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            limiter.snapshot(),
            AdmissionSnapshot {
                inflight: 1,
                queued: 0,
            }
        );

        drop(admitted);
        assert_eq!(limiter.snapshot().inflight, 0);
    }

    #[tokio::test]
    async fn queue_timeout_is_measured_from_queue_entry() {
        let queue_timeout = Duration::from_millis(25);
        let limiter = AdmissionLimiter::new(1, 1, queue_timeout);
        let active = limiter.acquire().await.unwrap();

        let error = limiter.acquire().await.unwrap_err();
        match error {
            AdmissionError::QueueTimeout { waited } => {
                assert!(waited >= queue_timeout);
            }
            other => panic!("expected queue timeout, got {other:?}"),
        }
        assert_eq!(limiter.snapshot().queued, 0);

        drop(active);
        assert_eq!(limiter.snapshot().inflight, 0);
    }

    #[tokio::test]
    async fn cancelling_a_waiter_releases_its_queue_slot() {
        let limiter = Arc::new(AdmissionLimiter::new(1, 1, Duration::from_secs(1)));
        let active = limiter.acquire().await.unwrap();

        let waiting_limiter = Arc::clone(&limiter);
        let waiter = tokio::spawn(async move { waiting_limiter.acquire().await });
        wait_for_queued(&limiter, 1).await;

        waiter.abort();
        let _ = waiter.await;
        wait_for_queued(&limiter, 0).await;
        assert_eq!(limiter.available_queue_slots(), 1);

        drop(active);
    }

    #[tokio::test]
    async fn shutdown_wakes_queued_waiters() {
        let limiter = Arc::new(AdmissionLimiter::new(1, 1, Duration::from_secs(60)));
        let active = limiter.acquire().await.unwrap();

        let waiting_limiter = Arc::clone(&limiter);
        let waiter = tokio::spawn(async move { waiting_limiter.acquire().await });
        wait_for_queued(&limiter, 1).await;

        limiter.begin_shutdown();
        let result = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(result, Err(AdmissionError::ShuttingDown)));
        assert_eq!(limiter.snapshot().queued, 0);

        drop(active);
        assert_eq!(limiter.snapshot().inflight, 0);
    }
}
