use super::{TokenBucket, Worker};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::info;

/// Manages dynamic capacity adjustment for the token bucket based on worker capabilities
pub struct CapacityManager {
    token_bucket: Arc<TokenBucket>,
    workers: Arc<RwLock<Vec<Arc<dyn Worker>>>>,
    update_interval: Duration,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
}

impl CapacityManager {
    /// Create a new capacity manager
    pub fn new(token_bucket: Arc<TokenBucket>, update_interval: Duration) -> Self {
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

        Self {
            token_bucket,
            workers: Arc::new(RwLock::new(Vec::new())),
            update_interval,
            shutdown_tx,
            shutdown_rx,
        }
    }

    /// Update the list of workers
    pub async fn update_workers(&self, workers: Vec<Arc<dyn Worker>>) {
        let mut w = self.workers.write().await;
        *w = workers;

        // Trigger immediate capacity recalculation
        self.recalculate_capacity().await;
    }

    /// Add a worker
    pub async fn add_worker(&self, worker: Arc<dyn Worker>) {
        let mut workers = self.workers.write().await;
        workers.push(worker);
        drop(workers);

        // Trigger immediate capacity recalculation
        self.recalculate_capacity().await;
    }

    /// Remove a worker by URL
    pub async fn remove_worker(&self, url: &str) {
        let mut workers = self.workers.write().await;
        workers.retain(|w| w.url() != url);
        drop(workers);

        // Trigger immediate capacity recalculation
        self.recalculate_capacity().await;
    }

    /// Recalculate and update token bucket capacity based on current workers
    async fn recalculate_capacity(&self) {
        let workers = self.workers.read().await;

        let mut total_capacity = 0;
        let mut known_capacity_count = 0;
        let mut healthy_workers = 0;

        for worker in workers.iter() {
            if worker.is_healthy() {
                healthy_workers += 1;

                // worker.capacity() returns the configured max concurrent requests
                // for that worker (e.g., from SGLang's --max-running-requests parameter)
                if let Some(capacity) = worker.capacity() {
                    total_capacity += capacity;
                    known_capacity_count += 1;
                }
            }
        }

        // If no workers have known capacity, use a default
        let new_capacity = if known_capacity_count > 0 {
            total_capacity
        } else if healthy_workers > 0 {
            // Default: assume each worker can handle 100 concurrent requests
            healthy_workers * 100
        } else {
            // No healthy workers, set minimal capacity
            1
        };

        // For refill rate, we can use the same value as capacity
        // This means we can handle 'capacity' new requests per second
        let new_refill_rate = new_capacity;

        // Update the token bucket
        self.token_bucket
            .update_parameters(new_capacity, new_refill_rate)
            .await;

        info!(
            "Updated token bucket capacity: {} (from {} healthy workers, {} with known capacity)",
            new_capacity, healthy_workers, known_capacity_count
        );
    }

    /// Start the background task that periodically updates capacity
    pub fn start(self: Arc<Self>) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self.update_interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            let mut shutdown_rx = self.shutdown_rx.clone();

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        self.recalculate_capacity().await;
                    }
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            info!("Capacity manager shutting down");
                            break;
                        }
                    }
                }
            }
        })
    }

    /// Shutdown the capacity manager
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[tokio::test]
    async fn test_capacity_calculation() {
        let token_bucket = Arc::new(TokenBucket::new(100, 100));
        let manager = Arc::new(CapacityManager::new(
            token_bucket.clone(),
            Duration::from_secs(10),
        ));

        // Add workers with different capacities
        let worker1 = Arc::new(BasicWorker::new(
            "http://worker1:8080".to_string(),
            WorkerType::Regular,
        ));
        worker1.set_healthy(true);

        let worker2 = Arc::new(BasicWorker::new(
            "http://worker2:8080".to_string(),
            WorkerType::Regular,
        ));
        worker2.set_healthy(true);

        manager.add_worker(worker1).await;
        manager.add_worker(worker2).await;

        // Since BasicWorker doesn't implement capacity(), it should use default
        let (capacity, rate) = token_bucket.get_parameters().await;
        assert_eq!(capacity, 200.0); // 2 workers * 100 default
        assert_eq!(rate, 200.0);
    }
}
