//! Workload-aware load balancing policy
//!
//! This policy checks the waiting queue size (num_waiting_reqs) for each worker
//! and filters out workers that exceed a threshold before selecting one.

use std::{
    collections::HashMap,
    env,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use rand::Rng;
use reqwest::Client;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use super::{get_healthy_worker_indices, LoadBalancingPolicy, WorkloadAwareConfig, SelectWorkerInfo};
use crate::core::Worker;

/// Cache entry for worker waiting count
#[derive(Clone, Debug)]
struct WaitingCountCache {
    num_waiting_reqs: usize,
    last_check_time: Instant,
}

/// Workload-aware selection policy
///
/// Selects workers randomly among healthy workers that have num_waiting_reqs
/// below the configured threshold. Workers with high waiting queue sizes are excluded.
#[derive(Debug)]
pub struct WorkloadAwarePolicy {
    /// Threshold for num_waiting_reqs - workers above this are considered busy
    num_waiting_reqs: usize,
    /// Cache for worker waiting counts (URL -> cache entry)
    /// Shared between main thread and background thread
    waiting_count_cache: Arc<Mutex<HashMap<String, WaitingCountCache>>>,
    /// Cache expiration threshold in seconds
    cache_expiration_secs: f64,
    /// Channel sender for triggering background async updates
    update_sender: mpsc::UnboundedSender<String>,
    /// Handle to the background thread
    background_handle: Option<thread::JoinHandle<()>>,
    /// Flag to signal the background thread to stop
    shutdown_flag: Arc<AtomicBool>,
}

impl WorkloadAwarePolicy {
    /// Create a new WorkloadAwarePolicy with default threshold of 10
    pub fn new() -> Self {
        Self::with_config(WorkloadAwareConfig {
            num_waiting_reqs: 10,
            api_key: None,
        })
    }

    /// Create a new WorkloadAwarePolicy with a custom threshold and optional API key
    pub fn with_config(config: WorkloadAwareConfig) -> Self {
        // Read cache expiration from environment variable, default to 1.0 seconds
        let cache_expiration_secs = env::var("SGL_ROUTER_CHECK_WORKLOAD_AWARE_INTERVAL")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(1.0);
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create workload aware HTTP client");

        // Create channel for triggering background updates
        let (tx, rx) = mpsc::unbounded_channel();
        let shutdown_flag = Arc::new(AtomicBool::new(false));

        // Create shared cache that will be used by both main thread and background thread
        let waiting_count_cache = Arc::new(Mutex::new(HashMap::<String, WaitingCountCache>::new()));

        // Clone necessary data for background thread
        let client_clone = client.clone();
        let api_key_clone = config.api_key.clone();
        let cache_clone = Arc::clone(&waiting_count_cache);
        let shutdown_clone = Arc::clone(&shutdown_flag);

        // Start background thread with tokio runtime
        let background_handle = thread::spawn(move || {
            // Create a new tokio runtime for the background thread
            let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
            rt.block_on(async move {
                let mut rx: mpsc::UnboundedReceiver<String> = rx;
                while !shutdown_clone.load(Ordering::Relaxed) {
                    tokio::select! {
                        // Wait for event to trigger update
                        worker_url_opt = rx.recv() => {
                            let worker_url_opt: Option<String> = worker_url_opt;
                            if let Some(worker_url) = worker_url_opt {
                                // Call async function to get num_waiting_reqs
                                let load_url = format!("{}/get_load", worker_url);
                                let mut request_builder = client_clone.get(&load_url);
                                if let Some(ref api_key) = api_key_clone {
                                    request_builder = request_builder.bearer_auth(api_key);
                                }

                                match request_builder.send().await {
                                    Ok(response) if response.status().is_success() => {
                                        if let Ok(json) = response.json::<Value>().await {
                                            let num_waiting_reqs = if let Some(obj) = json.as_object() {
                                                obj.get("num_waiting_reqs")
                                                    .and_then(|v| v.as_u64())
                                                    .map(|v| v as usize)
                                            } else if let Some(array) = json.as_array() {
                                                let sum: usize = array
                                                    .iter()
                                                    .filter_map(|entry| {
                                                        entry
                                                            .get("num_waiting_reqs")
                                                            .and_then(|v| v.as_u64())
                                                            .map(|v| v as usize)
                                                    })
                                                    .sum();
                                                Some(sum)
                                            } else {
                                                None
                                            };

                                            if let Some(queue_size) = num_waiting_reqs {
                                                let mut cache = cache_clone.lock().unwrap();
                                                cache.insert(
                                                    worker_url.clone(),
                                                    WaitingCountCache {
                                                        num_waiting_reqs: queue_size,
                                                        last_check_time: Instant::now(),
                                                    },
                                                );
                                                debug!(
                                                    "Background thread updated cache for worker {}: num_waiting_reqs={}",
                                                    worker_url, queue_size
                                                );
                                            }
                                        }
                                    }
                                    Ok(response) => {
                                        warn!(
                                            "Background thread: get_load request to {} returned status: {}",
                                            worker_url, response.status()
                                        );
                                    }
                                    Err(e) => {
                                        warn!(
                                            "Background thread: Failed to call /get_load on {}: {}",
                                            worker_url, e
                                        );
                                    }
                                }
                            } else {
                                // Channel closed, exit loop
                                break;
                            }
                        }
                        // Check shutdown flag periodically
                        _ = tokio::time::sleep(Duration::from_millis(100)) => {
                            // Continue loop to check shutdown flag
                        }
                    }
                }
            });
        });

        Self {
            num_waiting_reqs: config.num_waiting_reqs,
            waiting_count_cache,
            cache_expiration_secs,
            update_sender: tx,
            background_handle: Some(background_handle),
            shutdown_flag,
        }
    }

    /// Trigger background update for a worker
    pub fn trigger_update(&self, worker_url: &str) {
        if let Err(e) = self.update_sender.send(worker_url.to_string()) {
            warn!("Failed to send update event for worker {}: {}", worker_url, e);
        }
    }

    /// Query num_waiting_reqs from a worker's /get_load endpoint
    /// Uses caching to avoid frequent API calls
    fn get_num_waiting_reqs(&self, worker_url: &str) -> Option<usize> {
        let now = Instant::now();
        let cache = self.waiting_count_cache.lock().unwrap();
        if let Some(cached) = cache.get(worker_url) {
            let elapsed = now.duration_since(cached.last_check_time);
            if elapsed.as_secs_f64() < self.cache_expiration_secs {
                debug!(
                    "Using cached num_waiting_reqs for {}: {} (cached {}ms ago)",
                    worker_url,
                    cached.num_waiting_reqs,
                    elapsed.as_millis()
                );
            } else {
                self.trigger_update(worker_url);
            }
            return Some(cached.num_waiting_reqs);
        } else {
            self.trigger_update(worker_url);
            None
        }
    }
}

impl Default for WorkloadAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for WorkloadAwarePolicy {
    fn drop(&mut self) {
        // Signal shutdown
        self.shutdown_flag.store(true, Ordering::Relaxed);
        // Close the channel to wake up the background thread
        drop(self.update_sender.clone());
        // Wait for background thread to finish
        if let Some(handle) = self.background_handle.take() {
            let _ = handle.join();
        }
    }
}

impl LoadBalancingPolicy for WorkloadAwarePolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Filter workers by checking their num_waiting_reqs
        let mut available_indices = Vec::new();
        for &idx in &healthy_indices {
            let worker = &workers[idx];
            let worker_url = worker.url();

            // Trigger background update for this worker
            self.trigger_update(worker_url);

            // Query num_waiting_reqs from the worker (may use cached value)
            if let Some(num_waiting) = self.get_num_waiting_reqs(worker_url) {
                if num_waiting <= self.num_waiting_reqs {
                    available_indices.push(idx);
                    debug!(
                        "Worker {} has num_waiting_reqs={}, within threshold={}",
                        worker_url, num_waiting, self.num_waiting_reqs
                    );
                } else {
                    debug!(
                        "Worker {} has num_waiting_reqs={}, exceeds threshold={}, skipping",
                        worker_url, num_waiting, self.num_waiting_reqs
                    );
                }
            } else {
                // If we can't get load info, include the worker anyway (fail-open)
                // This ensures we don't exclude all workers if monitoring fails
                debug!(
                    "Could not get load info for worker {}, including anyway",
                    worker_url
                );
                available_indices.push(idx);
            }
        }

        if available_indices.is_empty() {
            warn!(
                "No workers available after filtering by workload (threshold={})",
                self.num_waiting_reqs
            );
            return None;
        }

        // Randomly select from available workers
        let mut rng = rand::rng();
        let random_idx = rng.random_range(0..available_indices.len());

        Some(available_indices[random_idx])
    }

    fn name(&self) -> &'static str {
        "workload_aware"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_aware_creation() {
        let policy = WorkloadAwarePolicy::new();
        assert_eq!(policy.name(), "workload_aware");
        assert_eq!(policy.num_waiting_reqs, 10);

        let policy = WorkloadAwarePolicy::with_config(WorkloadAwareConfig {
            num_waiting_reqs: 5,
            api_key: None,
        });
        assert_eq!(policy.num_waiting_reqs, 5);
    }

    #[test]
    fn test_workload_aware_with_api_key() {
        let policy = WorkloadAwarePolicy::with_config(WorkloadAwareConfig {
            num_waiting_reqs: 10,
            api_key: Some("test_key".to_string()),
        });
        assert_eq!(policy.num_waiting_reqs, 10);
    }
}
