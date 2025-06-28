//! Core worker trait and implementations for unified worker management
use crate::utils::api_path;
use super::error::WorkerError;
use dyn_clone::DynClone;
use futures::future::{BoxFuture, join_all};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{error, info};
use tokio::time::sleep;


const DEFAULT_HEALTH_CHECK_CACHE_TTL: Duration = Duration::from_secs(5);
const HEALTH_ENDPOINT: &str = "/health";


/// Worker types supported by the router
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerType {
    /// Regular worker for standard routing
    Regular,
    /// Decode worker for disaggregated prefill-decode systems
    Decode,
    /// Prefill worker for disaggregated prefill-decode systems
    Prefill(Option<u16>),
}

impl WorkerType {
    pub fn get_health_check_api(&self) -> &'static str {
        match self {
            _ => HEALTH_ENDPOINT,
        }
    }
}

/// Core trait for all worker types
pub trait Worker: Send + Sync + DynClone + fmt::Display {
    /// Get the worker's URL
    fn url(&self) -> &str;

    /// Get the worker's type
    fn worker_type(&self) -> WorkerType;

    /// Check if the worker is currently healthy
    fn is_healthy(&self) -> bool;

    /// Perform a health check on the worker
    fn check_health(&self) -> BoxFuture<'_, Result<(), WorkerError>>;

    /// Get the current load counter for this worker
    fn load(&self) -> Arc<AtomicUsize>;

    /// Update the health status of the worker
    fn update_health(&self, healthy: bool);
}
dyn_clone::clone_trait_object!(Worker);

/// Default implementation of the Worker trait
#[derive(Debug, Clone)]
struct WorkerCommon {
    url: String,
    healthy: Arc<AtomicBool>,
    load: Arc<AtomicUsize>,
    last_health_check: Arc<RwLock<Instant>>,
    health_check_ttl: Duration,
}

impl WorkerCommon {
    fn new(url: String, health_check_ttl: Duration) -> Self {
        Self {
            url,
            healthy: Arc::new(AtomicBool::new(true)),
            load: Arc::new(AtomicUsize::new(0)),
            last_health_check: Arc::new(RwLock::new(Instant::now())),
            health_check_ttl,
        }
    }
}

macro_rules! common_worker_methods {
    () => {
        fn url(&self) -> &str {
            &self.common.url
        }

        fn is_healthy(&self) -> bool {
            self.common.healthy.load(std::sync::atomic::Ordering::Relaxed)
        }

        fn check_health(&self) -> futures::future::BoxFuture<'_, Result<(), WorkerError>> {
            let url = self.common.url.clone();
            let health_url = api_path(&self.common.url, self.worker_type().get_health_check_api());
            let last_health_check = self.common.last_health_check.clone();
            let healthy = self.common.healthy.clone();
            Box::pin(async move {
                // Check if we need a fresh health check
                let needs_check = {
                    if let Ok(lock) = last_health_check.read() {
                        lock.elapsed() > self.common.health_check_ttl
                    } else {
                        true // If we can't read lock, assume we need a check
                    }
                };
                
                if !needs_check {
                    // Return cached result
                    if healthy.load(Ordering::Relaxed) {
                        return Ok(());
                    } else {
                        return Err(WorkerError::HealthCheckFailed {
                            url,
                            reason: "Worker is unhealthy (cached)".to_string(),
                        });
                    }
                }
                
                // Perform actual health check
                let client = reqwest::Client::new();
                let response = client.get(health_url).send().await?;
                
                // Update last check time
                if let Ok(mut lock) = last_health_check.write() {
                    *lock = Instant::now();
                }
                
                let is_healthy = response.status().is_success();
                healthy.store(is_healthy, Ordering::Relaxed);
                
                if !is_healthy {
                    return Err(WorkerError::HealthCheckFailed {
                        url,
                        reason: format!("Health check returned status: {}", response.status()),
                    });
                }
                
                Ok(())
            })
        }

        fn load(&self) -> std::sync::Arc<std::sync::atomic::AtomicUsize> {
            self.common.load.clone()
        }

        fn update_health(&self, healthy: bool) {
            self.common.healthy.store(healthy, std::sync::atomic::Ordering::Relaxed);
        }
    };
}

#[derive(Debug, Clone)]
pub struct RegularWorker {
    common: WorkerCommon,
}

impl RegularWorker {
    fn new(url: String, health_check_ttl: Duration) -> Self {
        Self {
            common: WorkerCommon::new(url, health_check_ttl),
        }
    }
}

impl fmt::Display for RegularWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegularWorker({})", self.common.url)
    }
}

impl Worker for RegularWorker {
    fn worker_type(&self) -> WorkerType {
        WorkerType::Regular
    }
    common_worker_methods!();
}

#[derive(Debug, Clone)]
pub struct DecodeWorker {
    common: WorkerCommon,
}

impl DecodeWorker {
    fn new(url: String, health_check_ttl: Duration) -> Self {
        Self {
            common: WorkerCommon::new(url, health_check_ttl),
        }
    }
}

impl fmt::Display for DecodeWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DecodeWorker({})", self.common.url)
    }
}

impl Worker for DecodeWorker {
    fn worker_type(&self) -> WorkerType {
        WorkerType::Decode
    }
    common_worker_methods!();
}

#[derive(Debug, Clone)]
pub struct PrefillWorker {
    common: WorkerCommon,
    bootstrap_port: Option<u16>
}

impl PrefillWorker {
    fn new(url: String, health_check_ttl: Duration, bootstrap_port: Option<u16>) -> Self {
        Self {
            common: WorkerCommon::new(url, health_check_ttl),
            bootstrap_port,
        }
    }
}

impl fmt::Display for PrefillWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.bootstrap_port {
            Some(port) => write!(f, "PrefillWorker({}, bootstrap_port={})", self.common.url, port),
            None => write!(f, "PrefillWorker({})", self.common.url),
        }
    }
}

impl Worker for PrefillWorker {
    fn worker_type(&self) -> WorkerType {
        WorkerType::Prefill(self.bootstrap_port)
    }
    common_worker_methods!();
}

pub struct WorkerFactory;

impl WorkerFactory {
    pub fn create_regular(url: String) -> Arc<dyn Worker> {
        Arc::new(RegularWorker::new(url, DEFAULT_HEALTH_CHECK_CACHE_TTL))
    }

    pub fn create_prefill(url: String, bootstrap_port: Option<u16>) -> Arc<dyn Worker> {
        Arc::new(PrefillWorker::new(url, DEFAULT_HEALTH_CHECK_CACHE_TTL, bootstrap_port))
    }

    pub fn create_decode(url: String) -> Arc<dyn Worker> {
        Arc::new(DecodeWorker::new(url, DEFAULT_HEALTH_CHECK_CACHE_TTL))
    }
}

pub mod worker_adapter {
    use super::*;

    pub fn from_regular_vec(urls: Vec<String>) -> Vec<Arc<dyn Worker>> {
        urls.iter().map(|url| WorkerFactory::create_regular(url.clone())).collect()
    }

    pub fn from_prefill_vec(urls: Vec<(String, Option<u16>)>) -> Vec<Arc<dyn Worker>> {
        urls.iter().map(|(url, bootstrap_port)| WorkerFactory::create_prefill(url.clone(), *bootstrap_port)).collect()
    }

    pub fn from_decode_vec(urls: Vec<String>) -> Vec<Arc<dyn Worker>> {
        urls.iter().map(|url| WorkerFactory::create_decode(url.clone())).collect()
    }
}

/// Wait for a vector of workers to become healthy
pub async fn wait_for_healthy_workers(
    workers: &[Arc<dyn Worker>],
    timeout_secs: u64,
    interval_secs: u64,
) -> Result<(), String> {
    let start_time = Instant::now();
    loop {
        if start_time.elapsed() > Duration::from_secs(timeout_secs) {
            let worker_info: Vec<String> = workers.iter().map(|w| w.to_string()).collect();
            error!(
                "Timeout {}s waiting for workers {:?} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                timeout_secs, worker_info
            );
            return Err(format!(
                "Timeout {}s waiting for workers {:?} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                timeout_secs, worker_info
            ));
        }

        // Run all health checks concurrently
        let health_futures: Vec<_> = workers.iter().map(|worker| worker.check_health()).collect();
        let health_results = workers.iter().zip(join_all(health_futures).await).collect::<Vec<_>>();

        let all_healthy = health_results.iter().all(|(_, result)| result.is_ok());
        let unhealthy_workers = health_results.into_iter().filter(|(_, result)| result.is_err()).map(|(worker, result)| (worker, result.unwrap_err())).collect::<Vec<_>>();

        if all_healthy {
            info!("All workers are healthy");
            return Ok(());
        } else {
            info!("Initializing workers:");
            for (worker, error) in unhealthy_workers.iter() {
                info!("  {} - {}", worker, error);
            }
            sleep(Duration::from_secs(interval_secs)).await;
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_regular_worker() {
        let worker = RegularWorker::new("http://localhost:8080".to_string(), DEFAULT_HEALTH_CHECK_CACHE_TTL);
        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), WorkerType::Regular);
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_prefill_worker() {
        let worker = DecodeWorker::new("http://localhost:8080".to_string(), DEFAULT_HEALTH_CHECK_CACHE_TTL);
        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), WorkerType::Decode);
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_decode_worker() {
        let worker = PrefillWorker::new("http://localhost:8080".to_string(), DEFAULT_HEALTH_CHECK_CACHE_TTL, Some(9000));
        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), WorkerType::Prefill(Some(9000)));
        assert!(worker.is_healthy());
    }
    
}