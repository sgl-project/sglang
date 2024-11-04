// src/router.rs

use std::fmt::Debug;

/// Generic Router trait that can be implemented with different policies
pub trait Router: Send + Sync + Debug {
    /// Select a worker URL based on the implementation's policy
    /// Returns None if no worker is available
    fn select(&self) -> Option<String>;

    // get first worker
    fn get_first(&self) -> Option<String>;
}

// Round Robin Router
#[derive(Debug)]
pub struct RoundRobinRouter {
    worker_urls: Vec<String>,
    current_index: std::sync::atomic::AtomicUsize, // AtomicUsize is a thread-safe integer
}

impl RoundRobinRouter {
    pub fn new(worker_urls: Vec<String>) -> Self {
        Self {
            worker_urls,
            current_index: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl Router for RoundRobinRouter {
    fn select(&self) -> Option<String> {
        if self.worker_urls.is_empty() {
            return None;
        }
        // Use relaxed because operation order doesn't matter in round robin
        let index = self.current_index.fetch_add(1, std::sync::atomic::Ordering::Relaxed) 
            % self.worker_urls.len();
        Some(self.worker_urls[index].clone())
    }

    fn get_first(&self) -> Option<String> {
        if self.worker_urls.is_empty() {
            return None;
        }
        Some(self.worker_urls[0].clone())
    }
}

// Random Router
#[derive(Debug)]
pub struct RandomRouter {
    worker_urls: Vec<String>,
}

impl RandomRouter {
    pub fn new(worker_urls: Vec<String>) -> Self {
        Self { worker_urls }
    }
}

impl Router for RandomRouter {
    fn select(&self) -> Option<String> {
        use rand::seq::SliceRandom;
        
        if self.worker_urls.is_empty() {
            return None;
        }
        
        self.worker_urls.choose(&mut rand::thread_rng()).cloned()
    }

    fn get_first(&self) -> Option<String> {
        if self.worker_urls.is_empty() {
            return None;
        }
        Some(self.worker_urls[0].clone())
    }
}

// create a router based on routing policy
pub fn create_router(worker_urls: Vec<String>, policy: String) -> Box<dyn Router> {
    match policy.to_lowercase().as_str() {
        "random" => Box::new(RandomRouter::new(worker_urls)),
        "round_robin" => Box::new(RoundRobinRouter::new(worker_urls)),
        _ => panic!("Unknown routing policy: {}. The available policies are 'random' and 'round_robin'", policy),
    }
}