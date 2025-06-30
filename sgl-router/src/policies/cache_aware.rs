use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use async_trait::async_trait;

use crate::openai_api_types::{
    ChatCompletionRequest, CompletionRequest, GenerateRequest, GenerationRequest,
};
use crate::tree::Tree;
use crate::worker::Worker;

use super::{LoadBalancing, RoutingError, RoutingPolicy};

pub struct CacheAwarePolicy {
    tree: Arc<Mutex<Tree>>,
    running_queue: Arc<Mutex<HashMap<String, usize>>>,
    processed_queue: Arc<Mutex<HashMap<String, usize>>>,
    cache_threshold: f32,
    balance_abs_threshold: usize,
    balance_rel_threshold: f32,
    _eviction_thread: Option<thread::JoinHandle<()>>,
}

pub struct CacheAwarePolicyConfig {
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub eviction_interval_secs: u64,
    pub max_tree_size: usize,
}

fn extract_text_from_json(request: &serde_json::Value) -> Option<String> {
    if let Ok(req) = serde_json::from_value::<CompletionRequest>(request.clone()) {
        return Some(req.extract_text_for_routing());
    }
    if let Ok(req) = serde_json::from_value::<ChatCompletionRequest>(request.clone()) {
        return Some(req.extract_text_for_routing());
    }
    if let Ok(req) = serde_json::from_value::<GenerateRequest>(request.clone()) {
        return Some(req.extract_text_for_routing());
    }
    None
}

impl CacheAwarePolicy {
    pub fn new(config: CacheAwarePolicyConfig, workers: &[Arc<dyn Worker>]) -> Self {
        let running_queue = Arc::new(Mutex::new(
            workers
                .iter()
                .map(|w| (w.get_url().to_string(), 0))
                .collect(),
        ));
        let processed_queue = Arc::new(Mutex::new(
            workers
                .iter()
                .map(|w| (w.get_url().to_string(), 0))
                .collect(),
        ));
        let tree = Arc::new(Mutex::new(Tree::new()));

        let tree_clone = Arc::clone(&tree);
        let eviction_thread = thread::spawn(move || loop {
            thread::sleep(Duration::from_secs(config.eviction_interval_secs));
            let locked_tree_clone = tree_clone.lock().unwrap();
            locked_tree_clone.evict_tenant_by_size(config.max_tree_size);
        });

        Self {
            tree,
            running_queue,
            processed_queue,
            cache_threshold: config.cache_threshold,
            balance_abs_threshold: config.balance_abs_threshold,
            balance_rel_threshold: config.balance_rel_threshold,
            _eviction_thread: Some(eviction_thread),
        }
    }
}

#[async_trait]
impl RoutingPolicy for CacheAwarePolicy {
    async fn select_single(
        &self,
        workers: &[Arc<dyn Worker>],
        request: &serde_json::Value,
    ) -> Result<Arc<dyn Worker>, RoutingError> {
        let healthy_workers = self.get_healthy_workers(workers);
        if healthy_workers.is_empty() {
            return Err(RoutingError::NoHealthyWorkers);
        }

        let text = extract_text_from_json(request).ok_or(RoutingError::TextExtractionFailed)?;

        let selected_worker_url = {
            let tree = self.tree.lock().unwrap();
            let mut running_queue = self.running_queue.lock().unwrap();

            let healthy_urls: Vec<String> = healthy_workers
                .iter()
                .map(|w| w.get_url().to_string())
                .collect();

            let loads: Vec<usize> = healthy_urls
                .iter()
                .map(|url| *running_queue.get(url).unwrap_or(&0))
                .collect();
            let max_load = *loads.iter().max().unwrap_or(&0);
            let min_load = *loads.iter().min().unwrap_or(&0);

            let is_imbalanced = max_load.saturating_sub(min_load) > self.balance_abs_threshold
                && (max_load as f32) > (min_load as f32 * self.balance_rel_threshold);

            let selected_url = if is_imbalanced {
                healthy_urls
                    .iter()
                    .min_by_key(|url| *running_queue.get(*url).unwrap_or(&usize::MAX))
                    .unwrap()
                    .clone()
            } else {
                let (matched_text, matched_worker) = tree.prefix_match(&text);
                let matched_rate =
                    matched_text.chars().count() as f32 / text.chars().count() as f32;

                if matched_rate > self.cache_threshold && matched_worker != "" {
                    matched_worker
                } else {
                    tree.get_smallest_tenant()
                }
            };

            if !healthy_urls.contains(&selected_url) {
                // Fallback if the selected worker is not healthy
                healthy_urls[0].clone()
            } else {
                running_queue
                    .entry(selected_url.clone())
                    .and_modify(|c| *c += 1);
                self.processed_queue
                    .lock()
                    .unwrap()
                    .entry(selected_url.clone())
                    .and_modify(|c| *c += 1);
                tree.insert(&text, &selected_url);
                selected_url
            }
        };

        healthy_workers
            .iter()
            .find(|w| w.get_url() == selected_worker_url)
            .cloned()
            .ok_or(RoutingError::NoHealthyWorkers) // Should not happen
    }

    async fn select_pair(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request: &serde_json::Value,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), RoutingError> {
        // For cache-aware, we can be smarter and select the decode worker
        // based on the prefill worker selection to maximize affinity.
        // However, for now, we just call select_single twice as a baseline.
        let p = self.select_single(prefill_workers, request).await?;
        let d = self.select_single(decode_workers, request).await?;
        Ok((p, d))
    }

    fn on_request_complete(&self, worker_url: &str, _success: bool) {
        if let Ok(mut queue) = self.running_queue.lock() {
            if let Some(count) = queue.get_mut(worker_url) {
                *count = count.saturating_sub(1);
            }
        }
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    // fn add_worker(&self, worker: Arc<dyn Worker>) {
    //     let worker_url = worker.get_url().to_string();
    //     self.running_queue
    //         .lock()
    //         .unwrap()
    //         .insert(worker_url.clone(), 0);
    //     self.processed_queue.lock().unwrap().insert(worker_url, 0);
    // }

    // fn remove_worker(&self, worker_url: &str) {
    //     self.running_queue.lock().unwrap().remove(worker_url);
    //     self.processed_queue.lock().unwrap().remove(worker_url);
    //     self.tree.lock().unwrap().remove_tenant(worker_url);
    // }
}

impl LoadBalancing for CacheAwarePolicy {}
