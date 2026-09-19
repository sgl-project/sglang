//! LMetric multiplication scheduling policy.
//!
//! LMetric scores each candidate as:
//!
//!     estimated_new_prefill_work(request, worker) * (worker_load + 1)
//!
//! This ports the core Blitz Router idea into the gateway using the existing
//! approximate prefix tree. The tree stores raw text, so the prefill estimate is
//! a character-count proxy until the gateway has tokenizer/KV-block visibility.

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use tracing::debug;

use super::{
    get_healthy_worker_indices, normalize_model_key, tree::Tree, utils::PeriodicTask,
    LMetricConfig, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::{Worker, WorkerType};

fn pool_tag(worker_type: &WorkerType) -> &'static str {
    match worker_type {
        WorkerType::Regular => "regular",
        WorkerType::Prefill { .. } => "prefill",
        WorkerType::Decode => "decode",
    }
}

fn make_tree_key(pool: &str, model: &str) -> String {
    format!("{}::{}", pool, model)
}

fn tree_key_for_worker(worker: &dyn Worker) -> String {
    make_tree_key(
        pool_tag(worker.worker_type()),
        normalize_model_key(worker.model_id()),
    )
}

/// LMetric routing policy.
///
/// Maintains one approximate prefix tree per `(worker pool, model)` and routes
/// to the healthy worker with the smallest multiplication score. Unlike
/// `cache_aware`, there is no threshold switch between cache affinity and load
/// balancing; both signals are combined directly by multiplication.
#[derive(Debug)]
pub struct LMetricPolicy {
    trees: Arc<DashMap<String, Arc<Tree>>>,
    _eviction_task: Option<PeriodicTask>,
}

impl LMetricPolicy {
    pub fn new() -> Self {
        Self::with_config(LMetricConfig::default())
    }

    pub fn with_config(config: LMetricConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        let eviction_task = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "LMetric eviction",
                move || {
                    for tree_ref in trees_clone.iter() {
                        let tree_key = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

                        debug!(
                            "LMetric cache eviction completed for {}, max_size: {}",
                            tree_key, max_tree_size
                        );
                    }
                },
            ))
        } else {
            None
        };

        Self {
            trees,
            _eviction_task: eviction_task,
        }
    }

    /// Initialize tree roots with worker URLs. This is idempotent.
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        let mut grouped: std::collections::HashMap<String, Vec<&Arc<dyn Worker>>> =
            std::collections::HashMap::new();
        for worker in workers {
            grouped
                .entry(tree_key_for_worker(worker.as_ref()))
                .or_default()
                .push(worker);
        }

        for (tree_key, pool_workers) in grouped {
            let tree = self.get_or_create_tree(&tree_key);
            for worker in pool_workers {
                tree.insert("", worker.url());
            }
        }
    }

    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = tree_key_for_worker(worker);
        let tree = self.get_or_create_tree(&tree_key);
        tree.insert("", worker.url());
    }

    pub fn remove_worker(&self, worker: &dyn Worker) {
        let tree_key = tree_key_for_worker(worker);
        if let Some(tree) = self.trees.get(&tree_key) {
            tree.remove_tenant(worker.url());
        }
    }

    pub fn remove_worker_by_url(&self, url: &str) {
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
    }

    pub fn evict_cache(&self, max_size: usize) {
        for tree_ref in self.trees.iter() {
            tree_ref.value().evict_tenant_by_size(max_size);
        }
    }

    fn get_or_create_tree(&self, tree_key: &str) -> Arc<Tree> {
        if let Some(tree) = self.trees.get(tree_key) {
            return tree.value().clone();
        }

        let tree = Arc::new(Tree::new());
        self.trees.insert(tree_key.to_string(), tree.clone());
        tree
    }

    fn seed_tree_with_workers(
        &self,
        tree: &Tree,
        workers: &[Arc<dyn Worker>],
        worker_indices: &[usize],
    ) {
        for &idx in worker_indices {
            tree.insert("", workers[idx].url());
        }
    }

    fn select_without_text(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        request_work: usize,
    ) -> Option<usize> {
        healthy_indices.iter().copied().min_by_key(|&idx| {
            let load = workers[idx].load();
            let score = request_work.saturating_mul(load.saturating_add(1));
            (score, load, idx)
        })
    }

    fn select_with_text(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        tree: &Tree,
        text: &str,
    ) -> Option<usize> {
        let input_work = text.chars().count();

        healthy_indices.iter().copied().min_by_key(|&idx| {
            let worker = &workers[idx];
            let matched_work = if input_work == 0 {
                0
            } else {
                tree.prefix_match_tenant_char_count(text, worker.url())
            };
            let new_prefill_work = input_work.saturating_sub(matched_work);
            let load = worker.load();
            let score = new_prefill_work.saturating_mul(load.saturating_add(1));

            (score, load, new_prefill_work, idx)
        })
    }
}

#[async_trait]
impl LoadBalancingPolicy for LMetricPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let all_healthy_indices = get_healthy_worker_indices(workers);
        if all_healthy_indices.is_empty() {
            return None;
        }

        let pivot = workers[all_healthy_indices[0]].as_ref();
        let tree_key = tree_key_for_worker(pivot);
        let healthy_indices: Vec<usize> = all_healthy_indices
            .into_iter()
            .filter(|&idx| tree_key_for_worker(workers[idx].as_ref()) == tree_key)
            .collect();
        if healthy_indices.is_empty() {
            return None;
        }

        let tree = self.get_or_create_tree(&tree_key);
        self.seed_tree_with_workers(&tree, workers, &healthy_indices);

        let selected_idx = if let Some(text) = info.request_text {
            self.select_with_text(workers, &healthy_indices, &tree, text)
        } else {
            let request_work = info.tokens.map_or(0, |tokens| tokens.len());
            self.select_without_text(workers, &healthy_indices, request_work)
        }?;

        if let Some(text) = info.request_text {
            tree.insert(text, workers[selected_idx].url());
        }

        workers[selected_idx].increment_processed();
        Some(selected_idx)
    }

    fn name(&self) -> &'static str {
        "lmetric"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.trees.clear();
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for LMetricPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn test_policy() -> LMetricPolicy {
        LMetricPolicy::with_config(LMetricConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        })
    }

    fn test_workers() -> Vec<Arc<dyn Worker>> {
        vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ]
    }

    #[tokio::test]
    async fn test_lmetric_reuses_cache_when_load_is_low() {
        let policy = test_policy();
        let workers = test_workers();
        policy.init_workers(&workers);

        let first = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("shared prefix"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let second = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("shared prefix with suffix"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(second, first);
    }

    #[tokio::test]
    async fn test_lmetric_multiplies_cache_work_by_load() {
        let policy = test_policy();
        let workers = test_workers();
        policy.init_workers(&workers);

        let cached_idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("abcdef"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let other_idx = 1 - cached_idx;

        for _ in 0..20 {
            workers[cached_idx].increment_load();
        }

        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("abcdefg"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(selected, other_idx);
    }

    #[tokio::test]
    async fn test_lmetric_without_text_uses_load() {
        let policy = test_policy();
        let workers = test_workers();
        workers[0].increment_load();
        workers[0].increment_load();
        let tokens = [1, 2, 3];

        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(selected, 1);
    }
}
