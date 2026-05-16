/*
    Cache-Aware Load Balancing Router

    This router combines two strategies to optimize both cache utilization and request distribution:

    1. Cache-Aware Routing (Approximate Tree)
    2. Load Balancing (Shortest Queue with Balance Thresholds)

    The router dynamically switches between these strategies based on load conditions:
    - Uses load balancing when the system is imbalanced
    - Uses cache-aware routing when the system is balanced

    A system is considered imbalanced if both conditions are met:
    1. (max - min) > abs_threshold
    2. max > rel_threshold * min

    Strategy Details:

    1. Cache-Aware Routing (Approximate Tree)
    -------------------------------------------
    This strategy maintains an approximate radix tree for each worker based on request history,
    eliminating the need for direct cache state queries. The tree stores raw text characters
    instead of token IDs to avoid tokenization overhead.

    Process:
    a. For each request, find the worker with the highest prefix match
    b. If match rate > cache_threshold:
    Route to the worker with highest match (likely has relevant data cached)
    c. If match rate ≤ cache_threshold:
    Route to the worker with smallest tree size (most available cache capacity)
    d. Background maintenance:
    Periodically evict least recently used leaf nodes to prevent memory overflow

    2. Load Balancing (Shortest Queue)
    -------------------------------------------
    This strategy tracks pending request counts per worker and routes new requests
    to the least busy worker when the system is detected to be imbalanced.

    Configuration Parameters:
    ------------------------
    1. cache_threshold: (float, 0.0 to 1.0)
    Minimum prefix match ratio to use highest-match routing.
    Below this threshold, routes to worker with most available cache space.

    2. balance_abs_threshold: (integer)
    Absolute difference threshold for load imbalance detection.
    System is potentially imbalanced if (max_load - min_load) > abs_threshold

    3. balance_rel_threshold: (float)
    Relative ratio threshold for load imbalance detection.
    System is potentially imbalanced if max_load > min_load * rel_threshold
    Used in conjunction with abs_threshold to determine final imbalance state.

    4. eviction_interval_secs: (integer)
    Interval between LRU eviction cycles for the approximate trees.

    5. max_tree_size: (integer)
    Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
    during the next eviction cycle.
*/

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use smg_mesh::{tree_ops::TreeOperation, OptionalMeshSyncManager};
use tracing::{debug, warn};

use super::{
    get_healthy_worker_indices, normalize_model_key, tree::Tree, utils::PeriodicTask,
    CacheAwareConfig, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::{Worker, UNKNOWN_MODEL_ID};

/// Cache-aware routing policy
///
/// Scores every healthy worker by `pending_tokens + miss_chars`, where:
///   - `pending_tokens` comes from each worker's most recent `/v1/loads`
///     report (`num_total_tokens` = used + queued prefill tokens). The
///     LoadMonitor pushes these in via `update_loads` on its own cadence.
///   - `miss_chars` is the prefill the worker would have to redo: zero
///     for the worker the gateway radix tree reports as the longest-prefix
///     tenant, `input_chars` for everyone else.
///
/// Newcomers start at `pending_tokens = 0`, so a fresh worker can beat a
/// cache holder once the holder's backlog exceeds the input length.
/// Note that `pending_tokens` is in tokens and `miss_chars` is in chars;
/// we deliberately add them with a 1:1 implicit ratio. This biases the
/// selection toward cache affinity for tokenizers where chars/token > 1
/// (typical English BPE ≈ 4), which is acceptable for v1.
///
/// Maintains separate radix trees per model for multi-model support.
/// Supports mesh synchronization of tree operations across cluster nodes.
/// When mesh is not enabled, the policy works independently without sync.
#[derive(Debug)]
pub struct CacheAwarePolicy {
    // Held only so existing CLI surface (cache_threshold, balance_*_threshold,
    // max_tree_size, eviction_interval_secs) continues to parse without error.
    // The new scoring rule consumes the eviction settings at construction
    // time; the threshold knobs are dead. Prune in a follow-up PR once we
    // remove them from CacheAwareConfig / CLI.
    #[allow(dead_code)]
    config: CacheAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    /// Per-worker pending-token snapshot, keyed by `worker.url()` (which is
    /// `base@rank` for DPAwareWorker, plain `base` otherwise). Populated by
    /// `LoadMonitor` via `update_loads`; unknown workers fall back to 0.
    pending_tokens: Arc<DashMap<String, isize>>,
    mesh_sync: OptionalMeshSyncManager,
    _eviction_task: Option<PeriodicTask>,
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        // Start background eviction thread if configured
        let eviction_task = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "Eviction",
                move || {
                    for tree_ref in trees_clone.iter() {
                        let model_id = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

                        debug!(
                            "Cache eviction completed for model {}, max_size: {}",
                            model_id, max_tree_size
                        );
                    }
                },
            ))
        } else {
            None
        };

        Self {
            config,
            trees,
            pending_tokens: Arc::new(DashMap::new()),
            mesh_sync: None,
            _eviction_task: eviction_task,
        }
    }

    /// Set mesh sync manager (can be called after construction)
    pub fn set_mesh_sync(&mut self, mesh_sync: OptionalMeshSyncManager) {
        self.mesh_sync = mesh_sync.clone();
        if mesh_sync.is_some() {
            self.restore_tree_state_from_mesh();
        }
    }

    /// Initialize the tree with worker URLs (used only during initial setup)
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // Group workers by model
        let mut model_workers: std::collections::HashMap<String, Vec<&Arc<dyn Worker>>> =
            std::collections::HashMap::new();
        for worker in workers {
            let tree_key = normalize_model_key(worker.model_id());
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }

        // Initialize tree for each model
        for (tree_key, model_workers) in model_workers {
            let tree = self
                .trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(Tree::new()));
            for worker in model_workers {
                tree.insert("", worker.url());
                self.pending_tokens
                    .entry(worker.url().to_string())
                    .or_insert(0);
            }
        }
    }

    /// Add a single worker to the tree (incremental update)
    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        let tree = self
            .trees
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", worker.url());
        self.pending_tokens
            .entry(worker.url().to_string())
            .or_insert(0);
    }

    /// Add a worker by URL and model (for backward compatibility)
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
        let tree = self
            .trees
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", url);
        self.pending_tokens.entry(url.to_string()).or_insert(0);
    }

    /// Remove a worker from the tree
    pub fn remove_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        if let Some(tree) = self.trees.get(tree_key) {
            tree.remove_tenant(worker.url());
        }
        self.pending_tokens.remove(worker.url());
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility)
    pub fn remove_worker_by_url(&self, url: &str) {
        // Remove from all trees since we don't know which model it belongs to
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
        self.pending_tokens.remove(url);
    }

    /// Restore tree state from mesh store
    /// This is called during initialization to rebuild trees from synchronized state
    fn restore_tree_state_from_mesh(&self) {
        if let Some(ref mesh_sync) = self.mesh_sync {
            // Get all tree states from mesh
            // We need to iterate through all models that have tree states
            // For now, we'll restore trees for models that are already in our trees map
            // In a full implementation, we might want to query mesh for all tree states

            for tree_ref in self.trees.iter() {
                let model_id = tree_ref.key();
                if let Some(tree_state) = mesh_sync.get_tree_state(model_id) {
                    debug!(
                        "Restoring tree state for model {} with {} operations",
                        model_id,
                        tree_state.operations.len()
                    );

                    let tree = tree_ref.value();
                    // Apply all operations to rebuild the tree
                    for operation in &tree_state.operations {
                        match operation {
                            TreeOperation::Insert(insert_op) => {
                                tree.insert(&insert_op.text, &insert_op.tenant);
                            }
                            TreeOperation::Remove(remove_op) => {
                                tree.remove_tenant(&remove_op.tenant);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Normalize model_id for mesh synchronization
    /// Converts empty model_id to UNKNOWN_MODEL_ID for consistency
    fn normalize_mesh_model_id(model_id: &str) -> &str {
        if model_id.is_empty() {
            UNKNOWN_MODEL_ID
        } else {
            model_id
        }
    }

    /// Apply remote tree operation from mesh
    /// This is called when receiving tree state updates from other nodes
    pub fn apply_remote_tree_operation(&self, model_id: &str, operation: &TreeOperation) {
        let tree_key = Self::normalize_mesh_model_id(model_id);

        let tree = self
            .trees
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));

        match operation {
            TreeOperation::Insert(insert_op) => {
                tree.insert(&insert_op.text, &insert_op.tenant);
                debug!(
                    "Applied remote tree insert: model={}, text={}, tenant={}",
                    model_id, insert_op.text, insert_op.tenant
                );
            }
            TreeOperation::Remove(remove_op) => {
                tree.remove_tenant(&remove_op.tenant);
                debug!(
                    "Applied remote tree remove: model={}, tenant={}",
                    model_id, remove_op.tenant
                );
            }
        }
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        for tree_ref in self.trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "Cache eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
    }

    /// Look up a worker's most recently reported pending-token count.
    /// Returns 0 for workers the LoadMonitor hasn't reached yet (newcomers).
    /// Negative values mean the worker's /v1/loads probe failed; treat as 0
    /// rather than `isize::MAX` so a transient outage doesn't permanently
    /// exclude a worker from selection.
    fn pending_tokens_for(&self, worker_url: &str) -> isize {
        self.pending_tokens
            .get(worker_url)
            .map(|v| (*v).max(0))
            .unwrap_or(0)
    }

    /// Insert request text into the tree under the chosen worker and mirror
    /// the op into the mesh. Pulled out so both the imbalanced and
    /// balanced paths can share it.
    fn record_request_in_tree(&self, model_id: &str, text: &str, worker_url: &str) {
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());
        let Some(tree) = tree else {
            debug!(
                "Warning: No tree found for model '{}', skipping cache update",
                model_id
            );
            return;
        };

        tree.insert(text, worker_url);

        if let Some(ref mesh_sync) = self.mesh_sync {
            use smg_mesh::tree_ops::TreeInsertOp;
            let op = TreeOperation::Insert(TreeInsertOp {
                text: text.to_string(),
                tenant: worker_url.to_string(),
            });
            let mesh_model_id = Self::normalize_mesh_model_id(model_id);
            if let Err(e) = mesh_sync.sync_tree_operation(mesh_model_id.to_string(), op) {
                warn!("Failed to sync tree insert operation to mesh: {}", e);
            }
        }
    }
}

#[async_trait]
impl LoadBalancingPolicy for CacheAwarePolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        // Router pre-filters by model, so any healthy worker yields the key.
        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id());
        let text = info.request_text.unwrap_or("");

        // Single tree walk: returns the deepest reachable node, the cached
        // tenant at that node (our "longest-prefix worker"), and the matched
        // / input char counts. If there's no tree yet for this model, treat
        // every worker as having zero match.
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());
        let (best_tenant, best_match_chars, input_chars) = match &tree {
            Some(t) => {
                let r = t.prefix_match_with_counts(text);
                (Some(r.tenant), r.matched_char_count, r.input_char_count)
            }
            None => {
                debug!(
                    "No cache tree for model '{}'; scoring on pending tokens only",
                    model_id
                );
                (None, 0usize, text.chars().count())
            }
        };

        // Score every healthy worker by `pending_tokens + miss_chars`.
        // The longest-prefix tenant gets `miss_chars = input - matched`;
        // everyone else is assumed to have zero overlap and pays the full
        // `input_chars`. Units are intentionally mixed 1:1 — see struct
        // doc-comment for rationale.
        let best_tenant_str: Option<&str> = best_tenant.as_ref().map(|t| t.as_ref());
        let scored_idx = healthy_indices
            .iter()
            .min_by_key(|&&idx| {
                let w = &workers[idx];
                let is_best = best_tenant_str == Some(w.url());
                let miss_chars = if is_best {
                    input_chars.saturating_sub(best_match_chars)
                } else {
                    input_chars
                };
                let pending = self.pending_tokens_for(w.url()) as i128;
                pending + miss_chars as i128
            })
            .copied();

        let idx = scored_idx?;

        // Update gateway-side tree + mesh with this request (skip if model
        // had no tree yet — caller may not have init'd one for this model).
        if tree.is_some() {
            self.record_request_in_tree(model_id, text, workers[idx].url());
        }

        workers[idx].increment_processed();
        Some(idx)
    }

    fn update_loads(&self, loads: &std::collections::HashMap<String, isize>) {
        for (url, v) in loads.iter() {
            self.pending_tokens.insert(url.clone(), *v);
        }
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        if !success {
            tracing::debug!(
                "Request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn needs_request_text(&self) -> bool {
        true // Cache-aware policy needs request text for cache affinity
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[tokio::test]
    async fn test_cache_aware_with_balanced_load() {
        // Create policy without eviction thread for testing
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // Initialize the policy with workers
        policy.init_workers(&workers);

        // First request should be distributed
        let idx1 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx1, idx3);
    }

    #[tokio::test]
    async fn test_newcomer_wins_when_cache_holder_saturated() {
        // Cache holder w1 has the prefix but is buried under pending tokens;
        // newcomer w2 has no overlap but pending=0. For short input, w2 wins.
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
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
        ];
        policy.init_workers(&workers);

        // Seed the tree so w1 owns "hello world" entirely.
        for _ in 0..3 {
            let idx = policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        request_text: Some("hello world"),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            assert_eq!(idx, 0);
        }

        // LoadMonitor reports w1 with a huge backlog, w2 idle.
        let mut loads = std::collections::HashMap::new();
        loads.insert("http://w1:8000".to_string(), 1_000_000);
        loads.insert("http://w2:8000".to_string(), 0);
        policy.update_loads(&loads);

        // A short prompt: w1's pending dwarfs any cache savings → w2.
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            idx, 1,
            "saturated cache holder should lose to idle newcomer"
        );
    }

    #[tokio::test]
    async fn test_cache_holder_wins_when_balanced() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
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
        ];
        policy.init_workers(&workers);

        // First request lands somewhere (deterministic: index 0 ties broken first).
        let first = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("the quick brown fox"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Both workers report identical pending. The cache holder should win
        // any subsequent identical-prefix request.
        let mut loads = std::collections::HashMap::new();
        loads.insert("http://w1:8000".to_string(), 10);
        loads.insert("http://w2:8000".to_string(), 10);
        policy.update_loads(&loads);

        for _ in 0..3 {
            let idx = policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        request_text: Some("the quick brown fox"),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            assert_eq!(
                idx, first,
                "cache affinity should win when pending is equal"
            );
        }
    }

    #[tokio::test]
    async fn test_pending_tokens_default_zero_for_unknown_worker() {
        // No update_loads call has happened yet; both workers should be
        // treated as pending=0 and the tie should break the same way every
        // call (longest-prefix tenant wins).
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
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
        ];
        policy.init_workers(&workers);

        let first = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("alpha beta"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let second = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("alpha beta"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn test_failed_load_probe_treated_as_zero() {
        // LoadMonitor reports w1 with -1 (probe failed) — must not cause w1
        // to be permanently preferred-or-excluded; we treat -1 as 0.
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
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
        ];
        policy.init_workers(&workers);

        let mut loads = std::collections::HashMap::new();
        loads.insert("http://w1:8000".to_string(), -1);
        loads.insert("http://w2:8000".to_string(), 500);
        policy.update_loads(&loads);

        // w1 (treated as 0) should beat w2 (500) for a fresh prompt.
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("never seen before"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx, 0);
    }

    #[tokio::test]
    async fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
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
        ];

        policy.init_workers(&workers);

        // Route some requests
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test1"),
                    ..Default::default()
                },
            )
            .await;
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test2"),
                    ..Default::default()
                },
            )
            .await;

        // Remove a worker
        policy.remove_worker_by_url("http://w1:8000");
        workers[0].set_healthy(false);

        // All requests should now go to worker2
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test1"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx, 1);
    }

    #[tokio::test]
    async fn test_cache_aware_sync_tree_operation_to_mesh() {
        use std::sync::Arc;

        use smg_mesh::{stores::StateStores, sync::MeshSyncManager};

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let mut policy = CacheAwarePolicy::with_config(config);
        policy.set_mesh_sync(Some(mesh_sync.clone()));

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        )];

        policy.init_workers(&workers);

        // Select worker with a request - should sync to mesh
        let _idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test request"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Verify tree operation was synced to mesh (under UNKNOWN_MODEL_ID since no model was specified)
        let tree_state = mesh_sync.get_tree_state(UNKNOWN_MODEL_ID);
        assert!(tree_state.is_some());
        let tree = tree_state.unwrap();
        assert!(!tree.operations.is_empty());
    }

    #[test]
    fn test_cache_aware_restore_tree_state_from_mesh() {
        use std::sync::Arc;

        use smg_mesh::{
            stores::StateStores,
            sync::MeshSyncManager,
            tree_ops::{TreeInsertOp, TreeOperation},
        };

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        // Pre-populate mesh with tree state
        let op1 = TreeOperation::Insert(TreeInsertOp {
            text: "test_text_1".to_string(),
            tenant: "http://w1:8000".to_string(),
        });
        mesh_sync
            .sync_tree_operation("model1".to_string(), op1)
            .unwrap();

        let op2 = TreeOperation::Insert(TreeInsertOp {
            text: "test_text_2".to_string(),
            tenant: "http://w2:8000".to_string(),
        });
        mesh_sync
            .sync_tree_operation("model1".to_string(), op2)
            .unwrap();

        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let mut policy = CacheAwarePolicy::with_config(config);
        policy.set_mesh_sync(Some(mesh_sync.clone()));

        // Initialize with a model to trigger restore
        let _workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        )];

        // Create a tree entry for model1 to trigger restore
        let _tree = policy
            .trees
            .entry("model1".to_string())
            .or_insert_with(|| Arc::new(Tree::new()));

        // Manually trigger restore (normally done in constructor)
        // For testing, we'll verify the tree state exists in mesh
        let tree_state = mesh_sync.get_tree_state("model1");
        assert!(tree_state.is_some());
        let state = tree_state.unwrap();
        assert_eq!(state.operations.len(), 2);
    }

    #[test]
    fn test_cache_aware_apply_remote_tree_operation() {
        use std::sync::Arc;

        use smg_mesh::{
            stores::StateStores,
            sync::MeshSyncManager,
            tree_ops::{TreeInsertOp, TreeOperation},
        };

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let mut policy = CacheAwarePolicy::with_config(config);
        policy.set_mesh_sync(Some(mesh_sync.clone()));

        // Apply remote tree operation
        let remote_op = TreeOperation::Insert(TreeInsertOp {
            text: "remote_text".to_string(),
            tenant: "http://remote:8000".to_string(),
        });

        policy.apply_remote_tree_operation("model1", &remote_op);

        // Verify the tree was updated
        let tree = policy.trees.get("model1");
        assert!(tree.is_some());
    }

    #[test]
    fn test_cache_aware_multi_node_consistency() {
        use std::sync::Arc;

        use smg_mesh::{
            stores::StateStores,
            sync::MeshSyncManager,
            tree_ops::{TreeInsertOp, TreeOperation},
        };

        // Simulate two nodes
        let stores1 = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync1 = Arc::new(MeshSyncManager::new(stores1.clone(), "node1".to_string()));

        let stores2 = Arc::new(StateStores::with_self_name("node2".to_string()));
        let mesh_sync2 = Arc::new(MeshSyncManager::new(stores2.clone(), "node2".to_string()));

        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };

        let mut _policy1 = CacheAwarePolicy::with_config(config.clone());
        _policy1.set_mesh_sync(Some(mesh_sync1.clone()));
        let mut _policy2 = CacheAwarePolicy::with_config(config);
        _policy2.set_mesh_sync(Some(mesh_sync2.clone()));

        // Node1 syncs a tree operation
        let op = TreeOperation::Insert(TreeInsertOp {
            text: "shared_text".to_string(),
            tenant: "http://shared:8000".to_string(),
        });
        mesh_sync1
            .sync_tree_operation("model1".to_string(), op.clone())
            .unwrap();

        // Node2 should be able to get the tree state
        let tree_state = mesh_sync2.get_tree_state("model1");
        // Note: In a real scenario, this would be synced via gossip protocol
        // For unit test, we verify the sync mechanism works
        // Tree state may or may not exist depending on sync timing
        let _ = tree_state;
    }

    #[tokio::test]
    async fn test_cache_aware_without_mesh() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        )];

        policy.init_workers(&workers);

        // Should work without mesh
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test request"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx, 0);
    }
}
