use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use async_trait::async_trait;
use crate::schedulers::{SchedulerPolicy, SelectRouterInfo};
use crate::routers::router_manager::{router_ids, RouterId};
use crate::core::WorkerRegistry;

/// Configuration for the ProportionScheduler.
#[derive(Debug, Clone)]
pub struct ProportionSchedulerConfig {
    /// The interval at which the crossover point is recalculated.
    pub adjust_interval: Duration,
    /// The time window of recent requests to consider for adjustment.
    pub adjust_window: Duration,
    /// Absolute difference in normalized load to trigger imbalance override.
    pub balance_abs_threshold: usize,
    /// Relative difference in normalized load to trigger imbalance override.
    pub balance_rel_threshold: f32,
    /// The performance weight of a single Regular worker relative to a PD pair (e.g., 0.4).
    pub regular_worker_weight: f32,
}

// impl Default for ProportionSchedulerConfig {
//     fn default() -> Self {
//         info!("默认方法");
//         Self {
//             adjust_interval: Duration::from_secs(1),
//             adjust_window: Duration::from_secs(2),
//             balance_abs_threshold: 1,
//             balance_rel_threshold: 1.001,
//             regular_worker_weight: 0.4,
//         }
//     }
// }

/// Represents a choice between the two primary resource routers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterChoice {
    Regular,
    Pd,
}

/// A record of a completed request for historical analysis.
#[derive(Debug, Clone)]
struct RequestRecord {
    token_count: usize,
    timestamp: Instant,
    router_id: RouterId,

}

/// A scheduler that routes requests between Regular and PD routers based on
/// proportional load distribution.
#[derive(Debug)]
pub struct ProportionScheduler {
    crossover_point: Arc<RwLock<usize>>,
    global_request_queue: Arc<RwLock<VecDeque<RequestRecord>>>,
    router_loads: Arc<RwLock<std::collections::HashMap<RouterId, usize>>>,
    worker_counts: Arc<RwLock<std::collections::HashMap<RouterId, usize>>>,
    config: ProportionSchedulerConfig,
    _adjustment_handle: tokio::task::JoinHandle<()>,
}

impl ProportionScheduler {
    pub fn new(config: ProportionSchedulerConfig, worker_registry: Arc<WorkerRegistry>) -> Self {
        info!(
            crossover_point = 512,
            config = ?config,
            "ProportionScheduler initialized."
        );
        let crossover_point = Arc::new(RwLock::new(512));
        let global_request_queue = Arc::new(RwLock::new(VecDeque::new()));
        let router_loads = Arc::new(RwLock::new(std::collections::HashMap::new()));
        let worker_counts = Arc::new(RwLock::new(std::collections::HashMap::new()));

        let adjustment_handle = Self::start_adjustment_task(
            Arc::clone(&crossover_point),
            Arc::clone(&global_request_queue),
            Arc::clone(&router_loads),
            Arc::clone(&worker_counts),
            Arc::clone(&worker_registry),
            config.clone(),
        );

        Self {
            crossover_point,
            global_request_queue,
            router_loads,
            worker_counts,
            config,
            _adjustment_handle: adjustment_handle,
        }
    }


    fn start_adjustment_task(
        crossover_point: Arc<RwLock<usize>>,
        global_request_queue: Arc<RwLock<VecDeque<RequestRecord>>>,
        router_loads: Arc<RwLock<std::collections::HashMap<RouterId, usize>>>,
        worker_counts: Arc<RwLock<std::collections::HashMap<RouterId, usize>>>,
        worker_registry: Arc<WorkerRegistry>,
        config: ProportionSchedulerConfig,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                debug!("Starting periodic scheduler adjustment task");
                tokio::time::sleep(config.adjust_interval).await;
                let (regular_workers, pd_workers) = worker_registry.get_scheduler_worker_counts();
                {
                    let mut counts_guard = worker_counts.write().unwrap();
                    counts_guard.insert(router_ids::HTTP_REGULAR, regular_workers);
                    counts_guard.insert(router_ids::HTTP_PD, pd_workers);
                }
                {
                    let mut queue_guard = global_request_queue.write().unwrap();
                    let mut loads_guard = router_loads.write().unwrap();
                    let now = Instant::now();
                    while let Some(req) = queue_guard.front() {
                        if now.duration_since(req.timestamp) > config.adjust_window {
                            if let Some(oldest_req) = queue_guard.pop_front() {
                                if let Some(load) = loads_guard.get_mut(&oldest_req.router_id) {
                                    *load = load.saturating_sub(oldest_req.token_count);
                                }
                            }
                        } else {
                            break;
                        }
                    }
                }

                let counts_guard = worker_counts.read().unwrap();
                let queue_guard = global_request_queue.read().unwrap();

                let pd_workers = *counts_guard.get(&router_ids::HTTP_PD).unwrap_or(&0) + *counts_guard.get(&router_ids::GRPC_PD).unwrap_or(&0);
                let regular_workers = *counts_guard.get(&router_ids::HTTP_REGULAR).unwrap_or(&0) + *counts_guard.get(&router_ids::GRPC_REGULAR).unwrap_or(&0);

                if queue_guard.is_empty() || pd_workers == 0 || regular_workers == 0 {
                    debug!(
                        is_queue_empty = queue_guard.is_empty(),
                        pd_workers,
                        regular_workers,
                        "Skipping adjustment due to empty queue or no workers."
                    );
                    continue;
                }

                let total_pd_weight = pd_workers as f32;
                let total_regular_weight = regular_workers as f32 * config.regular_worker_weight;
                let total_weight = total_pd_weight + total_regular_weight;

                if total_weight == 0.0 {
                    debug!("Skipping adjustment due to zero total worker weight.");
                    continue;
                }

                let ideal_regular_load_share = total_regular_weight / total_weight;

                let mut sorted_token_counts: Vec<_> = queue_guard.iter().map(|r| r.token_count).collect();
                sorted_token_counts.sort_unstable();

                let total_global_load: usize = sorted_token_counts.iter().sum();
                if total_global_load == 0 {
                    debug!("Skipping adjustment due to zero total load in queue.");
                    continue;
                }

                let mut accumulated_load: usize = 0;
                let mut new_crossover_point = *crossover_point.read().unwrap();

                for &token_count in &sorted_token_counts {
                    accumulated_load += token_count;
                    if (accumulated_load as f32 / total_global_load as f32) >= ideal_regular_load_share {
                        new_crossover_point = token_count;
                        break;
                    }
                }

                *crossover_point.write().unwrap() = new_crossover_point;
                info!(
                    new_crossover_point,
                    regular_workers,
                    pd_workers,
                    queue_size = queue_guard.len(),
                    "Crossover point adjusted."
                );
            }
        })
    }
}

#[async_trait]
impl SchedulerPolicy for ProportionScheduler {
    fn name(&self) -> &'static str {
        "proportion"
    }

    // `needs_request_text` is false, but we need tokens. This should be `needs_tokens`.
    // Let's assume we can get token_count from `info.tokens`.
    fn needs_request_text(&self) -> bool {
        true // Or we can change the trait to have `needs_tokens`
    }

    async fn select_router(
        &self,
        candidate_routers: &[RouterId],
        info: &SelectRouterInfo<'_>,
    ) -> Option<RouterId> {
        let token_count = info.tokens.map_or(0, |t| t.len());
        let choice: Option<RouterId> = {
            let loads_guard = self.router_loads.read().unwrap();
            let counts_guard = self.worker_counts.read().unwrap();

            let pd_load = *loads_guard.get(&router_ids::HTTP_PD).unwrap_or(&0);
            let regular_load = *loads_guard.get(&router_ids::HTTP_REGULAR).unwrap_or(&0);

            let regular_workers = *counts_guard.get(&router_ids::HTTP_REGULAR).unwrap_or(&0);
            let pd_workers = *counts_guard.get(&router_ids::HTTP_PD).unwrap_or(&0);

            if regular_workers == 0 && pd_workers == 0 {
                warn!("No regular or PD workers available to select from.");
                return None;
            }

            let total_pd_weight = pd_workers as f32;
            let total_regular_weight = regular_workers as f32 * self.config.regular_worker_weight;

            let norm_pd_load = if total_pd_weight > 0.0 { pd_load as f32 / total_pd_weight } else { f32::MAX };
            let norm_regular_load = if total_regular_weight > 0.0 { regular_load as f32 / total_regular_weight } else { f32::MAX };

            debug!(
                request_tokens = token_count,
                load.regular = regular_load,
                load.pd = pd_load,
                load.normalized_regular = norm_regular_load,
                load.normalized_pd = norm_pd_load,
                workers.regular = regular_workers,
                workers.pd = pd_workers,
                weight.regular = total_regular_weight,
                weight.pd = total_pd_weight,
                all_loads = ?*loads_guard,
                all_counts = ?*counts_guard,
                "Evaluating router selection logic."
            );


            let select_best_available = |choices: &[RouterId]| -> Option<RouterId> {
                choices.iter().find(|id| candidate_routers.contains(id)).cloned()
            };

            let imbalanced_choice = if (norm_pd_load - norm_regular_load).abs() > self.config.balance_abs_threshold as f32
                && (norm_pd_load > norm_regular_load * self.config.balance_rel_threshold || norm_regular_load > norm_pd_load * self.config.balance_rel_threshold)
            {
                debug!(
                    norm_regular_load,
                    norm_pd_load,
                    threshold.abs = self.config.balance_abs_threshold,
                    threshold.rel = self.config.balance_rel_threshold,
                    "Imbalance detected, overriding crossover point logic."
                );
                if norm_regular_load < norm_pd_load {
                    select_best_available(&[router_ids::HTTP_REGULAR, router_ids::GRPC_REGULAR])
                } else {
                    select_best_available(&[router_ids::HTTP_PD, router_ids::GRPC_PD])
                }
            } else {
                None
            };

            if let Some(id) = imbalanced_choice {
                Some(id)
            } else {
                let crossover = *self.crossover_point.read().unwrap();
                if token_count <= crossover {
                    select_best_available(&[router_ids::HTTP_REGULAR, router_ids::GRPC_REGULAR])
                        .or_else(|| select_best_available(&[router_ids::HTTP_PD, router_ids::GRPC_PD]))
                } else {
                    select_best_available(&[router_ids::HTTP_PD, router_ids::GRPC_PD])
                        .or_else(|| select_best_available(&[router_ids::HTTP_REGULAR, router_ids::GRPC_REGULAR]))
                }
            }

        };
        if let Some(ref chosen_id) = choice {
            self.global_request_queue.write().unwrap().push_back(RequestRecord {
                token_count,
                timestamp: Instant::now(),
                router_id: chosen_id.clone(),
            });
            debug!(
                request.tokens = token_count,
                router.id = ?chosen_id,
                "Request routed."
            );

            let mut loads_guard = self.router_loads.write().unwrap();
            *loads_guard.entry(chosen_id.clone()).or_insert(0) += token_count;
            info!("Updated load for {}: {}", chosen_id.as_str(), loads_guard.get(chosen_id).unwrap());
        } else{
            warn!(
                request.tokens = token_count,
                candidate_routers = ?candidate_routers,
                "Failed to select any router for the request."
            );
        }

        choice
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerRegistry, WorkerType};

    /// Helper: create a config with a long adjust_interval so the background task
    /// doesn't interfere with synchronous assertions.
    fn test_config() -> ProportionSchedulerConfig {
        ProportionSchedulerConfig {
            adjust_interval: Duration::from_secs(3600),
            adjust_window: Duration::from_secs(3600),
            balance_abs_threshold: 100,
            balance_rel_threshold: 1.5,
            regular_worker_weight: 0.4,
        }
    }

    /// Helper: create a WorkerRegistry with one Regular and one Prefill worker.
    fn registry_with_regular_and_prefill() -> Arc<WorkerRegistry> {
        let registry = Arc::new(WorkerRegistry::new());
        let regular: Box<dyn crate::core::Worker> = Box::new(
            BasicWorkerBuilder::new("http://regular:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        let prefill: Box<dyn crate::core::Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8000")
                .worker_type(WorkerType::Prefill { bootstrap_port: None })
                .build(),
        );
        registry.register(Arc::from(regular));
        registry.register(Arc::from(prefill));
        registry
    }

    /// Helper: seed internal worker_counts so select_router can work
    /// without waiting for the background adjustment task.
    fn seed_worker_counts(scheduler: &ProportionScheduler, regular: usize, pd: usize) {
        let mut counts = scheduler.worker_counts.write().unwrap();
        counts.insert(router_ids::HTTP_REGULAR, regular);
        counts.insert(router_ids::HTTP_PD, pd);
    }

    // ── 1. Basic properties ──────────────────────────────────────────

    #[test]
    fn test_scheduler_name() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let registry = registry_with_regular_and_prefill();
            let scheduler = ProportionScheduler::new(test_config(), registry);
            assert_eq!(scheduler.name(), "proportion");
        });
    }

    #[test]
    fn test_needs_request_text() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let registry = registry_with_regular_and_prefill();
            let scheduler = ProportionScheduler::new(test_config(), registry);
            assert!(scheduler.needs_request_text());
        });
    }

    // ── 2. No workers → None ─────────────────────────────────────────

    #[tokio::test]
    async fn test_no_workers_returns_none() {
        let registry = registry_with_regular_and_prefill();
        let scheduler = ProportionScheduler::new(test_config(), registry);
        // worker_counts stays empty (all zeros) → should return None
        let candidates = vec![router_ids::HTTP_REGULAR, router_ids::HTTP_PD];
        let tokens: Vec<u32> = vec![1; 100];
        let info = SelectRouterInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        let result = scheduler.select_router(&candidates, &info).await;
        assert!(result.is_none(), "Should return None when worker_counts are all zero");
    }

    // ── 3. Crossover-point routing ───────────────────────────────────

    #[tokio::test]
    async fn test_short_request_routes_to_regular() {
        let registry = registry_with_regular_and_prefill();
        let scheduler = ProportionScheduler::new(test_config(), registry);
        seed_worker_counts(&scheduler, 2, 2);

        let candidates = vec![router_ids::HTTP_REGULAR, router_ids::HTTP_PD];
        let tokens: Vec<u32> = vec![1; 100]; // 100 < crossover(512)
        let info = SelectRouterInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        let result = scheduler.select_router(&candidates, &info).await;
        assert_eq!(result, Some(router_ids::HTTP_REGULAR));
    }

    #[tokio::test]
    async fn test_long_request_routes_to_pd() {
        let registry = registry_with_regular_and_prefill();
        let scheduler = ProportionScheduler::new(test_config(), registry);
        seed_worker_counts(&scheduler, 2, 2);

        let candidates = vec![router_ids::HTTP_REGULAR, router_ids::HTTP_PD];
        let tokens: Vec<u32> = vec![1; 1000]; // 1000 > crossover(512)
        let info = SelectRouterInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        let result = scheduler.select_router(&candidates, &info).await;
        assert_eq!(result, Some(router_ids::HTTP_PD));
    }

    // ── 4. Fallback ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_fallback_when_only_pd_available() {
        let registry = registry_with_regular_and_prefill();
        let scheduler = ProportionScheduler::new(test_config(), registry);
        seed_worker_counts(&scheduler, 2, 2);

        let candidates = vec![router_ids::HTTP_PD]; // only PD
        let tokens: Vec<u32> = vec![1; 100]; // short, would prefer Regular
        let info = SelectRouterInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        let result = scheduler.select_router(&candidates, &info).await;
        assert_eq!(result, Some(router_ids::HTTP_PD));
    }

    #[tokio::test]
    async fn test_fallback_when_only_regular_available() {
        let registry = registry_with_regular_and_prefill();
        let scheduler = ProportionScheduler::new(test_config(), registry);
        seed_worker_counts(&scheduler, 2, 2);

        let candidates = vec![router_ids::HTTP_REGULAR]; // only Regular
        let tokens: Vec<u32> = vec![1; 1000]; // long, would prefer PD
        let info = SelectRouterInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        let result = scheduler.select_router(&candidates, &info).await;
        assert_eq!(result, Some(router_ids::HTTP_REGULAR));
    }

    // ── 5. Load tracking ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_load_tracking() {
        let registry = registry_with_regular_and_prefill();
        let scheduler = ProportionScheduler::new(test_config(), registry);
        seed_worker_counts(&scheduler, 2, 2);

        let candidates = vec![router_ids::HTTP_REGULAR, router_ids::HTTP_PD];

        // Short request (100 tokens) → Regular
        let tokens_short: Vec<u32> = vec![1; 100];
        let info_short = SelectRouterInfo {
            tokens: Some(&tokens_short),
            ..Default::default()
        };
        scheduler.select_router(&candidates, &info_short).await;

        // Long request (1000 tokens) → PD
        let tokens_long: Vec<u32> = vec![1; 1000];
        let info_long = SelectRouterInfo {
            tokens: Some(&tokens_long),
            ..Default::default()
        };
        scheduler.select_router(&candidates, &info_long).await;

        {
            let loads = scheduler.router_loads.read().unwrap();
            assert_eq!(*loads.get(&router_ids::HTTP_REGULAR).unwrap_or(&0), 100);
            assert_eq!(*loads.get(&router_ids::HTTP_PD).unwrap_or(&0), 1000);
        }
        {
            let queue = scheduler.global_request_queue.read().unwrap();
            assert_eq!(queue.len(), 2);
        }
    }

    // ── 6. Imbalance override ────────────────────────────────────────

    #[tokio::test]
    async fn test_imbalance_overrides_crossover() {
        let registry = registry_with_regular_and_prefill();
        let config = ProportionSchedulerConfig {
            adjust_interval: Duration::from_secs(3600),
            adjust_window: Duration::from_secs(3600),
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.01,
            regular_worker_weight: 1.0,
        };
        let scheduler = ProportionScheduler::new(config, registry);
        seed_worker_counts(&scheduler, 1, 1);

        // Inflate PD load so norm_pd >> norm_regular
        {
            let mut loads = scheduler.router_loads.write().unwrap();
            loads.insert(router_ids::HTTP_PD, 10000);
            loads.insert(router_ids::HTTP_REGULAR, 0);
        }

        let candidates = vec![router_ids::HTTP_REGULAR, router_ids::HTTP_PD];
        let tokens: Vec<u32> = vec![1; 1000]; // long, would go PD normally
        let info = SelectRouterInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        let result = scheduler.select_router(&candidates, &info).await;
        assert_eq!(
            result,
            Some(router_ids::HTTP_REGULAR),
            "Should override crossover and pick less-loaded Regular"
        );
    }

    // ── 7. Adjustment task recalculates crossover ────────────────────

    #[tokio::test]
    async fn test_adjustment_recalculates_crossover() {
        let registry = registry_with_regular_and_prefill();
        let config = ProportionSchedulerConfig {
            adjust_interval: Duration::from_millis(100),
            adjust_window: Duration::from_secs(60),
            balance_abs_threshold: 100,
            balance_rel_threshold: 1.5,
            regular_worker_weight: 0.5, // regular_share = 0.5/(0.5+1.0) ≈ 0.333
        };
        let scheduler = ProportionScheduler::new(config, registry);
        seed_worker_counts(&scheduler, 1, 1);

        // Populate queue: [100, 200, 300, 400, 500], total=1500
        {
            let mut queue = scheduler.global_request_queue.write().unwrap();
            let mut loads = scheduler.router_loads.write().unwrap();
            for &tc in &[100usize, 200, 300, 400, 500] {
                queue.push_back(RequestRecord {
                    token_count: tc,
                    timestamp: Instant::now(),
                    router_id: router_ids::HTTP_REGULAR,
                });
                *loads.entry(router_ids::HTTP_REGULAR).or_insert(0) += tc;
            }
        }

        let old_crossover = *scheduler.crossover_point.read().unwrap();

        // Wait for adjustment cycle
        tokio::time::sleep(Duration::from_millis(300)).await;

        let new_crossover = *scheduler.crossover_point.read().unwrap();
        // ideal_regular_share ≈ 0.333, sorted=[100,200,300,400,500]
        // accumulated: 100→6.7%, 300→20%, 600→40%≥33.3% → crossover=300
        assert_ne!(
            new_crossover, old_crossover,
            "Adjustment task should have recalculated crossover (old={old_crossover}, new={new_crossover})"
        );
        assert_eq!(new_crossover, 300, "Expected crossover=300 based on load distribution");
    }
}
