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