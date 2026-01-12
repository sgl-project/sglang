//! Least load balancing policy

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use tracing::debug;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::core::Worker;

/// Calculate request load based on request text length
pub fn calculate_request_load(info: &SelectWorkerInfo) -> isize {
    info.request_text.map(|t| t.len() as isize).unwrap_or(1)
}

/// Per-worker load state
#[derive(Debug, Default)]
struct WorkerState {
    /// Current total load
    load: isize,
    /// Pending request loads (FIFO queue for decrement)
    pending: VecDeque<isize>,
}

/// Least load policy
///
/// Selects the worker with the lowest load among all healthy workers.
/// Load is calculated based on request text length.
#[derive(Debug, Default)]
pub struct LeastLoadPolicy {
    state: Mutex<HashMap<String, WorkerState>>,
}

impl LeastLoadPolicy {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(HashMap::new()),
        }
    }
}

impl LoadBalancingPolicy for LeastLoadPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        let request_load = calculate_request_load(info);
        let mut state = self.state.lock().ok()?;

        // Find worker with minimum load
        let mut min_load = isize::MAX;
        let mut selected_idx = healthy_indices[0];

        for &idx in &healthy_indices {
            let url = workers[idx].url();
            let load = state.get(url).map(|s| s.load).unwrap_or(0);

            if load < min_load {
                min_load = load;
                selected_idx = idx;
            }
        }

        // Update selected worker's load
        let url = workers[selected_idx].url().to_string();
        let worker_state = state.entry(url.clone()).or_default();
        worker_state.load += request_load;
        worker_state.pending.push_back(request_load);

        debug!(
            "Least load selection: selected {} with load {}, added {}",
            url, min_load, request_load
        );

        drop(state);
        workers[selected_idx].increment_processed();

        Some(selected_idx)
    }

    fn on_request_complete(&self, worker_url: &str, _success: bool) {
        if let Ok(mut state) = self.state.lock() {
            if let Some(worker_state) = state.get_mut(worker_url) {
                let load_to_remove = worker_state.pending.pop_front().unwrap_or(1);
                worker_state.load = (worker_state.load - load_to_remove).max(0);
            }
        }
    }

    fn name(&self) -> &'static str {
        "least_load"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn update_loads(&self, loads: &HashMap<String, isize>) {
        if let Ok(mut state) = self.state.lock() {
            for (url, load) in loads {
                state.entry(url.clone()).or_default().load = *load;
            }
        }
    }

    fn reset(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.clear();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
