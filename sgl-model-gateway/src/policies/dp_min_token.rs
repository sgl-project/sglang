//! dp minimum tokens policy
//！
//! Engine support external DP dispatch
//！ this policy select dp_rank with the minimum number of tokens

use std::sync::Arc;

use async_trait::async_trait;

use super::DPRankLoadPolicy;
use crate::core::{Worker, WorkerLoadManager};

#[derive(Debug)]
pub struct MinimumTokensPolicy {
    worker_load_manager: Option<Arc<WorkerLoadManager>>,
}

impl MinimumTokensPolicy {
    pub fn new(worker_load_manager: Option<Arc<WorkerLoadManager>>) -> Self {
        Self {
            worker_load_manager,
        }
    }
}

#[async_trait]
impl DPRankLoadPolicy for MinimumTokensPolicy {
    async fn select_dp_rank(&self, worker: &dyn Worker, text_str: isize) -> Option<isize> {
        if let Some(worker_load) = self.worker_load_manager.as_ref() {
            let lowest_tokens_dp_rank = worker_load.get_lowest_dp_load(worker);
            if let Some(dp_rank) = lowest_tokens_dp_rank {
                worker_load.load_increment(worker, dp_rank, text_str);
            }
            return lowest_tokens_dp_rank;
        }
        None
    }
}
