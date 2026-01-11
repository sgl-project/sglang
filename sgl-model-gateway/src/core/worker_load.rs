//! Worker load
//!
//! Record and manage the DP group load of workers.
use std::{
    collections::HashMap,
    fmt::Debug,
    sync::RwLock,
};

use crate::core::Worker;

use tracing::debug;

#[derive(Debug, Default)]
pub struct WorkerLoadManager {
    // <worker, <dp_rank, loads>>
    dp_cached_loads: RwLock<HashMap<String, HashMap<isize, isize>>>,
}

impl WorkerLoadManager {
    pub fn new() -> Self {
        Self {
            dp_cached_loads: RwLock::new(HashMap::new()),
        }
    }

    pub fn update_dp_loads(&self, loads: &HashMap<String, HashMap<isize, isize>>) {
        debug!("WorkerLoadManager update_dp_loads map:{:?}", loads);
        if let Ok(mut cached) = self.dp_cached_loads.write() {
            *cached = loads.clone();
        }
    }

    pub fn get_lowest_dp_load(&self, worker: &dyn Worker) -> Option<isize> {
        if let Ok(cached_loads) = self.dp_cached_loads.read() {
            if let Some(loads) = cached_loads.get(worker.url()) {
                return loads
                    .iter()
                    .min_by_key(|&(_, load)| load)
                    .map(|(&rand_id, _)| rand_id);
            }
        }
        None
    }

    pub fn load_increment(&self, worker: &dyn Worker, dp_rank: isize, increment: isize) {
        // Add an increment to the load of dp group,
        // to prevent all request from being scheduled to the same DP group during the interval between two load reports.
        if let Ok(mut cached_loads) = self.dp_cached_loads.write() {
            debug!("WorkerLoadManager load_increment map:{:?}, increment:{}", cached_loads, increment);
            if let Some(loads) = cached_loads.get_mut(worker.url()) {
                if let Some(dp_load) = loads.get_mut(&dp_rank) {
                    *dp_load += increment;
                }
            }
        }
    }
}

#[cfg(test)]
mod dp_load_manager_tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_new_dp_load_manager_instance() {
        let dp_load_manager = WorkerLoadManager::new();
        let cached = dp_load_manager.dp_cached_loads.read().unwrap();
        assert!(cached.is_empty());
    }

    #[test]
    fn test_update_dp_load() {
        let manager = WorkerLoadManager::new();
        let mut loads = HashMap::new();

        // insert worker1_load
        let mut worker1_load = HashMap::new();
        worker1_load.insert(0, 2);
        worker1_load.insert(1, 1);
        loads.insert("http://worker1:8080".to_string(), worker1_load);

        // insert worker2.load
        let mut worker2_load = HashMap::new();
        worker2_load.insert(0, 3);
        loads.insert("http://worker2:8080".to_string(), worker2_load);

        // update
        manager.update_dp_loads(&loads);

        // assert
        let cached = manager.dp_cached_loads.read().unwrap();
        assert_eq!(cached.len(), 2);

        let worker2_cache = cached.get("http://worker2:8080").unwrap();
        assert_eq!(worker2_cache.get(&0), Some(&3));
    }

    #[test]
    fn test_get_lowest_dp_load() {
        let worker1 = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .api_key("test_api_key2")
            .build();

        let manager = WorkerLoadManager::new();
        let mut loads = HashMap::new();
        // insert worker1_load
        let mut worker1_load = HashMap::new();
        worker1_load.insert(0, 2);
        worker1_load.insert(1, 1);
        worker1_load.insert(3, 3);
        loads.insert(worker1.url().to_string(), worker1_load);
        manager.update_dp_loads(&loads);

        // Verify that the worker1 with the lowest load is dp_rank = 1
        assert_eq!(manager.get_lowest_dp_load(&worker1), Some(1));
    }

    #[test]
    fn test_load_increment() {
        let worker2 = BasicWorkerBuilder::new("http://worker2:8080")
            .worker_type(WorkerType::Regular)
            .api_key("test_api_key2")
            .build();

        let manager = WorkerLoadManager::new();
        manager.load_increment(&worker2, 0, 5);
        let cached = manager.dp_cached_loads.read().expect("Rwlock read1 failed");
        assert!(cached.get(worker2.url()).is_none());
        drop(cached);

        // insert worker2.load
        let mut worker2_load = HashMap::new();
        worker2_load.insert(0, 2);
        let mut loads = HashMap::new();
        loads.insert(worker2.url().to_string(), worker2_load);
        manager.update_dp_loads(&loads);

        // load increment
        manager.load_increment(&worker2, 0, 5);
        let cached = manager.dp_cached_loads.read().expect("Rwlock read2 failed");
        let worker2_cache = cached
            .get(worker2.url())
            .expect("worker2 not found in cache");
        // 2 + 5 = 7
        assert_eq!(worker2_cache.get(&0), Some(&7));
    }
}
