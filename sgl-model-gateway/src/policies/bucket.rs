use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, Mutex, RwLock},
    thread,
    time::{Duration, SystemTime},
};

use dashmap::DashMap;
use rand::Rng;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::{get_healthy_worker_indices, BucketConfig, LoadBalancingPolicy};
use crate::core::Worker;

#[derive(Debug)]
pub struct BucketPolicy {
    config: BucketConfig,
    buckets: Arc<DashMap<String, Arc<RwLock<Bucket>>>>,
    adjustment_handle: Option<thread::JoinHandle<()>>,
}

impl Default for BucketPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BucketPolicy {
    fn drop(&mut self) {
        if let Some(handle) = self.adjustment_handle.take() {
            drop(handle);
        }
    }
}

impl BucketPolicy {
    pub fn new() -> Self {
        Self::with_config(BucketConfig::default())
    }

    pub fn with_config(config: BucketConfig) -> Self {
        let buckets = Arc::new(DashMap::<String, Arc<RwLock<Bucket>>>::new());

        let adjustment_handle = {
            let buckets_clone = Arc::clone(&buckets);

            let interval_secs = config.bucket_adjust_interval_secs;

            Some(thread::spawn(move || loop {
                thread::sleep(Duration::from_secs(interval_secs as u64));

                for bucket_ref in buckets_clone.iter() {
                    let model_id = bucket_ref.key();
                    let bucket = bucket_ref.value();
                    match bucket.write() {
                        Ok(mut bucket_guard) => {
                            bucket_guard.adjust_boundary();
                        }
                        Err(e) => {
                            eprintln!(
                                "Failed to acquire write lock for bucket {}: {}",
                                model_id, e
                            );
                        }
                    }
                }
            }))
        };

        Self {
            config,
            buckets,
            adjustment_handle,
        }
    }

    pub fn init_prefill_worker_urls(&self, prefill_workers: &[Arc<dyn Worker>]) {
        // Group workers by model
        let mut model_workers: HashMap<String, Vec<&Arc<dyn Worker>>> = HashMap::new();
        for worker in prefill_workers {
            // Use "default" for unknown/empty model_ids for backward compatibility
            let model_id = worker.model_id();
            let model_key = if model_id.is_empty() || model_id == "unknown" {
                "default"
            } else {
                model_id
            };
            model_workers
                .entry(model_key.to_string())
                .or_default()
                .push(worker);
        }
        // Initialize bucket for each model
        for (model_key, model_workers) in model_workers {
            let bucket = self
                .buckets
                .entry(model_key)
                .or_insert_with(|| {
                    Arc::new(RwLock::new(Bucket::new(
                        self.config.bucket_adjust_interval_secs * 1000,
                    )))
                })
                .clone();

            let worker_urls: Vec<String> = model_workers
                .iter()
                .map(|worker| worker.url().to_string())
                .collect();

            let lock_result = bucket.write();
            if let Ok(mut bucket_guard) = lock_result {
                bucket_guard.init_prefill_worker_urls(worker_urls);
            } else {
                eprintln!("Failed to acquire write lock for bucket initialization");
            }
        }
    }

    pub fn add_prefill_url(&self, worker: &dyn Worker) {
        let model_id = worker.model_id();
        let model_key = if model_id.is_empty() || model_id == "unknown" {
            "default"
        } else {
            model_id
        };
        let bucket = self
            .buckets
            .entry(model_key.to_string())
            .or_insert_with(|| {
                Arc::new(RwLock::new(Bucket::new(
                    self.config.bucket_adjust_interval_secs * 1000,
                )))
            })
            .clone();

        let lock_result = bucket.write();
        if let Ok(mut bucket_guard) = lock_result {
            let worker_url = worker.url().to_string();

            let prefill_worker_urls_clone = {
                let mut prefill_worker_urls = bucket_guard.prefill_worker_urls.lock().unwrap();
                if !prefill_worker_urls.contains(&worker_url) {
                    prefill_worker_urls.push(worker_url.clone());
                }
                let cloned = prefill_worker_urls.clone();

                let mut chars_per_url = bucket_guard.chars_per_url.lock().unwrap();
                chars_per_url.entry(worker_url.clone()).or_insert(0);

                cloned
            };

            bucket_guard.init_prefill_worker_urls(prefill_worker_urls_clone);

            info!(
                "Added worker {} to bucket for model {}",
                worker_url, model_key
            );
        } else {
            error!(
                "Failed to acquire write lock for bucket of model {}",
                model_key
            );
        }
    }

    pub fn remove_prefill_url(&self, worker: &dyn Worker) {
        let model_id = worker.model_id();
        let model_key = if model_id.is_empty() || model_id == "unknown" {
            "default"
        } else {
            model_id
        };

        if let Some(bucket_entry) = self.buckets.get(model_key) {
            let bucket = bucket_entry.value();
            let worker_url = worker.url().to_string();

            let lock_result = bucket.write();
            if let Ok(mut bucket_guard) = lock_result {
                let (updated_len, updated_urls) = {
                    let mut prefill_worker_urls = bucket_guard.prefill_worker_urls.lock().unwrap();
                    prefill_worker_urls.retain(|u| u != &worker_url);
                    let len = prefill_worker_urls.len();
                    let urls_clone = prefill_worker_urls.clone();

                    let mut chars_per_url = bucket_guard.chars_per_url.lock().unwrap();
                    chars_per_url.remove(&worker_url);

                    (len, urls_clone)
                };

                bucket_guard.bucket_cnt = updated_len;

                if updated_len > 0 {
                    bucket_guard.init_prefill_worker_urls(updated_urls);
                }

                info!(
                    "Removed worker {} from bucket for model {} (remaining workers: {})",
                    worker_url, model_key, bucket_guard.bucket_cnt
                );
            } else {
                error!(
                    "Failed to acquire write lock for bucket of model {}",
                    model_key
                );
            }
        } else {
            warn!(
                "No bucket found for model {} when trying to remove worker",
                model_key
            );
        }
    }
}

impl LoadBalancingPolicy for BucketPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        let char_count = match request_text {
            None => 0,
            Some(text) => text.chars().count(),
        };

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model
        let first_model = workers[healthy_indices[0]].model_id();
        let model_key = if first_model.is_empty() || first_model == "unknown" {
            "default"
        } else {
            first_model
        };

        let bucket = self
            .buckets
            .get(model_key)
            .map(|entry| entry.value().clone());
        let prefill_url = if let Some(bucket) = bucket {
            let (choiced_url, chars_per_url_snapshot) = {
                let buc = bucket.read().unwrap();
                let chars_per_url_snapshot = buc.chars_per_url.lock().unwrap().clone();
                let choiced_url = buc.find_boundary(char_count);
                (choiced_url, chars_per_url_snapshot)
            };
            let max_load = chars_per_url_snapshot.values().copied().max().unwrap_or(0);
            let min_load = chars_per_url_snapshot.values().copied().min().unwrap_or(0);
            let abs_diff = max_load.saturating_sub(min_load);
            let rel_threshold = self.config.balance_rel_threshold * min_load as f32;
            let is_imbalanced =
                abs_diff > self.config.balance_abs_threshold && max_load as f32 > rel_threshold;
            debug!(
                "Current PD instance status | is_imbalanced={}",
                is_imbalanced
            );

            let mut rng = rand::rng();
            let prefill_url = if is_imbalanced {
                debug!("select prefill instance by Load Balance policy");
                let min_url = chars_per_url_snapshot
                    .iter()
                    .min_by_key(|(_, &chars)| chars)
                    .map(|(url, _)| url.clone())
                    .unwrap_or_else(|| {
                        let idx = rng.random_range(0..healthy_indices.len());
                        let url = workers[healthy_indices[idx]].url();
                        warn!("No URL found, randomly selecting: {}", url);
                        url.to_string()
                    });
                min_url
            } else {
                debug!("select prefill instance by Bucket policy");
                match choiced_url {
                    Some(url) if !url.is_empty() => url,
                    _ => {
                        let idx = rng.random_range(0..healthy_indices.len());
                        let selected_url = workers[healthy_indices[idx]].url();
                        warn!("Boundary not found, randomly selection: {}", selected_url);
                        selected_url.to_string()
                    }
                }
            };

            {
                let mut buc = bucket.write().unwrap();
                buc.post_process_request(char_count, prefill_url.clone());
            }

            prefill_url
        } else {
            warn!(
                "No bucket found for model {}, randomly selecting healthy worker",
                model_key
            );
            let mut rng = rand::rng();
            let idx = rng.random_range(0..healthy_indices.len());
            let selected_worker = &workers[healthy_indices[idx]];
            let prefill_url = selected_worker.url().to_string();
            prefill_url
        };

        workers.iter().position(|w| w.url() == prefill_url)
    }

    fn name(&self) -> &'static str {
        "bucket"
    }

    fn needs_request_text(&self) -> bool {
        true // Bucket policy needs request text
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug, Clone)]
pub struct Bucket {
    l_max: usize,
    bucket_cnt: usize,
    pub prefill_worker_urls: Arc<Mutex<Vec<String>>>,
    load_total: usize,
    pub period: usize,
    bucket_load: usize,
    boundary: Vec<Boundary>,
    request_list: VecDeque<SequencerRequest>,
    t_req_loads: HashMap<String, usize>,
    pub chars_per_url: Arc<Mutex<HashMap<String, usize>>>,
}

#[derive(Debug, Clone)]
pub struct SequencerRequest {
    pub id: String,
    pub char_cnt: usize,
    pub timestamp: SystemTime,
    pub prefill_worker_url: String,
}

#[derive(Debug, Clone)]
pub struct Boundary {
    pub url: String,
    pub range: [usize; 2],
}

impl Boundary {
    pub fn new(url: String, range: [usize; 2]) -> Self {
        Boundary { url, range }
    }
}

impl Bucket {
    pub fn new(period: usize) -> Self {
        let l_max = 4096;

        let bucket_cnt = 0;

        let load_total = 0;
        let bucket_load = 0;

        let t_req_loads = HashMap::new();
        let request_list = VecDeque::new();

        let initial_map = HashMap::new();

        let boundary = Vec::new();

        let prefill_worker_urls = Arc::new(Mutex::new(Vec::new()));

        Bucket {
            l_max,
            bucket_cnt,
            prefill_worker_urls,
            load_total,
            period,
            bucket_load,
            boundary,
            request_list,
            t_req_loads,
            chars_per_url: Arc::new(Mutex::new(initial_map)),
        }
    }

    pub fn init_prefill_worker_urls(&mut self, prefill_worker_urls: Vec<String>) {
        let bucket_cnt = prefill_worker_urls.len();
        self.bucket_cnt = bucket_cnt;
        let mut urls_lock = self.prefill_worker_urls.lock().unwrap();
        *urls_lock = prefill_worker_urls.clone();

        let mut chars_lock = self.chars_per_url.lock().unwrap();
        chars_lock.clear();

        for url in prefill_worker_urls.iter() {
            chars_lock.insert(url.clone(), 0);
        }

        let worker_cnt = bucket_cnt;
        let boundary = if worker_cnt == 0 {
            Vec::new()
        } else {
            let gap = self.l_max / worker_cnt;
            self.l_max = usize::MAX;
            prefill_worker_urls
                .iter()
                .enumerate()
                .map(|(i, url)| {
                    let min = i * gap;
                    let max = if i == worker_cnt - 1 {
                        self.l_max
                    } else {
                        (i + 1) * gap - 1
                    };
                    Boundary::new(url.clone(), [min, max])
                })
                .collect()
        };

        self.boundary = boundary;
        info!("Init boundary:{:?}", self.boundary);
    }

    pub fn post_process_request(&mut self, char_cnt: usize, prefill_url: String) {
        {
            let mut map = self.chars_per_url.lock().unwrap();
            *map.entry(prefill_url.clone()).or_insert(0) += char_cnt;
        }

        let now = SystemTime::now();
        let time_window_duration = Duration::from_millis(self.period as u64);
        let mut removed_load = 0;

        while let Some(req) = self.request_list.front() {
            let expired = match now.duration_since(req.timestamp) {
                Ok(duration) => duration > time_window_duration,
                Err(_) => true,
            };

            if !expired {
                break;
            }

            if let Some(removed_req) = self.request_list.pop_front() {
                self.t_req_loads.remove(&removed_req.id);
                removed_load += removed_req.char_cnt;

                let mut map = self.chars_per_url.lock().unwrap();
                if let Some(count) = map.get_mut(&removed_req.prefill_worker_url) {
                    *count = count.saturating_sub(removed_req.char_cnt);
                }
            }
        }

        self.load_total = self.load_total.saturating_sub(removed_load);

        let id = Uuid::new_v4().to_string();

        self.t_req_loads.insert(id.clone(), char_cnt);

        self.request_list.push_back(SequencerRequest {
            id,
            char_cnt,
            timestamp: now,
            prefill_worker_url: prefill_url,
        });

        self.load_total = self.load_total.saturating_add(char_cnt);
    }

    pub fn find_boundary(&self, char_count: usize) -> Option<String> {
        let mut left = 0;
        let mut right = self.boundary.len();
        let mut _steps = 0;

        while left < right {
            _steps += 1;
            let mid = left + (right - left) / 2;
            let range = self.boundary[mid].range;

            if char_count < range[0] {
                right = mid;
            } else if char_count > range[1] {
                left = mid + 1;
            } else {
                return Some(self.boundary[mid].url.clone());
            }
        }
        None
    }

    pub fn get_total_load(&self) -> usize {
        self.load_total
    }

    fn update_workers_cnt(&mut self) {
        let pwu = self.prefill_worker_urls.lock().unwrap();
        self.bucket_cnt = pwu.len();

        let mut char_map = self.chars_per_url.lock().unwrap();
        let current_urls: HashSet<_> = char_map.keys().cloned().collect();
        let new_urls: HashSet<_> = pwu.iter().cloned().collect();

        for url in new_urls.difference(&current_urls) {
            char_map.insert(url.clone(), 0);
        }

        for url in current_urls.difference(&new_urls) {
            if char_map.get(url) == Some(&0) {
                char_map.remove(url);
            }
        }
    }

    pub fn adjust_boundary(&mut self) {
        if self.t_req_loads.is_empty() {
            return;
        }

        self.update_workers_cnt();
        let worker_cnt = self.bucket_cnt;
        if worker_cnt == 0 {
            return;
        }
        let new_single_bucket_load = self.get_total_load() / worker_cnt;
        let old_single_bucket_load = self.bucket_load;

        if new_single_bucket_load <= 2 * old_single_bucket_load
            && (old_single_bucket_load <= 2 * new_single_bucket_load && old_single_bucket_load != 0)
        {
            info!("No need to adjust the bucket boundaries.");
            return;
        }

        info!("Before adjusting boundary | {:?}", self.boundary);
        self.bucket_load = new_single_bucket_load;
        let mut new_boundary = Vec::new();
        let mut hist_load: Vec<usize> = self.t_req_loads.values().cloned().collect();
        hist_load.sort();
        let mut upper_bound: usize = 0;
        let mut last_load_index: usize = 0;
        let max_value = usize::MAX;

        let worker_url = {
            let guard = self.prefill_worker_urls.lock().unwrap();
            (*guard).clone()
        };

        let mut iter = worker_url.iter().peekable();
        // let mut curr_worker_id = 0;
        while let Some(url) = iter.next() {
            if last_load_index >= hist_load.len() && iter.peek().is_none() {
                new_boundary.push(Boundary::new(url.clone(), [upper_bound, max_value]));
                break;
            }
            let mut load_accumulator = 0;
            let mut break_flag = false;
            for &load in hist_load[last_load_index..].iter() {
                load_accumulator += load;
                if load_accumulator >= new_single_bucket_load {
                    if iter.peek().is_none() {
                        new_boundary.push(Boundary::new(url.clone(), [upper_bound, max_value]));
                        break_flag = true;
                        break;
                    }
                    let real_load = upper_bound + new_single_bucket_load;
                    if load <= upper_bound {
                        new_boundary.push(Boundary::new(url.clone(), [upper_bound, real_load]));
                        upper_bound = real_load + 1;
                    } else {
                        new_boundary.push(Boundary::new(url.clone(), [upper_bound, load]));
                        upper_bound = load + 1;
                    }
                    last_load_index += 1;
                    break_flag = true;
                    break;
                } else {
                    last_load_index += 1;
                }
            }
            if !break_flag {
                let mut right_bound_value = upper_bound + new_single_bucket_load;
                if iter.peek().is_none() {
                    right_bound_value = max_value;
                    new_boundary.push(Boundary::new(url.clone(), [upper_bound, right_bound_value]));
                    break;
                }
                new_boundary.push(Boundary::new(url.clone(), [upper_bound, right_bound_value]));
                upper_bound = right_bound_value + 1;
            }
        }
        self.boundary = new_boundary;
        info!("After adjusting boundary | {:?}", self.boundary);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[tokio::test]
    async fn test_load_balancing_conditions() {
        // Test 1: Basic load balancing trigger
        let config = BucketConfig {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 10,
        };
        let policy = BucketPolicy::with_config(config);
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![
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
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // Initialize the policy with prefill_workers
        policy.init_prefill_worker_urls(&prefill_workers);

        // === Phase S1: Construct bucket boundaries ===
        // Requests len =33 -> Bucket 1(expected range: 0-33)
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(33)))
            .unwrap();
        // Two requests len =34 ->load balancing
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(34)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(34)))
            .unwrap();

        tokio::time::sleep(Duration::from_secs(11)).await;
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let lock_result = bucket.write();
                if let Ok(bucket_guard) = lock_result {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 33);
                    assert_eq!(bucket_guard.boundary[1].range[1], 67);
                } else {
                    error!(
                        "Failed to acquire write lock for bucket of model {}",
                        model_key
                    );
                }
            }
        }
        // === Phase S2: Validate load balancing ===
        // Three consecutive len=33 requests (Should route to different buckets)
        let idx_1 = policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(33)))
            .unwrap();
        let idx_2 = policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(33)))
            .unwrap();
        let idx_3 = policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(33)))
            .unwrap();
        assert_eq!(idx_1, 0, "Should not trigger load balancing");
        assert_ne!(idx_2, idx_3, "Should trigger load balancing");
        assert_ne!(idx_2, 0, "Should trigger load balancing");
        assert_ne!(idx_3, 0, "Should trigger load balancing");

        // Test 2: Not triggering when absolute threshold not met
        let config = BucketConfig {
            balance_abs_threshold: 30,
            balance_rel_threshold: 2.0,
            ..Default::default()
        };
        let policy = BucketPolicy::with_config(config);
        policy.init_prefill_worker_urls(&prefill_workers);

        // Create load difference below absolute threshold(20 + 8 = 28 < 30)
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(20)))
            .unwrap(); // worker1: 20
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(8)))
            .unwrap(); // worker1: 8

        // Next request should not use bucket scheduling (no load balancing)
        let idx = policy
            .select_worker(&prefill_workers, Some("request"))
            .unwrap();
        assert_eq!(
            idx, 0,
            "Should not trigger load balancing when relative threshold not met"
        );

        // Test 3: Not triggering when relative threshold not met
        let config = BucketConfig {
            balance_abs_threshold: 5,
            balance_rel_threshold: 3.0,
            ..Default::default()
        };
        let policy = BucketPolicy::with_config(config);
        policy.init_prefill_worker_urls(&prefill_workers);

        // Create load difference (but relative threshold not met)
        // Max/Min ratio = 15/5 = 3.0
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(15)))
            .unwrap(); // worker1: 15
        policy
            .select_worker(&prefill_workers, Some("short"))
            .unwrap(); // worker2: 5
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(10)))
            .unwrap(); // worker3: 10

        // Next request should use bucket scheduling (load balancing)
        let idx = policy
            .select_worker(&prefill_workers, Some("request"))
            .unwrap();
        assert_eq!(
            idx, 0,
            "Should not trigger load balancing when relative threshold not met"
        );
    }

    #[tokio::test]
    async fn test_adjust_boundary_1() {
        // Test configuration: Set high threshold to prevent load balancing policy.
        let config = BucketConfig {
            balance_abs_threshold: 300,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 3,
        };
        let policy = BucketPolicy::with_config(config);
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![
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
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // Initialize the policy with prefill_workers
        policy.init_prefill_worker_urls(&prefill_workers);

        // Initial boundary
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let lock_result = bucket.write();
                if let Ok(bucket_guard) = lock_result {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 1364);
                    assert_eq!(bucket_guard.boundary[1].range[1], 2729);
                } else {
                    error!(
                        "Failed to acquire write lock for bucket of model {}",
                        model_key
                    );
                }
            }
        }

        // ===Phase S1: Initial requests to trigger boundary adjustment ===
        // Send requests with lengths: [5, 10, 15, 20, 24, 26] (total = 100)
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(5)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(10)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(15)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(20)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(24)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(26)))
            .unwrap();

        tokio::time::sleep(Duration::from_secs(4)).await;
        // Verify boundaries adjusted to: [0, 20], [21, 26], [27, MAX]
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let lock_result = bucket.write();
                if let Ok(bucket_guard) = lock_result {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 20);
                    assert_eq!(bucket_guard.boundary[1].range[1], 26);
                } else {
                    error!(
                        "Failed to acquire write lock for bucket of model {}",
                        model_key
                    );
                }
            }
        }

        // ===Phase S2: Second set of  requests to trigger boundary adjustment ===
        // Send requests with lengths: [10, 20, 30, 40, 45, 57] (total = 202)
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(10)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(20)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(30)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(40)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(45)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(57)))
            .unwrap();

        tokio::time::sleep(Duration::from_secs(4)).await;
        // Verify boundaries adjusted to: [0, 40], [41, 57], [58, MAX]
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let lock_result = bucket.write();
                if let Ok(bucket_guard) = lock_result {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 40);
                    assert_eq!(bucket_guard.boundary[1].range[1], 57);
                } else {
                    error!(
                        "Failed to acquire write lock for bucket of model {}",
                        model_key
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_adjust_boundary_2() {
        let config = BucketConfig {
            balance_abs_threshold: 300,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 3,
        };
        let policy = BucketPolicy::with_config(config);
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![
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
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // Initialize the policy with prefill_workers
        policy.init_prefill_worker_urls(&prefill_workers);

        // Initial boundary
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let lock_result = bucket.write();
                if let Ok(bucket_guard) = lock_result {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 1364);
                    assert_eq!(bucket_guard.boundary[1].range[1], 2729);
                } else {
                    error!(
                        "Failed to acquire write lock for bucket of model {}",
                        model_key
                    );
                }
            }
        }

        // Send requests with char_count 20
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(20)))
            .unwrap();

        tokio::time::sleep(Duration::from_secs(4)).await;
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let lock_result = bucket.write();
                if let Ok(bucket_guard) = lock_result {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 20);
                    assert_eq!(bucket_guard.boundary[1].range[1], 27);
                } else {
                    error!(
                        "Failed to acquire write lock for bucket of model {}",
                        model_key
                    );
                }
            }
        }

        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(7)))
            .unwrap();

        tokio::time::sleep(Duration::from_secs(4)).await;
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let lock_result = bucket.write();
                if let Ok(bucket_guard) = lock_result {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 7);
                    assert_eq!(bucket_guard.boundary[1].range[1], 10);
                } else {
                    error!(
                        "Failed to acquire write lock for bucket of model {}",
                        model_key
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_not_adjust_boundary() {
        let config = BucketConfig {
            balance_abs_threshold: 300,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 3,
        };
        let policy = BucketPolicy::with_config(config);
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![
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
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // Initialize the policy with prefill_workers
        policy.init_prefill_worker_urls(&prefill_workers);

        // Initial boundary
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let lock_result = bucket.write();
                if let Ok(bucket_guard) = lock_result {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 1364);
                    assert_eq!(bucket_guard.boundary[1].range[1], 2729);
                } else {
                    error!(
                        "Failed to acquire write lock for bucket of model {}",
                        model_key
                    );
                }
            }
        }

        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(5)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(10)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(15)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(20)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(24)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(26)))
            .unwrap();

        tokio::time::sleep(Duration::from_secs(4)).await;
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let lock_result = bucket.write();
                if let Ok(bucket_guard) = lock_result {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 20);
                    assert_eq!(bucket_guard.boundary[1].range[1], 26);
                } else {
                    error!(
                        "Failed to acquire write lock for bucket of model {}",
                        model_key
                    );
                }
            }
        }

        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(10)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(20)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(30)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(32)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(45)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(55)))
            .unwrap();

        tokio::time::sleep(Duration::from_secs(4)).await;
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let lock_result = bucket.write();
                if let Ok(bucket_guard) = lock_result {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 20);
                    assert_eq!(bucket_guard.boundary[1].range[1], 26);
                } else {
                    error!(
                        "Failed to acquire write lock for bucket of model {}",
                        model_key
                    );
                }
            }
        }
    }
}
