use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, Mutex, RwLock},
    time::{Duration, SystemTime},
};

use rand::Rng;
use tracing::{info, warn};
use uuid::Uuid;

use super::{get_healthy_worker_indices, BucketConfig, LoadBalancingPolicy};
use crate::core::Worker;

#[derive(Debug)]
pub struct BucketPolicy {
    config: BucketConfig,
    bucket: Arc<RwLock<Bucket>>,
}

impl Default for BucketPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl BucketPolicy {
    pub fn new() -> Self {
        Self::with_config(BucketConfig::default())
    }

    pub fn with_config(config: BucketConfig) -> Self {
        let bucket = Arc::new(RwLock::new(Bucket::new(
            config.bucket_adjust_interval_secs * 1000,
        ))); // convert to ms

        let bucket_clone = Arc::clone(&bucket);
        tokio::spawn(async move {
            loop {
                {
                    let mut buc = bucket_clone.write().unwrap();
                    buc.adjust_boundary();
                }

                tokio::time::sleep(Duration::from_secs(
                    config.bucket_adjust_interval_secs as u64,
                ))
                .await;
            }
        });

        Self { config, bucket }
    }

    pub fn init_prefill_worker_urls(&self, prefill_workers: &[Arc<dyn Worker>]) {
        let prefill_worker_urls: Vec<String> = prefill_workers
            .iter()
            .map(|worker| worker.url().to_string())
            .collect();

        let mut bucket = self.bucket.write().unwrap();
        bucket.init_prefill_worker_urls(prefill_worker_urls);
    }

    pub fn add_prefill_url(&self, url: String) {
        let buc = self.bucket.write().unwrap();
        let mut prefill_worker_urls = buc.prefill_worker_urls.lock().unwrap();
        prefill_worker_urls.push(url);
    }

    pub fn remove_prefill_url(&self, url: &str) {
        let buc = self.bucket.write().unwrap();
        let mut prefill_worker_urls = buc.prefill_worker_urls.lock().unwrap();
        prefill_worker_urls.retain(|worker_url| worker_url != url);
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

        let buc_arc = Arc::clone(&self.bucket);
        let choiced_url_snapshot;
        let chars_per_url_snapshot;
        {
            let buc = buc_arc.read().unwrap();
            choiced_url_snapshot = buc.find_boundary(char_count);
            chars_per_url_snapshot = buc.chars_per_url.lock().unwrap().clone();
        }

        let max_load = chars_per_url_snapshot.values().copied().max().unwrap_or(0);
        let min_load = chars_per_url_snapshot.values().copied().min().unwrap_or(0);
        let abs_diff = max_load.saturating_sub(min_load);
        let rel_threshold = self.config.balance_rel_threshold * min_load as f32;

        //Load balancing is triggered when (max_load - min_load) > abs_threshold AND max_load > min_load * rel_threshold.
        // balance_abs_threshold = 1
        let is_imbalanced = 
            abs_diff > self.config.balance_abs_threshold && max_load as f32 > rel_threshold;
        info!(
            "Current PD instance status | is_imbalanced={}",
            is_imbalanced
        );
        let mut rng = rand::rng();
        let prefill_url = if is_imbalanced {
            info!("select prefill instance by Load Balance policy");
            let min_url = chars_per_url_snapshot
                .iter()
                .min_by_key(|(_, &chars)| chars)
                .map(|(url, _)| url.clone())
                .unwrap_or_else(|| {
                    let prefill_idx = rng.random_range(0..healthy_indices.len());
                    let url = workers[prefill_idx].url();
                    warn!("No URL found, randomly selecting: {}", url);
                    url.to_string()
                });
            min_url
        } else {
            info!("select prefill instance by Bucket policy");
            if choiced_url_snapshot.is_empty() {
                let prefill_idx = rng.random_range(0..healthy_indices.len());
                let selected_url = workers[prefill_idx].url();
                warn!("Boundary not found, randomly selection: {}", selected_url);
                selected_url.to_string()
            } else {
                choiced_url_snapshot
            }
        };

        {
            let mut buc = buc_arc.write().unwrap();
            buc.post_process_request(char_count, prefill_url.clone());
        }

        let prefill_idx = workers.iter().position(|w| w.url() == prefill_url)?;
        Some(prefill_idx)
    }

    fn select_worker_pair(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<(usize, usize)> {
        let prefill_idx = self.select_worker(prefill_workers, request_text)?;

        let healthy_decode = get_healthy_worker_indices(decode_workers);
        if healthy_decode.is_empty() {
            return None;
        }

        let mut rng = rand::rng();
        let decode_idx = rng.random_range(0..healthy_decode.len());

        Some((prefill_idx, decode_idx))
    }

    fn name(&self) -> &'static str {
        "bucket"
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

    pub fn find_boundary(&self, char_count: usize) -> String {
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
                return self.boundary[mid].url.clone();
            }
        }
        "".to_string()
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

        if self.bucket_cnt == 0 {
            return;
        }
        self.update_workers_cnt();
        let worker_cnt = self.bucket_cnt;
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

        tokio::time::sleep(Duration::from_secs(10)).await;
        {
            let bucket_guard = policy.bucket.read().unwrap();
            // Expected Boundary: [0, 33] [34, 67] [68, MAX]
            assert_eq!(bucket_guard.boundary[0].range[1], 33);
            assert_eq!(bucket_guard.boundary[1].range[1], 67);
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
        assert_eq!(idx_2, 1, "Should trigger load balancing");
        assert_eq!(idx_3, 2, "Should trigger load balancing");

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
            let bucket_guard = policy.bucket.read().unwrap();
            assert_eq!(bucket_guard.boundary[0].range[1], 1364);
            assert_eq!(bucket_guard.boundary[1].range[1], 2729);
        }

        // ===Phase S1: Initial requests to trigger boundary adjustment ===
        // Send requests with lengths: [5, 10, 15, 20, 24, 26] (total = 100)
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(5)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(10))).
            unwrap();
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

        tokio::time::sleep(Duration::from_secs(3)).await;
        // Verify boundaries adjusted to: [0, 20], [21, 26], [27, MAX]
        {
            let bucket_guard = policy.bucket.read().unwrap();
            assert_eq!(bucket_guard.boundary[0].range[1], 20);
            assert_eq!(bucket_guard.boundary[1].range[1], 26);
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

        tokio::time::sleep(Duration::from_secs(3)).await;
        // Verify boundaries adjusted to: [0, 40], [41, 57], [58, MAX]
        {
            let bucket_guard = policy.bucket.read().unwrap();
            assert_eq!(bucket_guard.boundary[0].range[1], 40);
            assert_eq!(bucket_guard.boundary[1].range[1], 57);
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
            let bucket_guard = policy.bucket.read().unwrap();
            assert_eq!(bucket_guard.boundary[0].range[1], 1364);
            assert_eq!(bucket_guard.boundary[1].range[1], 2729);
        }

        // Send requests with char_count 20
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(20)))
            .unwrap();

        tokio::time::sleep(Duration::from_secs(3)).await;
        {
            let bucket_guard = policy.bucket.read().unwrap();
            assert_eq!(bucket_guard.boundary[0].range[1], 20);
            assert_eq!(bucket_guard.boundary[1].range[1], 27);
        }

        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(7)))
            .unwrap();

        tokio::time::sleep(Duration::from_secs(3)).await;
        {
            let bucket_guard = policy.bucket.read().unwrap();
            assert_eq!(bucket_guard.boundary[0].range[1], 7);
            assert_eq!(bucket_guard.boundary[1].range[1], 10);
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
            let bucket_guard = policy.bucket.read().unwrap();
            assert_eq!(bucket_guard.boundary[0].range[1], 1364);
            assert_eq!(bucket_guard.boundary[1].range[1], 2729);
        }

        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(5)))
            .unwrap();
        policy
            .select_worker(&prefill_workers, Some(&*"a".repeat(10))).
            unwrap();
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

        tokio::time::sleep(Duration::from_secs(3)).await;
        {
            let bucket_guard = policy.bucket.read().unwrap();
            assert_eq!(bucket_guard.boundary[0].range[1], 20);
            assert_eq!(bucket_guard.boundary[1].range[1], 26);
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

        tokio::time::sleep(Duration::from_secs(3)).await;
        {
            let bucket_guard = policy.bucket.read().unwrap();
            assert_eq!(bucket_guard.boundary[0].range[1], 20);
            assert_eq!(bucket_guard.boundary[1].range[1], 26);
        }
    }
}
