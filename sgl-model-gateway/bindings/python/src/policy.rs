//! Gateway load-balancing policies for offline callers such as the simulator.
//!
//! `PyPolicy` runs a policy from `smg::policies` against workers described by
//! `PyWorkerInfo`, so a caller replaying a workload routes it with the same
//! code the gateway runs. The caller owns every worker's load: it is written
//! into the worker before each selection instead of being counted by the
//! request lifecycle.

use std::{
    collections::HashMap,
    sync::{atomic::Ordering, Arc},
};

use pyo3::{exceptions::PyValueError, prelude::*};
use smg::{
    config::{ManualAssignmentMode, PolicyConfig},
    core::{BasicWorker, BasicWorkerBuilder, HashRing, Worker, WorkerType},
    policies::{
        BucketPolicy, CacheAwarePolicy, LoadBalancingPolicy, PolicyFactory, SelectWorkerInfo,
    },
};

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyWorkerInfo {
    url: String,
    /// In-flight requests, the unit of `Worker::load`.
    load: usize,
    /// "regular", "prefill" or "decode". Cache-aware keeps one tree per pool.
    worker_type: String,
}

#[pymethods]
impl PyWorkerInfo {
    #[new]
    #[pyo3(signature = (url, load = 0, worker_type = String::from("regular")))]
    fn new(url: String, load: usize, worker_type: String) -> PyResult<Self> {
        parse_worker_type(&worker_type)?;
        Ok(Self {
            url,
            load,
            worker_type,
        })
    }
}

fn parse_worker_type(worker_type: &str) -> PyResult<WorkerType> {
    match worker_type {
        "regular" => Ok(WorkerType::Regular),
        "prefill" => Ok(WorkerType::Prefill {
            bootstrap_port: None,
        }),
        "decode" => Ok(WorkerType::Decode),
        other => Err(PyValueError::new_err(format!(
            "Unknown worker type {other:?}; expected regular, prefill or decode"
        ))),
    }
}

#[pyclass]
pub struct PyPolicy {
    policy: Arc<dyn LoadBalancingPolicy>,
    /// Workers persist across calls: policies key their state by URL, and the
    /// circuit breaker and processed counters live on the worker.
    workers: HashMap<String, Arc<BasicWorker>>,
    ring: Option<(Vec<String>, Arc<HashRing>)>,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl PyPolicy {
    /// Defaults match `sglang_router.router_args.RouterArgs`, i.e. what
    /// `launch_router` runs with when a flag is not given.
    #[new]
    #[pyo3(signature = (
        name,
        cache_threshold = 0.3,
        balance_abs_threshold = 64,
        balance_rel_threshold = 1.5,
        eviction_interval_secs = 60,
        max_tree_size = 2usize.pow(26),
        max_idle_secs = 4 * 3600,
        assignment_mode = String::from("random"),
        bucket_adjust_interval_secs = 5,
        load_check_interval_secs = 5,
        prefix_token_count = 256,
        load_factor = 1.25,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
        max_idle_secs: u64,
        assignment_mode: String,
        bucket_adjust_interval_secs: usize,
        load_check_interval_secs: u64,
        prefix_token_count: usize,
        load_factor: f64,
    ) -> PyResult<Self> {
        let config = match name {
            "random" => PolicyConfig::Random,
            "round_robin" => PolicyConfig::RoundRobin,
            "cache_aware" => PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            },
            "power_of_two" => PolicyConfig::PowerOfTwo {
                load_check_interval_secs,
            },
            "bucket" => PolicyConfig::Bucket {
                balance_abs_threshold,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            },
            "manual" => PolicyConfig::Manual {
                eviction_interval_secs,
                max_idle_secs,
                assignment_mode: match assignment_mode.as_str() {
                    "random" => ManualAssignmentMode::Random,
                    "min_load" => ManualAssignmentMode::MinLoad,
                    "min_group" => ManualAssignmentMode::MinGroup,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown assignment mode {other:?}"
                        )))
                    }
                },
            },
            "consistent_hashing" => PolicyConfig::ConsistentHashing,
            "prefix_hash" => PolicyConfig::PrefixHash {
                prefix_token_count,
                load_factor,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown routing policy {other:?}"
                )))
            }
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            policy: PolicyFactory::create_from_config(&config),
            workers: HashMap::new(),
            ring: None,
            runtime,
        })
    }

    fn name(&self) -> &'static str {
        self.policy.name()
    }

    fn needs_request_text(&self) -> bool {
        self.policy.needs_request_text()
    }

    /// Register the pool before the first request, as the gateway does when
    /// workers are added: cache-aware seeds its tree, bucket its boundaries.
    fn init_workers(&mut self, workers: Vec<PyRef<'_, PyWorkerInfo>>) -> PyResult<()> {
        let workers = self.resolve(&workers)?;
        let any = self.policy.as_any();
        if let Some(cache_aware) = any.downcast_ref::<CacheAwarePolicy>() {
            cache_aware.init_workers(&workers);
        } else if let Some(bucket) = any.downcast_ref::<BucketPolicy>() {
            bucket.init_prefill_worker_urls(&workers);
        }
        Ok(())
    }

    /// Index into `workers` of the selected worker, or None.
    ///
    /// `ring_urls` are the workers the gateway builds its consistent-hash ring
    /// from: every worker serving the model, of both PD roles.
    #[pyo3(signature = (workers, request_text = None, tokens = None, headers = None, ring_urls = None))]
    fn select_worker(
        &mut self,
        workers: Vec<PyRef<'_, PyWorkerInfo>>,
        request_text: Option<String>,
        tokens: Option<Vec<u32>>,
        headers: Option<HashMap<String, String>>,
        ring_urls: Option<Vec<String>>,
    ) -> PyResult<Option<usize>> {
        let workers = self.resolve(&workers)?;
        let headers = headers.map(to_header_map).transpose()?;
        let hash_ring = ring_urls.map(|urls| self.ring(urls));
        let info = SelectWorkerInfo {
            request_text: request_text.as_deref(),
            tokens: tokens.as_deref(),
            headers: headers.as_ref(),
            hash_ring,
        };
        Ok(self
            .runtime
            .block_on(self.policy.select_worker(&workers, &info)))
    }
}

impl PyPolicy {
    fn resolve(&mut self, infos: &[PyRef<'_, PyWorkerInfo>]) -> PyResult<Vec<Arc<dyn Worker>>> {
        infos
            .iter()
            .map(|info| {
                let worker_type = parse_worker_type(&info.worker_type)?;
                let worker = match self.workers.get(&info.url) {
                    Some(worker) if *worker.worker_type() == worker_type => Arc::clone(worker),
                    _ => {
                        let worker = Arc::new(
                            BasicWorkerBuilder::new(info.url.clone())
                                .worker_type(worker_type)
                                .build(),
                        );
                        self.workers.insert(info.url.clone(), Arc::clone(&worker));
                        worker
                    }
                };
                worker.load_counter.store(info.load, Ordering::Relaxed);
                Ok(worker as Arc<dyn Worker>)
            })
            .collect()
    }

    fn ring(&mut self, urls: Vec<String>) -> Arc<HashRing> {
        if let Some((cached, ring)) = &self.ring {
            if *cached == urls {
                return Arc::clone(ring);
            }
        }
        let workers: Vec<Arc<dyn Worker>> = urls
            .iter()
            .map(|url| Arc::new(BasicWorkerBuilder::new(url.clone()).build()) as Arc<dyn Worker>)
            .collect();
        let ring = Arc::new(HashRing::new(&workers));
        self.ring = Some((urls, Arc::clone(&ring)));
        ring
    }
}

fn to_header_map(headers: HashMap<String, String>) -> PyResult<http::HeaderMap> {
    let mut map = http::HeaderMap::new();
    for (name, value) in headers {
        let name = http::HeaderName::from_bytes(name.as_bytes())
            .map_err(|e| PyValueError::new_err(format!("Invalid header name {name:?}: {e}")))?;
        let value = http::HeaderValue::from_str(&value)
            .map_err(|e| PyValueError::new_err(format!("Invalid header value {value:?}: {e}")))?;
        map.insert(name, value);
    }
    Ok(map)
}
