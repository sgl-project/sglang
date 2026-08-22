// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Minimal Worker load integration for external KV placement routing.
//!
//! SGLang already publishes per-rank scheduler snapshots through `/v1/loads`.
//! The Router polls that upstream API and retains only a short-lived aggregate
//! per Worker. The `worker_generation` carried by the load snapshot must equal
//! the generation returned by the KV Indexer before cache placement and load
//! can be fused.

use crate::workers::WorkerRegistry;
use dashmap::DashMap;
use futures::future::join_all;
use serde::Deserialize;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(100);
pub const DEFAULT_FRESHNESS: Duration = Duration::from_millis(500);
const REQUEST_TIMEOUT: Duration = Duration::from_millis(400);

#[derive(Debug, Clone)]
pub struct WorkerLoad {
    pub worker_generation: String,
    pub total_requests: u64,
    pub token_usage: f64,
    received_at: Instant,
}

#[derive(Debug)]
pub struct WorkerLoadRegistry {
    entries: DashMap<String, WorkerLoad>,
    freshness: Duration,
}

impl Default for WorkerLoadRegistry {
    fn default() -> Self {
        Self::new(DEFAULT_FRESHNESS)
    }
}

impl WorkerLoadRegistry {
    pub fn new(freshness: Duration) -> Self {
        Self {
            entries: DashMap::new(),
            freshness,
        }
    }

    /// Return a fresh load for fallback scoring, independent of placement.
    pub fn fresh(&self, worker_address: &str) -> Option<WorkerLoad> {
        let load = self.entries.get(worker_address)?;
        (load.received_at.elapsed() <= self.freshness).then(|| load.clone())
    }

    /// Return load only when it describes the same process as the placement.
    pub fn fresh_for_generation(
        &self,
        worker_address: &str,
        placement_generation: &str,
    ) -> Option<WorkerLoad> {
        if placement_generation.is_empty() {
            return None;
        }
        self.fresh(worker_address).filter(|load| {
            !load.worker_generation.is_empty() && load.worker_generation == placement_generation
        })
    }

    fn record(&self, worker_address: String, mut load: WorkerLoad) {
        load.received_at = Instant::now();
        self.entries.insert(worker_address, load);
    }

    /// Publish one already-aggregated load sample into the registry.
    ///
    /// The HTTP poller is the production caller; keeping this transport-neutral
    /// boundary also makes the generation/freshness contract independently
    /// testable.
    pub fn update(
        &self,
        worker_address: &str,
        worker_generation: &str,
        total_requests: u64,
        token_usage: f64,
    ) -> bool {
        if worker_address.is_empty()
            || worker_generation.is_empty()
            || !token_usage.is_finite()
            || !(0.0..=1.0).contains(&token_usage)
        {
            return false;
        }
        self.record(
            worker_address.to_owned(),
            WorkerLoad {
                worker_generation: worker_generation.to_owned(),
                total_requests,
                token_usage,
                received_at: Instant::now(),
            },
        );
        true
    }

    fn retain_workers(&self, addresses: &HashSet<String>) {
        self.entries
            .retain(|address, _| addresses.contains(address));
    }
}

#[derive(Debug, Deserialize)]
struct LoadsResponse {
    loads: Vec<RankLoad>,
}

#[derive(Debug, Deserialize)]
struct RankLoad {
    #[serde(default)]
    worker_generation: String,
    #[serde(default)]
    num_running_reqs: u64,
    #[serde(default)]
    num_waiting_reqs: u64,
    #[serde(default)]
    token_usage: f64,
}

fn aggregate(response: LoadsResponse) -> Option<WorkerLoad> {
    let first = response.loads.first()?;
    if first.worker_generation.is_empty()
        || response
            .loads
            .iter()
            .any(|rank| rank.worker_generation != first.worker_generation)
    {
        return None;
    }

    let mut total_requests = 0_u64;
    let mut token_usage = 0.0_f64;
    for rank in &response.loads {
        if !rank.token_usage.is_finite() || !(0.0..=1.0).contains(&rank.token_usage) {
            return None;
        }
        total_requests = total_requests
            .saturating_add(rank.num_running_reqs)
            .saturating_add(rank.num_waiting_reqs);
        token_usage = token_usage.max(rank.token_usage);
    }

    Some(WorkerLoad {
        worker_generation: first.worker_generation.clone(),
        total_requests,
        token_usage,
        received_at: Instant::now(),
    })
}

async fn fetch(client: &reqwest::Client, worker_address: &str) -> Option<WorkerLoad> {
    let url = format!(
        "{}/v1/loads?include=core",
        worker_address.trim_end_matches('/')
    );
    let response = client.get(url).send().await.ok()?.error_for_status().ok()?;
    aggregate(response.json::<LoadsResponse>().await.ok()?)
}

async fn poll_once(client: &reqwest::Client, workers: &WorkerRegistry, loads: &WorkerLoadRegistry) {
    let worker_addresses: Vec<String> = workers
        .all()
        .into_iter()
        .map(|worker| worker.url.clone())
        .collect();
    let active: HashSet<_> = worker_addresses.iter().cloned().collect();
    loads.retain_workers(&active);

    let results = join_all(
        worker_addresses
            .iter()
            .map(|address| async move { (address.clone(), fetch(client, address).await) }),
    )
    .await;
    for (address, load) in results {
        match load {
            Some(load) => {
                loads.update(
                    &address,
                    &load.worker_generation,
                    load.total_requests,
                    load.token_usage,
                );
            }
            None => tracing::debug!(worker = %address, "Worker /v1/loads unavailable or invalid"),
        }
    }
}

pub fn spawn_poller(
    workers: Arc<WorkerRegistry>,
    loads: Arc<WorkerLoadRegistry>,
    poll_interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("Worker load HTTP client builds");
        let mut interval = tokio::time::interval(poll_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            poll_once(&client, &workers, &loads).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use axum::{routing::get, Json, Router};

    #[test]
    fn registry_fences_generation() {
        let registry = WorkerLoadRegistry::default();
        assert!(registry.update("http://worker", "generation-2", 3, 0.25));

        assert!(registry
            .fresh_for_generation("http://worker", "generation-1")
            .is_none());
        let load = registry
            .fresh_for_generation("http://worker", "generation-2")
            .expect("matching generation is fresh");
        assert_eq!(load.total_requests, 3);
    }

    #[test]
    fn aggregate_rejects_mixed_generations() {
        let response = LoadsResponse {
            loads: vec![
                RankLoad {
                    worker_generation: "g1".into(),
                    num_running_reqs: 1,
                    num_waiting_reqs: 2,
                    token_usage: 0.2,
                },
                RankLoad {
                    worker_generation: "g2".into(),
                    num_running_reqs: 3,
                    num_waiting_reqs: 4,
                    token_usage: 0.4,
                },
            ],
        };
        assert!(aggregate(response).is_none());
    }

    #[test]
    fn registry_expires_old_samples() {
        let registry = WorkerLoadRegistry::new(Duration::from_millis(10));
        registry.entries.insert(
            "http://worker".into(),
            WorkerLoad {
                worker_generation: "generation-1".into(),
                total_requests: 1,
                token_usage: 0.1,
                received_at: Instant::now() - Duration::from_millis(20),
            },
        );

        assert!(registry.fresh("http://worker").is_none());
    }

    #[tokio::test]
    async fn poller_reads_upstream_loads_shape() {
        let app = Router::new().route(
            "/v1/loads",
            get(|| async {
                Json(serde_json::json!({
                    "loads": [
                        {
                            "worker_generation": "generation-1",
                            "num_running_reqs": 2,
                            "num_waiting_reqs": 1,
                            "token_usage": 0.5
                        },
                        {
                            "worker_generation": "generation-1",
                            "num_running_reqs": 1,
                            "num_waiting_reqs": 0,
                            "token_usage": 0.75
                        }
                    ]
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let workers = WorkerRegistry::default();
        workers
            .add(WorkerSpec {
                id: WorkerId("worker".into()),
                url: format!("http://{address}"),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("model".into())],
                bootstrap_port: None,
            })
            .unwrap();
        let loads = WorkerLoadRegistry::default();
        let client = reqwest::Client::new();
        poll_once(&client, &workers, &loads).await;

        let load = loads
            .fresh_for_generation(&format!("http://{address}"), "generation-1")
            .expect("poll stores generation-fenced aggregate");
        assert_eq!(load.total_requests, 4);
        assert_eq!(load.token_usage, 0.75);
    }
}
