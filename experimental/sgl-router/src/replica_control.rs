//! Embedded Router load monitor and Worker/Indexer discovery control loop.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sgl_kv_indexer::fleet::{Generations, ReplicaFleet, StreamIdentity};
use sgl_kv_indexer::load_pb::load_monitor_server::{LoadMonitor, LoadMonitorServer};
use sgl_kv_indexer::load_pb::{LoadReportAck, LoadSample};
use sgl_kv_indexer::pb::StreamDescriptor;
use tonic::{Request, Response, Status};

use crate::policies::engine_load::{EngineLoadSnapshot, NativeCacheWorkerLoad};
use crate::policies::kv_events::BlockSizeOracle;
use crate::workers::WorkerRegistry;

const FRESHNESS: Duration = Duration::from_secs(2);

struct AuthorizedStream {
    address: String,
    epoch: String,
    token: String,
    lease: Instant,
    sample: Option<(LoadSample, Instant)>,
    previous: Option<LoadSample>,
}

#[derive(Clone, Default)]
pub struct RouterLoadMonitor {
    streams: Arc<Mutex<HashMap<StreamIdentity, AuthorizedStream>>>,
}

impl RouterLoadMonitor {
    pub fn into_server(self) -> LoadMonitorServer<Self> {
        LoadMonitorServer::new(self).max_decoding_message_size(16 * 1024)
    }

    fn authorize(&self, descriptor: &StreamDescriptor, epoch: &str, token: &str) {
        let key = descriptor.key.as_ref().expect("discovered key");
        let identity = (key.namespace.clone(), key.worker_id.clone(), key.dp_rank);
        let mut streams = self.streams.lock().unwrap();
        let entry = streams.entry(identity).or_insert_with(|| AuthorizedStream {
            address: descriptor.worker_address.clone(),
            epoch: epoch.into(),
            token: token.into(),
            lease: Instant::now(),
            sample: None,
            previous: None,
        });
        if entry.epoch != epoch
            || entry.token != token
            || entry.address != descriptor.worker_address
        {
            entry.sample = None;
            entry.previous = None;
        }
        entry.address = descriptor.worker_address.clone();
        entry.epoch = epoch.into();
        entry.token = token.into();
        entry.lease = Instant::now() + Duration::from_secs(6);
    }

    fn retain(&self, identities: &HashSet<StreamIdentity>) {
        self.streams
            .lock()
            .unwrap()
            .retain(|key, _| identities.contains(key));
    }

    /// Load and generations come from one lock-protected request snapshot.
    /// A Worker is available only when all registered DP streams are fresh.
    pub fn capture(&self) -> (EngineLoadSnapshot, Generations) {
        let now = Instant::now();
        let streams = self.streams.lock().unwrap();
        let mut grouped: HashMap<&str, Vec<(&StreamIdentity, &AuthorizedStream)>> = HashMap::new();
        for (key, stream) in streams.iter() {
            grouped
                .entry(&stream.address)
                .or_default()
                .push((key, stream));
        }
        let mut loads = HashMap::new();
        let mut generations = HashMap::new();
        for (address, ranks) in grouped {
            if ranks.iter().any(|(_, s)| {
                s.lease <= now
                    || s.sample
                        .as_ref()
                        .is_none_or(|(_, t)| now.saturating_duration_since(*t) > FRESHNESS)
            }) {
                continue;
            }
            let mut load = NativeCacheWorkerLoad {
                num_running_reqs: 0,
                num_waiting_reqs: 0,
                num_waiting_uncached_tokens: 0,
                num_used_tokens: 0,
                num_total_tokens: 0,
                max_total_num_tokens: 0,
                max_running_requests: 0,
                prefill_throughput_tokens_per_s: Some(0.0),
                estimated_prefill_queue_ms: None,
                captured_at: now,
            };
            for (key, stream) in ranks {
                let (sample, at) = stream.sample.as_ref().unwrap();
                load.num_running_reqs = load
                    .num_running_reqs
                    .saturating_add(sample.num_running_reqs);
                load.num_waiting_reqs = load
                    .num_waiting_reqs
                    .saturating_add(sample.num_waiting_reqs);
                load.num_waiting_uncached_tokens = load
                    .num_waiting_uncached_tokens
                    .saturating_add(sample.num_waiting_uncached_tokens);
                load.num_used_tokens = load.num_used_tokens.saturating_add(sample.num_used_tokens);
                load.num_total_tokens = load
                    .num_total_tokens
                    .saturating_add(sample.num_total_tokens);
                load.max_total_num_tokens = load
                    .max_total_num_tokens
                    .saturating_add(sample.max_total_num_tokens);
                load.max_running_requests = load
                    .max_running_requests
                    .saturating_add(sample.max_running_requests);
                load.captured_at = load.captured_at.min(*at);
                let throughput = stream.previous.as_ref().and_then(|prev| {
                    let tokens = sample
                        .total_prefill_uncached_tokens
                        .checked_sub(prev.total_prefill_uncached_tokens)?;
                    let us = sample
                        .total_prefill_busy_us
                        .checked_sub(prev.total_prefill_busy_us)?;
                    (us > 0).then_some(tokens as f64 * 1_000_000.0 / us as f64)
                });
                load.prefill_throughput_tokens_per_s = load
                    .prefill_throughput_tokens_per_s
                    .zip(throughput)
                    .map(|(a, b)| a + b);
                generations.insert(key.clone(), stream.epoch.clone());
            }
            load.estimated_prefill_queue_ms = load
                .prefill_throughput_tokens_per_s
                .filter(|v| *v > 0.0)
                .map(|v| load.num_waiting_uncached_tokens as f64 * 1000.0 / v);
            loads.insert(address.to_owned(), load);
        }
        (
            EngineLoadSnapshot::from_native_cache_workers(0, loads),
            generations,
        )
    }

    fn receive(&self, sample: LoadSample) -> Result<(), Status> {
        if sample.sample_age_ms >= 1000
            || [
                sample.generation_throughput,
                sample.cache_hit_rate,
                sample.utilization,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(Status::invalid_argument("stale or invalid load sample"));
        }
        let key = (
            sample.namespace.clone(),
            sample.worker_id.clone(),
            sample.dp_rank,
        );
        let mut streams = self.streams.lock().unwrap();
        let entry = streams
            .get_mut(&key)
            .ok_or_else(|| Status::permission_denied("unregistered Worker"))?;
        if entry.epoch != sample.worker_epoch
            || entry.token != sample.registration_token
            || entry.lease <= Instant::now()
        {
            return Err(Status::permission_denied(
                "expired registration or Worker generation",
            ));
        }
        if entry
            .sample
            .as_ref()
            .is_some_and(|(last, _)| last.sequence >= sample.sequence)
        {
            return Ok(());
        }
        let measured = Instant::now()
            .checked_sub(Duration::from_millis(sample.sample_age_ms))
            .unwrap_or_else(Instant::now);
        entry.previous = entry.sample.take().map(|(last, _)| last);
        entry.sample = Some((sample, measured));
        Ok(())
    }
}

#[tonic::async_trait]
impl LoadMonitor for RouterLoadMonitor {
    async fn report(
        &self,
        request: Request<tonic::Streaming<LoadSample>>,
    ) -> Result<Response<LoadReportAck>, Status> {
        let mut stream = request.into_inner();
        while let Some(sample) = tokio::time::timeout(Duration::from_secs(3), stream.message())
            .await
            .map_err(|_| Status::deadline_exceeded("load stream idle"))??
        {
            self.receive(sample)?;
        }
        Ok(Response::new(LoadReportAck {}))
    }
}

pub async fn run_control(
    registry: Arc<WorkerRegistry>,
    oracle: Arc<BlockSizeOracle>,
    fleet: Arc<ReplicaFleet>,
    monitor: RouterLoadMonitor,
    target: String,
) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    let token = uuid::Uuid::new_v4().to_string();
    let mut interval = tokio::time::interval(Duration::from_secs(2));
    loop {
        interval.tick().await;
        if let Err(error) = fleet.reload().await {
            tracing::warn!(%error, "retaining last valid Indexer endpoint list");
        }
        let mut tasks = tokio::task::JoinSet::new();
        for worker in registry.all() {
            let client = client.clone();
            let target = target.clone();
            let token = token.clone();
            tasks.spawn(async move {
                let streams =
                    sgl_kv_indexer::replica_bridge::discover(&client, &worker.url).await?;
                let registration: serde_json::Value = client
                    .post(format!(
                        "{}/v1/start_reporting",
                        worker.url.trim_end_matches('/')
                    ))
                    .json(
                        &serde_json::json!({"target": target, "token": token, "lease_seconds": 6}),
                    )
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await?;
                Ok::<_, anyhow::Error>((worker.url.clone(), streams, registration))
            });
        }
        let mut descriptors = HashMap::new();
        let mut identities = HashSet::new();
        let mut duplicate_identity = false;
        while let Some(result) = tasks.join_next().await {
            match result {
                Ok(Ok((url, streams, registration))) => {
                    // Never advertise a subset of an HTTP Worker's DP ranks.
                    let complete_registration =
                        registration["streams"].as_array().is_some_and(|ranks| {
                            ranks.len() == streams.len()
                                && streams.iter().all(|stream| {
                                    let key = stream.descriptor.key.as_ref().unwrap();
                                    ranks
                                        .iter()
                                        .filter(|rank| {
                                            rank["namespace"].as_str() == Some(&key.namespace)
                                                && rank["worker_id"].as_str()
                                                    == Some(&key.worker_id)
                                                && rank["dp_rank"].as_u64()
                                                    == Some(key.dp_rank as u64)
                                                && rank["worker_epoch"]
                                                    .as_str()
                                                    .is_some_and(|epoch| !epoch.is_empty())
                                        })
                                        .count()
                                        == 1
                                })
                        });
                    if !complete_registration {
                        tracing::warn!(%url, "incomplete load registration");
                        continue;
                    }
                    let mut accepted = Vec::new();
                    for stream in streams {
                        let descriptor = stream.descriptor;
                        let key = descriptor.key.as_ref().unwrap();
                        let identity = (key.namespace.clone(), key.worker_id.clone(), key.dp_rank);
                        let epoch = registration["streams"]
                            .as_array()
                            .and_then(|ranks| {
                                ranks.iter().find(|rank| {
                                    rank["namespace"].as_str() == Some(&key.namespace)
                                        && rank["worker_id"].as_str() == Some(&key.worker_id)
                                        && rank["dp_rank"].as_u64() == Some(key.dp_rank as u64)
                                })
                            })
                            .and_then(|rank| rank["worker_epoch"].as_str());
                        let Some(epoch) = epoch else {
                            continue;
                        };
                        if oracle.try_set(descriptor.page_size).is_err() {
                            continue;
                        }
                        oracle.set_bigram(descriptor.is_bigram);
                        monitor.authorize(&descriptor, epoch, &token);
                        if !identities.insert(identity) {
                            // Duplicate identity across Worker URLs is a config
                            // error. Fail this refresh closed for the entire set.
                            duplicate_identity = true;
                            accepted.clear();
                            break;
                        }
                        accepted.push(descriptor);
                    }
                    descriptors.insert(url, accepted);
                }
                Ok(Err(error)) => tracing::warn!(%error, "Worker reporting registration failed"),
                Err(error) => tracing::warn!(%error, "Worker control task failed"),
            }
        }
        if duplicate_identity {
            descriptors.clear();
            identities.clear();
        }
        monitor.retain(&identities);
        fleet.set_workers(descriptors);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sgl_kv_indexer::pb::StreamKey;

    fn descriptor(rank: u32) -> StreamDescriptor {
        StreamDescriptor {
            key: Some(StreamKey {
                namespace: "ns".into(),
                worker_id: "worker".into(),
                dp_rank: rank,
            }),
            worker_address: "http://worker".into(),
            ..Default::default()
        }
    }

    fn sample(rank: u32, epoch: &str, sequence: u64) -> LoadSample {
        LoadSample {
            namespace: "ns".into(),
            worker_id: "worker".into(),
            dp_rank: rank,
            worker_epoch: epoch.into(),
            registration_token: "token".into(),
            sequence,
            num_running_reqs: 3,
            max_total_num_tokens: 1000,
            max_running_requests: 100,
            ..Default::default()
        }
    }

    #[test]
    fn generation_change_and_old_replays_cannot_refresh_load() {
        let monitor = RouterLoadMonitor::default();
        monitor.authorize(&descriptor(0), "epoch1", "token");
        monitor.receive(sample(0, "epoch1", 2)).unwrap();
        let (load, generations) = monitor.capture();
        assert_eq!(
            load.fresh_load_for_url("http://worker")
                .unwrap()
                .num_running_reqs,
            3
        );
        assert_eq!(generations.values().next().unwrap(), "epoch1");
        let mut old = sample(0, "epoch1", 1);
        old.num_running_reqs = 0;
        monitor.receive(old).unwrap();
        assert_eq!(
            monitor
                .capture()
                .0
                .fresh_load_for_url("http://worker")
                .unwrap()
                .num_running_reqs,
            3
        );
        monitor.authorize(&descriptor(0), "epoch2", "token");
        assert!(monitor
            .capture()
            .0
            .fresh_load_for_url("http://worker")
            .is_none());
        assert!(monitor.receive(sample(0, "epoch1", 3)).is_err());
        monitor.receive(sample(0, "epoch2", 1)).unwrap();
        assert_eq!(monitor.capture().1.values().next().unwrap(), "epoch2");
    }

    #[test]
    fn every_rank_must_be_fresh_and_registered() {
        let monitor = RouterLoadMonitor::default();
        monitor.authorize(&descriptor(0), "e0", "token");
        monitor.authorize(&descriptor(1), "e1", "token");
        monitor.receive(sample(0, "e0", 1)).unwrap();
        assert!(monitor.capture().1.is_empty());
        monitor.receive(sample(1, "e1", 1)).unwrap();
        assert_eq!(
            monitor
                .capture()
                .0
                .fresh_load_for_url("http://worker")
                .unwrap()
                .num_running_reqs,
            6
        );
        let key = ("ns".into(), "worker".into(), 1);
        monitor
            .streams
            .lock()
            .unwrap()
            .get_mut(&key)
            .unwrap()
            .sample
            .as_mut()
            .unwrap()
            .1 = Instant::now() - Duration::from_secs(3);
        assert!(monitor.capture().1.is_empty());
        assert!(monitor.receive(sample(2, "e2", 1)).is_err());
        let mut stale = sample(1, "e1", 2);
        stale.sample_age_ms = 1000;
        assert!(monitor.receive(stale).is_err());
    }
}
