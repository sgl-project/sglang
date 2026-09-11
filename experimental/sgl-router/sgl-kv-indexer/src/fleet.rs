//! Stateless Router client: randomized replica attempts with exact coverage.

use std::collections::{HashMap, HashSet};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    RwLock,
};
use std::time::Duration;

use anyhow::{ensure, Result};
use rand::seq::SliceRandom;
use tokio::sync::Semaphore;
use tonic::transport::{Channel, Endpoint};

use crate::pb::kv_replica_client::KvReplicaClient;
use crate::pb::*;
use crate::{PrefixMatch, PrefixOutcome};

pub type StreamIdentity = (String, String, u32);
pub type Generations = HashMap<StreamIdentity, String>;

pub struct ReplicaFleet {
    source: String,
    endpoints: RwLock<Vec<(String, Channel)>>,
    descriptors: RwLock<HashMap<String, Vec<StreamDescriptor>>>,
    deadline: Duration,
    inflight: Semaphore,
    pub successful_queries: AtomicU64,
    pub fallback_queries: AtomicU64,
    pub failed_attempts: AtomicU64,
}

impl ReplicaFleet {
    /// Comma-separated endpoints, or @path to a hot-reloaded JSON string array.
    pub async fn new(source: String, deadline: Duration, max_inflight: usize) -> Result<Self> {
        ensure!(
            !deadline.is_zero() && max_inflight > 0,
            "invalid fleet query limits"
        );
        let fleet = Self {
            source,
            endpoints: RwLock::new(Vec::new()),
            descriptors: RwLock::new(HashMap::new()),
            deadline,
            inflight: Semaphore::new(max_inflight),
            successful_queries: AtomicU64::new(0),
            fallback_queries: AtomicU64::new(0),
            failed_attempts: AtomicU64::new(0),
        };
        fleet.reload().await?;
        Ok(fleet)
    }

    pub async fn reload(&self) -> Result<()> {
        let urls: Vec<String> = if let Some(path) = self.source.strip_prefix('@') {
            serde_json::from_slice(&tokio::fs::read(path).await?)?
        } else {
            self.source
                .split(',')
                .map(|s| s.trim().to_owned())
                .collect()
        };
        ensure!(urls.len() <= 128, "too many Indexer endpoints");
        let mut unique = HashSet::new();
        let mut next = Vec::new();
        let old = self.endpoints.read().unwrap();
        for url in urls {
            if !unique.insert(url.clone()) {
                continue;
            }
            let parsed = reqwest::Url::parse(&url)?;
            ensure!(
                ["http", "https"].contains(&parsed.scheme()) && parsed.host_str().is_some(),
                "invalid Indexer endpoint"
            );
            let channel = match old.iter().find(|(u, _)| *u == url) {
                Some((_, channel)) => channel.clone(),
                None => Endpoint::from_shared(url.clone())?
                    .connect_timeout(self.deadline)
                    .connect_lazy(),
            };
            next.push((url, channel));
        }
        drop(old);
        *self.endpoints.write().unwrap() = next;
        Ok(())
    }

    pub fn set_workers(&self, descriptors: HashMap<String, Vec<StreamDescriptor>>) {
        *self.descriptors.write().unwrap() = descriptors;
    }

    /// The returned match is accepted only when every eligible stream has a
    /// current, fresh load generation and the replica covers that exact set.
    pub async fn query(
        &self,
        hashes: Vec<i64>,
        model: &str,
        page_size: u32,
        is_bigram: bool,
        workers: &[String],
        generations: &Generations,
    ) -> PrefixOutcome {
        let result = self
            .query_inner(hashes, model, page_size, is_bigram, workers, generations)
            .await;
        match result {
            Some(result) => {
                self.successful_queries.fetch_add(1, Ordering::Relaxed);
                result
            }
            None => {
                self.fallback_queries.fetch_add(1, Ordering::Relaxed);
                PrefixOutcome::Empty
            }
        }
    }

    async fn query_inner(
        &self,
        mut hashes: Vec<i64>,
        model: &str,
        page_size: u32,
        is_bigram: bool,
        workers: &[String],
        generations: &Generations,
    ) -> Option<PrefixOutcome> {
        let _permit = self.inflight.try_acquire().ok()?;
        let expected: Vec<StreamDescriptor> = {
            let descriptors = self.descriptors.read().ok()?;
            let mut expected = Vec::new();
            for url in workers {
                let streams = descriptors.get(url)?;
                if streams.is_empty() {
                    return None;
                }
                expected.extend(streams.iter().cloned());
            }
            expected
        };
        let namespace = expected.first()?.key.as_ref()?.namespace.clone();
        for descriptor in &expected {
            let key = descriptor.key.as_ref()?;
            if descriptor.model != model
                || descriptor.page_size != page_size
                || descriptor.is_bigram != is_bigram
                || descriptor.hash_schema_version != 1
                || key.namespace != namespace
                || !generations.contains_key(&(
                    key.namespace.clone(),
                    key.worker_id.clone(),
                    key.dp_rank,
                ))
            {
                return None;
            }
        }
        // Reserve substantial room for explicit coverage identities in the 8MiB
        // transport envelope; policy keeps the original prompt denominator.
        hashes.truncate(750_000);
        let request = ReplicaPrefixRequest {
            namespace,
            model: model.into(),
            page_size,
            is_bigram,
            hash_schema_version: 1,
            hashes,
            eligible_streams: expected.iter().filter_map(|s| s.key.clone()).collect(),
            max_blocks: 0,
        };
        let mut endpoints = self.endpoints.read().ok()?.clone();
        endpoints.shuffle(&mut rand::thread_rng());
        let expires = tokio::time::Instant::now() + self.deadline;
        let count = endpoints.len();
        for (i, (endpoint, channel)) in endpoints.into_iter().enumerate() {
            let remaining = expires.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            // One failed replica cannot consume the entire multi-replica budget.
            let attempt = remaining / (count - i) as u32;
            let mut client =
                KvReplicaClient::new(channel).max_decoding_message_size(8 * 1024 * 1024);
            let mut rpc = tonic::Request::new(request.clone());
            rpc.set_timeout(attempt);
            let response = tokio::time::timeout(attempt, client.match_prefix(rpc)).await;
            if let Ok(Ok(response)) = response {
                let response = response.into_inner();
                if let Some(outcome) = validate_response(&request, &expected, generations, response)
                {
                    tracing::debug!(%endpoint, "accepted complete READY replica coverage");
                    return Some(outcome);
                }
            }
            self.failed_attempts.fetch_add(1, Ordering::Relaxed);
        }
        None
    }
}

fn validate_response(
    request: &ReplicaPrefixRequest,
    expected: &[StreamDescriptor],
    generations: &Generations,
    response: ReplicaPrefixResponse,
) -> Option<PrefixOutcome> {
    if !response.complete || response.coverage.len() != expected.len() {
        return None;
    }
    let mut covered = HashSet::new();
    for coverage in &response.coverage {
        let descriptor = coverage.stream.as_ref()?;
        let key = descriptor.key.as_ref()?;
        let identity = (key.namespace.clone(), key.worker_id.clone(), key.dp_rank);
        if !coverage.ready
            || !covered.insert(identity.clone())
            || generations.get(&identity)? != &coverage.worker_epoch
        {
            return None;
        }
        let configured = expected.iter().find(|d| d.key == descriptor.key)?;
        if configured.worker_address != descriptor.worker_address
            || configured.model != descriptor.model
            || configured.page_size != descriptor.page_size
            || configured.is_bigram != descriptor.is_bigram
            || configured.hash_schema_version != descriptor.hash_schema_version
        {
            return None;
        }
    }
    // An HTTP Worker may internally dispatch to any DP rank. Its safe cache
    // signal is the minimum across all its ranks, including explicit misses.
    let mut by_worker: HashMap<String, (String, u32)> = HashMap::new();
    for descriptor in expected {
        let key = descriptor.key.as_ref()?;
        let prefix = response
            .matches
            .iter()
            .filter(|m| {
                m.worker_id == key.worker_id
                    && m.dp_rank == key.dp_rank
                    && m.worker_address == descriptor.worker_address
            })
            .map(|m| m.matched_prefix_blocks)
            .min()
            .unwrap_or(0);
        if prefix as usize > request.hashes.len() {
            return None;
        }
        by_worker
            .entry(descriptor.worker_address.clone())
            .and_modify(|(_, p)| *p = (*p).min(prefix))
            .or_insert((key.worker_id.clone(), prefix));
    }
    let mut matches: Vec<_> = by_worker
        .into_iter()
        .filter(|(_, (_, prefix))| *prefix > 0)
        .map(
            |(address, (worker_id, matched_prefix_blocks))| PrefixMatch {
                address,
                worker_id,
                matched_prefix_blocks,
            },
        )
        .collect();
    matches.sort_by_key(|m| std::cmp::Reverse(m.matched_prefix_blocks));
    match matches.first() {
        Some(first) => Some(PrefixOutcome::Matched {
            best_prefix_blocks: first.matched_prefix_blocks,
            matches,
        }),
        None => Some(PrefixOutcome::Empty),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (
        ReplicaPrefixRequest,
        Vec<StreamDescriptor>,
        Generations,
        ReplicaPrefixResponse,
    ) {
        let descriptors: Vec<_> = (0..2)
            .map(|rank| StreamDescriptor {
                key: Some(StreamKey {
                    namespace: "ns".into(),
                    worker_id: "worker".into(),
                    dp_rank: rank,
                }),
                worker_address: "http://worker".into(),
                model: "model".into(),
                page_size: 4,
                hash_schema_version: 1,
                ..Default::default()
            })
            .collect();
        let request = ReplicaPrefixRequest {
            namespace: "ns".into(),
            hashes: vec![1, 2, 3],
            ..Default::default()
        };
        let generations = (0..2)
            .map(|rank| (("ns".into(), "worker".into(), rank), format!("epoch{rank}")))
            .collect();
        let response = ReplicaPrefixResponse {
            complete: true,
            coverage: descriptors
                .iter()
                .map(|descriptor| StreamCoverage {
                    stream: Some(descriptor.clone()),
                    worker_epoch: format!("epoch{}", descriptor.key.as_ref().unwrap().dp_rank),
                    ready: true,
                    watermark: 10,
                })
                .collect(),
            matches: (0..2)
                .map(|rank| ReplicaPrefixMatch {
                    worker_address: "http://worker".into(),
                    worker_id: "worker".into(),
                    dp_rank: rank,
                    matched_prefix_blocks: 3 - rank,
                })
                .collect(),
        };
        (request, descriptors, generations, response)
    }

    #[test]
    fn multi_rank_routing_uses_minimum_including_misses() {
        let (request, descriptors, generations, mut response) = fixture();
        assert!(matches!(
            validate_response(&request, &descriptors, &generations, response.clone()),
            Some(PrefixOutcome::Matched {
                best_prefix_blocks: 2,
                ..
            })
        ));
        response.matches.pop();
        assert_eq!(
            validate_response(&request, &descriptors, &generations, response),
            Some(PrefixOutcome::Empty)
        );
    }

    #[test]
    fn partial_duplicate_wrong_generation_and_schema_fail_closed() {
        let (request, descriptors, generations, response) = fixture();
        let mut broken = response.clone();
        broken.coverage.pop();
        assert!(validate_response(&request, &descriptors, &generations, broken).is_none());
        let mut broken = response.clone();
        broken.coverage[1] = broken.coverage[0].clone();
        assert!(validate_response(&request, &descriptors, &generations, broken).is_none());
        let mut broken = response.clone();
        broken.coverage[0].worker_epoch = "old".into();
        assert!(validate_response(&request, &descriptors, &generations, broken).is_none());
        let mut broken = response;
        broken.coverage[0].stream.as_mut().unwrap().page_size = 8;
        assert!(validate_response(&request, &descriptors, &generations, broken).is_none());
    }
}
