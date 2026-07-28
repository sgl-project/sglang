// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use sgl_kv_indexer::pb::kv_indexer_server::KvIndexerServer;
use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvActionType,
    GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse, MatchExternalKvRequest,
    MatchExternalKvResponse,
};
use sgl_kv_indexer::{shutdown_signal, KvIndexerBackend, KvIndexerService};
use tonic::transport::Server;
use tonic::Status;
use tracing::info;

/// A small stateful backend for joint debugging: it keeps the live set of
/// (tier, hash) blocks in memory and logs running totals on every apply, so the
/// indexer side of the SGLang -> bridge -> indexer chain is observable.
#[derive(Default)]
struct LoggingKvIndexerBackend {
    live: Mutex<HashSet<(i32, String)>>,
    seqs: Mutex<HashMap<String, u64>>,
}

impl LoggingKvIndexerBackend {
    fn total(&self) -> usize {
        self.live.lock().unwrap().len()
    }
}

#[tonic::async_trait]
impl KvIndexerBackend for LoggingKvIndexerBackend {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        let (mut reported, mut revoked, mut cleared) = (0usize, 0usize, 0usize);
        {
            let mut live = self.live.lock().unwrap();
            for action in &request.actions {
                match ExternalKvActionType::try_from(action.r#type) {
                    Ok(ExternalKvActionType::ActionReport) => {
                        for hash in &action.hashes {
                            if live.insert((action.tier, hash.clone())) {
                                reported += 1;
                            }
                        }
                    }
                    Ok(ExternalKvActionType::ActionRevoke) => {
                        for hash in &action.hashes {
                            if live.remove(&(action.tier, hash.clone())) {
                                revoked += 1;
                            }
                        }
                    }
                    Ok(ExternalKvActionType::ActionClearAllAtTier) => {
                        let before = live.len();
                        live.retain(|(tier, _)| *tier != action.tier);
                        cleared += before - live.len();
                    }
                    _ => {}
                }
            }
        }
        info!(
            worker = %request.worker_id,
            seq = request.seq,
            reported,
            revoked,
            cleared,
            live_total = self.total(),
            "APPLY external kv batch"
        );
        let checkpoint = if request.actions.is_empty() {
            self.seqs.lock().unwrap().get(&request.worker_id).copied()
        } else {
            self.seqs
                .lock()
                .unwrap()
                .insert(request.worker_id.clone(), request.seq);
            Some(request.seq)
        };
        Ok(ApplyExternalKvBatchResponse {
            last_applied_seq: checkpoint.unwrap_or_default(),
            duplicate: false,
            has_applied_seq: checkpoint.is_some(),
        })
    }

    async fn match_external_kv(
        &self,
        _request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        Ok(MatchExternalKvResponse { matches: vec![] })
    }

    async fn get_external_kv_hit_counts(
        &self,
        _request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        Ok(GetExternalKvHitCountsResponse { entries: vec![] })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let addr = std::env::var("KV_INDEXER_LISTEN_ADDR")
        .unwrap_or_else(|_| "[::1]:50051".to_string())
        .parse::<SocketAddr>()?;

    let backend = select_backend().await?;

    // Standard gRPC health service (grpc.health.v1). A background prober flips
    // the status to reflect backend readiness (e.g. Redis connectivity), so
    // k8s / the router can tell whether this indexer can actually serve.
    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    tokio::spawn(health_prober(health_reporter, backend.clone()));

    let service = KvIndexerServer::new(KvIndexerService::new(backend));

    info!(%addr, "starting SGLang KV Indexer gRPC server");
    Server::builder()
        .add_service(health_service)
        .add_service(service)
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;

    Ok(())
}

/// The concrete gRPC service type, used to name it in the health registry.
type GrpcService = KvIndexerServer<KvIndexerService<Arc<dyn KvIndexerBackend>>>;

/// How often readiness is re-probed against the backend.
const HEALTH_PROBE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// Periodically reflects backend readiness into the gRPC health service, for
/// both the named `KvIndexer` service and the overall (empty-name) status that
/// k8s' built-in gRPC probe checks by default.
async fn health_prober(
    reporter: tonic_health::server::HealthReporter,
    backend: Arc<dyn KvIndexerBackend>,
) {
    use tonic_health::ServingStatus::{NotServing, Serving};
    let mut tick = tokio::time::interval(HEALTH_PROBE_INTERVAL);
    let mut last: Option<bool> = None;
    loop {
        tick.tick().await;
        let ok = backend.health().await;
        if last == Some(ok) {
            continue;
        }
        last = Some(ok);
        let status = if ok { Serving } else { NotServing };
        reporter.set_service_status("", status).await;
        reporter
            .set_service_status(<GrpcService as tonic::server::NamedService>::NAME, status)
            .await;
        if !ok {
            tracing::warn!("backend not ready: reporting NOT_SERVING");
        }
    }
}

/// Selects the storage backend from `KV_INDEXER_BACKEND`:
///   * `logging` — in-memory, logs running totals; for joint debugging.
///   * `redis` — the Redis backend (requires the `redis-backend` cargo feature).
///
/// The variable is required: silently defaulting to a fake backend makes a
/// misconfigured production process look healthy while returning no real matches.
/// The Redis backend lives behind the feature so the default build stays light;
/// requesting `redis` without it is a loud startup error rather than a silent
/// fallback.
async fn select_backend(
) -> Result<Arc<dyn KvIndexerBackend>, Box<dyn std::error::Error + Send + Sync>> {
    let backend = match std::env::var("KV_INDEXER_BACKEND") {
        Ok(value) => value,
        Err(_) => {
            return Err(
                "KV_INDEXER_BACKEND is required; set it explicitly to redis or logging".into(),
            )
        }
    };
    match backend.as_str() {
        "logging" => {
            info!("using logging backend");
            Ok(Arc::new(LoggingKvIndexerBackend::default()))
        }
        "redis" => {
            #[cfg(feature = "redis-backend")]
            {
                info!("using redis backend");
                let backend = sgl_kv_indexer::RedisKvIndexerBackend::from_env().await?;
                Ok(Arc::new(backend))
            }
            #[cfg(not(feature = "redis-backend"))]
            {
                Err(
                    "KV_INDEXER_BACKEND=redis requires building with --features redis-backend"
                        .into(),
                )
            }
        }
        other => Err(format!("unknown KV_INDEXER_BACKEND: {other}").into()),
    }
}
