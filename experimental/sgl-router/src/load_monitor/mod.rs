// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Router-owned load reporting, ingestion, immutable snapshots, and renewal.

pub mod proto;

use crate::config::LoadMonitorConfig;
use crate::discovery::{WorkerId, WorkerMode};
use crate::workers::Worker;
use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use parking_lot::RwLock;
use proto::load_monitor_service_server::{LoadMonitorService, LoadMonitorServiceServer};
use proto::{LoadReport, RankLoad, ReportStatus, WorkerType};
use rand::Rng;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

/// Engine endpoint used to start or renew reporting.
pub const START_REPORTING_PATH: &str = "/v1/start_reporting";
/// Requested engine report cadence.
pub const REPORT_INTERVAL: Duration = Duration::from_secs(1);
/// Router-receipt age after which a report stops being schedulable.
pub const STALE_AFTER: Duration = Duration::from_secs(3);
/// Engine-side reporting lease renewed by the Router.
pub const LEASE_TTL: Duration = Duration::from_secs(15);
/// Timeout applied to each registration HTTP request.
pub const REGISTRATION_HTTP_TIMEOUT: Duration = Duration::from_secs(2);
/// Initial registration retry delay.
pub const RECONNECT_INITIAL: Duration = Duration::from_millis(200);
/// Maximum exponential registration retry delay before jitter.
pub const RECONNECT_MAX: Duration = Duration::from_secs(5);
/// Maximum random delay added to registration retries.
pub const RECONNECT_JITTER_MAX: Duration = Duration::from_millis(500);

/// Lightweight internal result used while validating and binding an ingest
/// stream.
///
/// Boxing keeps the error branch small; the gRPC boundary converts it back to
/// the protocol-level [`Status`] returned to the engine.
type IngestResult<T> = std::result::Result<T, Box<Status>>;

/// Boxes a gRPC status for propagation through internal ingest helpers.
///
/// The caller supplies the fully classified status, and the returned boxed
/// value is unboxed exactly once by the tonic service boundary.
fn ingest_status(status: Status) -> Box<Status> {
    Box::new(status)
}

/// Freshness classification exposed by snapshots and consumed by scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Freshness {
    /// The worker has not delivered any report since it was registered.
    Missing,
    /// The report is explicit stale, locally invalid for scheduling, or too old.
    Stale,
    /// The engine explicitly reported that it cannot obtain load.
    Unreachable,
    /// The report is healthy and younger than [`STALE_AFTER`].
    Fresh,
}

/// Fully owned per-rank load values retained for diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RankSnapshot {
    pub dp_rank: i32,
    pub snapshot_time_unix_ms: i64,
    pub num_running_reqs: u64,
    pub num_waiting_reqs: u64,
    pub num_waiting_uncached_tokens: u64,
    pub num_used_tokens: u64,
    pub num_total_tokens: u64,
    pub max_total_num_tokens: u64,
    pub max_running_requests: u64,
    pub token_usage: f64,
    pub gen_throughput: f64,
    pub cache_hit_rate: f64,
    pub utilization: f64,
    pub prefill_throughput: f64,
}

/// Aggregated worker load used by policies and exposed for diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct AggregateLoad {
    pub rank_count: usize,
    pub num_running_reqs: u64,
    pub num_waiting_reqs: u64,
    pub total_requests: u64,
    pub num_waiting_uncached_tokens: u64,
    pub num_used_tokens: u64,
    pub num_total_tokens: u64,
    pub max_total_num_tokens: u64,
    pub max_running_requests: u64,
    pub free_tokens: u64,
    pub available_slots: u64,
    pub queue_pressure: f64,
    pub request_utilization: f64,
    pub weighted_token_usage: f64,
    pub max_rank_token_usage: f64,
    pub gen_throughput: f64,
    pub prefill_throughput: f64,
}

/// Owned worker entry returned by one immutable snapshot capture.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WorkerSnapshot {
    pub worker_id: String,
    pub url: String,
    pub mode: WorkerMode,
    pub model_ids: Vec<String>,
    pub freshness: Freshness,
    pub source_instance_id: Option<String>,
    pub sequence_id: Option<u64>,
    pub report_time_unix_ms: Option<i64>,
    pub last_error: Option<String>,
    pub received_at: Option<String>,
    pub expires_at: Option<String>,
    pub aggregate: Option<AggregateLoad>,
    pub ranks: Vec<RankSnapshot>,
}

/// HTTP-facing immutable view captured under one store read lock.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LoadMonitorSnapshot {
    pub enabled: bool,
    pub version: u64,
    pub captured_at: Option<String>,
    pub workers: Vec<WorkerSnapshot>,
}

impl LoadMonitorSnapshot {
    /// Returns the fresh aggregate load for `worker_id` in this snapshot.
    ///
    /// The result is cloned so policy candidates remain fully owned and cannot
    /// observe later store mutations.
    pub fn fresh_load(&self, worker_id: &WorkerId) -> Option<AggregateLoad> {
        self.workers
            .iter()
            .find(|worker| worker.worker_id == worker_id.0 && worker.freshness == Freshness::Fresh)
            .and_then(|worker| worker.aggregate.clone())
    }
}

#[derive(Debug, Clone)]
struct WorkerTarget {
    id: WorkerId,
    url: String,
    origin: String,
    mode: WorkerMode,
    model_ids: Vec<String>,
}

impl WorkerTarget {
    /// Builds the reporting identity for one registered Router worker.
    ///
    /// # Errors
    ///
    /// Returns an error when the worker URL cannot be reduced to a unique
    /// `host:port` origin.
    fn from_worker(worker: &Arc<Worker>) -> Result<Self> {
        Ok(Self {
            id: worker.id.clone(),
            url: worker.url.clone(),
            origin: normalize_origin(&worker.url)?,
            mode: worker.mode(),
            model_ids: worker
                .model_ids
                .iter()
                .map(|model| model.0.clone())
                .collect(),
        })
    }

    /// Returns whether a target preserves the engine identity that owns load
    /// report sequence and source-retirement state.
    fn same_identity(&self, other: &Self) -> bool {
        self.url == other.url && self.mode == other.mode
    }
}

#[derive(Debug, Clone)]
struct AcceptedReport {
    source_instance_id: String,
    sequence_id: u64,
    report_time_unix_ms: i64,
    status: ReportStatus,
    last_error: Option<String>,
    received_at: SystemTime,
    locally_stale: bool,
    aggregate: AggregateLoad,
    ranks: Vec<RankSnapshot>,
}

#[derive(Debug)]
struct WorkerState {
    target: WorkerTarget,
    report: Option<AcceptedReport>,
    active_source: Option<String>,
    active_session: Option<u64>,
    retired_sources: HashSet<String>,
}

impl WorkerState {
    /// Creates a missing-load state for a newly discovered worker.
    fn new(target: WorkerTarget) -> Self {
        Self {
            target,
            report: None,
            active_source: None,
            active_session: None,
            retired_sources: HashSet::new(),
        }
    }
}

#[derive(Debug, Default)]
struct StoreState {
    version: u64,
    workers: HashMap<WorkerId, WorkerState>,
    origin_to_id: HashMap<String, WorkerId>,
    duplicate_origins: HashSet<String>,
}

#[derive(Debug)]
struct RegistrationTask {
    identity: WorkerTarget,
    cancel: CancellationToken,
    waiting_for_topology: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

#[derive(Debug)]
struct MonitorInner {
    config: LoadMonitorConfig,
    callback_port: u16,
    client: reqwest::Client,
    store: RwLock<StoreState>,
    registrations: Mutex<HashMap<WorkerId, RegistrationTask>>,
    next_session: AtomicU64,
    active_streams: AtomicUsize,
    stream_change: Notify,
    shutting_down: AtomicBool,
}

/// Shared load-monitor handle used by discovery, gRPC, HTTP, and scheduling.
#[derive(Debug, Clone)]
pub struct LoadMonitor {
    inner: Arc<MonitorInner>,
}

impl LoadMonitor {
    /// Constructs a disabled monitor used when no gRPC listener is running.
    pub fn disabled() -> Self {
        Self::new_inner(LoadMonitorConfig::default(), 0)
            .expect("disabled load-monitor HTTP client must build")
    }

    /// Constructs an enabled monitor after the gRPC listener has selected its
    /// actual callback port.
    ///
    /// # Errors
    ///
    /// Returns an error if the registration HTTP client cannot be built.
    fn new_enabled(config: LoadMonitorConfig, callback_port: u16) -> Result<Self> {
        Self::new_inner(config, callback_port)
    }

    /// Constructs the shared monitor state and bounded registration client.
    ///
    /// # Errors
    ///
    /// Returns an error if `reqwest` cannot construct a rustls HTTP client.
    fn new_inner(config: LoadMonitorConfig, callback_port: u16) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(REGISTRATION_HTTP_TIMEOUT)
            .build()
            .context("build load-monitor registration client")?;
        Ok(Self {
            inner: Arc::new(MonitorInner {
                config,
                callback_port,
                client,
                store: RwLock::new(StoreState::default()),
                registrations: Mutex::new(HashMap::new()),
                next_session: AtomicU64::new(1),
                active_streams: AtomicUsize::new(0),
                stream_change: Notify::new(),
                shutting_down: AtomicBool::new(false),
            }),
        })
    }

    /// Returns whether active load monitoring is enabled.
    pub fn enabled(&self) -> bool {
        self.inner.config.enabled
    }

    /// Captures a fully owned, deterministically sorted snapshot.
    ///
    /// Freshness is evaluated exactly once using Router wall-clock receipt
    /// time, so every consumer of the returned value observes the same view.
    pub fn snapshot(&self) -> LoadMonitorSnapshot {
        if !self.enabled() {
            return LoadMonitorSnapshot {
                enabled: false,
                version: 0,
                captured_at: None,
                workers: Vec::new(),
            };
        }
        let captured = SystemTime::now();
        let store = self.inner.store.read();
        let mut workers = store
            .workers
            .values()
            .map(|state| worker_snapshot(state, captured))
            .collect::<Vec<_>>();
        workers.sort_by(|left, right| left.worker_id.cmp(&right.worker_id));
        LoadMonitorSnapshot {
            enabled: true,
            version: store.version,
            captured_at: Some(format_time(captured)),
            workers,
        }
    }

    /// Reconciles the complete Router worker registry into monitor state and
    /// per-worker registration renewal tasks.
    ///
    /// Workers whose URL and role are unchanged preserve accepted reports,
    /// sequence state, and retired sources. Removed or identity-changed workers
    /// lose that state and have their prior renewal task cancelled.
    pub async fn reconcile(&self, workers: Vec<Arc<Worker>>) {
        if !self.enabled() || self.inner.shutting_down.load(Ordering::Acquire) {
            return;
        }
        let mut targets = HashMap::new();
        for worker in workers {
            match WorkerTarget::from_worker(&worker) {
                Ok(target) => {
                    targets.insert(target.id.clone(), target);
                }
                Err(error) => tracing::error!(
                    worker_id = %worker.id,
                    worker_url = %worker.url,
                    error = %error,
                    "load monitor: worker URL has no reportable origin",
                ),
            }
        }

        {
            let mut store = self.inner.store.write();
            let mut changed = false;
            store.workers.retain(|id, _| {
                let keep = targets.contains_key(id);
                changed |= !keep;
                keep
            });
            for (id, target) in &targets {
                match store.workers.get_mut(id) {
                    Some(state) if state.target.same_identity(target) => {
                        changed |= state.target.model_ids != target.model_ids;
                        state.target.model_ids.clone_from(&target.model_ids);
                    }
                    Some(state) => {
                        *state = WorkerState::new(target.clone());
                        changed = true;
                    }
                    None => {
                        store
                            .workers
                            .insert(id.clone(), WorkerState::new(target.clone()));
                        changed = true;
                    }
                }
            }
            let mut origin_members: HashMap<String, Vec<WorkerId>> = HashMap::new();
            for target in targets.values() {
                origin_members
                    .entry(target.origin.clone())
                    .or_default()
                    .push(target.id.clone());
            }
            let mut next_origins = HashMap::new();
            let mut duplicate_origins = HashSet::new();
            for (origin, ids) in origin_members {
                if ids.len() == 1 {
                    next_origins.insert(origin, ids[0].clone());
                } else {
                    tracing::error!(
                        %origin,
                        worker_ids = ?ids,
                        "load monitor: duplicate normalized worker origin; rejecting report streams",
                    );
                    duplicate_origins.insert(origin);
                }
            }
            if store.origin_to_id != next_origins {
                store.origin_to_id = next_origins;
                changed = true;
            }
            if store.duplicate_origins != duplicate_origins {
                store.duplicate_origins = duplicate_origins;
                changed = true;
            }
            if changed {
                store.version = store.version.wrapping_add(1);
            }
        }

        self.reconcile_registration_tasks(targets).await;
    }

    /// Stops every Start Reporting renewal without sending an explicit stop.
    ///
    /// The engine closes its gRPC stream after the existing lease expires.
    pub async fn stop_registrations(&self) {
        self.inner.shutting_down.store(true, Ordering::Release);
        let mut tasks = self.inner.registrations.lock().await;
        for task in tasks.values() {
            task.cancel.cancel();
        }
        let handles = tasks
            .drain()
            .map(|(_, task)| task.handle)
            .collect::<Vec<_>>();
        drop(tasks);
        for handle in handles {
            let _ = handle.await;
        }
    }

    /// Waits until all engine report streams have closed, bounded by one lease
    /// TTL. A timeout is expected for engines that do not honor lease expiry.
    pub async fn wait_for_streams_or_lease_expiry(&self) {
        let wait = async {
            loop {
                let changed = self.inner.stream_change.notified();
                if self.inner.active_streams.load(Ordering::Acquire) == 0 {
                    break;
                }
                changed.await;
            }
        };
        if tokio::time::timeout(LEASE_TTL, wait).await.is_err() {
            tracing::warn!(
                active_streams = self.inner.active_streams.load(Ordering::Acquire),
                "load monitor: report streams remained open after one lease TTL; forcing shutdown",
            );
        }
    }

    /// Reconciles per-worker HTTP renewal tasks against the current topology.
    async fn reconcile_registration_tasks(&self, targets: HashMap<WorkerId, WorkerTarget>) {
        let mut tasks = self.inner.registrations.lock().await;
        tasks.retain(|id, task| {
            let keep = targets
                .get(id)
                .is_some_and(|target| task.identity.same_identity(target))
                && !task.waiting_for_topology.load(Ordering::Acquire)
                && !task.handle.is_finished();
            if !keep {
                task.cancel.cancel();
                task.handle.abort();
            }
            keep
        });

        for (id, target) in targets {
            if tasks.contains_key(&id) {
                continue;
            }
            let cancel = CancellationToken::new();
            let monitor = self.clone();
            let target_for_task = target.clone();
            let cancel_for_task = cancel.clone();
            let waiting_for_topology = Arc::new(AtomicBool::new(false));
            let waiting_for_task = Arc::clone(&waiting_for_topology);
            let handle = tokio::spawn(async move {
                monitor
                    .run_registration_loop(target_for_task, cancel_for_task, waiting_for_task)
                    .await;
            });
            tasks.insert(
                id,
                RegistrationTask {
                    identity: target,
                    cancel,
                    waiting_for_topology,
                    handle,
                },
            );
        }
    }

    /// Renews one engine's reporting lease until cancellation or a terminal
    /// client-side HTTP response.
    ///
    /// `target` identifies the engine, `cancel` stops the worker task, and
    /// `waiting_for_topology` publishes terminal-4xx state to a concurrent
    /// reconcile before this task's join handle necessarily becomes finished.
    async fn run_registration_loop(
        &self,
        target: WorkerTarget,
        cancel: CancellationToken,
        waiting_for_topology: Arc<AtomicBool>,
    ) {
        let mut backoff = RECONNECT_INITIAL;
        loop {
            let result = self.register_once(&target).await;
            let delay = match result {
                Ok(RegistrationOutcome::Renewed) => {
                    backoff = RECONNECT_INITIAL;
                    REPORT_INTERVAL
                }
                Ok(RegistrationOutcome::Retry) => {
                    let jitter_ms =
                        rand::thread_rng().gen_range(0..=RECONNECT_JITTER_MAX.as_millis() as u64);
                    let delay = backoff + Duration::from_millis(jitter_ms);
                    backoff = (backoff * 2).min(RECONNECT_MAX);
                    delay
                }
                Err(error) => {
                    tracing::warn!(
                        worker_id = %target.id,
                        error = %error,
                        "load monitor: Start Reporting transport failure",
                    );
                    let jitter_ms =
                        rand::thread_rng().gen_range(0..=RECONNECT_JITTER_MAX.as_millis() as u64);
                    let delay = backoff + Duration::from_millis(jitter_ms);
                    backoff = (backoff * 2).min(RECONNECT_MAX);
                    delay
                }
                Ok(RegistrationOutcome::WaitForTopology) => {
                    // Publish the terminal state before the task completes so
                    // a concurrent topology generation cannot miss the
                    // restart window by observing an unfinished JoinHandle.
                    waiting_for_topology.store(true, Ordering::Release);
                    return;
                }
            };
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(delay) => {}
            }
        }
    }

    /// Sends one unauthenticated `/v1/start_reporting` lease request.
    ///
    /// # Errors
    ///
    /// Returns configuration or HTTP transport failures so the caller can
    /// apply bounded exponential retry.
    async fn register_once(&self, target: &WorkerTarget) -> Result<RegistrationOutcome> {
        let report_ip = self
            .inner
            .config
            .report_ip
            .as_deref()
            .ok_or_else(|| anyhow!("enabled load monitor has no report IP"))?;
        let url = format!(
            "{}{}",
            target.url.trim_end_matches('/'),
            START_REPORTING_PATH
        );
        let body = StartReportingRequest {
            ip: report_ip,
            port: self.inner.callback_port,
            report_interval_ms: REPORT_INTERVAL.as_millis() as u64,
            lease_ttl_ms: LEASE_TTL.as_millis() as u64,
        };
        let response = self.inner.client.post(&url).json(&body).send().await?;
        let status = response.status();
        if status.is_success() {
            return Ok(RegistrationOutcome::Renewed);
        }
        let response_body = response.text().await.unwrap_or_default();
        if status.as_u16() == 429 || status.is_server_error() {
            tracing::warn!(
                worker_id = %target.id,
                %status,
                body = %response_body,
                "load monitor: Start Reporting retryable response",
            );
            return Ok(RegistrationOutcome::Retry);
        }
        tracing::error!(
            worker_id = %target.id,
            %status,
            body = %response_body,
            "load monitor: Start Reporting rejected; waiting for next topology generation",
        );
        Ok(RegistrationOutcome::WaitForTopology)
    }

    /// Binds a stream identity from its first report and atomically accepts the
    /// report when it is valid.
    ///
    /// # Errors
    ///
    /// Returns gRPC `invalid_argument` for unknown origins, role mismatches,
    /// retired sources, duplicate streams, and invalid rank fields.
    fn begin_stream(&self, report: LoadReport) -> IngestResult<StreamBinding> {
        if self.inner.shutting_down.load(Ordering::Acquire) {
            return Err(ingest_status(Status::unavailable(
                "load monitor is shutting down",
            )));
        }
        let worker = report.worker.as_ref().ok_or_else(|| {
            ingest_status(Status::invalid_argument(
                "first report is missing worker identity",
            ))
        })?;
        let origin = normalize_origin(&worker.worker_addr)
            .map_err(|error| ingest_status(Status::invalid_argument(error.to_string())))?;
        let source = report.source_instance_id.clone();
        if source.is_empty() {
            return Err(ingest_status(Status::invalid_argument(
                "source_instance_id must be non-empty",
            )));
        }
        let role = WorkerType::try_from(worker.worker_type)
            .map_err(|_| ingest_status(Status::invalid_argument("unknown worker_type")))?;
        let session = self.inner.next_session.fetch_add(1, Ordering::Relaxed);
        let mut store = self.inner.store.write();
        if store.duplicate_origins.contains(&origin) {
            return Err(ingest_status(Status::invalid_argument(format!(
                "duplicate worker origin {origin}"
            ))));
        }
        let id = store.origin_to_id.get(&origin).cloned().ok_or_else(|| {
            ingest_status(Status::invalid_argument(format!(
                "unknown worker origin {origin}"
            )))
        })?;
        let state = store.workers.get_mut(&id).ok_or_else(|| {
            ingest_status(Status::invalid_argument(
                "worker was removed during stream bind",
            ))
        })?;
        if role != worker_type_for_mode(state.target.mode) {
            return Err(ingest_status(Status::invalid_argument(
                "reported worker role does not match discovery",
            )));
        }
        if state.retired_sources.contains(&source) {
            return Err(ingest_status(Status::failed_precondition(
                "source_instance_id has been retired",
            )));
        }
        if state.active_source.as_deref() == Some(source.as_str()) && state.active_session.is_some()
        {
            return Err(ingest_status(Status::already_exists(
                "duplicate stream for worker origin",
            )));
        }
        let same_source = state.active_source.as_deref() == Some(source.as_str());
        let duplicate_sequence = same_source
            && state
                .report
                .as_ref()
                .is_some_and(|current| report.sequence_id <= current.sequence_id);
        let accepted = if duplicate_sequence {
            None
        } else {
            Some(validate_report(&report, SystemTime::now())?)
        };
        if let Some(previous) = state.active_source.replace(source.clone()) {
            if previous != source {
                state.retired_sources.insert(previous);
            }
        }
        state.active_session = Some(session);
        if let Some(accepted) = accepted {
            state.report = Some(accepted);
            store.version = store.version.wrapping_add(1);
        }
        Ok(StreamBinding {
            id,
            origin,
            role,
            source,
            session,
        })
    }

    /// Applies a subsequent report after verifying immutable stream identity.
    ///
    /// Duplicate and out-of-order sequence numbers are ignored without closing
    /// the stream. A superseded stream is closed on its next message.
    fn apply_stream_report(&self, binding: &StreamBinding, report: LoadReport) -> IngestResult<()> {
        let worker = report.worker.as_ref().ok_or_else(|| {
            ingest_status(Status::invalid_argument(
                "report is missing worker identity",
            ))
        })?;
        let origin = normalize_origin(&worker.worker_addr)
            .map_err(|error| ingest_status(Status::invalid_argument(error.to_string())))?;
        let role = WorkerType::try_from(worker.worker_type)
            .map_err(|_| ingest_status(Status::invalid_argument("unknown worker_type")))?;
        if origin != binding.origin
            || role != binding.role
            || report.source_instance_id != binding.source
        {
            return Err(ingest_status(Status::invalid_argument(
                "worker_addr, worker_type, and source_instance_id must remain stable",
            )));
        }
        let mut store = self.inner.store.write();
        let state = store
            .workers
            .get_mut(&binding.id)
            .ok_or_else(|| ingest_status(Status::not_found("worker was removed")))?;
        if state.active_session != Some(binding.session) {
            return Err(ingest_status(Status::aborted(
                "stream source was superseded",
            )));
        }
        if state
            .report
            .as_ref()
            .is_some_and(|current| report.sequence_id <= current.sequence_id)
        {
            return Ok(());
        }
        state.report = Some(validate_report(&report, SystemTime::now())?);
        store.version = store.version.wrapping_add(1);
        Ok(())
    }

    /// Clears the active stream marker only when it still belongs to the
    /// ending stream session.
    fn end_stream(&self, binding: &StreamBinding) {
        let mut store = self.inner.store.write();
        if let Some(state) = store.workers.get_mut(&binding.id) {
            if state.active_session == Some(binding.session) {
                state.active_session = None;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegistrationOutcome {
    Renewed,
    Retry,
    WaitForTopology,
}

#[derive(Debug, Serialize)]
struct StartReportingRequest<'a> {
    ip: &'a str,
    port: u16,
    report_interval_ms: u64,
    lease_ttl_ms: u64,
}

#[derive(Debug)]
struct StreamBinding {
    id: WorkerId,
    origin: String,
    role: WorkerType,
    source: String,
    session: u64,
}

#[derive(Debug)]
struct StreamCountGuard {
    inner: Arc<MonitorInner>,
}

impl StreamCountGuard {
    /// Increments the live gRPC stream count for graceful shutdown tracking.
    fn new(inner: Arc<MonitorInner>) -> Self {
        inner.active_streams.fetch_add(1, Ordering::AcqRel);
        Self { inner }
    }
}

impl Drop for StreamCountGuard {
    /// Decrements the live stream count and wakes shutdown waiters.
    fn drop(&mut self) {
        self.inner.active_streams.fetch_sub(1, Ordering::AcqRel);
        self.inner.stream_change.notify_waiters();
    }
}

#[tonic::async_trait]
impl LoadMonitorService for LoadMonitor {
    /// Receives one engine's client-streaming load reports.
    ///
    /// The first message binds immutable stream identity; later messages must
    /// preserve it. Sequence duplicates are ignored, while invalid identity or
    /// rank data closes the stream with a precise gRPC status.
    async fn report(
        &self,
        request: Request<tonic::Streaming<LoadReport>>,
    ) -> Result<Response<()>, Status> {
        let _count = StreamCountGuard::new(Arc::clone(&self.inner));
        let mut stream = request.into_inner();
        let first = stream
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("report stream is empty"))?;
        let binding = self.begin_stream(first).map_err(|status| *status)?;
        let result = async {
            while let Some(report) = stream.message().await? {
                self.apply_stream_report(&binding, report)
                    .map_err(|status| *status)?;
            }
            Ok(Response::new(()))
        }
        .await;
        self.end_stream(&binding);
        result
    }
}

/// Running gRPC server and its cancellation handle.
#[derive(Debug)]
pub struct GrpcServerHandle {
    local_addr: SocketAddr,
    cancel: CancellationToken,
    join: JoinHandle<Result<(), tonic::transport::Error>>,
}

impl GrpcServerHandle {
    /// Returns the actual bound listener address, including an ephemeral port.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Stops renewals, waits one lease window for streams, then terminates the
    /// gRPC server and joins its task.
    pub async fn shutdown(self, monitor: &LoadMonitor) {
        monitor.stop_registrations().await;
        monitor.wait_for_streams_or_lease_expiry().await;
        self.cancel.cancel();
        match self.join.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => tracing::error!(%error, "load monitor gRPC server failed"),
            Err(error) => tracing::error!(%error, "load monitor gRPC task failed"),
        }
    }
}

/// Binds and starts the independent load-monitor gRPC listener.
///
/// Binding completes before the monitor is returned, so registration requests
/// always advertise the actual listening port.
///
/// # Errors
///
/// Returns an error if the address cannot bind, the local address cannot be
/// read, or the registration client cannot be constructed.
pub async fn bind_and_serve(config: LoadMonitorConfig) -> Result<(LoadMonitor, GrpcServerHandle)> {
    if !config.enabled {
        return Err(anyhow!("cannot bind a disabled load monitor"));
    }
    let bind = format!("{}:{}", config.bind_host, config.bind_port);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("bind load-monitor gRPC listener {bind}"))?;
    let local_addr = listener
        .local_addr()
        .context("read load-monitor local address")?;
    let monitor = LoadMonitor::new_enabled(config, local_addr.port())?;
    let service = LoadMonitorServiceServer::new(monitor.clone());
    let cancel = CancellationToken::new();
    let cancel_for_server = cancel.clone();
    let join = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
                cancel_for_server.cancelled().await;
            })
            .await
    });
    Ok((
        monitor,
        GrpcServerHandle {
            local_addr,
            cancel,
            join,
        },
    ))
}

/// Converts a worker URL or reported address into canonical `host:port` form.
///
/// # Errors
///
/// Returns an error for missing hosts, missing ports, unsupported URL syntax,
/// or values that cannot be parsed even after adding an `http://` prefix.
fn normalize_origin(value: &str) -> Result<String> {
    let normalized = if value.contains("://") {
        value.to_owned()
    } else {
        format!("http://{value}")
    };
    let parsed = url::Url::parse(&normalized)
        .with_context(|| format!("invalid worker address {value:?}"))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow!("worker address {value:?} has no host"))?;
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| anyhow!("worker address {value:?} has no port"))?;
    if host.contains(':') {
        Ok(format!("[{host}]:{port}"))
    } else {
        Ok(format!("{host}:{port}"))
    }
}

/// Maps Router discovery roles to the protobuf worker role contract.
fn worker_type_for_mode(mode: WorkerMode) -> WorkerType {
    match mode {
        WorkerMode::Plain => WorkerType::Regular,
        WorkerMode::Prefill => WorkerType::Prefill,
        WorkerMode::Decode => WorkerType::Decode,
    }
}

/// Validates and converts all ranks for one worker report into an owned value.
///
/// # Errors
///
/// Returns `invalid_argument` for an invalid source or timestamp, unspecified
/// status, malformed unreachable payload, duplicate ranks, negative counters,
/// invalid capacity relations, or non-finite/negative throughput values.
fn validate_report(report: &LoadReport, received_at: SystemTime) -> IngestResult<AcceptedReport> {
    if report.source_instance_id.is_empty() {
        return Err(ingest_status(Status::invalid_argument(
            "source_instance_id must be non-empty",
        )));
    }
    if report.report_time_unix_ms < 0 {
        return Err(ingest_status(Status::invalid_argument(
            "report_time_unix_ms must be non-negative",
        )));
    }
    let status = ReportStatus::try_from(report.status)
        .map_err(|_| ingest_status(Status::invalid_argument("unknown report status")))?;
    if status == ReportStatus::Unspecified {
        return Err(ingest_status(Status::invalid_argument(
            "report status must be specified",
        )));
    }
    let mut seen = HashSet::new();
    let mut ranks = Vec::with_capacity(report.ranks.len());
    if status == ReportStatus::Unreachable {
        if !report.ranks.is_empty() {
            return Err(ingest_status(Status::invalid_argument(
                "unreachable report must not contain ranks",
            )));
        }
        if report.last_error.as_deref().is_none_or(str::is_empty) {
            return Err(ingest_status(Status::invalid_argument(
                "unreachable report must contain last_error",
            )));
        }
    } else {
        if report.ranks.is_empty() {
            return Err(ingest_status(Status::invalid_argument(
                "healthy or stale report must contain at least one DP rank",
            )));
        }
        for rank in &report.ranks {
            if !seen.insert(rank.dp_rank) {
                return Err(ingest_status(Status::invalid_argument(format!(
                    "duplicate dp_rank {}",
                    rank.dp_rank
                ))));
            }
            ranks.push(validate_rank(rank)?);
        }
    }
    ranks.sort_by_key(|rank| rank.dp_rank);
    let aggregate = aggregate_ranks(&ranks);
    let locally_stale = status == ReportStatus::Healthy
        && ranks
            .iter()
            .any(|rank| rank.max_total_num_tokens == 0 || rank.max_running_requests == 0);
    let mut last_error = report.last_error.clone();
    if locally_stale && last_error.as_deref().is_none_or(str::is_empty) {
        last_error = Some("engine reported a rank with zero token or request capacity".to_string());
    }
    Ok(AcceptedReport {
        source_instance_id: report.source_instance_id.clone(),
        sequence_id: report.sequence_id,
        report_time_unix_ms: report.report_time_unix_ms,
        status,
        last_error,
        received_at,
        locally_stale,
        aggregate,
        ranks,
    })
}

/// Validates one protobuf rank and converts signed counters to owned values.
///
/// # Errors
///
/// Returns `invalid_argument` for negative counters, capacity violations,
/// duplicate handling performed by the caller, or non-finite/negative floats.
fn validate_rank(rank: &RankLoad) -> IngestResult<RankSnapshot> {
    if rank.dp_rank < 0 {
        return Err(ingest_status(Status::invalid_argument(
            "dp_rank must be non-negative",
        )));
    }
    if rank.snapshot_time_unix_ms < 0 {
        return Err(ingest_status(Status::invalid_argument(
            "snapshot_time_unix_ms must be non-negative",
        )));
    }
    let counters = [
        ("num_running_reqs", rank.num_running_reqs),
        ("num_waiting_reqs", rank.num_waiting_reqs),
        (
            "num_waiting_uncached_tokens",
            rank.num_waiting_uncached_tokens,
        ),
        ("num_used_tokens", rank.num_used_tokens),
        ("num_total_tokens", rank.num_total_tokens),
        ("max_total_num_tokens", rank.max_total_num_tokens),
        ("max_running_requests", rank.max_running_requests),
    ];
    for (name, value) in counters {
        if value < 0 {
            return Err(ingest_status(Status::invalid_argument(format!(
                "{name} must be non-negative"
            ))));
        }
    }
    if rank.num_used_tokens > rank.max_total_num_tokens {
        return Err(ingest_status(Status::invalid_argument(
            "num_used_tokens cannot exceed max_total_num_tokens",
        )));
    }
    if rank.num_running_reqs > rank.max_running_requests {
        return Err(ingest_status(Status::invalid_argument(
            "num_running_reqs cannot exceed max_running_requests",
        )));
    }
    let finite_floats = [
        ("token_usage", rank.token_usage),
        ("cache_hit_rate", rank.cache_hit_rate),
        ("utilization", rank.utilization),
    ];
    for (name, value) in finite_floats {
        if !value.is_finite() {
            return Err(ingest_status(Status::invalid_argument(format!(
                "{name} must be finite"
            ))));
        }
    }
    for (name, value) in [
        ("gen_throughput", rank.gen_throughput),
        ("prefill_throughput", rank.prefill_throughput),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(ingest_status(Status::invalid_argument(format!(
                "{name} must be finite and non-negative"
            ))));
        }
    }
    Ok(RankSnapshot {
        dp_rank: rank.dp_rank,
        snapshot_time_unix_ms: rank.snapshot_time_unix_ms,
        num_running_reqs: rank.num_running_reqs as u64,
        num_waiting_reqs: rank.num_waiting_reqs as u64,
        num_waiting_uncached_tokens: rank.num_waiting_uncached_tokens as u64,
        num_used_tokens: rank.num_used_tokens as u64,
        num_total_tokens: rank.num_total_tokens as u64,
        max_total_num_tokens: rank.max_total_num_tokens as u64,
        max_running_requests: rank.max_running_requests as u64,
        token_usage: rank.token_usage,
        gen_throughput: rank.gen_throughput,
        cache_hit_rate: rank.cache_hit_rate,
        utilization: rank.utilization,
        prefill_throughput: rank.prefill_throughput,
    })
}

/// Sums rank loads and derives scheduling and diagnostic utilization values.
fn aggregate_ranks(ranks: &[RankSnapshot]) -> AggregateLoad {
    let mut aggregate = AggregateLoad {
        rank_count: ranks.len(),
        ..AggregateLoad::default()
    };
    let mut weighted_token_usage = 0.0;
    for rank in ranks {
        aggregate.num_running_reqs += rank.num_running_reqs;
        aggregate.num_waiting_reqs += rank.num_waiting_reqs;
        aggregate.num_waiting_uncached_tokens += rank.num_waiting_uncached_tokens;
        aggregate.num_used_tokens += rank.num_used_tokens;
        aggregate.num_total_tokens += rank.num_total_tokens;
        aggregate.max_total_num_tokens += rank.max_total_num_tokens;
        aggregate.max_running_requests += rank.max_running_requests;
        aggregate.gen_throughput += rank.gen_throughput;
        aggregate.prefill_throughput += rank.prefill_throughput;
        weighted_token_usage += rank.token_usage * rank.max_total_num_tokens as f64;
        aggregate.max_rank_token_usage = aggregate.max_rank_token_usage.max(rank.token_usage);
    }
    aggregate.total_requests = aggregate
        .num_running_reqs
        .saturating_add(aggregate.num_waiting_reqs);
    aggregate.free_tokens = aggregate
        .max_total_num_tokens
        .saturating_sub(aggregate.num_used_tokens);
    aggregate.available_slots = aggregate
        .max_running_requests
        .saturating_sub(aggregate.num_running_reqs);
    if aggregate.gen_throughput > 0.0 {
        aggregate.queue_pressure =
            aggregate.num_waiting_uncached_tokens as f64 / aggregate.gen_throughput;
    }
    if aggregate.max_running_requests > 0 {
        aggregate.request_utilization =
            aggregate.num_running_reqs as f64 / aggregate.max_running_requests as f64;
    }
    if aggregate.max_total_num_tokens > 0 {
        aggregate.weighted_token_usage =
            weighted_token_usage / aggregate.max_total_num_tokens as f64;
    }
    aggregate
}

/// Creates one worker's diagnostic entry at a fixed capture time.
fn worker_snapshot(state: &WorkerState, captured: SystemTime) -> WorkerSnapshot {
    let Some(report) = &state.report else {
        return WorkerSnapshot {
            worker_id: state.target.id.0.clone(),
            url: state.target.url.clone(),
            mode: state.target.mode,
            model_ids: state.target.model_ids.clone(),
            freshness: Freshness::Missing,
            source_instance_id: None,
            sequence_id: None,
            report_time_unix_ms: None,
            last_error: None,
            received_at: None,
            expires_at: None,
            aggregate: None,
            ranks: Vec::new(),
        };
    };
    let age = captured
        .duration_since(report.received_at)
        .unwrap_or(Duration::ZERO);
    let freshness = match report.status {
        ReportStatus::Unreachable => Freshness::Unreachable,
        ReportStatus::Stale | ReportStatus::Unspecified => Freshness::Stale,
        ReportStatus::Healthy if report.locally_stale || age >= STALE_AFTER => Freshness::Stale,
        ReportStatus::Healthy => Freshness::Fresh,
    };
    WorkerSnapshot {
        worker_id: state.target.id.0.clone(),
        url: state.target.url.clone(),
        mode: state.target.mode,
        model_ids: state.target.model_ids.clone(),
        freshness,
        source_instance_id: Some(report.source_instance_id.clone()),
        sequence_id: Some(report.sequence_id),
        report_time_unix_ms: Some(report.report_time_unix_ms),
        last_error: report.last_error.clone(),
        received_at: Some(format_time(report.received_at)),
        expires_at: Some(format_time(report.received_at + STALE_AFTER)),
        aggregate: Some(report.aggregate.clone()),
        ranks: report.ranks.clone(),
    }
}

/// Formats a system time as an RFC3339 UTC diagnostic timestamp.
fn format_time(time: SystemTime) -> String {
    let millis = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as i64;
    DateTime::<Utc>::from_timestamp_millis(millis)
        .unwrap_or(DateTime::<Utc>::UNIX_EPOCH)
        .to_rfc3339_opts(SecondsFormat::Millis, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerSpec};

    /// Builds one Router worker for store and snapshot tests.
    fn test_worker(id: &str, mode: WorkerMode) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.to_string()),
            url: format!("http://{id}:30000"),
            mode,
            model_ids: vec![ModelId("model".to_string())],
            bootstrap_port: None,
        }))
    }

    /// Builds a healthy report with one valid rank.
    fn test_report(origin: &str, source: &str, sequence: u64, mode: WorkerMode) -> LoadReport {
        LoadReport {
            source_instance_id: source.to_string(),
            sequence_id: sequence,
            report_time_unix_ms: 123,
            worker: Some(proto::Worker {
                worker_addr: origin.to_string(),
                worker_type: worker_type_for_mode(mode) as i32,
                model: Some("model".to_string()),
                zone: None,
            }),
            status: ReportStatus::Healthy as i32,
            last_error: None,
            ranks: vec![RankLoad {
                dp_rank: 0,
                snapshot_time_unix_ms: 123,
                num_running_reqs: 2,
                num_waiting_reqs: 3,
                num_waiting_uncached_tokens: 4,
                num_used_tokens: 20,
                num_total_tokens: 24,
                max_total_num_tokens: 100,
                max_running_requests: 10,
                token_usage: 0.2,
                gen_throughput: 5.0,
                cache_hit_rate: 0.5,
                utilization: 0.7,
                prefill_throughput: 6.0,
            }],
        }
    }

    /// Creates an enabled in-memory monitor without starting network servers.
    fn test_monitor() -> LoadMonitor {
        LoadMonitor::new_enabled(
            LoadMonitorConfig {
                enabled: true,
                bind_host: "127.0.0.1".to_string(),
                bind_port: 0,
                report_ip: Some("127.0.0.1".to_string()),
            },
            12345,
        )
        .unwrap()
    }

    /// Disabled snapshots preserve the documented exact empty shape.
    #[test]
    fn disabled_snapshot_has_exact_empty_shape() {
        let json = serde_json::to_value(LoadMonitor::disabled().snapshot()).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"enabled":false,"version":0,"captured_at":null,"workers":[]})
        );
    }

    /// Rank aggregation sums counters and both throughput fields.
    #[test]
    fn aggregate_sums_rank_loads() {
        let first =
            validate_rank(&test_report("w:30000", "s", 1, WorkerMode::Plain).ranks[0]).unwrap();
        let mut second = first.clone();
        second.dp_rank = 1;
        let aggregate = aggregate_ranks(&[first, second]);
        assert_eq!(aggregate.total_requests, 10);
        assert_eq!(aggregate.free_tokens, 160);
        assert_eq!(aggregate.available_slots, 16);
        assert_eq!(aggregate.gen_throughput, 10.0);
        assert_eq!(aggregate.prefill_throughput, 12.0);
    }

    /// Invalid prefill throughput is rejected before the store changes.
    #[test]
    fn rejects_non_finite_prefill_throughput() {
        let mut rank = test_report("w:30000", "s", 1, WorkerMode::Plain).ranks[0];
        rank.prefill_throughput = f64::NAN;
        assert!(validate_rank(&rank).is_err());
    }

    /// Invalid counts, capacity relations, duplicate ranks, and floats are
    /// rejected before any worker snapshot can be replaced.
    #[test]
    fn rejects_invalid_rank_contract_categories() {
        let base = test_report("worker:30000", "source", 1, WorkerMode::Plain);
        let mut cases = Vec::new();

        let mut negative_count = base.clone();
        negative_count.ranks[0].num_running_reqs = -1;
        cases.push(("negative count", negative_count));

        let mut token_capacity = base.clone();
        token_capacity.ranks[0].num_used_tokens = 101;
        cases.push(("token capacity", token_capacity));

        let mut request_capacity = base.clone();
        request_capacity.ranks[0].num_running_reqs = 11;
        cases.push(("request capacity", request_capacity));

        let mut duplicate_rank = base.clone();
        duplicate_rank.ranks.push(duplicate_rank.ranks[0]);
        cases.push(("duplicate rank", duplicate_rank));

        let mut infinite_metric = base;
        infinite_metric.ranks[0].utilization = f64::INFINITY;
        cases.push(("infinite metric", infinite_metric));

        for (category, report) in cases {
            let error = validate_report(&report, SystemTime::now()).unwrap_err();
            assert_eq!(
                error.code(),
                tonic::Code::InvalidArgument,
                "category {category}"
            );
        }
    }

    /// Negative engine timestamps are rejected even though Router receipt
    /// time remains authoritative for freshness.
    #[test]
    fn rejects_negative_report_timestamp() {
        let mut report = test_report("w:30000", "s", 1, WorkerMode::Plain);
        report.report_time_unix_ms = -1;
        assert!(validate_report(&report, SystemTime::now()).is_err());
    }

    /// Unreachable reports carry only an explanatory error and no rank set.
    #[test]
    fn validates_unreachable_report_shape() {
        let mut report = test_report("w:30000", "s", 1, WorkerMode::Plain);
        report.status = ReportStatus::Unreachable as i32;
        report.last_error = Some("scheduler unavailable".to_string());
        assert!(validate_report(&report, SystemTime::now()).is_err());

        report.ranks.clear();
        assert!(validate_report(&report, SystemTime::now()).is_ok());
        report.last_error = None;
        assert!(validate_report(&report, SystemTime::now()).is_err());
    }

    /// Duplicate and out-of-order sequences leave the latest accepted report.
    #[tokio::test]
    async fn ignores_non_increasing_sequence_without_closing_stream() {
        let monitor = test_monitor();
        monitor
            .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
            .await;
        let binding = monitor
            .begin_stream(test_report("worker:30000", "source", 2, WorkerMode::Plain))
            .unwrap();
        monitor
            .apply_stream_report(
                &binding,
                test_report("worker:30000", "source", 1, WorkerMode::Plain),
            )
            .unwrap();
        assert_eq!(monitor.snapshot().workers[0].sequence_id, Some(2));
        monitor.stop_registrations().await;
    }

    /// Duplicate discovery origins remain visible but cannot bind a report
    /// stream to an arbitrary WorkerId.
    #[tokio::test]
    async fn duplicate_origin_rejects_stream_binding() {
        let monitor = test_monitor();
        let first = test_worker("worker", WorkerMode::Plain);
        let second = Arc::new(Worker::new(WorkerSpec {
            id: WorkerId("worker-copy".to_string()),
            url: first.url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("model".to_string())],
            bootstrap_port: None,
        }));
        monitor.reconcile(vec![first, second]).await;

        let error = monitor
            .begin_stream(test_report("worker:30000", "source", 1, WorkerMode::Plain))
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert!(error.message().contains("duplicate worker origin"));
        assert_eq!(monitor.snapshot().workers.len(), 2);
        monitor.stop_registrations().await;
    }

    /// First-message lookup and subsequent identity checks reject unknown,
    /// role-mismatched, or changing report streams.
    #[tokio::test]
    async fn stream_identity_is_bound_and_immutable() {
        let monitor = test_monitor();
        monitor
            .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
            .await;

        let unknown = test_report("unknown:30000", "source", 1, WorkerMode::Plain);
        assert!(monitor.begin_stream(unknown).is_err());
        let wrong_role = test_report("worker:30000", "source", 1, WorkerMode::Decode);
        assert!(monitor.begin_stream(wrong_role).is_err());

        let binding = monitor
            .begin_stream(test_report("worker:30000", "source", 1, WorkerMode::Plain))
            .unwrap();
        let changed_source = test_report("worker:30000", "other-source", 2, WorkerMode::Plain);
        assert!(monitor
            .apply_stream_report(&binding, changed_source)
            .is_err());
        let changed_origin = test_report("other:30000", "source", 2, WorkerMode::Plain);
        assert!(monitor
            .apply_stream_report(&binding, changed_origin)
            .is_err());
        monitor.end_stream(&binding);
        monitor.stop_registrations().await;
    }

    /// A new source takes ownership and permanently retires the previous one.
    #[tokio::test]
    async fn new_source_retires_old_source_until_worker_recreated() {
        let monitor = test_monitor();
        monitor
            .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
            .await;
        let old = monitor
            .begin_stream(test_report("worker:30000", "old", 1, WorkerMode::Plain))
            .unwrap();
        let _new = monitor
            .begin_stream(test_report("worker:30000", "new", 1, WorkerMode::Plain))
            .unwrap();
        assert!(monitor
            .apply_stream_report(
                &old,
                test_report("worker:30000", "old", 2, WorkerMode::Plain)
            )
            .is_err());
        assert!(monitor
            .begin_stream(test_report("worker:30000", "old", 3, WorkerMode::Plain))
            .is_err());
        monitor.stop_registrations().await;
    }

    /// Healthy reports with zero capacity retain diagnostics but are stale.
    #[tokio::test]
    async fn zero_capacity_healthy_report_is_locally_stale() {
        let monitor = test_monitor();
        monitor
            .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
            .await;
        let mut report = test_report("worker:30000", "source", 1, WorkerMode::Plain);
        report.ranks[0].num_running_reqs = 0;
        report.ranks[0].num_used_tokens = 0;
        report.ranks[0].max_running_requests = 0;
        report.ranks[0].max_total_num_tokens = 0;
        monitor.begin_stream(report).unwrap();
        let snapshot = monitor.snapshot();
        assert_eq!(snapshot.workers[0].freshness, Freshness::Stale);
        assert_eq!(snapshot.workers[0].ranks.len(), 1);
        assert_eq!(
            snapshot.workers[0].last_error.as_deref(),
            Some("engine reported a rank with zero token or request capacity")
        );
        monitor.stop_registrations().await;
    }

    /// Freshness expiration uses Router receipt time rather than engine time.
    #[tokio::test]
    async fn freshness_expires_from_router_receipt_time() {
        let monitor = test_monitor();
        monitor
            .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
            .await;
        monitor
            .begin_stream(test_report("worker:30000", "source", 1, WorkerMode::Plain))
            .unwrap();
        {
            let mut store = monitor.inner.store.write();
            store
                .workers
                .get_mut(&WorkerId("worker".to_string()))
                .unwrap()
                .report
                .as_mut()
                .unwrap()
                .received_at = SystemTime::now() - STALE_AFTER - Duration::from_millis(1);
        }
        assert_eq!(monitor.snapshot().workers[0].freshness, Freshness::Stale);
        monitor.stop_registrations().await;
    }

    /// An owned snapshot cannot change after a later report replaces the store.
    #[tokio::test]
    async fn captured_snapshot_is_immutable_across_updates() {
        let monitor = test_monitor();
        monitor
            .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
            .await;
        let binding = monitor
            .begin_stream(test_report("worker:30000", "source", 1, WorkerMode::Plain))
            .unwrap();
        let first = monitor.snapshot();
        monitor
            .apply_stream_report(
                &binding,
                test_report("worker:30000", "source", 2, WorkerMode::Plain),
            )
            .unwrap();
        let second = monitor.snapshot();
        assert_eq!(first.workers[0].sequence_id, Some(1));
        assert_eq!(second.workers[0].sequence_id, Some(2));
        assert!(second.version > first.version);
        monitor.stop_registrations().await;
    }

    /// A request's owned candidates keep one snapshot version while a later
    /// report can switch least-load routing for the next request.
    #[tokio::test]
    async fn immutable_snapshot_drives_load_based_routing_switch() {
        use crate::policies::load_based::LoadBasedPolicy;
        use crate::policies::{policy_candidates, Policy, SelectionContext};

        let monitor = test_monitor();
        let worker_a = test_worker("worker-a", WorkerMode::Plain);
        let worker_b = test_worker("worker-b", WorkerMode::Plain);
        monitor
            .reconcile(vec![Arc::clone(&worker_a), Arc::clone(&worker_b)])
            .await;

        let mut report_a = test_report("worker-a:30000", "source-a", 1, WorkerMode::Plain);
        report_a.ranks[0].num_running_reqs = 1;
        report_a.ranks[0].num_waiting_reqs = 0;
        let mut report_b = test_report("worker-b:30000", "source-b", 1, WorkerMode::Plain);
        report_b.ranks[0].num_running_reqs = 8;
        report_b.ranks[0].num_waiting_reqs = 0;
        let binding_a = monitor.begin_stream(report_a).unwrap();
        let binding_b = monitor.begin_stream(report_b).unwrap();

        let first_snapshot = monitor.snapshot();
        let first_candidates = policy_candidates(
            vec![Arc::clone(&worker_a), Arc::clone(&worker_b)],
            &first_snapshot,
        );
        let model = ModelId("model".to_string());
        let context = SelectionContext::new(&model, None);
        let policy = LoadBasedPolicy::new();
        assert_eq!(
            policy.select(&first_candidates, &context).unwrap().id,
            worker_a.id
        );

        let mut next_a = test_report("worker-a:30000", "source-a", 2, WorkerMode::Plain);
        next_a.ranks[0].num_running_reqs = 9;
        next_a.ranks[0].num_waiting_reqs = 0;
        let mut next_b = test_report("worker-b:30000", "source-b", 2, WorkerMode::Plain);
        next_b.ranks[0].num_running_reqs = 0;
        next_b.ranks[0].num_waiting_reqs = 0;
        monitor.apply_stream_report(&binding_a, next_a).unwrap();
        monitor.apply_stream_report(&binding_b, next_b).unwrap();

        // Candidates already created for the first request remain pinned even
        // after both worker reports have changed.
        assert_eq!(
            policy.select(&first_candidates, &context).unwrap().id,
            worker_a.id
        );
        let second_snapshot = monitor.snapshot();
        assert!(second_snapshot.version > first_snapshot.version);
        let second_candidates = policy_candidates(
            vec![Arc::clone(&worker_a), Arc::clone(&worker_b)],
            &second_snapshot,
        );
        assert_eq!(
            policy.select(&second_candidates, &context).unwrap().id,
            worker_b.id
        );

        monitor.end_stream(&binding_a);
        monitor.end_stream(&binding_b);
        monitor.stop_registrations().await;
    }

    /// Exercises actual HTTP registration, an ephemeral gRPC listener, stream
    /// ingestion, and immutable snapshot publication without engine auth.
    #[tokio::test]
    async fn fake_engine_registration_and_grpc_report_form_complete_loop() {
        use axum::routing::post;
        use axum::{Json, Router};
        use tokio::sync::mpsc;

        let (registration_tx, mut registration_rx) = mpsc::channel(4);
        let app = Router::new().route(
            START_REPORTING_PATH,
            post(
                move |headers: axum::http::HeaderMap, Json(body): Json<serde_json::Value>| {
                    let registration_tx = registration_tx.clone();
                    async move {
                        registration_tx
                            .send((
                                body,
                                headers.contains_key(axum::http::header::AUTHORIZATION),
                            ))
                            .await
                            .unwrap();
                        axum::http::StatusCode::OK
                    }
                },
            ),
        );
        let engine_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let engine_addr = engine_listener.local_addr().unwrap();
        let engine_server = tokio::spawn(async move {
            axum::serve(engine_listener, app).await.unwrap();
        });

        let config = LoadMonitorConfig {
            enabled: true,
            bind_host: "127.0.0.1".to_string(),
            bind_port: 0,
            report_ip: Some("127.0.0.1".to_string()),
        };
        let (monitor, grpc) = bind_and_serve(config).await.unwrap();
        let worker = Arc::new(Worker::new(WorkerSpec {
            id: WorkerId("worker".to_string()),
            url: format!("http://{engine_addr}"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("model".to_string())],
            bootstrap_port: None,
        }));
        monitor.reconcile(vec![worker]).await;

        let (registration, has_authorization) =
            tokio::time::timeout(Duration::from_secs(3), registration_rx.recv())
                .await
                .unwrap()
                .unwrap();
        assert!(!has_authorization, "Router must not send a Bearer token");
        assert_eq!(registration["ip"], "127.0.0.1");
        assert_eq!(
            registration["port"].as_u64(),
            Some(grpc.local_addr().port() as u64)
        );
        assert_eq!(registration["report_interval_ms"].as_u64(), Some(1000));
        assert_eq!(registration["lease_ttl_ms"].as_u64(), Some(15000));

        let mut client = proto::load_monitor_service_client::LoadMonitorServiceClient::connect(
            format!("http://{}", grpc.local_addr()),
        )
        .await
        .unwrap();
        let report = test_report(
            &engine_addr.to_string(),
            "fake-engine",
            1,
            WorkerMode::Plain,
        );
        client
            .report(tokio_stream::iter(vec![report]))
            .await
            .unwrap();
        let snapshot = monitor.snapshot();
        assert_eq!(snapshot.workers[0].freshness, Freshness::Fresh);
        assert_eq!(
            snapshot.workers[0]
                .aggregate
                .as_ref()
                .unwrap()
                .total_requests,
            5
        );
        let renewal = tokio::time::timeout(Duration::from_secs(3), registration_rx.recv())
            .await
            .unwrap();
        assert!(renewal.is_some(), "Router must renew the reporting lease");

        grpc.shutdown(&monitor).await;
        engine_server.abort();
    }

    /// Retryable HTTP responses back off, terminal 4xx responses pause until
    /// the next topology reconcile, and removal clears monitor state.
    #[tokio::test]
    async fn registration_retry_terminal_response_and_removal_reconcile() {
        use axum::routing::post;
        use axum::Router;
        use tokio::sync::mpsc;

        let attempts = Arc::new(AtomicUsize::new(0));
        let (attempt_tx, mut attempt_rx) = mpsc::channel(8);
        let attempts_for_handler = Arc::clone(&attempts);
        let app = Router::new().route(
            START_REPORTING_PATH,
            post(move || {
                let attempt_tx = attempt_tx.clone();
                let attempt = attempts_for_handler.fetch_add(1, Ordering::AcqRel) + 1;
                async move {
                    attempt_tx.send(attempt).await.unwrap();
                    match attempt {
                        1 => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        2 => axum::http::StatusCode::TOO_MANY_REQUESTS,
                        3 => axum::http::StatusCode::BAD_REQUEST,
                        _ => axum::http::StatusCode::OK,
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let engine_addr = listener.local_addr().unwrap();
        let engine_server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let monitor = LoadMonitor::new_enabled(
            LoadMonitorConfig {
                enabled: true,
                bind_host: "127.0.0.1".to_string(),
                bind_port: 0,
                report_ip: Some("127.0.0.1".to_string()),
            },
            3456,
        )
        .unwrap();
        let worker = Arc::new(Worker::new(WorkerSpec {
            id: WorkerId("worker".to_string()),
            url: format!("http://{engine_addr}"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("model".to_string())],
            bootstrap_port: None,
        }));
        monitor.reconcile(vec![Arc::clone(&worker)]).await;

        for expected in 1..=3 {
            let actual = tokio::time::timeout(Duration::from_secs(3), attempt_rx.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(actual, expected);
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(1200), attempt_rx.recv())
                .await
                .is_err(),
            "terminal 4xx must pause registration until topology reconcile"
        );

        monitor.reconcile(vec![worker]).await;
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), attempt_rx.recv())
                .await
                .unwrap(),
            Some(4)
        );
        monitor.reconcile(Vec::new()).await;
        assert!(monitor.snapshot().workers.is_empty());

        monitor.stop_registrations().await;
        engine_server.abort();
    }
}
