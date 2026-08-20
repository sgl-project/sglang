// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Router-owned Worker reporter sessions and immutable load snapshots.

pub mod proto;

use crate::config::LoadMonitorConfig;
use crate::discovery::{WorkerId, WorkerMode};
use crate::workers::Worker;
use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use parking_lot::RwLock;
use proto::load_monitor_service_client::LoadMonitorServiceClient;
use proto::{
    router_frame, worker_frame, KeepAlive, LoadReport, RankLoad, RegisterRequest, ReportStatus,
    RouterFrame, StopRequest,
};
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU16;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::transport::Endpoint;
use tonic::Status;

/// Requested engine report cadence.
pub const REPORT_INTERVAL: Duration = Duration::from_secs(1);
/// Router-receipt age after which a report stops being schedulable.
pub const STALE_AFTER: Duration = Duration::from_secs(3);
/// Engine-side reporting lease renewed by the Router.
pub const LEASE_TTL: Duration = Duration::from_secs(15);
/// Timeout applied while connecting to one Worker reporter.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
/// Timeout for the Worker's registration ack after the stream is established.
pub const ACK_TIMEOUT: Duration = Duration::from_secs(2);
/// Initial gRPC reconnection delay.
pub const RECONNECT_INITIAL: Duration = Duration::from_millis(200);
/// Maximum exponential gRPC reconnection delay before jitter.
pub const RECONNECT_MAX: Duration = Duration::from_secs(5);
/// Maximum random delay added to reconnection retries.
pub const RECONNECT_JITTER_MAX: Duration = Duration::from_millis(500);

/// Lightweight internal result used while validating and binding an ingest
/// report.
type IngestResult<T> = std::result::Result<T, Box<Status>>;

/// Boxes a gRPC status for propagation through internal ingest helpers.
///
/// The caller supplies the fully classified status. Boxing keeps validation
/// error paths small without changing the public snapshot types.
fn ingest_status(status: Status) -> Box<Status> {
    Box::new(status)
}

/// Freshness classification exposed by immutable monitor snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// The worker has not delivered any report since it was registered.
    Missing,
    /// The report is explicit stale, locally invalid for scheduling, or too old.
    Stale,
    /// Engine cannot obtain load; ages to [`Freshness::Stale`] after [`STALE_AFTER`].
    Unreachable,
    /// The report is healthy and younger than [`STALE_AFTER`].
    Fresh,
}

/// Fully owned per-rank load values retained for diagnostics.
#[derive(Debug, Clone, PartialEq)]
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
}

/// Aggregated worker load used by policies and exposed for diagnostics.
#[derive(Debug, Clone, Default, PartialEq)]
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
}

/// Owned worker entry returned by one immutable snapshot capture.
#[derive(Debug, Clone, PartialEq)]
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

/// Internal immutable view captured under one store read lock.
#[derive(Debug, Clone, PartialEq)]
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
    reporter_endpoint: String,
    mode: WorkerMode,
    model_ids: Vec<String>,
}

impl WorkerTarget {
    /// Builds the reporting identity and paired reporter endpoint for a Worker.
    ///
    /// Port: `/server_info` value wins, else `fallback_port`; neither => not
    /// monitored.
    ///
    /// # Errors
    ///
    /// Missing reporter port or an unparsable worker URL.
    fn from_worker(worker: &Arc<Worker>, fallback_port: Option<NonZeroU16>) -> Result<Self> {
        let reporter_port = worker
            .reporter_port()
            .and_then(NonZeroU16::new)
            .or(fallback_port)
            .ok_or_else(|| {
                anyhow!(
                    "worker has no load reporter port (absent from /server_info and no \
                     --load-reporter-port); load monitoring disabled for this worker"
                )
            })?;
        Ok(Self {
            id: worker.id.clone(),
            url: worker.url.clone(),
            reporter_endpoint: reporter_endpoint(&worker.url, reporter_port)?,
            mode: worker.mode(),
            model_ids: worker
                .model_ids
                .iter()
                .map(|model| model.0.clone())
                .collect(),
        })
    }

    /// Returns whether an existing client task still targets the same Worker.
    fn same_identity(&self, other: &Self) -> bool {
        self.url == other.url
            && self.reporter_endpoint == other.reporter_endpoint
            && self.mode == other.mode
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
}

impl WorkerState {
    /// Creates a missing-load state for a newly discovered worker.
    fn new(target: WorkerTarget) -> Self {
        Self {
            target,
            report: None,
            active_source: None,
            active_session: None,
        }
    }
}

#[derive(Debug, Default)]
struct StoreState {
    version: u64,
    workers: HashMap<WorkerId, WorkerState>,
}

#[derive(Debug)]
struct ReporterTask {
    identity: WorkerTarget,
    cancel: CancellationToken,
    handle: JoinHandle<()>,
}

#[derive(Debug)]
struct MonitorInner {
    config: LoadMonitorConfig,
    router_id: String,
    store: RwLock<StoreState>,
    sessions: Mutex<HashMap<WorkerId, ReporterTask>>,
    next_session: AtomicU64,
    shutting_down: AtomicBool,
}

/// Shared load-monitor handle used by discovery and snapshot consumers.
#[derive(Debug, Clone)]
pub struct LoadMonitor {
    inner: Arc<MonitorInner>,
}

impl LoadMonitor {
    /// Constructs a disabled monitor with no Worker reporter sessions.
    pub fn disabled() -> Self {
        Self::new(LoadMonitorConfig::default())
    }

    /// Constructs a monitor from validated Router configuration.
    pub fn new(config: LoadMonitorConfig) -> Self {
        Self {
            inner: Arc::new(MonitorInner {
                config,
                router_id: format!("sgl-router-{}", uuid::Uuid::new_v4()),
                store: RwLock::new(StoreState::default()),
                sessions: Mutex::new(HashMap::new()),
                next_session: AtomicU64::new(1),
                shutting_down: AtomicBool::new(false),
            }),
        }
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
    /// per-Worker outbound reporter tasks.
    ///
    /// Workers whose URL and role are unchanged preserve accepted reports,
    /// sequence state, and retired sources. Removed or identity-changed workers
    /// lose that state and have their prior reporter task cancelled.
    pub async fn reconcile(&self, workers: Vec<Arc<Worker>>) {
        if !self.enabled() || self.inner.shutting_down.load(Ordering::Acquire) {
            return;
        }
        let fallback_port = self.inner.config.reporter_port;
        let mut targets = HashMap::new();
        for worker in workers {
            match WorkerTarget::from_worker(&worker, fallback_port) {
                Ok(target) => {
                    targets.insert(target.id.clone(), target);
                }
                Err(_error) if worker.reporter_port().is_none() && fallback_port.is_none() => {
                    tracing::debug!(
                        worker_id = %worker.id,
                        worker_url = %worker.url,
                        "load monitor: Worker has no load reporter port; monitoring disabled for this Worker",
                    );
                }
                Err(error) => tracing::error!(
                    worker_id = %worker.id,
                    worker_url = %worker.url,
                    error = %error,
                    "load monitor: cannot derive Worker reporter endpoint",
                ),
            }
        }
        let task_targets = self.update_store(targets);
        self.reconcile_reporter_tasks(task_targets).await;
    }

    /// Updates Worker snapshot state and excludes duplicate endpoints from
    /// client-task creation.
    fn update_store(
        &self,
        mut targets: HashMap<WorkerId, WorkerTarget>,
    ) -> HashMap<WorkerId, WorkerTarget> {
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
        if changed {
            store.version = store.version.wrapping_add(1);
        }
        drop(store);

        let mut endpoint_members: HashMap<String, Vec<WorkerId>> = HashMap::new();
        for target in targets.values() {
            endpoint_members
                .entry(target.reporter_endpoint.clone())
                .or_default()
                .push(target.id.clone());
        }
        for (endpoint, ids) in endpoint_members {
            if ids.len() > 1 {
                tracing::error!(
                    %endpoint,
                    worker_ids = ?ids,
                    "load monitor: duplicate reporter endpoint; skipping all duplicate sessions",
                );
                for id in ids {
                    targets.remove(&id);
                }
            }
        }
        targets
    }

    /// Reconciles one long-lived outbound gRPC task per unique Worker endpoint.
    async fn reconcile_reporter_tasks(&self, targets: HashMap<WorkerId, WorkerTarget>) {
        let mut tasks = self.inner.sessions.lock().await;
        let stale_ids = tasks
            .iter()
            .filter_map(|(id, task)| {
                let keep = targets
                    .get(id)
                    .is_some_and(|target| task.identity.same_identity(target))
                    && !task.handle.is_finished();
                (!keep).then(|| id.clone())
            })
            .collect::<Vec<_>>();
        let mut stale_handles = Vec::with_capacity(stale_ids.len());
        for id in stale_ids {
            if let Some(task) = tasks.remove(&id) {
                task.cancel.cancel();
                stale_handles.push(task.handle);
            }
        }
        drop(tasks);
        for handle in stale_handles {
            let _ = handle.await;
        }

        let mut tasks = self.inner.sessions.lock().await;
        for (id, target) in targets {
            if tasks.contains_key(&id) {
                continue;
            }
            let cancel = CancellationToken::new();
            let monitor = self.clone();
            let target_for_task = target.clone();
            let cancel_for_task = cancel.clone();
            let handle = tokio::spawn(async move {
                monitor
                    .run_reporter_loop(target_for_task, cancel_for_task)
                    .await;
            });
            tasks.insert(
                id,
                ReporterTask {
                    identity: target,
                    cancel,
                    handle,
                },
            );
        }
    }

    /// Connects and reconnects one Worker's reporter until cancellation.
    async fn run_reporter_loop(&self, target: WorkerTarget, cancel: CancellationToken) {
        let mut backoff = RECONNECT_INITIAL;
        loop {
            let started = tokio::time::Instant::now();
            let result = self.monitor_once(&target, &cancel).await;
            if cancel.is_cancelled() {
                return;
            }
            if let Err(error) = result {
                tracing::warn!(
                    worker_id = %target.id,
                    reporter_endpoint = %target.reporter_endpoint,
                    error = %error,
                    "load monitor: Worker reporter session ended; reconnecting",
                );
            }
            if started.elapsed() >= REPORT_INTERVAL {
                backoff = RECONNECT_INITIAL;
            }
            let jitter_ms =
                rand::thread_rng().gen_range(0..=RECONNECT_JITTER_MAX.as_millis() as u64);
            let delay = backoff + Duration::from_millis(jitter_ms);
            backoff = (backoff * 2).min(RECONNECT_MAX);
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(delay) => {}
            }
        }
    }

    /// Runs one Router-initiated bidirectional reporter session.
    ///
    /// # Errors
    ///
    /// Returns connection, handshake, stream, or report-validation failures so
    /// the caller can reconnect with bounded backoff.
    async fn monitor_once(&self, target: &WorkerTarget, cancel: &CancellationToken) -> Result<()> {
        let endpoint = Endpoint::from_shared(target.reporter_endpoint.clone())?
            .connect_timeout(CONNECT_TIMEOUT);
        let channel = tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            result = endpoint.connect() => result?,
        };
        let mut client = LoadMonitorServiceClient::new(channel);
        let (request_tx, request_rx) = mpsc::channel(8);
        request_tx
            .send(RouterFrame {
                payload: Some(router_frame::Payload::Register(RegisterRequest {
                    router_id: self.inner.router_id.clone(),
                    report_interval_ms: REPORT_INTERVAL.as_millis() as i64,
                    lease_ttl_ms: LEASE_TTL.as_millis() as i64,
                })),
            })
            .await
            .map_err(|_| anyhow!("reporter request stream closed before registration"))?;
        let response = tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            result = client.monitor(ReceiverStream::new(request_rx)) => result?,
        };
        let mut stream = response.into_inner();
        let first = tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            _ = tokio::time::sleep(ACK_TIMEOUT) => {
                return Err(anyhow!(
                    "Worker did not send registration ack within {ACK_TIMEOUT:?}"
                ));
            }
            result = stream.message() => result?,
        }
        .ok_or_else(|| anyhow!("Worker closed reporter stream before registration ack"))?;
        let registered = match first.payload {
            Some(worker_frame::Payload::Registered(registered)) => registered,
            Some(worker_frame::Payload::Error(error)) => {
                return Err(anyhow!(
                    "Worker rejected registration: {}: {}",
                    error.code,
                    error.message
                ));
            }
            _ => return Err(anyhow!("first WorkerFrame must be registered or error")),
        };
        if registered.lease_ttl_ms <= 0
            || registered.renew_after_ms <= 0
            || registered.renew_after_ms >= registered.lease_ttl_ms
        {
            return Err(anyhow!(
                "Worker returned invalid lease timing: renew_after_ms must be positive and less than lease_ttl_ms"
            ));
        }
        let renew_after = Duration::from_millis(registered.renew_after_ms as u64);
        let session = self.begin_session(&target.id)?;
        let result = self
            .run_registered_session(
                target,
                session,
                request_tx,
                &mut stream,
                renew_after,
                cancel,
            )
            .await;
        self.end_session(&target.id, session);
        result
    }

    /// Processes reports and keep-alive deadlines for an acknowledged session.
    ///
    /// # Errors
    ///
    /// Returns stream, protocol, send, or load-report validation failures.
    async fn run_registered_session(
        &self,
        target: &WorkerTarget,
        session: u64,
        request_tx: mpsc::Sender<RouterFrame>,
        stream: &mut tonic::Streaming<proto::WorkerFrame>,
        renew_after: Duration,
        cancel: &CancellationToken,
    ) -> Result<()> {
        let keep_alive = tokio::time::sleep(renew_after);
        tokio::pin!(keep_alive);
        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    let _ = request_tx.try_send(RouterFrame {
                        payload: Some(router_frame::Payload::Stop(StopRequest {})),
                    });
                    return Ok(());
                }
                _ = &mut keep_alive => {
                    request_tx.send(RouterFrame {
                        payload: Some(router_frame::Payload::KeepAlive(KeepAlive {})),
                    }).await.map_err(|_| anyhow!("Worker closed reporter request stream"))?;
                    keep_alive.as_mut().reset(tokio::time::Instant::now() + renew_after);
                }
                frame = stream.message() => {
                    let frame = frame?.ok_or_else(|| anyhow!("Worker closed reporter response stream"))?;
                    match frame.payload {
                        Some(worker_frame::Payload::Report(report)) => {
                            self.apply_report(&target.id, session, report)
                                .map_err(|status| anyhow!("invalid Worker report ({:?}): {}", status.code(), status.message()))?;
                        }
                        Some(worker_frame::Payload::Error(error)) => {
                            return Err(anyhow!("Worker reporter error: {}: {}", error.code, error.message));
                        }
                        Some(worker_frame::Payload::Registered(_)) => {
                            return Err(anyhow!("Worker sent duplicate registration ack"));
                        }
                        None => return Err(anyhow!("Worker sent an empty reporter frame")),
                    }
                }
            }
        }
    }

    /// Marks a newly acknowledged outbound stream as the Worker's active session.
    ///
    /// # Errors
    ///
    /// Returns `not_found` when discovery removed the Worker during handshake.
    fn begin_session(&self, worker_id: &WorkerId) -> IngestResult<u64> {
        let session = self.inner.next_session.fetch_add(1, Ordering::Relaxed);
        let mut store = self.inner.store.write();
        let state = store.workers.get_mut(worker_id).ok_or_else(|| {
            ingest_status(Status::not_found("worker was removed during handshake"))
        })?;
        state.active_session = Some(session);
        Ok(session)
    }

    /// Validates and publishes one report for its discovery-bound Worker.
    ///
    /// Report metadata never chooses the Worker identity: `worker_id` comes
    /// from the outbound task created by discovery. Duplicate or out-of-order
    /// sequences from the same source are ignored.
    fn apply_report(
        &self,
        worker_id: &WorkerId,
        session: u64,
        report: LoadReport,
    ) -> IngestResult<()> {
        let accepted = validate_report(&report, SystemTime::now())?;
        let mut store = self.inner.store.write();
        let state = store
            .workers
            .get_mut(worker_id)
            .ok_or_else(|| ingest_status(Status::not_found("worker was removed")))?;
        if state.active_session != Some(session) {
            return Err(ingest_status(Status::aborted(
                "reporter session was superseded",
            )));
        }
        let same_source =
            state.active_source.as_deref() == Some(accepted.source_instance_id.as_str());
        if same_source
            && state
                .report
                .as_ref()
                .is_some_and(|current| accepted.sequence_id <= current.sequence_id)
        {
            return Ok(());
        }
        state.active_source = Some(accepted.source_instance_id.clone());
        state.report = Some(accepted);
        store.version = store.version.wrapping_add(1);
        Ok(())
    }

    /// Clears the active marker only when it still belongs to this session.
    fn end_session(&self, worker_id: &WorkerId, session: u64) {
        let mut store = self.inner.store.write();
        if let Some(state) = store.workers.get_mut(worker_id) {
            if state.active_session == Some(session) {
                state.active_session = None;
            }
        }
    }

    /// Cancels all Worker sessions, sends best-effort stop frames, and joins tasks.
    pub async fn shutdown(&self) {
        self.inner.shutting_down.store(true, Ordering::Release);
        let mut tasks = self.inner.sessions.lock().await;
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
}

/// Converts a discovered inference URL into its paired h2c reporter endpoint.
///
/// # Errors
///
/// Returns an error when the inference URL is invalid or has no host.
fn reporter_endpoint(worker_url: &str, reporter_port: NonZeroU16) -> Result<String> {
    let parsed = url::Url::parse(worker_url)
        .with_context(|| format!("invalid Worker URL {worker_url:?}"))?;
    match parsed.host() {
        Some(url::Host::Ipv6(host)) => Ok(format!("http://[{host}]:{reporter_port}")),
        Some(host) => Ok(format!("http://{host}:{reporter_port}")),
        None => Err(anyhow!("Worker URL {worker_url:?} has no host")),
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
    for (name, value) in [("gen_throughput", rank.gen_throughput)] {
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
        ReportStatus::Unreachable if age >= STALE_AFTER => Freshness::Stale,
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
mod tests;
