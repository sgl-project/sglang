// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use crate::workers::{WireProtocol, Worker, WorkerRegistry};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;
use uuid::Uuid;

const CONTROL_CHANNEL_CAPACITY: usize = 256;
const MAX_TOMBSTONES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerSource {
    Api,
    Kubernetes,
    Static,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalWorkerSource {
    Kubernetes,
    Static,
}

impl From<ExternalWorkerSource> for WorkerSource {
    fn from(value: ExternalWorkerSource) -> Self {
        match value {
            ExternalWorkerSource::Kubernetes => Self::Kubernetes,
            ExternalWorkerSource::Static => Self::Static,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerInput {
    pub url: String,
    pub mode: WorkerMode,
    #[serde(default)]
    pub model_ids: Vec<ModelId>,
    #[serde(default)]
    pub bootstrap_port: Option<u16>,
    #[serde(default)]
    pub version_group: Option<String>,
    #[serde(default)]
    pub services: BTreeSet<String>,
}

impl WorkerInput {
    fn into_spec(self, id: WorkerId) -> WorkerSpec {
        WorkerSpec {
            id,
            url: self.url,
            mode: self.mode,
            model_ids: self.model_ids,
            bootstrap_port: self.bootstrap_port,
            version_group: self.version_group,
            services: self.services,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerControlKind {
    Create,
    Replace,
    Remove,
}

impl WorkerControlKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Replace => "replace",
            Self::Remove => "remove",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationState {
    Pending,
    Processing,
    Succeeded,
    Failed,
}

impl OperationState {
    fn is_in_progress(self) -> bool {
        matches!(self, Self::Pending | Self::Processing)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OperationView {
    pub id: String,
    pub kind: WorkerControlKind,
    pub state: OperationState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AcceptedOperation {
    pub worker_id: WorkerId,
    pub operation_id: String,
    pub revision: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerRuntimeView {
    pub registered: bool,
    pub resolved_model_ids: Vec<ModelId>,
    pub mode: WorkerMode,
    pub healthy: bool,
    pub active_requests: usize,
    pub circuit_breaker_state: &'static str,
    pub protocol: &'static str,
    pub grpc_available: bool,
}

impl WorkerRuntimeView {
    fn from_worker(worker: &Worker) -> Self {
        let breaker = worker.breaker.snapshot();
        Self {
            registered: true,
            resolved_model_ids: worker.model_ids.clone(),
            mode: worker.mode(),
            healthy: breaker.admit,
            active_requests: worker.router_inflight_load(),
            circuit_breaker_state: match breaker.state_code {
                0 => "closed",
                1 => "open",
                2 => "half_open",
                _ => unreachable!("invalid circuit breaker state"),
            },
            protocol: match worker.protocol() {
                WireProtocol::Http1 => "http1",
                WireProtocol::H2c => "h2c",
            },
            grpc_available: worker.grpc().is_some(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerView {
    pub id: WorkerId,
    pub url: String,
    pub mode: WorkerMode,
    pub model_ids: Vec<ModelId>,
    pub bootstrap_port: Option<u16>,
    pub version_group: Option<String>,
    pub services: BTreeSet<String>,
    pub source: WorkerSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation: Option<OperationView>,
    pub deleted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime: Option<WorkerRuntimeView>,
}

#[derive(Debug, Clone, Default)]
pub struct WorkerListFilter {
    pub model_id: Option<ModelId>,
    pub mode: Option<WorkerMode>,
    pub source: Option<WorkerSource>,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum WorkerControlError {
    #[error("{0}")]
    Invalid(String),
    #[error("missing If-Match worker revision")]
    MissingRevision,
    #[error("invalid If-Match worker revision")]
    InvalidRevision,
    #[error("worker {0} not found")]
    NotFound(WorkerId),
    #[error("{message}")]
    Conflict { code: &'static str, message: String },
    #[error("worker control channel unavailable")]
    Unavailable,
}

impl WorkerControlError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Invalid(_) => "invalid_worker_spec",
            Self::MissingRevision => "missing_worker_revision",
            Self::InvalidRevision => "invalid_worker_revision",
            Self::NotFound(_) => "worker_not_found",
            Self::Conflict { code, .. } => code,
            Self::Unavailable => "worker_control_unavailable",
        }
    }
}

#[derive(Debug, Clone)]
pub enum WorkerControlAction {
    Create(WorkerSpec),
    Replace { spec: WorkerSpec },
    Remove { worker_id: WorkerId },
}

impl WorkerControlAction {
    pub fn kind(&self) -> WorkerControlKind {
        match self {
            Self::Create(_) => WorkerControlKind::Create,
            Self::Replace { .. } => WorkerControlKind::Replace,
            Self::Remove { .. } => WorkerControlKind::Remove,
        }
    }

    pub fn worker_id(&self) -> &WorkerId {
        match self {
            Self::Create(spec) | Self::Replace { spec, .. } => &spec.id,
            Self::Remove { worker_id, .. } => worker_id,
        }
    }
}

#[derive(Debug)]
pub struct WorkerControlCommand {
    pub operation_id: String,
    pub action: WorkerControlAction,
    pub submitted_at: Instant,
}

impl WorkerControlCommand {
    pub fn worker_id(&self) -> WorkerId {
        self.action.worker_id().clone()
    }
}

#[derive(Debug, Clone)]
struct ManagedWorker {
    spec: WorkerSpec,
    revision: u64,
    operation: OperationView,
    deleted: bool,
}

#[derive(Debug, Default)]
struct ControlState {
    workers: HashMap<WorkerId, ManagedWorker>,
    operation_workers: HashMap<String, WorkerId>,
    tombstones: VecDeque<WorkerId>,
}

#[derive(Debug)]
pub struct WorkerControlService {
    registry: Arc<WorkerRegistry>,
    external_source: ExternalWorkerSource,
    tx: mpsc::Sender<WorkerControlCommand>,
    state: Mutex<ControlState>,
}

impl WorkerControlService {
    pub fn new(
        registry: Arc<WorkerRegistry>,
        external_source: ExternalWorkerSource,
    ) -> (Arc<Self>, mpsc::Receiver<WorkerControlCommand>) {
        let (tx, rx) = mpsc::channel(CONTROL_CHANNEL_CAPACITY);
        (
            Arc::new(Self {
                registry,
                external_source,
                tx,
                state: Mutex::new(ControlState::default()),
            }),
            rx,
        )
    }

    pub(crate) fn registry(&self) -> Arc<WorkerRegistry> {
        Arc::clone(&self.registry)
    }

    pub(crate) fn has_managed_url(&self, url: &str) -> bool {
        self.state
            .lock()
            .unwrap()
            .workers
            .iter()
            .any(|(id, record)| {
                !record.deleted
                    && ((record.operation.state != OperationState::Failed
                        && worker_urls_equal(&record.spec.url, url))
                        || self
                            .registry
                            .get(id)
                            .is_some_and(|worker| worker_urls_equal(&worker.url, url)))
            })
    }

    pub fn create(&self, mut input: WorkerInput) -> Result<AcceptedOperation, WorkerControlError> {
        input.url = normalized_worker_url(&input.url)?;
        let worker_id = WorkerId(Uuid::new_v4().to_string());
        let operation_id = Uuid::new_v4().to_string();
        let spec = input.into_spec(worker_id.clone());
        let operation = OperationView {
            id: operation_id.clone(),
            kind: WorkerControlKind::Create,
            state: OperationState::Pending,
            error: None,
        };
        let command = WorkerControlCommand {
            operation_id: operation_id.clone(),
            action: WorkerControlAction::Create(spec.clone()),
            submitted_at: Instant::now(),
        };

        let mut state = self.state.lock().unwrap();
        self.reject_duplicate_url(&state, &spec.url, None)?;
        state.workers.insert(
            worker_id.clone(),
            ManagedWorker {
                spec,
                revision: 1,
                operation,
                deleted: false,
            },
        );
        state
            .operation_workers
            .insert(operation_id.clone(), worker_id.clone());
        if self.tx.try_send(command).is_err() {
            state.workers.remove(&worker_id);
            state.operation_workers.remove(&operation_id);
            return Err(WorkerControlError::Unavailable);
        }
        Ok(AcceptedOperation {
            worker_id,
            operation_id,
            revision: 1,
        })
    }

    pub fn replace(
        &self,
        worker_id: &WorkerId,
        mut input: WorkerInput,
        expected_revision: u64,
    ) -> Result<AcceptedOperation, WorkerControlError> {
        input.url = normalized_worker_url(&input.url)?;
        let mut state = self.state.lock().unwrap();
        let previous = self.api_worker(&state, worker_id)?.clone();
        ensure_mutation_ready(&previous, expected_revision)?;
        if input.url != previous.spec.url {
            return Err(conflict(
                "worker_url_immutable",
                "worker URL is immutable; delete and recreate the worker",
            ));
        }
        self.reject_duplicate_url(&state, &input.url, Some(worker_id))?;

        let operation_id = Uuid::new_v4().to_string();
        let revision = previous.revision + 1;
        let spec = input.into_spec(worker_id.clone());
        let operation = OperationView {
            id: operation_id.clone(),
            kind: WorkerControlKind::Replace,
            state: OperationState::Pending,
            error: None,
        };
        let command = WorkerControlCommand {
            operation_id: operation_id.clone(),
            action: WorkerControlAction::Replace { spec: spec.clone() },
            submitted_at: Instant::now(),
        };
        state.operation_workers.remove(&previous.operation.id);
        state.workers.insert(
            worker_id.clone(),
            ManagedWorker {
                spec,
                revision,
                operation,
                deleted: false,
            },
        );
        state
            .operation_workers
            .insert(operation_id.clone(), worker_id.clone());
        if self.tx.try_send(command).is_err() {
            state.operation_workers.remove(&operation_id);
            state
                .operation_workers
                .insert(previous.operation.id.clone(), worker_id.clone());
            state.workers.insert(worker_id.clone(), previous);
            return Err(WorkerControlError::Unavailable);
        }
        Ok(AcceptedOperation {
            worker_id: worker_id.clone(),
            operation_id,
            revision,
        })
    }

    pub fn remove(
        &self,
        worker_id: &WorkerId,
        expected_revision: u64,
    ) -> Result<AcceptedOperation, WorkerControlError> {
        let mut state = self.state.lock().unwrap();
        let previous = self.api_worker(&state, worker_id)?.clone();
        ensure_mutation_ready(&previous, expected_revision)?;

        let operation_id = Uuid::new_v4().to_string();
        let revision = previous.revision + 1;
        let operation = OperationView {
            id: operation_id.clone(),
            kind: WorkerControlKind::Remove,
            state: OperationState::Pending,
            error: None,
        };
        let command = WorkerControlCommand {
            operation_id: operation_id.clone(),
            action: WorkerControlAction::Remove {
                worker_id: worker_id.clone(),
            },
            submitted_at: Instant::now(),
        };
        state.operation_workers.remove(&previous.operation.id);
        let current = state.workers.get_mut(worker_id).unwrap();
        current.revision = revision;
        current.operation = operation;
        state
            .operation_workers
            .insert(operation_id.clone(), worker_id.clone());
        if self.tx.try_send(command).is_err() {
            state.operation_workers.remove(&operation_id);
            state
                .operation_workers
                .insert(previous.operation.id.clone(), worker_id.clone());
            state.workers.insert(worker_id.clone(), previous);
            return Err(WorkerControlError::Unavailable);
        }
        Ok(AcceptedOperation {
            worker_id: worker_id.clone(),
            operation_id,
            revision,
        })
    }

    pub fn get(&self, worker_id: &WorkerId) -> Option<WorkerView> {
        let managed = self.state.lock().unwrap().workers.get(worker_id).cloned();
        if let Some(record) = managed {
            return Some(self.managed_view(record));
        }
        self.registry
            .get(worker_id)
            .map(|worker| self.external_view(&worker))
    }

    pub fn list(&self, filter: &WorkerListFilter) -> Vec<WorkerView> {
        let records: Vec<ManagedWorker> = self
            .state
            .lock()
            .unwrap()
            .workers
            .values()
            .filter(|record| !record.deleted)
            .cloned()
            .collect();
        let managed_ids: HashSet<WorkerId> = records
            .iter()
            .map(|record| record.spec.id.clone())
            .collect();
        let mut views: Vec<WorkerView> = records
            .into_iter()
            .map(|record| self.managed_view(record))
            .chain(
                self.registry
                    .all()
                    .into_iter()
                    .filter(|worker| !managed_ids.contains(&worker.id))
                    .map(|worker| self.external_view(&worker)),
            )
            .filter(|view| matches_filter(view, filter))
            .collect();
        views.sort_by(|a, b| a.id.0.cmp(&b.id.0));
        views
    }

    pub fn mark_processing(&self, operation_id: &str) {
        self.update_operation(operation_id, OperationState::Processing, None);
    }

    pub fn mark_succeeded(&self, operation_id: &str) {
        let mut state = self.state.lock().unwrap();
        let Some(worker_id) = state.operation_workers.get(operation_id).cloned() else {
            return;
        };
        let Some(record) = state.workers.get_mut(&worker_id) else {
            return;
        };
        if record.operation.id != operation_id {
            return;
        }
        record.operation.state = OperationState::Succeeded;
        record.operation.error = None;
        if record.operation.kind == WorkerControlKind::Remove {
            record.deleted = true;
            state.tombstones.push_back(worker_id);
            while state.tombstones.len() > MAX_TOMBSTONES {
                if let Some(expired) = state.tombstones.pop_front() {
                    if let Some(removed) = state.workers.remove(&expired) {
                        state.operation_workers.remove(&removed.operation.id);
                    }
                }
            }
        }
    }

    pub fn mark_failed(&self, operation_id: &str, error: String) {
        self.update_operation(operation_id, OperationState::Failed, Some(error));
    }

    fn update_operation(
        &self,
        operation_id: &str,
        operation_state: OperationState,
        error: Option<String>,
    ) {
        let mut state = self.state.lock().unwrap();
        let Some(worker_id) = state.operation_workers.get(operation_id) else {
            return;
        };
        let worker_id = worker_id.clone();
        let Some(record) = state.workers.get_mut(&worker_id) else {
            return;
        };
        if record.operation.id == operation_id {
            record.operation.state = operation_state;
            record.operation.error = error;
        }
    }

    fn api_worker<'a>(
        &self,
        state: &'a ControlState,
        worker_id: &WorkerId,
    ) -> Result<&'a ManagedWorker, WorkerControlError> {
        match state.workers.get(worker_id) {
            Some(record) if !record.deleted => Ok(record),
            Some(_) => Err(WorkerControlError::NotFound(worker_id.clone())),
            None if self.registry.get(worker_id).is_some() => Err(conflict(
                "worker_not_api_managed",
                format!("worker {worker_id} is managed by discovery and is read-only"),
            )),
            None => Err(WorkerControlError::NotFound(worker_id.clone())),
        }
    }

    fn reject_duplicate_url(
        &self,
        state: &ControlState,
        url: &str,
        excluding: Option<&WorkerId>,
    ) -> Result<(), WorkerControlError> {
        let managed_conflict = state.workers.values().any(|record| {
            !record.deleted && record.spec.url == url && excluding != Some(&record.spec.id)
        });
        let runtime_conflict = self
            .registry
            .all()
            .iter()
            .any(|worker| worker_urls_equal(&worker.url, url) && excluding != Some(&worker.id));
        if managed_conflict || runtime_conflict {
            return Err(conflict(
                "worker_url_conflict",
                format!("worker URL {url} is already registered"),
            ));
        }
        Ok(())
    }

    fn managed_view(&self, record: ManagedWorker) -> WorkerView {
        let runtime = (!record.deleted)
            .then(|| self.registry.get(&record.spec.id))
            .flatten()
            .map(|worker| WorkerRuntimeView::from_worker(&worker));
        WorkerView {
            id: record.spec.id,
            url: record.spec.url,
            mode: record.spec.mode,
            model_ids: record.spec.model_ids,
            bootstrap_port: record.spec.bootstrap_port,
            version_group: record.spec.version_group,
            services: record.spec.services,
            source: WorkerSource::Api,
            revision: Some(record.revision),
            operation: Some(record.operation),
            deleted: record.deleted,
            runtime,
        }
    }

    fn external_view(&self, worker: &Worker) -> WorkerView {
        WorkerView {
            id: worker.id.clone(),
            url: worker.url.clone(),
            mode: worker.mode(),
            model_ids: worker.model_ids.clone(),
            bootstrap_port: worker.bootstrap_port(),
            version_group: worker.version_group().map(str::to_owned),
            services: worker.services(),
            source: self.external_source.into(),
            revision: None,
            operation: None,
            deleted: false,
            runtime: Some(WorkerRuntimeView::from_worker(worker)),
        }
    }
}

fn ensure_mutation_ready(
    record: &ManagedWorker,
    expected_revision: u64,
) -> Result<(), WorkerControlError> {
    if record.operation.state.is_in_progress() {
        return Err(conflict(
            "worker_operation_in_progress",
            "another operation is already in progress for this worker",
        ));
    }
    if record.revision != expected_revision {
        return Err(conflict(
            "stale_revision",
            format!(
                "expected revision {expected_revision}, current revision is {}",
                record.revision
            ),
        ));
    }
    Ok(())
}

fn normalized_worker_url(raw: &str) -> Result<String, WorkerControlError> {
    let parsed = url::Url::parse(raw)
        .map_err(|error| WorkerControlError::Invalid(format!("invalid worker URL: {error}")))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(WorkerControlError::Invalid(
            "worker URL must use http or https and include a host".into(),
        ));
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(WorkerControlError::Invalid(
            "worker URL must not include a query or fragment".into(),
        ));
    }
    if !matches!(parsed.path(), "" | "/") {
        return Err(WorkerControlError::Invalid(
            "worker URL must not include a path".into(),
        ));
    }
    Ok(parsed.as_str().trim_end_matches('/').to_owned())
}

pub(crate) fn worker_urls_equal(left: &str, right: &str) -> bool {
    match (normalized_worker_url(left), normalized_worker_url(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => left.trim_end_matches('/') == right.trim_end_matches('/'),
    }
}

fn conflict(code: &'static str, message: impl Into<String>) -> WorkerControlError {
    WorkerControlError::Conflict {
        code,
        message: message.into(),
    }
}

fn matches_filter(view: &WorkerView, filter: &WorkerListFilter) -> bool {
    let runtime_models = view
        .runtime
        .as_ref()
        .map(|runtime| &runtime.resolved_model_ids);
    let runtime_mode = view.runtime.as_ref().map(|runtime| runtime.mode);
    filter
        .model_id
        .as_ref()
        .is_none_or(|model| runtime_models.unwrap_or(&view.model_ids).contains(model))
        && filter
            .mode
            .is_none_or(|mode| runtime_mode.unwrap_or(view.mode) == mode)
        && filter.source.is_none_or(|source| view.source == source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode};
    use crate::workers::WorkerRegistry;
    use std::sync::Arc;

    fn input(url: &str) -> WorkerInput {
        WorkerInput {
            url: url.into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
            version_group: None,
            services: Default::default(),
        }
    }

    #[tokio::test]
    async fn create_records_pending_operation_and_sends_command() {
        let registry = Arc::new(WorkerRegistry::default());
        let (service, mut rx) = WorkerControlService::new(registry, ExternalWorkerSource::Static);

        let accepted = service
            .create(input("http://worker-1:30000"))
            .expect("create accepted");
        let command = rx.recv().await.expect("control command");

        assert_eq!(accepted.revision, 1);
        assert_eq!(accepted.worker_id, command.worker_id());
        assert_eq!(accepted.operation_id, command.operation_id);
        let view = service.get(&accepted.worker_id).expect("managed worker");
        assert_eq!(
            view.operation.expect("operation").state,
            OperationState::Pending
        );
        assert_eq!(view.source, WorkerSource::Api);
    }

    #[test]
    fn create_rejects_duplicate_url() {
        let registry = Arc::new(WorkerRegistry::default());
        let (service, _rx) = WorkerControlService::new(registry, ExternalWorkerSource::Static);

        service
            .create(input("http://worker-1:30000"))
            .expect("first create accepted");
        let error = service
            .create(input("http://worker-1:30000"))
            .expect_err("duplicate rejected");

        assert_eq!(error.code(), "worker_url_conflict");
    }

    #[test]
    fn create_rejects_equivalent_discovery_url() {
        let registry = Arc::new(WorkerRegistry::default());
        registry
            .add(WorkerSpec {
                id: WorkerId("discovered".into()),
                url: "http://worker-1:30000/".into(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                ..Default::default()
            })
            .expect("registry add");
        let (service, _rx) = WorkerControlService::new(registry, ExternalWorkerSource::Static);

        let error = service
            .create(input("http://worker-1:30000"))
            .expect_err("equivalent URL rejected");

        assert_eq!(error.code(), "worker_url_conflict");
    }

    #[test]
    fn replace_rejects_stale_revision_and_url_change() {
        let registry = Arc::new(WorkerRegistry::default());
        let (service, _rx) = WorkerControlService::new(registry, ExternalWorkerSource::Static);
        let created = service
            .create(input("http://worker-1:30000"))
            .expect("create accepted");
        service.mark_succeeded(&created.operation_id);

        let stale = service
            .replace(&created.worker_id, input("http://worker-1:30000"), 0)
            .expect_err("stale revision rejected");
        assert_eq!(stale.code(), "stale_revision");

        let changed = service
            .replace(&created.worker_id, input("http://worker-2:30000"), 1)
            .expect_err("URL change rejected");
        assert_eq!(changed.code(), "worker_url_immutable");
    }

    #[test]
    fn mutation_rejects_discovery_managed_worker() {
        let registry = Arc::new(WorkerRegistry::default());
        registry
            .add(WorkerSpec {
                id: WorkerId("discovered".into()),
                url: "http://worker-1:30000".into(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                ..Default::default()
            })
            .expect("registry add");
        let (service, _rx) = WorkerControlService::new(registry, ExternalWorkerSource::Kubernetes);

        let error = service
            .remove(&WorkerId("discovered".into()), 1)
            .expect_err("discovery worker is read only");

        assert_eq!(error.code(), "worker_not_api_managed");
    }

    #[test]
    fn successful_remove_keeps_tombstone_visible() {
        let registry = Arc::new(WorkerRegistry::default());
        let (service, _rx) = WorkerControlService::new(registry, ExternalWorkerSource::Static);
        let created = service
            .create(input("http://worker-1:30000"))
            .expect("create accepted");
        service.mark_succeeded(&created.operation_id);
        let removed = service
            .remove(&created.worker_id, 1)
            .expect("remove accepted");

        service.mark_succeeded(&removed.operation_id);

        let view = service.get(&created.worker_id).expect("tombstone retained");
        assert!(view.deleted);
        assert!(view.runtime.is_none());
        assert_eq!(
            view.operation.expect("operation").state,
            OperationState::Succeeded
        );
    }

    #[test]
    fn list_filters_api_workers_by_resolved_runtime_model() {
        let registry = Arc::new(WorkerRegistry::default());
        let (service, _rx) =
            WorkerControlService::new(Arc::clone(&registry), ExternalWorkerSource::Static);
        let mut unresolved = input("http://worker-1:30000");
        unresolved.model_ids.clear();
        let created = service.create(unresolved).expect("create accepted");
        service.mark_succeeded(&created.operation_id);
        registry
            .add(WorkerSpec {
                id: created.worker_id,
                url: "http://worker-1:30000".into(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("resolved".into())],
                ..Default::default()
            })
            .expect("runtime worker");

        let workers = service.list(&WorkerListFilter {
            model_id: Some(ModelId("resolved".into())),
            mode: Some(WorkerMode::Decode),
            source: Some(WorkerSource::Api),
        });

        assert_eq!(workers.len(), 1);
    }
}
