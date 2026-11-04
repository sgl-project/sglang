//! Single-writer Mooncake soft-pin owner. A deployment must route every writer
//! in a protection domain through this owner; local epochs provide no fencing.
//!
//! Claims outlive submission operations. The spawned task advances RPCs and TTLs
//! without inference traffic. Transport timeouts never prove backend quiescence.
use crate::models::PrefixTarget;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
    pin::Pin,
    sync::Arc,
    time::Duration,
};
use tokio::{
    sync::{mpsc, oneshot, watch},
    time::Instant,
};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectionDomain {
    pub cluster: String,
    pub tenant_id: String,
    pub model_namespace: String,
    pub storage_namespace: String,
    pub layout_id: String,
}

/// Supplied by a trusted Engine/storage-layout resolver, never from event i64
/// hashes. Every object needed to recover this scope must appear in `keys`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedStorageScope {
    pub component: String,
    pub start_token: u64,
    pub end_token: u64,
    pub dependency_start: u64,
    pub keys: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedStorageTarget {
    pub domain: ProtectionDomain,
    pub scopes: Vec<ResolvedStorageScope>,
}
/// P2 trusted Engine `storage_targets` response. This describes key identity;
/// existence stays unknown until each Mooncake mutation returns a key receipt.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineStorageTargets {
    pub description_version: String,
    pub namespace: String,
    pub compatibility: EngineStorageCompatibility,
    pub key_salt_encoding: String,
    pub page_size: u64,
    pub location: crate::protocol::Location,
    pub existence: String,
    pub pages: Vec<EngineStoragePage>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineStorageCompatibility {
    pub cache_salt: Option<String>,
    pub extra_key: Option<String>,
    pub layout_id: Option<String>,
    #[serde(default)]
    pub is_multimodal: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineStoragePage {
    pub start_token: u64,
    pub end_token: u64,
    pub component: String,
    pub storage_digest: String,
    pub keys: Vec<String>,
    pub object_bytes: Vec<u64>,
}
impl EngineStorageTargets {
    /// Call only on an authenticated response to the exact requested target and
    /// context. The response does not echo tokens; transport/request correlation
    /// remains the resolver's responsibility. The encoding is obtained from live
    /// Engine discovery; it is never guessed from the shape of a cache salt.
    pub fn into_resolved(
        self,
        requested: &PrefixTarget,
        domain: &ProtectionDomain,
        expected_salt_encoding: &str,
    ) -> Result<ResolvedStorageTarget, ProtectionError> {
        if self.description_version != "1.0"
            || self.existence != "unknown"
            || self.namespace != requested.compatibility.namespace
            || self.namespace != domain.model_namespace
            || self.compatibility.layout_id.as_deref()
                != Some(requested.compatibility.layout_id.as_str())
            || self.compatibility.layout_id.as_deref() != Some(domain.layout_id.as_str())
            || self.compatibility.cache_salt != requested.compatibility.cache_salt
            || self.compatibility.extra_key.is_some()
            || self.compatibility.is_multimodal
            || self.page_size != requested.compatibility.page_size
            || self.key_salt_encoding != expected_salt_encoding
            || !matches!(
                expected_salt_encoding,
                "sglang.kv.weight.v1" | "owner_assertion"
            )
            || self.location.tier != "storage"
            || self.location.engine_id.is_some()
            || self.location.backend.as_deref() != Some("mooncake")
            || self.location.backend_namespace.as_deref() != Some(domain.storage_namespace.as_str())
            || self.pages.is_empty()
            || self.pages.len() > 4096
        {
            return Err(ProtectionError::ResolverScopeMismatch);
        }
        let mut scopes = Vec::with_capacity(self.pages.len());
        for page in self.pages {
            if page.storage_digest.len() != 64
                || !page.storage_digest.bytes().all(|b| b.is_ascii_hexdigit())
                || page.keys.is_empty()
                || page.keys.len() != page.object_bytes.len()
                || page.object_bytes.contains(&0)
                || page.end_token.checked_sub(page.start_token) != Some(self.page_size)
            {
                return Err(ProtectionError::ResolverScopeMismatch);
            }
            scopes.push(ResolvedStorageScope {
                component: page.component,
                start_token: page.start_token,
                end_token: page.end_token,
                dependency_start: 0,
                keys: page.keys,
            });
        }
        Ok(ResolvedStorageTarget {
            domain: domain.clone(),
            scopes,
        })
    }
}
pub trait StorageTargetResolver: Send + Sync + 'static {
    fn resolve<'a>(
        &'a self,
        target: &'a PrefixTarget,
    ) -> BoxFuture<'a, Result<ResolvedStorageTarget, ProtectionError>>;
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimIdentity {
    pub caller_id: String,
    pub task_incarnation: String,
    pub claim_id: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimMode {
    Acquire,
    Renew,
    Release,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectionRequest {
    pub identity: ClaimIdentity,
    pub revision: u64,
    pub mode: ClaimMode,
    pub target: PrefixTarget,
    pub ttl_seconds: Option<f64>,
    pub required_protection: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtectionError {
    UnsupportedProtection,
    InvalidRequest,
    DomainMismatch,
    ResolverScopeMismatch,
    Capacity,
    StaleRevision,
    Conflict,
    UnknownClaim,
    TerminalClaim,
    RecoveryRequired,
    Closed,
}
impl std::fmt::Display for ProtectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ProtectionError {}

/// Configuration assertion, not a lease or a backend fence. Unknown recovery
/// cannot be cleared by elapsed time. Start FreshDomain only after externally
/// establishing there are no old writers/RPCs/claims requiring preservation.
#[derive(Clone, Debug)]
pub enum RecoveryMode {
    FreshDomain { deployment_assertion: String },
    UnknownDomain { reason: String },
}
#[derive(Clone, Debug)]
pub struct ProtectionConfig {
    pub domain: ProtectionDomain,
    pub owner_epoch: String,
    pub recovery: RecoveryMode,
    pub max_claims: usize,
    pub max_tombstones: usize,
    pub max_objects: usize,
    pub max_objects_per_claim: usize,
    pub max_claims_per_object: usize,
    pub max_identity_bytes: usize,
    pub command_capacity: usize,
    pub max_inflight_batches: usize,
    pub max_batches_per_tick: usize,
    pub batch_size: usize,
    /// Configured against the deployed/probed server; no assumed server limit.
    pub max_ttl: Duration,
    pub rpc_observation_timeout: Duration,
    pub retry_delay: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendMutation {
    SoftPin { ttl_ms: u64 },
    SoftUnpin,
}
#[derive(Clone, Debug)]
pub struct BackendBatch {
    pub tenant_id: String,
    pub keys: Vec<String>,
    pub mutation: BackendMutation,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyResult {
    pub key: String,
    pub error_code: i64,
}
/// Quiescent means this exact RPC finished applying. Unknown includes lost or
/// invalid responses and timeouts, even when the client request was cancelled.
#[derive(Clone, Debug)]
pub enum BackendReply {
    Quiescent(Vec<KeyResult>),
    Unknown(String),
}
pub trait SoftPinBackend: Send + Sync + 'static {
    /// Confirmed API support, separate from a configured TTL maximum.
    fn api_verified(&self) -> bool;
    fn max_ttl_ms(&self) -> u64;
    fn mutate<'a>(&'a self, batch: BackendBatch) -> BoxFuture<'a, BackendReply>;
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClaimPhase {
    Desired,
    Active,
    Partial,
    Released,
    Expired,
    Unknown,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScopeProtection {
    pub component: String,
    pub start_token: u64,
    pub end_token: u64,
    pub dependency_start: u64,
    pub confirmed_remaining_ms: Option<u64>,
    pub missing: bool,
    pub uncertain: bool,
    pub backend_error_codes: Vec<i64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtectionClaimView {
    pub identity: ClaimIdentity,
    pub revision: u64,
    pub owner_epoch: String,
    pub state_version: u64,
    pub phase: ClaimPhase,
    pub tombstone: bool,
    pub desired_remaining_ms: u64,
    pub effective_protection: Option<String>,
    pub confirmed_remaining_ms: Option<u64>,
    pub converged: bool,
    /// No pending, outstanding or indeterminate mutation touches these objects.
    pub backend_quiescent: bool,
    pub uncertain: bool,
    pub scopes: Vec<ScopeProtection>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtectionSnapshot {
    pub owner_epoch: String,
    pub state_version: u64,
    pub recovery_unknown: bool,
    pub recovery_reason: Option<String>,
    pub claim: Option<ProtectionClaimView>,
    pub objects: usize,
    pub inflight_batches: usize,
    pub uncertain_objects: usize,
    /// Only backend_soft_pin is ever advertised, and none during unknown recovery.
    pub supported_protection: Vec<String>,
    pub max_ttl_ms: u64,
}

#[derive(Clone)]
pub struct BackendProtectionOwner {
    commands: mpsc::Sender<Command>,
    resolver: Arc<dyn StorageTargetResolver>,
    config: ProtectionConfig,
    changes: watch::Receiver<u64>,
}
impl BackendProtectionOwner {
    pub fn spawn(
        config: ProtectionConfig,
        backend: Arc<dyn SoftPinBackend>,
        resolver: Arc<dyn StorageTargetResolver>,
    ) -> Result<Self, ProtectionError> {
        validate_config(&config)?;
        if !matches!(config.recovery, RecoveryMode::UnknownDomain { .. }) && !backend.api_verified()
        {
            return Err(ProtectionError::UnsupportedProtection);
        }
        if duration_ms(config.max_ttl) > backend.max_ttl_ms() {
            return Err(ProtectionError::InvalidRequest);
        }
        let (commands, rx) = mpsc::channel(config.command_capacity);
        let (changes_tx, changes) = watch::channel(0);
        let state = OwnerState::new(config.clone(), backend, changes_tx);
        tokio::spawn(state.run(rx));
        Ok(Self {
            commands,
            resolver,
            config,
            changes,
        })
    }
    /// Parsing and resolution occur before actor admission and create no claim.
    /// Cancellation of this future never implicitly releases an admitted claim.
    pub async fn submit(
        &self,
        request: ProtectionRequest,
    ) -> Result<ProtectionClaimView, ProtectionError> {
        validate_request(&request, &self.config)?;
        // Release and renew bind the original resolved scope in the actor. A
        // disappearing resolver must not prevent release of an existing claim.
        let target = if request.mode == ClaimMode::Acquire
            && self
                .query(Some(request.identity.clone()))
                .await?
                .claim
                .is_none()
        {
            Some(self.resolver.resolve(&request.target).await?)
        } else {
            None
        };
        let (tx, rx) = oneshot::channel();
        self.commands
            .send(Command::Submit(request, target, tx))
            .await
            .map_err(|_| ProtectionError::Closed)?;
        rx.await.map_err(|_| ProtectionError::Closed)?
    }
    pub async fn query(
        &self,
        identity: Option<ClaimIdentity>,
    ) -> Result<ProtectionSnapshot, ProtectionError> {
        let (tx, rx) = oneshot::channel();
        self.commands
            .send(Command::Query(identity, tx))
            .await
            .map_err(|_| ProtectionError::Closed)?;
        rx.await.map_err(|_| ProtectionError::Closed)
    }
    pub fn changes(&self) -> watch::Receiver<u64> {
        self.changes.clone()
    }
    pub async fn next_deadline(&self) -> Result<Option<Instant>, ProtectionError> {
        let (tx, rx) = oneshot::channel();
        self.commands
            .send(Command::Deadline(tx))
            .await
            .map_err(|_| ProtectionError::Closed)?;
        rx.await.map_err(|_| ProtectionError::Closed)
    }
    /// Trusted administrative reconciliation only. The caller must have verified
    /// the old RPC cannot apply later (backend receipt or execution isolation).
    /// `confirmed_remaining` is a conservative backend observation taken now.
    /// Mooncake's pin/unpin HTTP API itself supplies no such query/fencing.
    pub async fn reconcile_object_after_quiescence(
        &self,
        key: String,
        confirmed_remaining: Option<Duration>,
        evidence: String,
    ) -> Result<(), ProtectionError> {
        let (tx, rx) = oneshot::channel();
        self.commands
            .send(Command::Reconcile(key, confirmed_remaining, evidence, tx))
            .await
            .map_err(|_| ProtectionError::Closed)?;
        rx.await.map_err(|_| ProtectionError::Closed)?
    }
}

enum Command {
    Submit(
        ProtectionRequest,
        Option<ResolvedStorageTarget>,
        oneshot::Sender<Result<ProtectionClaimView, ProtectionError>>,
    ),
    Query(Option<ClaimIdentity>, oneshot::Sender<ProtectionSnapshot>),
    Deadline(oneshot::Sender<Option<Instant>>),
    Reconcile(
        String,
        Option<Duration>,
        String,
        oneshot::Sender<Result<(), ProtectionError>>,
    ),
}
struct Claim {
    request: ProtectionRequest,
    resolved: ResolvedStorageTarget,
    deadline: Instant,
    terminal: Option<ClaimPhase>,
}
struct Object {
    claims: BTreeSet<ClaimIdentity>,
    version: u64,
    confirmed: Option<Instant>,
    applied_desired: Option<Instant>,
    last_error: Option<i64>,
    uncertainty: Option<String>,
    inflight: Option<InFlight>,
    retry_at: Option<Instant>,
}
#[derive(Clone)]
struct InFlight {
    batch: u64,
    version: u64,
    sent: Instant,
    timeout: Instant,
    timed_out: bool,
    mutation: BackendMutation,
    desired: Option<Instant>,
}
struct Completion {
    id: u64,
    keys: Vec<String>,
    reply: BackendReply,
}
struct OwnerState {
    config: ProtectionConfig,
    backend: Arc<dyn SoftPinBackend>,
    changes: watch::Sender<u64>,
    claims: BTreeMap<ClaimIdentity, Claim>,
    objects: BTreeMap<String, Object>,
    version: u64,
    next_batch: u64,
    inflight_batches: usize,
}
impl OwnerState {
    fn new(
        config: ProtectionConfig,
        backend: Arc<dyn SoftPinBackend>,
        changes: watch::Sender<u64>,
    ) -> Self {
        Self {
            config,
            backend,
            changes,
            claims: BTreeMap::new(),
            objects: BTreeMap::new(),
            version: 0,
            next_batch: 0,
            inflight_batches: 0,
        }
    }
    fn changed(&mut self) {
        self.version += 1;
        self.changes.send_replace(self.version);
    }
    fn recovery_unknown(&self) -> bool {
        matches!(self.config.recovery, RecoveryMode::UnknownDomain { .. })
    }
    async fn run(mut self, mut commands: mpsc::Receiver<Command>) {
        let (completion_tx, mut completions) = mpsc::channel(self.config.max_inflight_batches);
        let mut commands_open = true;
        loop {
            self.expire(Instant::now());
            self.dispatch(&completion_tx);
            let deadline = self.deadline();
            tokio::select! {
                command = commands.recv(), if commands_open => match command {
                    Some(Command::Submit(request, target, tx)) => { let result = self.submit(request, target); let _ = tx.send(result); }
                    Some(Command::Query(identity, tx)) => { let _ = tx.send(self.snapshot(identity.as_ref())); }
                    Some(Command::Deadline(tx)) => { let _ = tx.send(self.deadline()); }
                    Some(Command::Reconcile(key, remaining, evidence, tx)) => { let result = self.reconcile(key, remaining, evidence); let _ = tx.send(result); }
                    None => commands_open = false,
                },
                completion = completions.recv(), if self.inflight_batches > 0 => {
                    if let Some(completion) = completion { self.complete(completion); }
                },
                _ = async { match deadline { Some(at) => tokio::time::sleep_until(at).await, None => std::future::pending::<()>().await } } => {},
            }
            // Keep driving live claims after the last operation handle disappears.
            if !commands_open
                && self.inflight_batches == 0
                && self.claims.values().all(|c| c.terminal.is_some())
                && self.objects.values().all(|o| {
                    o.uncertainty.is_some() || (o.confirmed.is_none() && o.inflight.is_none())
                })
            {
                break;
            }
        }
    }
    fn submit(
        &mut self,
        request: ProtectionRequest,
        target: Option<ResolvedStorageTarget>,
    ) -> Result<ProtectionClaimView, ProtectionError> {
        if self.recovery_unknown() {
            return Err(ProtectionError::RecoveryRequired);
        }
        let now = Instant::now();
        if let Some(old) = self.claims.get(&request.identity) {
            if request.revision < old.request.revision {
                return Err(ProtectionError::StaleRevision);
            }
            if request.revision == old.request.revision {
                return if request == old.request {
                    Ok(self.claim_view(&request.identity, now).unwrap())
                } else {
                    Err(ProtectionError::Conflict)
                };
            }
            if old.terminal.is_some() {
                return Err(ProtectionError::TerminalClaim);
            }
            if request.target != old.request.target {
                return Err(ProtectionError::Conflict);
            }
            if request.mode == ClaimMode::Acquire {
                return Err(ProtectionError::Conflict);
            }
        } else if request.mode == ClaimMode::Release {
            // A release may overtake its acquire on the transport. Preserve a
            // bounded revision fence even when no local claim was observed.
            if self.claims.len() >= self.config.max_tombstones {
                return Err(ProtectionError::Capacity);
            }
            let identity = request.identity.clone();
            self.claims.insert(
                identity.clone(),
                Claim {
                    request,
                    resolved: ResolvedStorageTarget {
                        domain: self.config.domain.clone(),
                        scopes: vec![],
                    },
                    deadline: now,
                    terminal: Some(ClaimPhase::Released),
                },
            );
            self.changed();
            return Ok(self.claim_view(&identity, now).unwrap());
        } else if request.mode != ClaimMode::Acquire {
            return Err(ProtectionError::UnknownClaim);
        }
        let resolved = if let Some(old) = self.claims.get(&request.identity) {
            old.resolved.clone()
        } else {
            let target = target.ok_or(ProtectionError::ResolverScopeMismatch)?;
            validate_resolved(&target, &request.target, &self.config)?;
            let keys = target_keys(&target);
            // Reserve eventual tombstones at admission. Never evict a revision
            // fence and thereby make an old acquire able to resurrect a claim.
            let live = self
                .claims
                .values()
                .filter(|c| c.terminal.is_none())
                .count();
            if live >= self.config.max_claims
                || self.claims.len() >= self.config.max_tombstones
                || self.objects.len()
                    + keys
                        .iter()
                        .filter(|k| !self.objects.contains_key(*k))
                        .count()
                    > self.config.max_objects
            {
                return Err(ProtectionError::Capacity);
            }
            for key in &keys {
                if self
                    .objects
                    .get(key)
                    .is_some_and(|o| o.claims.len() >= self.config.max_claims_per_object)
                {
                    return Err(ProtectionError::Capacity);
                }
            }
            target
        };
        let deadline = if request.mode == ClaimMode::Release {
            now
        } else {
            now + Duration::from_secs_f64(request.ttl_seconds.unwrap())
        };
        let terminal = (request.mode == ClaimMode::Release).then_some(ClaimPhase::Released);
        for key in target_keys(&resolved) {
            let object = self.objects.entry(key).or_insert_with(|| Object {
                claims: BTreeSet::new(),
                version: 0,
                confirmed: None,
                applied_desired: None,
                last_error: None,
                uncertainty: None,
                inflight: None,
                retry_at: None,
            });
            if terminal.is_some() {
                object.claims.remove(&request.identity);
            } else {
                object.claims.insert(request.identity.clone());
            }
            object.version += 1;
            object.retry_at = None;
        }
        let identity = request.identity.clone();
        self.claims.insert(
            identity.clone(),
            Claim {
                request,
                resolved,
                deadline,
                terminal,
            },
        );
        self.changed();
        Ok(self.claim_view(&identity, now).unwrap())
    }
    fn expire(&mut self, now: Instant) {
        let expired: Vec<_> = self
            .claims
            .iter()
            .filter(|(_, c)| c.terminal.is_none() && c.deadline <= now)
            .map(|(id, _)| id.clone())
            .collect();
        let mut changed = false;
        for identity in expired {
            let claim = self.claims.get_mut(&identity).unwrap();
            claim.terminal = Some(ClaimPhase::Expired);
            for key in target_keys(&claim.resolved) {
                if let Some(object) = self.objects.get_mut(&key) {
                    object.claims.remove(&identity);
                    object.version += 1;
                    object.retry_at = None;
                }
            }
            changed = true;
        }
        for object in self.objects.values_mut() {
            if object.confirmed.is_some_and(|until| until <= now) {
                object.confirmed = None;
                changed = true;
            }
            if let Some(inflight) = object.inflight.as_mut() {
                if !inflight.timed_out && inflight.timeout <= now {
                    inflight.timed_out = true;
                    object.uncertainty = Some("rpc_observation_timeout".into());
                    object.confirmed = None;
                    changed = true;
                }
            }
        }
        self.objects.retain(|_, o| {
            !(o.claims.is_empty()
                && o.inflight.is_none()
                && o.uncertainty.is_none()
                && o.confirmed.is_none()
                && o.applied_desired.is_none())
        });
        if changed {
            self.changed();
        }
    }
    fn desired(&self, object: &Object) -> Option<Instant> {
        object
            .claims
            .iter()
            .filter_map(|id| self.claims.get(id))
            .filter(|c| c.terminal.is_none())
            .map(|c| c.deadline)
            .max()
    }
    fn dispatch(&mut self, completions: &mpsc::Sender<Completion>) {
        if self.recovery_unknown() {
            return;
        }
        let now = Instant::now();
        let mut candidates: Vec<(BackendMutation, Vec<(String, Option<Instant>)>)> = Vec::new();
        for (key, object) in &self.objects {
            if object.inflight.is_some()
                || object.uncertainty.is_some()
                || object.retry_at.is_some_and(|at| at > now)
            {
                continue;
            }
            let desired = self.desired(object);
            if desired == object.applied_desired {
                continue;
            }
            let mutation = match desired {
                Some(until) if until > now => BackendMutation::SoftPin {
                    ttl_ms: ceil_ms(until - now).min(duration_ms(self.config.max_ttl)),
                },
                _ => BackendMutation::SoftUnpin,
            };
            if let Some((_, keys)) = candidates
                .iter_mut()
                .find(|(m, k)| *m == mutation && k.len() < self.config.batch_size)
            {
                keys.push((key.clone(), desired));
            } else if candidates.len()
                < self
                    .config
                    .max_batches_per_tick
                    .min(self.config.max_inflight_batches - self.inflight_batches)
            {
                candidates.push((mutation, vec![(key.clone(), desired)]));
            }
        }
        for (mutation, keys) in candidates {
            self.next_batch += 1;
            let id = self.next_batch;
            let sent = Instant::now();
            let key_names: Vec<_> = keys.iter().map(|(key, _)| key.clone()).collect();
            for (key, desired) in keys {
                let object = self.objects.get_mut(&key).unwrap();
                // An in-flight unpin cannot support a newly acquired claim.
                match mutation {
                    BackendMutation::SoftUnpin => object.confirmed = None,
                    BackendMutation::SoftPin { ttl_ms } => {
                        object.confirmed = object
                            .confirmed
                            .map(|until| until.min(sent + Duration::from_millis(ttl_ms)));
                    }
                }
                object.inflight = Some(InFlight {
                    batch: id,
                    version: object.version,
                    sent,
                    timeout: sent + self.config.rpc_observation_timeout,
                    timed_out: false,
                    mutation: mutation.clone(),
                    desired,
                });
                object.retry_at = None;
            }
            self.inflight_batches += 1;
            let backend = self.backend.clone();
            let tx = completions.clone();
            let tenant_id = self.config.domain.tenant_id.clone();
            tokio::spawn(async move {
                let reply = backend
                    .mutate(BackendBatch {
                        tenant_id,
                        keys: key_names.clone(),
                        mutation,
                    })
                    .await;
                let _ = tx
                    .send(Completion {
                        id,
                        keys: key_names,
                        reply,
                    })
                    .await;
            });
            self.changed();
        }
    }
    fn complete(&mut self, completion: Completion) {
        self.inflight_batches -= 1;
        let now = Instant::now();
        let mut results = BTreeMap::new();
        let unknown = match completion.reply {
            BackendReply::Unknown(reason) => Some(reason),
            BackendReply::Quiescent(keys) => {
                let mut duplicate = false;
                for key in keys {
                    if results.insert(key.key, key.error_code).is_some() {
                        duplicate = true;
                    }
                }
                duplicate.then(|| "duplicate_backend_results".into())
            }
        };
        for key in completion.keys {
            let Some(object) = self.objects.get_mut(&key) else {
                continue;
            };
            let Some(inflight) = object.inflight.take() else {
                continue;
            };
            if inflight.batch != completion.id {
                object.inflight = Some(inflight);
                continue;
            }
            if let Some(reason) = unknown.as_ref() {
                object.uncertainty = Some(reason.clone());
                object.confirmed = None;
                continue;
            }
            let Some(code) = results.remove(&key) else {
                object.uncertainty = Some("missing_backend_result".into());
                object.confirmed = None;
                continue;
            };
            object.uncertainty = None; // The exact RPC's final receipt is quiescence.
            object.last_error = (code != 0).then_some(code);
            if code == 0 || (code == -704 && inflight.mutation == BackendMutation::SoftUnpin) {
                object.confirmed = match inflight.mutation {
                    BackendMutation::SoftPin { ttl_ms } => {
                        Some(inflight.sent + Duration::from_millis(ttl_ms))
                            .filter(|until| *until > now)
                    }
                    BackendMutation::SoftUnpin => None,
                };
                object.applied_desired = inflight.desired;
            } else {
                // Per-key errors confirm no protection for that key. Retain the
                // desired scope and retry with bounded delay, independently.
                object.confirmed = None;
                object.retry_at = Some(now + self.config.retry_delay);
            }
            if object.version != inflight.version {
                object.retry_at = None;
            }
        }
        // Unknown objects remain resident forever rather than losing recovery
        // information. Known idle objects have no references to protect.
        self.objects.retain(|_, o| {
            !(o.claims.is_empty()
                && o.inflight.is_none()
                && o.uncertainty.is_none()
                && o.confirmed.is_none()
                && o.applied_desired.is_none())
        });
        self.changed();
    }
    fn reconcile(
        &mut self,
        key: String,
        remaining: Option<Duration>,
        evidence: String,
    ) -> Result<(), ProtectionError> {
        if self.recovery_unknown() {
            return Err(ProtectionError::RecoveryRequired);
        }
        if evidence.trim().is_empty() || remaining.is_some_and(|d| d > self.config.max_ttl) {
            return Err(ProtectionError::InvalidRequest);
        }
        let object = self
            .objects
            .get_mut(&key)
            .ok_or(ProtectionError::UnknownClaim)?;
        // Cannot dismiss a task that still owns the old future. Its final receipt
        // remains responsible for completion and batch accounting.
        if object.inflight.is_some() {
            return Err(ProtectionError::Conflict);
        }
        object.uncertainty = None;
        object.confirmed = remaining
            .filter(|d| !d.is_zero())
            .map(|d| Instant::now() + d);
        object.applied_desired = None;
        object.retry_at = None;
        object.version += 1;
        self.changed();
        Ok(())
    }
    fn deadline(&self) -> Option<Instant> {
        self.claims
            .values()
            .filter(|c| c.terminal.is_none())
            .map(|c| c.deadline)
            .chain(self.objects.values().filter_map(|o| o.confirmed))
            .chain(self.objects.values().filter_map(|o| o.retry_at))
            .chain(
                self.objects
                    .values()
                    .filter_map(|o| o.inflight.as_ref())
                    .filter(|i| !i.timed_out)
                    .map(|i| i.timeout),
            )
            .min()
    }
    fn claim_view(&self, identity: &ClaimIdentity, now: Instant) -> Option<ProtectionClaimView> {
        let claim = self.claims.get(identity)?;
        let terminal = claim.terminal.clone();
        let scopes: Vec<_> = claim
            .resolved
            .scopes
            .iter()
            .map(|scope| {
                // A page whose dependency prefix is missing/unknown cannot be
                // reported recoverable even if its own K/V objects were pinned.
                let required: BTreeSet<_> = claim
                    .resolved
                    .scopes
                    .iter()
                    .filter(|prior| {
                        prior.component == scope.component
                            && prior.end_token > scope.dependency_start
                            && prior.start_token < scope.end_token
                    })
                    .flat_map(|part| part.keys.iter())
                    .collect();
                let objects: Vec<_> = required
                    .iter()
                    .filter_map(|key| self.objects.get(*key))
                    .collect();
                let uncertain = objects.iter().any(|o| o.uncertainty.is_some());
                let missing = objects.iter().any(|o| o.last_error == Some(-704));
                let backend_error_codes = objects
                    .iter()
                    .filter_map(|o| o.last_error)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let confirmed =
                    if terminal.is_none() && !uncertain && objects.len() == required.len() {
                        objects
                            .iter()
                            .map(|o| o.confirmed)
                            .collect::<Option<Vec<_>>>()
                            .and_then(|deadlines| deadlines.into_iter().min())
                            .filter(|until| *until > now)
                    } else {
                        None
                    };
                ScopeProtection {
                    component: scope.component.clone(),
                    start_token: scope.start_token,
                    end_token: scope.end_token,
                    dependency_start: scope.dependency_start,
                    confirmed_remaining_ms: confirmed.map(|until| {
                        duration_ms(until.min(claim.deadline).saturating_duration_since(now))
                    }),
                    missing,
                    uncertain,
                    backend_error_codes,
                }
            })
            .collect();
        let confirmed_count = scopes
            .iter()
            .filter(|s| s.confirmed_remaining_ms.is_some_and(|ms| ms > 0))
            .count();
        let confirmed_remaining_ms = if confirmed_count == scopes.len() {
            scopes.iter().filter_map(|s| s.confirmed_remaining_ms).min()
        } else {
            None
        };
        let phase = terminal.clone().unwrap_or_else(|| {
            if scopes.iter().any(|s| s.uncertain) {
                ClaimPhase::Unknown
            } else if confirmed_count == scopes.len() {
                ClaimPhase::Active
            } else if confirmed_count > 0 {
                ClaimPhase::Partial
            } else {
                ClaimPhase::Desired
            }
        });
        let converged = terminal.is_none()
            && target_keys(&claim.resolved).iter().all(|key| {
                self.objects.get(key).is_some_and(|o| {
                    o.uncertainty.is_none()
                        && o.confirmed.is_some_and(|until| until >= claim.deadline)
                })
            });
        let backend_quiescent = target_keys(&claim.resolved).iter().all(|key| {
            self.objects.get(key).is_none_or(|o| {
                o.inflight.is_none()
                    && o.uncertainty.is_none()
                    && o.applied_desired == self.desired(o)
            })
        });
        let uncertain = scopes.iter().any(|s| s.uncertain);
        Some(ProtectionClaimView {
            identity: identity.clone(),
            revision: claim.request.revision,
            owner_epoch: self.config.owner_epoch.clone(),
            state_version: self.version,
            phase,
            tombstone: terminal.is_some(),
            desired_remaining_ms: if terminal.is_none() {
                duration_ms(claim.deadline.saturating_duration_since(now))
            } else {
                0
            },
            effective_protection: (confirmed_count > 0).then(|| "backend_soft_pin".into()),
            confirmed_remaining_ms,
            converged,
            backend_quiescent,
            uncertain,
            scopes,
        })
    }
    fn snapshot(&self, identity: Option<&ClaimIdentity>) -> ProtectionSnapshot {
        ProtectionSnapshot {
            owner_epoch: self.config.owner_epoch.clone(),
            state_version: self.version,
            recovery_unknown: self.recovery_unknown(),
            recovery_reason: match &self.config.recovery {
                RecoveryMode::UnknownDomain { reason } => Some(reason.clone()),
                _ => None,
            },
            claim: identity.and_then(|id| self.claim_view(id, Instant::now())),
            objects: self.objects.len(),
            inflight_batches: self.inflight_batches,
            uncertain_objects: self
                .objects
                .values()
                .filter(|o| o.uncertainty.is_some())
                .count(),
            supported_protection: if self.recovery_unknown() {
                vec![]
            } else {
                vec!["backend_soft_pin".into()]
            },
            max_ttl_ms: duration_ms(self.config.max_ttl),
        }
    }
}
fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u64::MAX as u128) as u64
}
fn ceil_ms(duration: Duration) -> u64 {
    duration
        .as_nanos()
        .div_ceil(1_000_000)
        .min(u64::MAX as u128) as u64
}
fn target_keys(target: &ResolvedStorageTarget) -> BTreeSet<String> {
    target
        .scopes
        .iter()
        .flat_map(|s| s.keys.iter().cloned())
        .collect()
}
fn validate_config(c: &ProtectionConfig) -> Result<(), ProtectionError> {
    if c.owner_epoch.is_empty()
        || [
            &c.domain.cluster,
            &c.domain.tenant_id,
            &c.domain.model_namespace,
            &c.domain.storage_namespace,
            &c.domain.layout_id,
        ]
        .iter()
        .any(|s| s.is_empty())
        || c.max_claims == 0
        || c.max_tombstones < c.max_claims
        || c.max_objects == 0
        || c.max_objects_per_claim == 0
        || c.max_claims_per_object == 0
        || c.max_identity_bytes == 0
        || c.command_capacity == 0
        || c.max_inflight_batches == 0
        || c.max_batches_per_tick == 0
        || !(1..=4096).contains(&c.batch_size)
        || duration_ms(c.max_ttl) == 0
        || c.max_ttl.as_nanos() % 1_000_000 != 0
        || c.max_ttl > Duration::from_secs(31_536_000)
        || c.rpc_observation_timeout.is_zero()
        || c.retry_delay.is_zero()
    {
        return Err(ProtectionError::InvalidRequest);
    }
    match &c.recovery {
        RecoveryMode::FreshDomain {
            deployment_assertion,
        } if deployment_assertion.trim().is_empty() => Err(ProtectionError::InvalidRequest),
        RecoveryMode::UnknownDomain { reason } if reason.trim().is_empty() => {
            Err(ProtectionError::InvalidRequest)
        }
        _ => Ok(()),
    }
}
fn validate_request(r: &ProtectionRequest, c: &ProtectionConfig) -> Result<(), ProtectionError> {
    if r.required_protection != "backend_soft_pin" {
        return Err(ProtectionError::UnsupportedProtection);
    }
    if r.revision > crate::protocol::MAX_INTEGER
        || [
            &r.identity.caller_id,
            &r.identity.task_incarnation,
            &r.identity.claim_id,
        ]
        .iter()
        .any(|s| s.is_empty() || s.len() > c.max_identity_bytes)
    {
        return Err(ProtectionError::InvalidRequest);
    }
    match (r.mode, r.ttl_seconds) {
        (ClaimMode::Release, None) => {}
        (ClaimMode::Acquire | ClaimMode::Renew, Some(ttl))
            if ttl.is_finite() && ttl > 0.0 && ttl <= c.max_ttl.as_secs_f64() && ttl >= 0.001 => {}
        _ => return Err(ProtectionError::InvalidRequest),
    }
    let action = crate::protocol::Action {
        action_id: "protection-validation".into(),
        action_type: "sglang.pin".into(),
        action_version: "1.0".into(),
        payload: serde_json::json!({"mode": match r.mode { ClaimMode::Acquire => "acquire", ClaimMode::Renew => "renew", ClaimMode::Release => "release" }, "target": r.target, "location":{"tier":"storage","engine_id":null,"backend":"mooncake","backend_namespace":null},"claim_id":r.identity.claim_id,"revision":r.revision,"ttl_seconds":r.ttl_seconds,"required_protection":r.required_protection}),
    };
    crate::protocol::parse_action(&action).map_err(|_| ProtectionError::InvalidRequest)?;
    Ok(())
}
fn validate_resolved(
    r: &ResolvedStorageTarget,
    p: &PrefixTarget,
    c: &ProtectionConfig,
) -> Result<(), ProtectionError> {
    if r.domain != c.domain {
        return Err(ProtectionError::DomainMismatch);
    }
    if r.domain.model_namespace != p.compatibility.namespace
        || r.domain.layout_id != p.compatibility.layout_id
    {
        return Err(ProtectionError::DomainMismatch);
    }
    if p.start_token != 0
        || r.scopes.is_empty()
        || r.scopes.len() > c.max_objects_per_claim
        || target_keys(r).len() > c.max_objects_per_claim
    {
        return Err(ProtectionError::ResolverScopeMismatch);
    }
    let page = p.compatibility.page_size;
    for component in &p.compatibility.components {
        let mut scopes: Vec<_> = r
            .scopes
            .iter()
            .filter(|s| &s.component == component)
            .collect();
        scopes.sort_by_key(|s| s.start_token);
        let mut frontier = p.start_token;
        for s in scopes {
            if s.start_token != frontier
                || s.end_token <= s.start_token
                || s.end_token > p.end_token
                || s.start_token % page != 0
                || s.end_token % page != 0
                || s.dependency_start > s.start_token
                || s.dependency_start % page != 0
                || s.keys.is_empty()
                || s.keys.len() > c.max_objects_per_claim
                || s.keys
                    .iter()
                    .any(|k| k.is_empty() || k.len() > c.max_identity_bytes)
                || s.keys.iter().collect::<BTreeSet<_>>().len() != s.keys.len()
            {
                return Err(ProtectionError::ResolverScopeMismatch);
            }
            frontier = s.end_token;
        }
        if frontier != p.end_token {
            return Err(ProtectionError::ResolverScopeMismatch);
        }
    }
    if r.scopes
        .iter()
        .any(|s| !p.compatibility.components.contains(&s.component))
    {
        return Err(ProtectionError::ResolverScopeMismatch);
    }
    Ok(())
}

/// Concrete Mooncake 0.3.14 admin adapter. HTTP errors and malformed/missing
/// receipts are unknown effects. No retries occur inside the transport.
pub struct MooncakeAdminBackend {
    client: reqwest::Client,
    base: reqwest::Url,
    max_ttl_ms: u64,
    max_response_bytes: usize,
    api_verified: bool,
}
impl MooncakeAdminBackend {
    pub fn new(
        base: &str,
        timeout: Duration,
        configured_max_ttl_ms: u64,
        max_response_bytes: usize,
    ) -> Result<Self, ProtectionError> {
        let mut base = reqwest::Url::parse(base).map_err(|_| ProtectionError::InvalidRequest)?;
        if !matches!(base.scheme(), "http" | "https")
            || base.host_str().is_none()
            || base.query().is_some()
            || base.fragment().is_some()
            || !base.username().is_empty()
            || base.password().is_some()
            || timeout.is_zero()
            || configured_max_ttl_ms == 0
            || max_response_bytes == 0
        {
            return Err(ProtectionError::InvalidRequest);
        }
        base.set_path("/");
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| ProtectionError::InvalidRequest)?;
        Ok(Self {
            client,
            base,
            max_ttl_ms: configured_max_ttl_ms,
            max_response_bytes,
            api_verified: false,
        })
    }
    /// Probes both deployed admin endpoints using a caller-owned key that is
    /// known absent. Pin/unpin must both return Mooncake's OBJECT_NOT_FOUND
    /// (-704). API discovery creates no positive claim or residency observation.
    /// The configured TTL bound remains an explicit deployment constraint.
    pub async fn verify_api_with_missing_key(
        &mut self,
        tenant_id: &str,
        private_missing_key: &str,
    ) -> Result<(), ProtectionError> {
        self.api_verified = false;
        for mutation in [
            BackendMutation::SoftPin { ttl_ms: 1 },
            BackendMutation::SoftUnpin,
        ] {
            let reply = self
                .mutate(BackendBatch {
                    tenant_id: tenant_id.into(),
                    keys: vec![private_missing_key.into()],
                    mutation,
                })
                .await;
            if !matches!(reply, BackendReply::Quiescent(ref rows) if rows.len() == 1 && rows[0].key == private_missing_key && rows[0].error_code == -704)
            {
                return Err(ProtectionError::UnsupportedProtection);
            }
        }
        self.api_verified = true;
        Ok(())
    }
}
impl SoftPinBackend for MooncakeAdminBackend {
    fn api_verified(&self) -> bool {
        self.api_verified
    }
    fn max_ttl_ms(&self) -> u64 {
        self.max_ttl_ms
    }
    fn mutate<'a>(&'a self, batch: BackendBatch) -> BoxFuture<'a, BackendReply> {
        Box::pin(async move {
            if batch.keys.is_empty()
                || batch.keys.len() > 4096
                || batch.tenant_id.is_empty()
                || batch.keys.iter().any(|k| k.is_empty())
                || batch.keys.iter().collect::<BTreeSet<_>>().len() != batch.keys.len()
            {
                return BackendReply::Unknown("invalid_backend_batch".into());
            }
            let (path, body) = match batch.mutation {
                BackendMutation::SoftPin { ttl_ms } if (1..=self.max_ttl_ms).contains(&ttl_ms) => (
                    "api/v1/soft_pin",
                    serde_json::json!({"tenant_id":batch.tenant_id,"keys":batch.keys,"ttl_ms":ttl_ms}),
                ),
                BackendMutation::SoftPin { .. } => {
                    return BackendReply::Unknown("unsupported_backend_ttl".into())
                }
                BackendMutation::SoftUnpin => (
                    "api/v1/soft_unpin",
                    serde_json::json!({"tenant_id":batch.tenant_id,"keys":batch.keys}),
                ),
            };
            let url = self.base.join(path).expect("fixed relative endpoint");
            let mut response = match self.client.post(url).json(&body).send().await {
                Ok(r) => r,
                Err(e) => return BackendReply::Unknown(format!("http_transport:{e}")),
            };
            if !response.status().is_success() {
                return BackendReply::Unknown(format!("http_status:{}", response.status()));
            }
            let mut bytes = Vec::new();
            loop {
                match response.chunk().await {
                    Ok(Some(chunk))
                        if bytes.len().saturating_add(chunk.len()) <= self.max_response_bytes =>
                    {
                        bytes.extend_from_slice(&chunk)
                    }
                    Ok(Some(_)) => return BackendReply::Unknown("response_budget".into()),
                    Ok(None) => break,
                    Err(e) => return BackendReply::Unknown(format!("http_body:{e}")),
                }
            }
            #[derive(Deserialize)]
            struct WireResult {
                key: String,
                error_code: i64,
            }
            #[derive(Deserialize)]
            struct WireReply {
                results: Vec<WireResult>,
            }
            match serde_json::from_slice::<WireReply>(&bytes) {
                Ok(reply) => BackendReply::Quiescent(
                    reply
                        .results
                        .into_iter()
                        .map(|r| KeyResult {
                            key: r.key,
                            error_code: r.error_code,
                        })
                        .collect(),
                ),
                Err(e) => BackendReply::Unknown(format!("http_receipt:{e}")),
            }
        })
    }
}
