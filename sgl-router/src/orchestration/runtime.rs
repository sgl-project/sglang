use super::feedback::{Effect, FeedbackCache};
use super::{
    client::{confirmed_end, Capabilities, EngineClient},
    OrchestrationConfig,
};
use crate::{
    discovery::{ModelId, WorkerMode},
    workers::{Worker, WorkerRegistry},
};
use axum::http::{HeaderMap, HeaderValue};
use bytes::Bytes;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sgl_kv_orchestrator::backend_protection::{
    BackendProtectionOwner, ClaimIdentity, ClaimMode, ProtectionRequest,
};
use sgl_kv_orchestrator::{
    models::{PrefixTarget, SessionIdentity},
    operations::{DemandBook, DemandId},
    preparation::{Estimate, PreparationCosts},
    protocol::{parse_action, Action},
    session_index::{
        Association, AssociationProvenance, ContentKey, ContentRegistration, RequestIdentity,
        ResourceKind, SessionIndex,
    },
    target_resolver::{FrozenTarget, Selector},
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KvContext {
    pub session_id: String,
    pub incarnation: String,
    pub branch_id: String,
    pub revision: u64,
    pub branch_revision: u64,
    pub request_id: String,
    pub request_incarnation: String,
}
impl KvContext {
    fn task(&self, caller: &str) -> SessionIdentity {
        SessionIdentity {
            caller_id: caller.into(),
            session_id: self.session_id.clone(),
            incarnation: self.incarnation.clone(),
        }
    }
    fn request(&self) -> RequestIdentity {
        RequestIdentity {
            request_id: self.request_id.clone(),
            incarnation: self.request_incarnation.clone(),
        }
    }
}
#[derive(Clone)]
struct Ready {
    url: String,
    prefix: PrefixTarget,
    end: u64,
    until: Instant,
}
#[derive(Default)]
struct Published {
    ready: Vec<Ready>,
}
enum Message {
    Api(Value, oneshot::Sender<Value>),
    Request {
        context: KvContext,
        tokens: Vec<u32>,
        salt: Option<String>,
        worker: String,
    },
    Routed {
        context: KvContext,
        worker: String,
    },
}
/// Cloneable bounded admission handle. All I/O happens in the actor, never under
/// the publication lock or the normal inference task. Queue pressure drops only
/// optional association/preparation signals; inference retains its ordinary path.
pub struct Runtime {
    tx: mpsc::Sender<Message>,
    published: Arc<RwLock<Published>>,
    config: OrchestrationConfig,
    traffic: Arc<AtomicU64>,
}
impl Runtime {
    pub fn start(
        config: OrchestrationConfig,
        registry: Arc<WorkerRegistry>,
        model: ModelId,
        auth: Option<HeaderValue>,
    ) -> Arc<Self> {
        let (tx, rx) = mpsc::channel(config.queue_capacity);
        let published = Arc::new(RwLock::new(Published::default()));
        let traffic = Arc::new(AtomicU64::new(0));
        let runtime = Arc::new(Self {
            tx,
            published: published.clone(),
            config: config.clone(),
            traffic: traffic.clone(),
        });
        let actor = Actor::new(config, registry, model, auth, published, traffic);
        tokio::spawn(actor.run(rx));
        runtime
    }
    pub fn authorized(&self, headers: &HeaderMap) -> bool {
        headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.strip_prefix("Bearer ") == Some(self.config.control_bearer.as_str()))
    }
    pub async fn api(&self, value: Value) -> Value {
        let (tx, rx) = oneshot::channel();
        if self.tx.try_send(Message::Api(value, tx)).is_err() {
            return error("capacity", "orchestration queue full");
        }
        match tokio::time::timeout(
            Duration::from_millis(
                self.config
                    .rpc_timeout_ms
                    .saturating_mul((self.config.max_workers + 4) as u64),
            ),
            rx,
        )
        .await
        {
            Ok(Ok(value)) => value,
            _ => error(
                "unknown",
                "control response unavailable; query with the same message_id",
            ),
        }
    }
    /// Strip Router-only metadata before Engine deserialization. The configured
    /// caller is trusted; body caller_id is never consulted. Incarnations are
    /// explicit, so legacy session_id alone remains an unknown lifecycle signal.
    pub fn extract(&self, body: Bytes) -> Result<(Bytes, Option<KvContext>), String> {
        let mut value: Value = serde_json::from_slice(&body).map_err(|e| e.to_string())?;
        let Some(meta) = value.as_object_mut().and_then(|v| v.remove("kv_context")) else {
            return Ok((body, None));
        };
        let meta: KvContext = serde_json::from_value(meta).map_err(|e| e.to_string())?;
        if value
            .get("session_id")
            .is_some_and(|s| s != &json!(meta.session_id))
        {
            return Err("session_id conflicts with kv_context".into());
        }
        for id in [
            &meta.session_id,
            &meta.incarnation,
            &meta.branch_id,
            &meta.request_id,
            &meta.request_incarnation,
        ] {
            if id.is_empty() || id.len() > self.config.limits.max_identifier_bytes {
                return Err("invalid kv_context identity".into());
            }
        }
        Ok((
            Bytes::from(serde_json::to_vec(&value).map_err(|e| e.to_string())?),
            Some(meta),
        ))
    }
    pub fn associate(
        &self,
        context: KvContext,
        tokens: Vec<u32>,
        salt: Option<String>,
        worker: String,
    ) -> bool {
        if tokens.len() > 65_536 {
            return false;
        }
        self.tx
            .try_send(Message::Request {
                context,
                tokens,
                salt,
                worker,
            })
            .is_ok()
    }
    pub fn note_dispatch(&self) {
        self.traffic.fetch_add(1, Ordering::Relaxed);
    }
    pub fn routed(&self, context: KvContext, worker: String) {
        let _ = self.tx.try_send(Message::Routed { context, worker });
    }
    /// Advisory evidence only. Final selection/admission is performed by existing
    /// policy. Host epoch/worker health is checked again by those candidate filters.
    pub fn prepared_depths(&self, tokens: &[u32], salt: Option<&str>) -> Vec<(String, u64)> {
        let Some(view) = self.published.try_read() else {
            return vec![];
        };
        view.ready
            .iter()
            .filter(|p| {
                p.until > Instant::now()
                    && p.prefix.compatibility.cache_salt.as_deref() == salt
                    && p.prefix
                        .token_ids
                        .as_ref()
                        .is_some_and(|prefix| tokens.starts_with(&prefix[..p.end as usize]))
            })
            .map(|p| (p.url.clone(), p.end))
            .collect()
    }
}
fn error(code: &str, reason: impl ToString) -> Value {
    json!({"success":false,"error":{"code":code,"reason":reason.to_string()}})
}
fn ok(data: Value) -> Value {
    json!({"success":true,"data":data})
}
fn digest(value: &impl Serialize) -> String {
    fn canonical(value: Value) -> Value {
        match value {
            Value::Object(map) => {
                let sorted: BTreeMap<_, _> = map
                    .into_iter()
                    .map(|(key, value)| (key, canonical(value)))
                    .collect();
                Value::Object(sorted.into_iter().collect())
            }
            Value::Array(values) => Value::Array(values.into_iter().map(canonical).collect()),
            value => value,
        }
    }
    format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&canonical(serde_json::to_value(value).expect("identity")))
                .expect("serializable identity")
        )
    )
}
fn preparation_owner(url: &str, caps: &Capabilities) -> String {
    digest(&(
        url,
        &caps.engine_id,
        &caps.epoch,
        &caps.compatibility,
        &caps.public["storage_backend"],
        &caps.public["backend_namespace"],
    ))
}
fn task(value: &Value, caller: &str) -> Result<SessionIdentity, String> {
    Ok(SessionIdentity {
        caller_id: caller.into(),
        session_id: value["session_id"]
            .as_str()
            .ok_or("session_id required")?
            .into(),
        incarnation: value["incarnation"]
            .as_str()
            .ok_or("incarnation required")?
            .into(),
    })
}
fn validate_input(value: &Value) -> Result<(), String> {
    const MAX_TIME_MS: u64 = 86_400_000;
    for key in ["expected_wait_ms", "available_ms", "delay_ms", "ttl_ms"] {
        if let Some(number) = value.get(key) {
            if number.as_u64().is_none_or(|n| n > MAX_TIME_MS) {
                return Err(format!("{key} requires an integer in 0..86400000"));
            }
        }
    }
    for key in ["revision", "branch_revision", "attempt"] {
        if let Some(number) = value.get(key) {
            if number
                .as_u64()
                .is_none_or(|n| n > sgl_kv_orchestrator::protocol::MAX_INTEGER)
            {
                return Err(format!("{key} exceeds the safe integer domain"));
            }
        }
    }
    if let Some(ttl) = value.get("ttl_seconds").filter(|ttl| !ttl.is_null()) {
        if ttl
            .as_f64()
            .is_none_or(|n| !n.is_finite() || n <= 0.0 || n > 31_536_000.0)
        {
            return Err("invalid ttl_seconds".into());
        }
    }
    for tokens in [
        value.get("input_ids"),
        value.get("target").and_then(|v| v.get("token_ids")),
    ]
    .into_iter()
    .flatten()
    .filter(|v| !v.is_null())
    {
        let tokens = tokens.as_array().ok_or("tokens must be a flat array")?;
        if tokens.len() > 65_536
            || tokens
                .iter()
                .any(|token| token.as_u64().is_none_or(|n| n > u32::MAX as u64))
        {
            return Err("token capacity/type violation".into());
        }
    }
    if let Some(revision) = value
        .get("target")
        .and_then(|target| target.get("revision"))
    {
        if revision
            .as_u64()
            .is_none_or(|n| n > sgl_kv_orchestrator::protocol::MAX_INTEGER)
        {
            return Err("invalid selector revision".into());
        }
    }
    if value["target"]["blocks"]
        .as_array()
        .is_some_and(|blocks| blocks.len() > 4096)
    {
        return Err("block descriptor capacity".into());
    }
    Ok(())
}
struct Job {
    id: String,
    url: String,
    caps: Capabilities,
    prefix: PrefixTarget,
    action: String,
    payload: Value,
    context: Value,
    result: Value,
    operation_ref: Option<Value>,
    claim_view: Value,
    submitted: bool,
    attempted: bool,
    message_id: String,
    releasing_pin: bool,
    cancel_sent: bool,
    quiescent: bool,
    due: Instant,
    deadline: Instant,
    last_error: Option<String>,
    started: Option<Instant>,
    cost_recorded: bool,
    traffic_at_start: u64,
    cache_watermark: Value,
    ready_end: u64,
    ready_at: Option<Instant>,
    preparation_estimate: Option<Estimate>,
    ready_pin_eligible: bool,
    ready_pin: Option<Box<Job>>,
    ready_pin_error: Option<String>,
}
impl Job {
    fn material_bytes(&self) -> usize {
        let tokens = self.prefix.token_ids.as_ref().map_or(0, Vec::len)
            + self.payload["target"]["token_ids"]
                .as_array()
                .map_or(0, Vec::len);
        tokens.saturating_mul(4).saturating_add(
            self.ready_pin
                .as_ref()
                .map_or(0, |pin| pin.material_bytes()),
        )
    }

    fn pin_inactive(&self) -> bool {
        self.quiescent
            && ((!self.attempted && !self.releasing_pin)
                || (self.last_error.is_none()
                    && (self.pin_refused()
                        || matches!(
                            self.claim_view["state"].as_str(),
                            Some("released" | "expired" | "invalidated")
                        ))))
    }

    fn pin_refused(&self) -> bool {
        // Only internal children use a fresh claim key for each episode.
        // A refused explicit renewal can still have an older active claim.
        self.id.starts_with("ready-pin-")
            && self.quiescent
            && !self.releasing_pin
            && matches!(
                self.result["update"]["status"].as_str(),
                Some("missing" | "deferred" | "unsupported" | "failed" | "cancelled")
            )
            && self.result["public_result"]["claim"].is_null()
    }
}
struct Intent {
    fingerprint: String,
    admission_gaps: Vec<String>,
    jobs: Vec<String>,
    frozen: FrozenTarget,
    expires: Instant,
    released: bool,
    claims: Vec<ClaimIdentity>,
}
struct ClaimRecord {
    task: SessionIdentity,
    request: ProtectionRequest,
    view: Value,
    release_requested: bool,
    interests: BTreeSet<DemandId>,
    feedback_unknown: Option<String>,
    error: Option<String>,
}
struct Actor {
    config: OrchestrationConfig,
    client: EngineClient,
    registry: Arc<WorkerRegistry>,
    model: ModelId,
    published: Arc<RwLock<Published>>,
    index: SessionIndex,
    book: DemandBook,
    jobs: BTreeMap<String, Job>,
    intents: BTreeMap<DemandId, Intent>,
    caps: BTreeMap<String, Capabilities>,
    last_route: BTreeMap<SessionIdentity, String>,
    tick_cursor: usize,
    wire_incarnation: String,
    next_wire_message: u64,
    events: BTreeMap<DemandId, (String, Value)>,
    cost_ms_per_token: f64,
    preparation_costs: Option<PreparationCosts>,
    traffic: Arc<AtomicU64>,
    backend_owner: Option<BackendProtectionOwner>,
    backend_error: Option<String>,
    claims: BTreeMap<ClaimIdentity, ClaimRecord>,
    feedback: FeedbackCache,
}
impl Actor {
    fn new(
        config: OrchestrationConfig,
        registry: Arc<WorkerRegistry>,
        model: ModelId,
        auth: Option<HeaderValue>,
        published: Arc<RwLock<Published>>,
        traffic: Arc<AtomicU64>,
    ) -> Self {
        Self {
            client: EngineClient::new(config.rpc_timeout_ms, config.max_engine_bytes, auth),
            index: SessionIndex::new(config.limits.clone()),
            book: DemandBook::new(config.limits.max_intents, config.max_operations),
            config: config.clone(),
            registry,
            model,
            published,
            jobs: BTreeMap::new(),
            intents: BTreeMap::new(),
            caps: BTreeMap::new(),
            last_route: BTreeMap::new(),
            tick_cursor: 0,
            wire_incarnation: uuid::Uuid::new_v4().to_string(),
            next_wire_message: 0,
            events: BTreeMap::new(),
            cost_ms_per_token: config.initial_prefetch_ms_per_token,
            preparation_costs: config
                .preparation_policy
                .clone()
                .map(|policy| PreparationCosts::new(policy).expect("validated preparation config")),
            traffic,
            backend_owner: None,
            backend_error: None,
            claims: BTreeMap::new(),
            feedback: FeedbackCache::new(
                config.max_workers,
                config.max_operations,
                config.max_engine_bytes,
            ),
        }
    }
    async fn run(mut self, mut rx: mpsc::Receiver<Message>) {
        match super::protection::start(
            &self.config,
            self.registry.clone(),
            self.model.clone(),
            self.client.clone(),
        )
        .await
        {
            Ok(owner) => self.backend_owner = owner,
            Err(reason) => self.backend_error = Some(reason),
        }
        let mut tick = tokio::time::interval(Duration::from_millis(self.config.poll_ms));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                message=rx.recv()=>match message {
                    Some(Message::Api(value,reply))=> { let result=self.api(value).await;let _=reply.send(result); }
                    Some(Message::Request {context,tokens,salt,worker})=> { if let Err(reason)=self.record(context,tokens,salt,worker).await { tracing::debug!(%reason,"KV association unavailable"); } }
                    Some(Message::Routed {context,worker})=> { self.changed_route(&context.task(&self.config.caller_id),&worker); }
                    None=>break,
                },
                _=tick.tick()=>self.pump().await,
            }
            self.publish();
        }
        // Actor shutdown never fabricates quiescence. Engine leases/deadlines
        // remain authoritative; explicit Router cleanup drains while running.
    }
    fn workers(&self) -> Vec<Arc<Worker>> {
        let mut workers: Vec<_> = self
            .registry
            .healthy_workers_for(&self.model)
            .into_iter()
            .filter(|w| w.mode() == WorkerMode::Plain || w.mode() == WorkerMode::Prefill)
            .collect();
        workers.sort_by(|a, b| {
            a.router_inflight_load()
                .cmp(&b.router_inflight_load())
                .then_with(|| a.id.0.cmp(&b.id.0))
        });
        workers.truncate(self.config.max_workers);
        workers
    }
    async fn refresh(&mut self, url: &str) -> Result<Capabilities, String> {
        match self.client.capabilities(url).await {
            Ok(caps) => {
                let effect = self.feedback.bind(url, &caps);
                self.invalidate_feedback(url, effect, "discovered owner/source change");
                if self.caps.get(url).is_some_and(|old| {
                    old.epoch != caps.epoch
                        || old.generation != caps.generation
                        || old.compatibility != caps.compatibility
                }) {
                    for job in self.jobs.values_mut().filter(|j| j.url == url) {
                        job.ready_end = 0;
                        job.ready_at = None;
                    }
                }
                if self.caps.contains_key(url) || self.caps.len() < self.config.max_workers {
                    self.caps.insert(url.into(), caps.clone());
                }
                Ok(caps)
            }
            Err(e) => {
                let effect = self.feedback.fail(url, "capabilities unavailable");
                self.invalidate_feedback(url, effect, "capabilities unavailable");
                self.caps.remove(url);
                for job in self.jobs.values_mut().filter(|j| j.url == url) {
                    job.ready_end = 0;
                    job.ready_at = None;
                }
                Err(e)
            }
        }
    }
    fn invalidate_feedback(&mut self, url: &str, effect: Effect, reason: &str) {
        for job in self.jobs.values_mut().filter(|job| job.url == url) {
            if effect.ready {
                job.ready_end = 0;
                job.ready_at = None;
            }
            if effect.claims {
                job.claim_view = Value::Null;
            }
        }
        if effect.storage {
            for claim in self.claims.values_mut() {
                claim.feedback_unknown = Some(reason.into());
                suppress_protection(&mut claim.view);
            }
        }
        // Both normal routing paths read this publication. No I/O holds its lock.
        self.publish();
    }
    async fn poll_feedback(&mut self, url: &str, caps: &Capabilities) -> Result<(), String> {
        let limit = self.config.queue_capacity.min(256).max(1);
        let request = self
            .feedback
            .command(url, caps.context(&self.config.caller_id), limit);
        let result = match request {
            Ok(None) => return Ok(()),
            Ok(Some(command)) => match self.client.command(url, &command).await {
                Ok(page) => self
                    .feedback
                    .apply(url, &self.config.caller_id, &page, limit),
                Err(reason) => Err(reason),
            },
            Err(reason) => Err(reason),
        };
        match result {
            Ok(effect) => {
                self.invalidate_feedback(url, effect, "feedback invalidation/gap");
                Ok(())
            }
            Err(reason) => {
                let effect = self.feedback.fail(url, &reason);
                self.invalidate_feedback(url, effect, &reason);
                Err(reason)
            }
        }
    }
    async fn record(
        &mut self,
        context: KvContext,
        tokens: Vec<u32>,
        salt: Option<String>,
        worker: String,
    ) -> Result<(), String> {
        if !self.workers().iter().any(|w| w.url == worker) {
            return Err("worker unavailable".into());
        }
        let caps = self.refresh(&worker).await?;
        let task = context.task(&self.config.caller_id);
        if self.index.session_revision(&task).is_none() {
            self.index
                .start_session(task.clone(), None)
                .map_err(|e| format!("{e:?}"))?;
        }
        let mut compatibility = caps.compatibility;
        compatibility.cache_salt = salt;
        let page = compatibility.page_size as usize;
        let end = tokens.len() / page * page;
        if end == 0 {
            return Err("no complete page".into());
        }
        let mut parent = None;
        let mut handles = Vec::new();
        // Page-prefix digests are adapter identities, not Engine storage keys.
        // Full content compatibility participates in every digest via ContentKey.
        for (i, chunk) in tokens[..end].chunks(page).enumerate() {
            let key = ContentKey::new(
                compatibility.clone(),
                digest(&(parent.as_ref().map(ContentKey::digest), chunk)),
            )
            .map_err(|e| format!("{e:?}"))?;
            self.index
                .register_content(ContentRegistration {
                    key: key.clone(),
                    parent: parent.clone(),
                    end_token: ((i + 1) * page) as u64,
                    suffix_tokens: Some(chunk.to_vec()),
                })
                .map_err(|e| format!("{e:?}"))?;
            handles.push(self.index.content(&key));
            parent = Some(key);
        }
        self.index
            .associate(Association {
                session: task.clone(),
                session_revision: context.revision,
                branch_id: context.branch_id.clone(),
                branch_revision: context.branch_revision,
                content: parent.expect("complete page"),
                parent_frontier: None,
                request: Some(context.request()),
                provenance: AssociationProvenance::RenderedRequest,
            })
            .map_err(|e| format!("{e:?}"))?;
        self.changed_route(&task, &worker);
        Ok(())
    }
    fn changed_route(&mut self, task: &SessionIdentity, worker: &str) {
        if self.index.session_revision(task).is_none() {
            return;
        }
        self.last_route.insert(task.clone(), worker.into());
        // Only this task's useless preparation interests are withdrawn. Shared
        // jobs and other tasks remain. Ordinary cache stays under Engine lifetime.
        let ids: Vec<_> = self
            .intents
            .iter()
            .filter(|(d, i)| {
                &d.task == task
                    && !i.released
                    && !i.jobs.is_empty()
                    && i.jobs.iter().all(|id| {
                        self.jobs
                            .get(id)
                            .is_some_and(|j| j.action == "sglang.prefetch" && j.url != worker)
                    })
            })
            .map(|(d, _)| d.clone())
            .collect();
        for demand in ids {
            self.release(&demand);
        }
    }
    fn release(&mut self, demand: &DemandId) {
        self.book.release(demand);
        if let Some(intent) = self.intents.get_mut(demand) {
            intent.released = true;
            for identity in &intent.claims {
                if let Some(claim) = self.claims.get_mut(identity) {
                    claim.interests.remove(demand);
                    if claim.interests.is_empty() {
                        claim.release_requested = true;
                    }
                }
            }
        }
        let _ = self
            .index
            .untrack_resource(&demand.task, ResourceKind::Intent, &demand.id);
    }
    fn resolve(&mut self, target: &Value, task: &SessionIdentity) -> Result<FrozenTarget, String> {
        let selector = match target["kind"].as_str().unwrap_or("session") {
            "session" => Selector::Session(task.clone()),
            "branch" => Selector::Branch {
                session: task.clone(),
                branch_id: target["branch_id"]
                    .as_str()
                    .ok_or("branch_id required")?
                    .into(),
                revision: target["revision"].as_u64(),
            },
            "request" => Selector::Request {
                session: task.clone(),
                request: RequestIdentity {
                    request_id: target["request_id"]
                        .as_str()
                        .ok_or("request_id required")?
                        .into(),
                    incarnation: target["request_incarnation"]
                        .as_str()
                        .ok_or("request_incarnation required")?
                        .into(),
                },
            },
            "prefix" => {
                let prefix: PrefixTarget =
                    serde_json::from_value(target.clone()).map_err(|e| e.to_string())?;
                // Reuse P1 structural/dependency validation before creating state.
                let action = Action {
                    action_id: "validate".into(),
                    action_type: "sglang.prefetch".into(),
                    action_version: "1.0".into(),
                    payload: json!({"target":prefix,"source":{"tier":"storage"},"destination":{"tier":"host"}}),
                };
                parse_action(&action).map_err(|e| format!("{e:?}"))?;
                let key = ContentKey::new(prefix.compatibility.clone(), digest(&prefix))
                    .map_err(|e| format!("{e:?}"))?;
                self.index
                    .register_content(ContentRegistration {
                        key: key.clone(),
                        parent: None,
                        end_token: prefix.end_token,
                        suffix_tokens: prefix.token_ids,
                    })
                    .map_err(|e| format!("{e:?}"))?;
                Selector::Prefix {
                    content: key,
                    start_token: prefix.start_token,
                    end_token: prefix.end_token,
                }
            }
            _ => return Err("unsupported selector".into()),
        };
        Ok(self.index.resolve(selector))
    }
    async fn api(&mut self, value: Value) -> Value {
        if let Err(reason) = validate_input(&value) {
            return error("invalid_payload", reason);
        }
        let event_key = if value["kind"] == "session_event" {
            task(value.get("task").unwrap_or(&value), &self.config.caller_id)
                .ok()
                .zip(value["event_id"].as_str())
                .map(|(task, id)| DemandId {
                    task,
                    id: id.into(),
                })
        } else {
            None
        };
        let fingerprint = digest(&value);
        if let Some(key) = &event_key {
            if let Some((old, receipt)) = self.events.get(key) {
                return if old == &fingerprint {
                    receipt.clone()
                } else {
                    error("conflict", "event ID payload conflict")
                };
            }
            if self.events.len() >= self.config.max_operations {
                return error("capacity", "lifecycle replay budget exhausted");
            }
            if matches!(
                value["event"].as_str(),
                Some("tool_wait" | "resume_expected" | "tool_done")
            ) && self.index.session_revision(&key.task).is_some_and(|r| {
                value["revision"]
                    .as_u64()
                    .is_none_or(|revision| revision <= r)
            }) {
                return error("conflict", "stale lifecycle revision");
            }
        }
        let result = match self.apply(value.clone()).await {
            Ok(data) => ok(data),
            Err(e) => error(
                if e.to_lowercase().contains("capacity") {
                    "capacity"
                } else {
                    "rejected"
                },
                e,
            ),
        };
        if result["success"] == true {
            if let Some(key) = event_key {
                if matches!(
                    value["event"].as_str(),
                    Some("tool_wait" | "resume_expected" | "tool_done")
                ) {
                    let _ = self
                        .index
                        .advance_revision(&key.task, value["revision"].as_u64().unwrap_or(0));
                }
                self.events.insert(key, (fingerprint, result.clone()));
            }
        }
        result
    }
    async fn apply(&mut self, value: Value) -> Result<Value, String> {
        let kind = value["kind"].as_str().ok_or("kind required")?;
        if kind == "feedback" {
            return Ok(self.feedback.view());
        }
        if kind == "capabilities" {
            let backend = match &self.backend_owner {
                Some(owner) => {
                    serde_json::to_value(owner.query(None).await.map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?
                }
                None => Value::Null,
            };
            return Ok(
                json!({"enabled":true,"caller_id":self.config.caller_id,"engine_session_required":false,"lifecycle":"explicit","protection_owner":if self.backend_owner.is_some(){"configured"}else{"unavailable"},"backend_protection":backend,"protection_error":self.backend_error,"feedback":self.feedback.view(),"prefetch_ms_per_token":self.cost_ms_per_token,"preparation_policy":self.config.preparation_policy,"preparation_cost_groups":self.preparation_costs.as_ref().map(|model|model.groups()),"index":format!("{:?}",self.index.usage())}),
            );
        }
        let task = task(value.get("task").unwrap_or(&value), &self.config.caller_id)?;
        let id = value["message_id"]
            .as_str()
            .or_else(|| value["event_id"].as_str())
            .unwrap_or("");
        if id.is_empty() || id.len() > self.config.limits.max_identifier_bytes {
            return Err("message_id/event_id required within identifier budget".into());
        }
        let demand = DemandId {
            task: task.clone(),
            id: id.into(),
        };
        match kind {
            "query" => {
                let jobs = self
                    .intents
                    .get(&demand)
                    .ok_or("unknown intent")?
                    .jobs
                    .clone();
                for id in jobs {
                    self.poll_job(&id).await;
                }
                if let Some(intent) = self.intents.get(&demand) {
                    for identity in intent.claims.clone() {
                        self.poll_claim(&identity).await;
                    }
                }
                Ok(self.intent_view(&demand))
            }
            "cancel" => {
                if !self.intents.contains_key(&demand) {
                    return Err("unknown intent".into());
                }
                self.release(&demand);
                Ok(self.intent_view(&demand))
            }
            "ack" => {
                let ids = self
                    .intents
                    .get(&demand)
                    .ok_or("unknown intent")?
                    .jobs
                    .clone();
                if !self.intents[&demand].released
                    && (!self.intents[&demand].claims.is_empty()
                        || ids.iter().any(|id| {
                            self.jobs.get(id).is_some_and(|job| {
                                job.action == "sglang.pin" && job.payload["mode"] != "release"
                            })
                        }))
                {
                    return Err("cancel or release the PIN intent before acknowledging".into());
                }
                self.release(&demand);
                for id in &ids {
                    self.poll_job(id).await;
                }
                let identities = self.intents[&demand].claims.clone();
                for identity in &identities {
                    self.poll_claim(identity).await;
                }
                if ids
                    .iter()
                    .any(|id| self.jobs.get(id).is_some_and(|j| !self.job_quiescent(j)))
                    || identities.iter().any(|id| {
                        self.claims
                            .get(id)
                            .is_some_and(|c| c.view["backend_quiescent"] != true)
                    })
                {
                    return Err("operation/claim not quiescent".into());
                }
                self.intents.remove(&demand);
                for id in ids {
                    if self.book.forget_job(&id) {
                        self.jobs.remove(&id);
                    }
                }
                self.index.collect_garbage();
                Ok(json!({"acknowledged":true}))
            }
            "session_event" => {
                let event = value["event"].as_str().ok_or("event required")?;
                let revision = value["revision"].as_u64().ok_or("revision required")?;
                match event {
                    "task_started" => {
                        let parent = value
                            .get("parent")
                            .map(|p| task_fn(p, &self.config.caller_id))
                            .transpose()?;
                        self.index
                            .start_session(task, parent)
                            .map_err(|e| format!("{e:?}"))?;
                        Ok(json!({"state":"active","lifecycle":"explicit"}))
                    }
                    "task_finished" | "task_cancelled" => {
                        let cleanup = self
                            .index
                            .finish_task_with_cascade(
                                &task,
                                revision,
                                value["cascade"].as_bool().unwrap_or(false),
                            )
                            .map_err(|e| format!("{e:?}"))?;
                        for ended in &cleanup.tasks {
                            self.book.release_task(ended);
                            self.last_route.remove(ended);
                            for claim in self
                                .claims
                                .values_mut()
                                .filter(|claim| &claim.task == ended)
                            {
                                claim.release_requested = true;
                            }
                            for (d, i) in self.intents.iter_mut().filter(|(d, _)| &d.task == ended)
                            {
                                let _ = d;
                                i.released = true;
                            }
                        }
                        Ok(
                            json!({"state":"cleanup_requested","tasks":cleanup.tasks,"claims":cleanup.claims.iter().map(|c|json!({"task":c.task,"claim_id":c.id})).collect::<Vec<_>>(),
                            "intents":cleanup.intents.iter().map(|i|json!({"task":i.task,"intent_id":i.id})).collect::<Vec<_>>(),"quiescent":false}),
                        )
                    }
                    "request_observed" => {
                        let tokens: Vec<u32> = serde_json::from_value(value["input_ids"].clone())
                            .map_err(|e| e.to_string())?;
                        if tokens.len() > 65_536 {
                            return Err("token capacity".into());
                        }
                        let context = KvContext {
                            session_id: task.session_id.clone(),
                            incarnation: task.incarnation.clone(),
                            branch_id: value["branch_id"]
                                .as_str()
                                .ok_or("branch_id required")?
                                .into(),
                            revision,
                            branch_revision: value["branch_revision"]
                                .as_u64()
                                .ok_or("branch_revision required")?,
                            request_id: value["request_id"]
                                .as_str()
                                .ok_or("request_id required")?
                                .into(),
                            request_incarnation: value["request_incarnation"]
                                .as_str()
                                .ok_or("request_incarnation required")?
                                .into(),
                        };
                        self.record(
                            context,
                            tokens,
                            value["cache_salt"].as_str().map(str::to_string),
                            value["worker_url"]
                                .as_str()
                                .ok_or("worker_url required")?
                                .into(),
                        )
                        .await?;
                        Ok(json!({"state":"associated","residency":"unknown"}))
                    }
                    "request_finished" => {
                        let r: RequestIdentity = RequestIdentity {
                            request_id: value["request_id"]
                                .as_str()
                                .ok_or("request_id required")?
                                .into(),
                            incarnation: value["request_incarnation"]
                                .as_str()
                                .ok_or("request_incarnation required")?
                                .into(),
                        };
                        self.index
                            .forget_request(&task, &r, revision)
                            .map_err(|e| format!("{e:?}"))?;
                        Ok(json!({"state":"request_unbound","task_state":"active"}))
                    }
                    "tool_wait" | "resume_expected" | "tool_done" => {
                        if self.index.session_revision(&task).is_none() {
                            return Err("unknown task incarnation".into());
                        }
                        let target =
                            self.resolve(value.get("target").unwrap_or(&Value::Null), &task)?;
                        if event == "tool_done" && self.preparation_costs.is_some() {
                            let prefixes: Vec<_> = target
                                .known_scope()
                                .iter()
                                .filter_map(|scope| self.index.materialize(scope.scope()).ok())
                                .collect();
                            for (demand, intent) in &self.intents {
                                if demand.task != task || intent.released {
                                    continue;
                                }
                                for id in &intent.jobs {
                                    if let Some(job) = self.jobs.get_mut(id) {
                                        if !job.quiescent
                                            && job.action == "sglang.prefetch"
                                            && value["worker_url"]
                                                .as_str()
                                                .is_none_or(|url| url == job.url)
                                            && prefixes.iter().any(|prefix| {
                                                prefix.compatibility == job.prefix.compatibility
                                                    && prefix.token_ids == job.prefix.token_ids
                                            })
                                        {
                                            job.due = Instant::now();
                                        }
                                    }
                                }
                            }
                        }
                        let tokens = target
                            .known_scope()
                            .iter()
                            .map(|s| s.scope().end_token())
                            .max()
                            .unwrap_or(0);
                        let owner = value["worker_url"]
                            .as_str()
                            .or_else(|| self.last_route.get(&task).map(String::as_str))
                            .and_then(|url| {
                                self.caps.get(url).map(|caps| preparation_owner(url, caps))
                            })
                            .unwrap_or_default();
                        let estimate = self.preparation_costs.as_ref().map(|model| {
                            model.estimate(
                                &owner,
                                tokens,
                                self.config.initial_prefetch_ms_per_token,
                                self.config.lead_margin_ms,
                            )
                        });
                        let cost = estimate.map_or_else(
                            || (tokens as f64 * self.cost_ms_per_token).ceil() as u64,
                            |estimate| estimate.preparation_ms,
                        );
                        let lead = estimate.map_or_else(
                            || cost.saturating_add(self.config.lead_margin_ms),
                            |estimate| estimate.lead_ms,
                        );
                        if event == "tool_done"
                            && cost
                                > value["available_ms"]
                                    .as_u64()
                                    .unwrap_or(self.config.max_resume_prefetch_ms)
                        {
                            return Ok(
                                json!({"state":"on_demand_fallback","estimated_ms":cost,"reason":"preparation too late"}),
                            );
                        }
                        if event == "resume_expected"
                            && (value["expected_wait_ms"].as_u64().unwrap_or(0) == 0
                                || (self.preparation_costs.is_none()
                                    && value["expected_wait_ms"].as_u64().unwrap_or(0) < lead))
                        {
                            return Ok(
                                json!({"state":"on_demand_fallback","estimated_ms":cost,"reason":"insufficient preparation lead"}),
                            );
                        }
                        if event == "tool_wait" && value["retention"].is_object() {
                            let retention = &value["retention"];
                            let pin_id = format!("policy-pin-{}", digest(&(&task, id)));
                            let pin = json!({"kind":"submit","task":value.get("task").unwrap_or(&value),"message_id":pin_id,
                                "action":"sglang.pin","target":value.get("target").cloned().unwrap_or(Value::Null),
                                "location":{"tier":retention["tier"]},"mode":"acquire","claim_id":pin_id,"revision":revision,
                                "ttl_seconds":retention["ttl_seconds"],"required_protection":retention["required_protection"]});
                            let pin_demand = DemandId {
                                task: task.clone(),
                                id: pin_id,
                            };
                            let retained = self.submit(pin, task.clone(), pin_demand).await;
                            if !value["pressure"].as_bool().unwrap_or(false) {
                                return Ok(
                                    json!({"retention":retained.unwrap_or_else(|e|error("unsupported",e)),"demote":"skipped_no_pressure"}),
                                );
                            }
                        }
                        if event == "tool_wait"
                            && value["source"]["tier"] == "host"
                            && self
                                .config
                                .preparation_policy
                                .as_ref()
                                .is_some_and(|policy| {
                                    value["expected_wait_ms"]
                                        .as_u64()
                                        .unwrap_or(0)
                                        .saturating_sub(lead)
                                        <= policy.min_cold_residency_ms
                                })
                        {
                            return Ok(
                                json!({"state":"skipped","reason":"insufficient expected cold residency","estimated_preparation_ms":cost}),
                            );
                        }
                        if event == "tool_wait"
                            && (!value["pressure"].as_bool().unwrap_or(false)
                                || value["expected_wait_ms"].as_u64().unwrap_or(0)
                                    < self.config.poll_ms)
                        {
                            return Ok(
                                json!({"state":"skipped","reason":"insufficient pressure/wait evidence"}),
                            );
                        }
                        let mut command = value.clone();
                        command["kind"] = json!("submit");
                        command["message_id"] = json!(id);
                        if event == "resume_expected" {
                            command["delay_ms"] = json!(value["expected_wait_ms"]
                                .as_u64()
                                .unwrap_or(0)
                                .saturating_sub(lead));
                        }
                        command["action"] = json!(if event == "tool_wait" {
                            "sglang.demote"
                        } else {
                            "sglang.prefetch"
                        });
                        let demote_source = if self.preparation_costs.is_some()
                            && value["source"]["tier"] == "host"
                        {
                            "host"
                        } else {
                            "gpu"
                        };
                        command["source"] =
                            json!({"tier":if event=="tool_wait"{demote_source}else{"storage"}});
                        command["destination"] =
                            json!({"tier":if event=="tool_wait"{"storage"}else{"host"}});
                        if event == "tool_done" && self.preparation_costs.is_none() {
                            let ids: Vec<_> = self
                                .intents
                                .keys()
                                .filter(|d| d.task == task)
                                .cloned()
                                .collect();
                            for d in ids {
                                for job in self.intents[&d].jobs.clone() {
                                    if let Some(j) = self.jobs.get_mut(&job) {
                                        if j.action == "sglang.prefetch" {
                                            j.due = Instant::now();
                                        }
                                    }
                                }
                            }
                        }
                        let mut result = self.submit(command, task, demand).await?;
                        if self.preparation_costs.is_some() {
                            result["preparation_estimate"] = result["operations"]
                                .as_array()
                                .and_then(|jobs| jobs.first())
                                .map(|job| job["preparation_estimate"].clone())
                                .unwrap_or(Value::Null);
                        }
                        Ok(result)
                    }
                    _ => Err("unsupported lifecycle signal".into()),
                }
            }
            "submit" => self.submit(value, task, demand).await,
            _ => Err("unsupported Router control kind".into()),
        }
    }
    async fn submit(
        &mut self,
        value: Value,
        task: SessionIdentity,
        demand: DemandId,
    ) -> Result<Value, String> {
        let admitted_at = Instant::now();
        let fingerprint = digest(&value);
        if let Some(intent) = self.intents.get(&demand) {
            return if intent.fingerprint == fingerprint {
                Ok(self.intent_view(&demand))
            } else {
                Err("message ID payload conflict".into())
            };
        }
        if self.intents.len() >= self.config.max_operations {
            return Err("intent replay capacity; acknowledge completed intents".into());
        }
        if self.index.session_revision(&task).is_none() {
            return Err("task_started required".into());
        }
        let frozen = self.resolve(value.get("target").unwrap_or(&Value::Null), &task)?;
        let mut unique = BTreeSet::new();
        let mut planned_bytes = 0usize;
        for scope in frozen.known_scope() {
            if unique.insert((
                scope.scope().key(),
                scope.scope().start_token(),
                scope.scope().end_token(),
            )) {
                planned_bytes = planned_bytes
                    .checked_add(
                        usize::try_from(scope.scope().end_token())
                            .ok()
                            .and_then(|n| n.checked_mul(4))
                            .ok_or("token material capacity")?,
                    )
                    .ok_or("token material capacity")?;
            }
        }
        if unique.len() > self.config.max_operations
            || planned_bytes > self.config.limits.max_token_material_bytes
        {
            return Err("fanout/token material capacity".into());
        }

        drop(unique);
        let action = value["action"].as_str().ok_or("action required")?;
        if !matches!(action, "sglang.prefetch" | "sglang.demote" | "sglang.pin") {
            return Err("unsupported action".into());
        }
        let source = value["source"]["tier"].as_str().unwrap_or("storage");
        let destination = value["destination"]["tier"].as_str().unwrap_or("host");
        let pin_tier = value["location"]["tier"].as_str().unwrap_or("host");
        if action == "sglang.pin" && pin_tier == "storage" {
            return self
                .submit_l3(value, task, demand, fingerprint, frozen)
                .await;
        }
        let mut workers = self.workers();
        if let Some(url) = value["worker_url"].as_str() {
            workers.retain(|w| w.url == url);
        }
        // A read-only affinity preference. No policy.select, load guard, pending
        // stake or session-affinity commit is used for preparation placement.
        if let Some(preferred) = self.last_route.get(&task) {
            workers.sort_by_key(|w| (w.url != *preferred, w.router_inflight_load()));
        }
        let mut available = Vec::new();
        for w in workers {
            if let Ok(caps) = self.refresh(&w.url).await {
                available.push((w.url.clone(), caps));
            }
        }
        let due = Instant::now()
            + Duration::from_millis(
                value["delay_ms"]
                    .as_u64()
                    .unwrap_or(0)
                    .min(self.config.operation_ttl_ms),
            );
        let mut expires = Instant::now()
            + Duration::from_millis(
                value["ttl_ms"]
                    .as_u64()
                    .unwrap_or(self.config.operation_ttl_ms)
                    .min(self.config.operation_ttl_ms),
            );
        if action == "sglang.pin" {
            if let Some(ttl) = value["ttl_seconds"]
                .as_f64()
                .filter(|ttl| ttl.is_finite() && *ttl > 0.0 && *ttl <= 31_536_000.0)
            {
                expires = Instant::now() + Duration::from_secs_f64(ttl);
            }
        }
        let mut planned = BTreeMap::new();
        let mut ids = BTreeSet::new();
        let mut missing = Vec::new();
        for resolved in frozen.known_scope() {
            let prefix = match self.index.materialize(resolved.scope()) {
                Ok(p) => p,
                Err(gaps) => {
                    missing.push(format!("{gaps:?}"));
                    continue;
                }
            };
            let Some((url, caps)) = available.iter().find(|(_, c)| {
                c.compatible(&prefix)
                    && if action == "sglang.pin" {
                        c.public["actions"][action]
                            .as_array()
                            .is_some_and(|a| a.iter().any(|v| v == "1.0"))
                    } else {
                        c.path(action, source, destination)
                    }
            }) else {
                missing.push("no healthy compatible executable worker".into());
                continue;
            };
            let preparation_estimate = self.preparation_costs.as_ref().and_then(|model| {
                (action == "sglang.prefetch"
                    && source == "storage"
                    && destination == "host"
                    && value["event"] == "resume_expected")
                    .then(|| {
                        model.estimate(
                            &preparation_owner(url, caps),
                            prefix.end_token,
                            self.config.initial_prefetch_ms_per_token,
                            self.config.lead_margin_ms,
                        )
                    })
            });
            let job_due = preparation_estimate.map_or(due, |estimate| {
                admitted_at
                    + Duration::from_millis(
                        value["expected_wait_ms"]
                            .as_u64()
                            .unwrap_or(0)
                            .saturating_sub(estimate.lead_ms),
                    )
            });
            let mut src = value["source"].clone();
            let mut dst = value["destination"].clone();
            for location in [&mut src, &mut dst] {
                if location["tier"] == "storage" {
                    location["backend"] = caps.public["storage_backend"].clone();
                    location["backend_namespace"] = caps.public["backend_namespace"].clone();
                } else if location.is_object() {
                    location["engine_id"] = json!(caps.engine_id);
                }
            }
            let payload = if action == "sglang.pin" {
                json!({"target":prefix,"location":{"tier":pin_tier,"engine_id":caps.engine_id},"mode":value["mode"],
                "claim_id":format!("claim-{}",digest(&(self.config.caller_id.as_str(),&task,&value["claim_id"],&prefix))),"revision":value["revision"],
                "ttl_seconds":value["ttl_seconds"],"required_protection":value.get("required_protection").cloned().unwrap_or(json!("eviction_exclusion"))})
            } else {
                json!({"target":prefix,"source":src,"destination":dst,"allow_partial":value["allow_partial"].as_bool().unwrap_or(false),"timeout_ms":self.config.operation_ttl_ms})
            };
            let parsed = Action {
                action_id: "transfer".into(),
                action_type: action.into(),
                action_version: "1.0".into(),
                payload: payload.clone(),
            };
            parse_action(&parsed).map_err(|e| format!("{e:?}"))?;
            // Exact immutable spec + owner epoch coalesces across tasks. PINs
            // retain their claim identity, so different claims never merge.
            let id = format!(
                "job-{}",
                digest(&(
                    url,
                    &caps.epoch,
                    action,
                    &payload,
                    value["attempt"].as_u64().unwrap_or(0)
                ))
            );
            ids.insert(id.clone());
            if !self.jobs.contains_key(&id) {
                // Concurrent interests share the logical job. A new lifecycle
                // episode needs a fresh Engine envelope, while lost replies
                // must retry the same envelope within this episode.
                self.next_wire_message = self
                    .next_wire_message
                    .checked_add(1)
                    .ok_or("wire message sequence exhausted")?;
                let message_id = format!(
                    "op-{}",
                    digest(&(&self.wire_incarnation, &id, self.next_wire_message))
                );
                planned.insert(
                    id.clone(),
                    Job {
                        id: id.clone(),
                        url: url.clone(),
                        caps: caps.clone(),
                        prefix,
                        action: action.into(),
                        payload,
                        context: caps.context(&self.config.caller_id),
                        result: Value::Null,
                        operation_ref: None,
                        claim_view: Value::Null,
                        submitted: false,
                        attempted: false,
                        message_id,
                        releasing_pin: false,
                        cancel_sent: false,
                        quiescent: false,
                        due: job_due,
                        deadline: expires,
                        last_error: None,
                        started: None,
                        cost_recorded: false,
                        traffic_at_start: 0,
                        cache_watermark: Value::Null,
                        ready_end: 0,
                        ready_at: None,
                        preparation_estimate,
                        ready_pin_eligible: false,
                        ready_pin: None,
                        ready_pin_error: None,
                    },
                );
            }
        }
        if (!missing.is_empty()
            || !frozen.missing_scope().is_empty()
            || !frozen.materialization_required().is_empty())
            && !value["allow_partial"].as_bool().unwrap_or(false)
        {
            ids.clear();
            planned.clear();
            missing.push("full request not admitted: allow_partial is false".into());
        }
        let bytes: usize = self
            .jobs
            .values()
            .chain(planned.values())
            .map(Job::material_bytes)
            .sum();
        if bytes > self.config.limits.max_token_material_bytes
            || self.jobs.len()
                + planned.len()
                + self
                    .jobs
                    .values()
                    .filter(|job| job.ready_pin.is_some())
                    .count()
                > self.config.max_operations
        {
            return Err("operation token material capacity".into());
        }
        let pin_claims: Vec<_> = planned
            .values()
            .filter(|job| job.action == "sglang.pin")
            .filter(|job| {
                !self
                    .jobs
                    .values()
                    .any(|old| old.payload["claim_id"] == job.payload["claim_id"])
            })
            .map(|job| job.payload["claim_id"].as_str().unwrap_or("").to_string())
            .collect();
        if self.index.usage().claims.saturating_add(pin_claims.len())
            > self.config.limits.max_claims
        {
            return Err("Router claim capacity".into());
        }
        self.book
            .attach(demand.clone(), ids.clone())
            .map_err(|e| format!("{e:?}"))?;
        if let Err(e) = self
            .index
            .track_resource(&task, ResourceKind::Intent, demand.id.clone())
        {
            self.book.release(&demand);
            for id in planned.keys() {
                let _ = self.book.mark_quiescent(id);
                self.book.forget_job(id);
            }
            return Err(format!("{e:?}"));
        }
        for id in pin_claims {
            self.index
                .track_resource(&task, ResourceKind::Claim, id)
                .map_err(|e| format!("{e:?}"))?;
        }
        self.jobs.extend(planned);
        if self
            .config
            .preparation_policy
            .as_ref()
            .is_some_and(|policy| {
                policy.ready_pin_ttl_ms > 0
                    && action == "sglang.prefetch"
                    && source == "storage"
                    && destination == "host"
                    && matches!(
                        value["event"].as_str(),
                        Some("resume_expected" | "tool_done")
                    )
            })
        {
            for id in &ids {
                self.jobs.get_mut(id).unwrap().ready_pin_eligible = true;
            }
        }
        self.intents.insert(
            demand.clone(),
            Intent {
                fingerprint,
                admission_gaps: missing,
                jobs: ids.into_iter().collect(),
                frozen,
                expires,
                released: false,
                claims: Vec::new(),
            },
        );
        Ok(self.intent_view(&demand))
    }
    async fn submit_l3(
        &mut self,
        value: Value,
        task: SessionIdentity,
        demand: DemandId,
        fingerprint: String,
        frozen: FrozenTarget,
    ) -> Result<Value, String> {
        let owner = self.backend_owner.clone().ok_or_else(|| {
            self.backend_error
                .clone()
                .unwrap_or("L3 owner not configured".into())
        })?;
        let mode = match value["mode"].as_str() {
            Some("acquire") => ClaimMode::Acquire,
            Some("renew") => ClaimMode::Renew,
            Some("release") => ClaimMode::Release,
            _ => return Err("pin mode required".into()),
        };
        let revision = value["revision"]
            .as_u64()
            .ok_or("claim revision required")?;
        let claim_id = value["claim_id"]
            .as_str()
            .filter(|s| !s.is_empty() && s.len() <= self.config.limits.max_identifier_bytes)
            .ok_or("claim_id required")?;
        let mut requests = Vec::new();
        for scope in frozen.known_scope() {
            let prefix = self
                .index
                .materialize(scope.scope())
                .map_err(|g| format!("{g:?}"))?;
            let identity = ClaimIdentity {
                caller_id: self.config.caller_id.clone(),
                task_incarnation: digest(&task),
                claim_id: digest(&(claim_id, &prefix)),
            };
            requests.push(ProtectionRequest {
                identity,
                revision,
                mode,
                target: prefix,
                ttl_seconds: value["ttl_seconds"].as_f64(),
                required_protection: value["required_protection"]
                    .as_str()
                    .unwrap_or("eviction_exclusion")
                    .into(),
            });
        }
        if !frozen.missing_scope().is_empty() || requests.is_empty() {
            return Err("L3 target scope incomplete".into());
        }
        let new = requests
            .iter()
            .filter(|r| !self.claims.contains_key(&r.identity))
            .count();
        if self.claims.len() + new > self.config.limits.max_claims {
            return Err("Router claim capacity".into());
        }
        let retained: usize = self
            .claims
            .values()
            .map(|c| {
                c.request
                    .target
                    .token_ids
                    .as_ref()
                    .map_or(0, |t| t.len() * 4)
            })
            .sum();
        let incoming: usize = requests
            .iter()
            .filter(|r| !self.claims.contains_key(&r.identity))
            .map(|r| r.target.token_ids.as_ref().map_or(0, |t| t.len() * 4))
            .sum();
        if retained.saturating_add(incoming) > self.config.limits.max_token_material_bytes {
            return Err("claim token material capacity".into());
        }
        self.index
            .track_resource(&task, ResourceKind::Intent, demand.id.clone())
            .map_err(|e| format!("{e:?}"))?;
        let mut identities = Vec::new();
        for request in requests {
            let identity = request.identity.clone();
            let tracked =
                self.index
                    .track_resource(&task, ResourceKind::Claim, identity.claim_id.clone());
            if tracked.is_err() {
                return Err("task claim tracking capacity".into());
            }
            let mut interests = self
                .claims
                .get(&identity)
                .map_or_else(BTreeSet::new, |c| c.interests.clone());
            if request.mode == ClaimMode::Release {
                interests.clear();
            } else {
                interests.insert(demand.clone());
            }
            let (view, error) = match owner.submit(request.clone()).await {
                Ok(view) => (serde_json::to_value(view).map_err(|e| e.to_string())?, None),
                Err(e) => (Value::Null, Some(e.to_string())),
            };
            self.claims.insert(
                identity.clone(),
                ClaimRecord {
                    task: task.clone(),
                    request,
                    view,
                    release_requested: false,
                    interests,
                    feedback_unknown: self
                        .claims
                        .get(&identity)
                        .and_then(|claim| claim.feedback_unknown.clone()),
                    error,
                },
            );
            identities.push(identity);
        }
        self.intents.insert(
            demand.clone(),
            Intent {
                fingerprint,
                admission_gaps: vec![],
                jobs: vec![],
                frozen,
                expires: Instant::now()
                    + value["ttl_seconds"]
                        .as_f64()
                        .filter(|ttl| ttl.is_finite() && *ttl > 0.0 && *ttl <= 31_536_000.0)
                        .map(Duration::from_secs_f64)
                        .unwrap_or(Duration::from_millis(self.config.operation_ttl_ms)),
                released: false,
                claims: identities,
            },
        );
        Ok(self.intent_view(&demand))
    }
    async fn poll_claim(&mut self, identity: &ClaimIdentity) {
        let Some(owner) = self.backend_owner.clone() else {
            return;
        };
        let Some(mut claim) = self.claims.remove(identity) else {
            return;
        };
        if claim.release_requested && claim.request.mode != ClaimMode::Release {
            let mut release = claim.request.clone();
            release.mode = ClaimMode::Release;
            release.ttl_seconds = None;
            release.revision = release.revision.saturating_add(1);
            match owner.submit(release.clone()).await {
                Ok(view) => {
                    claim.request = release;
                    claim.view = serde_json::to_value(view).unwrap_or(Value::Null);
                    claim.error = None;
                }
                Err(e) => claim.error = Some(e.to_string()),
            }
        }
        match owner.query(Some(identity.clone())).await {
            Ok(snapshot) => {
                if let Some(view) = snapshot.claim {
                    claim.view = serde_json::to_value(view).unwrap_or(Value::Null);
                }
            }
            Err(e) => claim.error = Some(e.to_string()),
        }
        if claim.feedback_unknown.is_some() {
            suppress_protection(&mut claim.view);
        }
        self.claims.insert(identity.clone(), claim);
    }
    fn intent_view(&self, demand: &DemandId) -> Value {
        let Some(intent) = self.intents.get(demand) else {
            return json!({"state":"unknown"});
        };
        let now = Instant::now();
        json!({"message_id":demand.id,"task":demand.task,"snapshot_revision":intent.frozen.snapshot_revision(),
            "known_scope":intent.frozen.known_scope().iter().map(|s|json!({"branch":s.snapshot().map(|b|b.id().branch_id.clone()),"end_token":s.scope().end_token(),"content":s.scope().key().digest()})).collect::<Vec<_>>(),
            "missing_scope":intent.frozen.missing_scope().iter().map(|s|format!("{s:?}")).collect::<Vec<_>>(),
            "materialization_required":intent.frozen.materialization_required().iter().map(|s|format!("{s:?}")).collect::<Vec<_>>(),
            "admission_gaps":intent.admission_gaps,"claims":intent.claims.iter().filter_map(|id|self.claims.get(id)).map(|c|json!({"claim":c.view,"error":c.error,"feedback_unknown":c.feedback_unknown,"release_requested":c.release_requested})).collect::<Vec<_>>(),
            "released":intent.released,"quiescent":intent.claims.iter().all(|id|self.claims.get(id).is_none_or(|c|c.view["backend_quiescent"]==true)) && intent.jobs.iter().all(|id|self.jobs.get(id).is_none_or(|j|self.job_quiescent(j))),
            "operations":intent.jobs.iter().filter_map(|id|self.jobs.get(id)).map(|j|json!({"child_id":j.id,"worker_url":j.url,"engine_id":j.caps.engine_id,"owner_epoch":j.caps.epoch,
                "action":j.action,"result":j.result,"preparation_estimate":j.preparation_estimate,"cache_watermark":j.cache_watermark,"observation_age_ms":j.ready_at.map(|at|at.elapsed().as_millis()),"claim":j.claim_view,"error":j.last_error,"quiescent":self.job_quiescent(j),"cancel_requested":self.book.job(&j.id).is_some_and(|b|b.cancel_requested),
                "auto_pin":j.ready_pin.as_ref().map(|pin|json!({"child_id":pin.id,"result":pin.result,"state":if pin.pin_refused(){json!("not_acquired")}else{pin.claim_view["state"].clone()},"claim":pin.claim_view,"error":pin.last_error,"quiescent":self.job_quiescent(j),"release_requested":self.book.job(&j.id).is_some_and(|b|b.cancel_requested)||now>=pin.deadline})),"auto_pin_error":j.ready_pin_error,
                "ready_scope":if !intent.released && self.feedback.current(&j.url,&j.cache_watermark) && j.ready_at.is_some_and(|at|now.duration_since(at)<Duration::from_millis(self.config.poll_ms*2)) && j.ready_end>0 {json!({"start_token":0,"end_token":j.ready_end,"tier":j.payload["destination"]["tier"],"provenance":"engine_scoped_query"})}else{Value::Null}})).collect::<Vec<_>>()})
    }
    async fn pump(&mut self) {
        let now = Instant::now();
        let expired: Vec<_> = self
            .intents
            .iter()
            .filter(|(_, i)| !i.released && i.expires <= now)
            .map(|(d, _)| d.clone())
            .collect();
        for d in expired {
            self.release(&d);
        }
        self.tick_cursor = self.tick_cursor.wrapping_add(1);
        if let Some(identity) = self
            .claims
            .keys()
            .nth(self.tick_cursor % self.claims.len().max(1))
            .cloned()
        {
            self.poll_claim(&identity).await;
        }
        let healthy: BTreeSet<_> = self.workers().into_iter().map(|w| w.url.clone()).collect();
        self.caps.retain(|url, _| healthy.contains(url));
        for job in self.jobs.values_mut().filter(|j| !healthy.contains(&j.url)) {
            job.ready_end = 0;
            job.ready_at = None;
            job.last_error = Some("worker unavailable".into());
        }
        let candidates = self.workers();
        if let Some(worker) = candidates.get(self.tick_cursor % candidates.len().max(1)) {
            if let Ok(caps) = self.refresh(&worker.url).await {
                let _ = self.poll_feedback(&worker.url, &caps).await;
            }
        }
        let ids: Vec<_> = self.jobs.keys().cloned().collect();
        if !ids.is_empty() {
            let id = ids[self.tick_cursor % ids.len()].clone();
            self.poll_job(&id).await;
        }
    }
    async fn poll_job(&mut self, id: &str) {
        let Some(mut job) = self.jobs.remove(id) else {
            return;
        };
        let cancel = self.book.job(id).is_some_and(|b| b.cancel_requested);
        let result = self.advance(&mut job, cancel).await;
        if let Err(e) = result {
            job.last_error = Some(e);
            job.ready_end = 0;
            job.ready_at = None;
            job.claim_view = Value::Null;
        }
        self.poll_ready_pin(&mut job, cancel).await;
        if self.job_quiescent(&job) {
            let _ = self.book.mark_quiescent(id);
        }
        self.jobs.insert(id.into(), job);
    }
    fn job_quiescent(&self, job: &Job) -> bool {
        job.quiescent
            && job.ready_pin.as_ref().is_none_or(|pin| {
                pin.pin_inactive()
                    || (pin.quiescent
                        && pin.last_error.is_none()
                        && Instant::now() < pin.deadline
                        && !self
                            .book
                            .job(&job.id)
                            .is_some_and(|book| book.cancel_requested))
            })
    }

    async fn poll_ready_pin(&mut self, job: &mut Job, cancel: bool) {
        if job.ready_pin_eligible
            && job.ready_pin.is_none()
            && !cancel
            && job.last_error.is_none()
            && job.quiescent
            && job.result["update"]["status"] == "completed"
            && job.ready_end == job.prefix.end_token
        {
            job.ready_pin_eligible = false;
            let policy = self.config.preparation_policy.as_ref().unwrap();
            let reserved: u64 = self
                .jobs
                .values()
                .filter_map(|j| j.ready_pin.as_ref())
                .filter(|pin| !pin.pin_inactive())
                .map(|pin| pin.prefix.end_token)
                .sum();
            let material: usize = self.jobs.values().map(Job::material_bytes).sum();
            let children = self.jobs.values().filter(|j| j.ready_pin.is_some()).count();
            if reserved.saturating_add(job.prefix.end_token) > policy.auto_pin_max_tokens {
                job.ready_pin_error = Some("auto PIN token capacity".into());
            } else if material
                .saturating_add(job.material_bytes())
                .saturating_add(job.prefix.end_token as usize * 8)
                > self.config.limits.max_token_material_bytes
                || self.jobs.len() + children + 2 > self.config.max_operations
                || self.index.usage().claims + children + 1 > self.config.limits.max_claims
            {
                job.ready_pin_error = Some("auto PIN operation/material/claim capacity".into());
            } else {
                let id = format!("ready-pin-{}", digest(&job.message_id));
                let ttl_ms = policy.ready_pin_ttl_ms;
                let now = Instant::now();
                job.ready_pin = Some(Box::new(Job {
                    id: id.clone(),
                    url: job.url.clone(),
                    caps: job.caps.clone(),
                    prefix: job.prefix.clone(),
                    action: "sglang.pin".into(),
                    payload: json!({"target":job.prefix,"location":{"tier":"host","engine_id":job.caps.engine_id},"mode":"acquire",
                        "claim_id":id,"revision":1,"ttl_seconds":ttl_ms as f64/1000.0,"required_protection":"eviction_exclusion"}),
                    context: job.context.clone(),
                    result: Value::Null,
                    operation_ref: None,
                    claim_view: Value::Null,
                    submitted: false,
                    attempted: false,
                    message_id: id,
                    releasing_pin: false,
                    cancel_sent: false,
                    quiescent: false,
                    due: now,
                    deadline: now + Duration::from_millis(ttl_ms),
                    last_error: None,
                    started: None,
                    cost_recorded: false,
                    traffic_at_start: 0,
                    cache_watermark: Value::Null,
                    ready_end: 0,
                    ready_at: None,
                    preparation_estimate: None,
                    ready_pin_eligible: false,
                    ready_pin: None,
                    ready_pin_error: None,
                }));
            }
        }
        if let Some(pin) = &mut job.ready_pin {
            if pin.pin_inactive() {
                return;
            }
            let release = cancel || Instant::now() >= pin.deadline;
            if let Err(error) = self.advance(pin, release).await {
                pin.last_error = Some(error);
                pin.claim_view = Value::Null;
            }
        }
    }
    async fn advance(&mut self, job: &mut Job, cancel: bool) -> Result<(), String> {
        if !job.submitted && !job.attempted && !job.releasing_pin && cancel {
            job.quiescent = true;
            return Ok(());
        }
        if cancel
            && job.action == "sglang.pin"
            && job.payload["mode"] != "release"
            && job.quiescent
            && !job.releasing_pin
        {
            if self.jobs.values().any(|other| {
                other.url == job.url
                    && other.action == "sglang.pin"
                    && other.payload["claim_id"] == job.payload["claim_id"]
                    && self
                        .book
                        .job(&other.id)
                        .is_some_and(|interest| !interest.interests.is_empty())
            }) {
                return Ok(());
            }
            let current = self.client.command(&job.url, &json!({"kind":"claim","context":job.context,"claim_id":job.payload["claim_id"]})).await?;
            if current["owner_epoch"] != job.caps.epoch {
                return Err("stale claim owner epoch".into());
            }
            if matches!(
                current["state"].as_str(),
                Some("released" | "expired" | "invalidated")
            ) {
                job.claim_view = current;
                job.last_error = None;
                return Ok(());
            }
            job.payload["mode"] = json!("release");
            job.payload["revision"] = json!(current["revision"]
                .as_u64()
                .ok_or("claim revision absent")?
                .saturating_add(1));
            job.payload["ttl_seconds"] = Value::Null;
            job.message_id = format!("{}-release", job.id);
            job.submitted = false;
            job.attempted = false;
            job.quiescent = false;
            job.releasing_pin = true;
            job.result = Value::Null;
        }
        if Instant::now() < job.due {
            return Ok(());
        }
        if !self.workers().iter().any(|w| w.url == job.url) {
            return Err("worker unavailable".into());
        }
        let live = self.refresh(&job.url).await?;
        if live.epoch != job.caps.epoch || live.compatibility != job.caps.compatibility {
            return Err(
                "owner epoch/compatibility changed; reconcile old operation separately".into(),
            );
        }
        self.poll_feedback(&job.url, &live).await?;
        job.caps.generation = live.generation;
        if !job.submitted {
            if !job.attempted && !job.releasing_pin && Instant::now() >= job.deadline {
                job.quiescent = true;
                return Err("expired before submission".into());
            }
            // Stable IDs survive lost HTTP responses; retries submit the same
            // envelope/context. Never make a new physical attempt implicitly.
            if job.started.is_none() {
                job.started = Some(Instant::now());
                job.traffic_at_start = self.traffic.load(Ordering::Relaxed);
            }
            job.attempted = true;
            let reply=self.client.command(&job.url,&json!({"kind":"submit","context":job.context,"envelope":{"protocol_version":"0.1","message_id":job.message_id,
                "actions":[{"action_id":"transfer","action_type":job.action,"action_version":"1.0","payload":job.payload}]}})).await?;
            job.submitted = true;
            self.update_result(job, reply)?;
        }
        if cancel && !job.releasing_pin && !job.cancel_sent && !job.quiescent {
            if let Some(reference) = &job.operation_ref {
                let reply = self
                    .client
                    .command(&job.url, &json!({"kind":"cancel","ref":reference}))
                    .await?;
                job.cancel_sent = true;
                if reply.get("update").is_some() {
                    self.update_result(job, json!({"results":[reply]}))?;
                }
            }
        }
        let reply = self
            .client
            .command(
                &job.url,
                &json!({"kind":"query","context":job.context,"message_id":job.message_id}),
            )
            .await?;
        self.update_result(job, reply)?;
        job.ready_end = 0;
        job.ready_at = None;
        if job.action == "sglang.pin" {
            // A terminal refusal establishes that no claim was acquired. The
            // Engine returns the operation result, without a claim to query.
            if job.pin_refused() {
                job.claim_view = Value::Null;
                job.last_error = None;
                return Ok(());
            }
            let ticket = self.feedback.ticket(&job.url)?;
            let claim=self.client.command(&job.url,&json!({"kind":"claim","context":job.context,"claim_id":job.payload["claim_id"]})).await?;
            if claim["owner_epoch"] != job.caps.epoch {
                return Err("stale claim owner epoch".into());
            }
            self.poll_feedback(&job.url, &live).await?;
            if !self.feedback.accept_claim(&job.url, &ticket) {
                return Err("claim query crossed feedback invalidation".into());
            }
            job.claim_view = claim;
        }
        if job.action == "sglang.prefetch" && !cancel {
            // A terminal result alone cannot establish current location.
            let tier = job.payload["destination"]["tier"]
                .as_str()
                .unwrap_or("host");
            let ticket = self.feedback.ticket(&job.url)?;
            let snapshot = self
                .client
                .residency(&job.url, &live, &self.config.caller_id, &job.prefix, tier)
                .await?;
            self.poll_feedback(&job.url, &live).await?;
            let effect =
                self.feedback
                    .accept_query(&job.url, &ticket, &snapshot["cache_watermark"])?;
            self.invalidate_feedback(&job.url, effect, "newer scoped query watermark");
            job.cache_watermark = snapshot
                .get("cache_watermark")
                .cloned()
                .unwrap_or(Value::Null);
            job.ready_end = confirmed_end(&snapshot, &live, &job.prefix, tier)?;
            job.ready_at = Some(Instant::now());
            if job.ready_end == job.prefix.end_token && !job.cost_recorded {
                if let Some(start) = job.started {
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    if let Some(model) = &mut self.preparation_costs {
                        if job.payload["source"]["tier"] == "storage"
                            && job.payload["destination"]["tier"] == "host"
                            && job.result["update"]["status"] == "completed"
                        {
                            model.observe(
                                &preparation_owner(&job.url, &job.caps),
                                job.prefix.end_token,
                                elapsed_ms,
                            );
                            job.cost_recorded = true;
                        }
                    }
                    if self.preparation_costs.is_none()
                        && job.traffic_at_start == self.traffic.load(Ordering::Relaxed)
                    {
                        let observed = elapsed_ms / job.prefix.end_token as f64;
                        self.cost_ms_per_token = 0.8 * self.cost_ms_per_token + 0.2 * observed;
                        job.cost_recorded = true;
                    }
                }
            }
        }
        if !cancel
            && job.quiescent
            && !job.cost_recorded
            && job.action == "sglang.prefetch"
            && job.payload["source"]["tier"] == "storage"
            && job.payload["destination"]["tier"] == "host"
            && job.result["update"]["status"] != "completed"
        {
            if let (Some(model), Some(start)) = (&mut self.preparation_costs, job.started) {
                model.observe_incomplete(
                    &preparation_owner(&job.url, &job.caps),
                    job.prefix.end_token,
                    start.elapsed().as_secs_f64() * 1000.0,
                );
                job.cost_recorded = true;
            }
        }
        job.last_error = None;
        Ok(())
    }
    fn update_result(&self, job: &mut Job, reply: Value) -> Result<(), String> {
        let result = reply["results"]
            .as_array()
            .and_then(|a| a.iter().find(|r| r["ref"]["action_id"] == "transfer"))
            .ok_or("operation result unavailable")?;
        if result["ref"]["context"] != job.context || result["ref"]["message_id"] != job.message_id
        {
            return Err("operation identity mismatch".into());
        }
        let old = job.result["sequence"].as_u64().unwrap_or(0);
        let new = result["sequence"]
            .as_u64()
            .ok_or("operation sequence absent")?;
        if new < old {
            return Err("stale operation result".into());
        }
        job.operation_ref = Some(result["ref"].clone());
        job.quiescent = result["update"]["quiescent"].as_bool().unwrap_or(false);
        job.result = result.clone();
        Ok(())
    }
    fn publish(&self) {
        let now = Instant::now();
        let healthy: BTreeSet<_> = self.workers().iter().map(|w| w.url.clone()).collect();
        let ready = self
            .jobs
            .values()
            .filter(|j| {
                j.ready_end > 0
                    && self.feedback.current(&j.url, &j.cache_watermark)
                    && self
                        .book
                        .job(&j.id)
                        .is_some_and(|b| !b.interests.is_empty())
                    && healthy.contains(&j.url)
                    && self
                        .caps
                        .get(&j.url)
                        .is_some_and(|c| c.epoch == j.caps.epoch)
            })
            .filter_map(|j| {
                let at = j.ready_at?;
                let until = at + Duration::from_millis(self.config.poll_ms * 2);
                (until > now).then(|| Ready {
                    url: j.url.clone(),
                    prefix: j.prefix.clone(),
                    end: j.ready_end,
                    until,
                })
            })
            .collect();
        *self.published.write() = Published { ready };
    }
}
fn task_fn(value: &Value, caller: &str) -> Result<SessionIdentity, String> {
    task(value, caller)
}

/// Keep cleanup/drain identities, suppress any positive protection after missing
/// backend history. Querying a cached backend-owner receipt does not re-probe
/// the underlying objects and cannot repair such an invalidation by itself.
fn suppress_protection(view: &mut Value) {
    if !view.is_object() {
        return;
    }
    if matches!(view["phase"].as_str(), Some("released" | "expired")) {
        return;
    }
    view["phase"] = json!("unknown");
    view["uncertain"] = json!(true);
    view["effective_protection"] = Value::Null;
    view["confirmed_remaining_ms"] = Value::Null;
    if let Some(scopes) = view["scopes"].as_array_mut() {
        for scope in scopes {
            scope["uncertain"] = json!(true);
            scope["confirmed_remaining_ms"] = Value::Null;
        }
    }
}
