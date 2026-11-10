use super::feedback::{Effect, FeedbackCache};
use super::{
    client::{confirmed_end, Capabilities, EngineClient},
    OrchestrationConfig,
};
use crate::{
    models::{PrefixTarget, SessionIdentity},
    operations::{DemandBook, DemandId},
    protocol::{parse_action, Action},
    session_index::{
        Association, AssociationProvenance, ContentKey, ContentRegistration, RequestIdentity,
        ResourceKind, SessionIndex,
    },
    target_resolver::{FrozenTarget, Selector},
};
use bytes::Bytes;
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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
enum Message {
    Api(Value, oneshot::Sender<Value>),
    Routed {
        context: KvContext,
        tokens: Option<Vec<u32>>,
        salt: Option<String>,
        worker: String,
    },
}
/// Bounded admission handle. Queue pressure drops optional route observations;
/// inference retains its ordinary path.
pub struct Runtime {
    tx: mpsc::Sender<Message>,
    dropped_routes: Arc<AtomicU64>,
    config: OrchestrationConfig,
}
impl Runtime {
    pub fn start(
        config: OrchestrationConfig,
        workers: impl Fn() -> Vec<String> + Send + 'static,
        auth: Option<HeaderValue>,
    ) -> Arc<Self> {
        let (tx, rx) = mpsc::channel(config.queue_capacity);
        let dropped_routes = Arc::new(AtomicU64::new(0));
        let runtime = Arc::new(Self {
            tx,
            dropped_routes: dropped_routes.clone(),
            config: config.clone(),
        });
        let actor = Actor::new(config, Box::new(workers), auth, dropped_routes);
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
    pub fn routed(
        &self,
        context: &KvContext,
        worker: &str,
        tokens: Option<&[u32]>,
        salt: Option<&str>,
    ) {
        let Ok(permit) = self.tx.try_reserve() else {
            self.dropped_routes.fetch_add(1, Ordering::Relaxed);
            return;
        };
        permit.send(Message::Routed {
            context: context.clone(),
            worker: worker.into(),
            tokens: tokens.filter(|t| t.len() <= 65_536).map(<[u32]>::to_vec),
            salt: salt.map(str::to_string),
        });
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
    if value
        .get("ttl_ms")
        .is_some_and(|v| v.as_u64().is_none_or(|n| n > MAX_TIME_MS))
    {
        return Err("ttl_ms requires an integer in 0..86400000".into());
    }
    if value.get("delay_ms").is_some() {
        return Err("send prepare when the prefetch should start".into());
    }
    for key in ["revision", "branch_revision", "attempt"] {
        if let Some(number) = value.get(key) {
            if number
                .as_u64()
                .is_none_or(|n| n > crate::protocol::MAX_INTEGER)
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
            .is_none_or(|n| n > crate::protocol::MAX_INTEGER)
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
    deadline: Instant,
    last_error: Option<String>,
    cache_watermark: Value,
    ready_end: u64,
    ready_at: Option<Instant>,
}
struct Intent {
    fingerprint: String,
    admission_gaps: Vec<String>,
    jobs: Vec<String>,
    frozen: FrozenTarget,
    expires: Instant,
    released: bool,
}
struct Actor {
    config: OrchestrationConfig,
    client: EngineClient,
    workers: Box<dyn Fn() -> Vec<String> + Send>,
    dropped_routes: Arc<AtomicU64>,
    index: SessionIndex,
    book: DemandBook,
    jobs: BTreeMap<String, Job>,
    intents: BTreeMap<DemandId, Intent>,
    caps: BTreeMap<String, Capabilities>,
    last_route: BTreeMap<SessionIdentity, String>,
    tick_cursor: usize,
    events: BTreeMap<DemandId, (String, Value)>,
    feedback: FeedbackCache,
}
impl Actor {
    fn new(
        config: OrchestrationConfig,
        workers: Box<dyn Fn() -> Vec<String> + Send>,
        auth: Option<HeaderValue>,
        dropped_routes: Arc<AtomicU64>,
    ) -> Self {
        Self {
            client: EngineClient::new(config.rpc_timeout_ms, config.max_engine_bytes, auth),
            index: SessionIndex::new(config.limits.clone()),
            book: DemandBook::new(config.limits.max_intents, config.max_operations),
            config: config.clone(),
            workers,
            dropped_routes,
            jobs: BTreeMap::new(),
            intents: BTreeMap::new(),
            caps: BTreeMap::new(),
            last_route: BTreeMap::new(),
            tick_cursor: 0,
            events: BTreeMap::new(),
            feedback: FeedbackCache::new(
                config.max_workers,
                config.max_operations,
                config.max_engine_bytes,
            ),
        }
    }
    async fn run(mut self, mut rx: mpsc::Receiver<Message>) {
        let mut tick = tokio::time::interval(Duration::from_millis(self.config.poll_ms));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                message=rx.recv()=>match message {
                    Some(Message::Api(value,reply))=> { let result=self.api(value).await;let _=reply.send(result); }
                    Some(Message::Routed { context, tokens, salt, worker }) => {
                        if let Some(tokens) = tokens {
                            if let Err(reason) = self.record(context.clone(), tokens, salt, worker.clone()).await {
                                tracing::debug!(%reason, "KV association unavailable");
                            }
                        }
                        self.changed_route(&context.task(&self.config.caller_id), &worker);
                    }
                    None=>break,
                },
                _=tick.tick()=>self.pump().await,
            }
        }
        // Actor shutdown never fabricates quiescence. Engine leases/deadlines
        // remain authoritative; explicit Router cleanup drains while running.
    }
    fn workers(&self) -> Vec<String> {
        let mut workers = (self.workers)();
        workers.truncate(self.config.max_workers);
        workers
    }
    async fn refresh(&mut self, url: &str) -> Result<Capabilities, String> {
        match self.client.capabilities(url).await {
            Ok(caps) => {
                let effect = self.feedback.bind(url, &caps);
                self.invalidate_feedback(url, effect);
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
                self.invalidate_feedback(url, effect);
                self.caps.remove(url);
                for job in self.jobs.values_mut().filter(|j| j.url == url) {
                    job.ready_end = 0;
                    job.ready_at = None;
                }
                Err(e)
            }
        }
    }
    fn invalidate_feedback(&mut self, url: &str, effect: Effect) {
        for job in self.jobs.values_mut().filter(|job| job.url == url) {
            if effect.ready {
                job.ready_end = 0;
                job.ready_at = None;
            }
            if effect.claims {
                job.claim_view = Value::Null;
            }
        }
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
                self.invalidate_feedback(url, effect);
                Ok(())
            }
            Err(reason) => {
                let effect = self.feedback.fail(url, &reason);
                self.invalidate_feedback(url, effect);
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
        if !self.workers().iter().any(|w| *w == worker) {
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
                Some("tool_wait" | "prepare" | "resume")
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
                    Some("tool_wait" | "prepare" | "resume")
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
            return Ok(json!({
                "enabled": true,
                "caller_id": self.config.caller_id,
                "engine_session_required": false,
                "lifecycle": "explicit",
                "pin_tier": "host",
                "dropped_route_events": self.dropped_routes.load(Ordering::Relaxed),
                "feedback": self.feedback.view(),
                "index": format!("{:?}", self.index.usage())
            }));
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
                    && ids.iter().any(|id| {
                        self.jobs.get(id).is_some_and(|job| {
                            job.action == "sglang.pin" && job.payload["mode"] != "release"
                        })
                    })
                {
                    return Err("cancel or release the PIN intent before acknowledging".into());
                }
                self.release(&demand);
                for id in &ids {
                    self.poll_job(id).await;
                }
                if ids
                    .iter()
                    .any(|id| self.jobs.get(id).is_some_and(|j| !j.quiescent))
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
                        let worker = value["worker_url"].as_str().ok_or("worker_url required")?;
                        self.record(
                            context,
                            tokens,
                            value["cache_salt"].as_str().map(str::to_string),
                            worker.into(),
                        )
                        .await?;
                        self.changed_route(&task, worker);
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
                    "tool_wait" | "prepare" | "resume" => {
                        let (action, source, destination) = if event == "tool_wait" {
                            if value["pressure"] != true {
                                return Ok(
                                    json!({"state":"skipped","reason":"pressure not asserted"}),
                                );
                            }
                            ("sglang.demote", "host", "storage")
                        } else {
                            ("sglang.prefetch", "storage", "host")
                        };
                        let mut command = value.clone();
                        command["kind"] = json!("submit");
                        command["message_id"] = json!(id);
                        command["action"] = json!(action);
                        command["source"] = json!({"tier":source});
                        command["destination"] = json!({"tier":destination});
                        self.submit(command, task, demand).await
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
        if action == "sglang.pin" && pin_tier != "host" {
            return Err("only host PIN is supported".into());
        }
        let mut workers = self.workers();
        if let Some(url) = value["worker_url"].as_str() {
            workers.retain(|w| w == url);
        }
        // A read-only affinity preference. No policy.select, load guard, pending
        // stake or session-affinity commit is used for preparation placement.
        if let Some(preferred) = self.last_route.get(&task) {
            workers.sort_by_key(|url| url != preferred);
        }
        let mut available = Vec::new();
        for url in workers {
            if let Ok(caps) = self.refresh(&url).await {
                available.push((url, caps));
            }
        }
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
                        message_id: id.clone(),
                        releasing_pin: false,
                        cancel_sent: false,
                        quiescent: false,
                        deadline: expires,
                        last_error: None,
                        cache_watermark: Value::Null,
                        ready_end: 0,
                        ready_at: None,
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
            .map(|j| j.prefix.token_ids.as_ref().map_or(0, |t| t.len() * 4))
            .sum();
        if bytes > self.config.limits.max_token_material_bytes {
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
        self.intents.insert(
            demand.clone(),
            Intent {
                fingerprint,
                admission_gaps: missing,
                jobs: ids.into_iter().collect(),
                frozen,
                expires,
                released: false,
            },
        );
        Ok(self.intent_view(&demand))
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
            "admission_gaps":intent.admission_gaps,
            "released":intent.released,"quiescent":intent.jobs.iter().all(|id|self.jobs.get(id).is_none_or(|j|j.quiescent)),
            "operations":intent.jobs.iter().filter_map(|id|self.jobs.get(id)).map(|j|json!({"child_id":j.id,"worker_url":j.url,"engine_id":j.caps.engine_id,"owner_epoch":j.caps.epoch,
                "action":j.action,"result":j.result,"cache_watermark":j.cache_watermark,"observation_age_ms":j.ready_at.map(|at|at.elapsed().as_millis()),"claim":j.claim_view,"error":j.last_error,"quiescent":j.quiescent,"cancel_requested":self.book.job(&j.id).is_some_and(|b|b.cancel_requested),
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
        let healthy: BTreeSet<_> = self.workers().into_iter().collect();
        self.caps.retain(|url, _| healthy.contains(url));
        for job in self.jobs.values_mut().filter(|j| !healthy.contains(&j.url)) {
            job.ready_end = 0;
            job.ready_at = None;
            job.last_error = Some("worker unavailable".into());
        }
        let candidates = self.workers();
        if let Some(worker) = candidates.get(self.tick_cursor % candidates.len().max(1)) {
            if let Ok(caps) = self.refresh(worker).await {
                let _ = self.poll_feedback(worker, &caps).await;
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
        let result = self.advance(&mut job).await;
        if let Err(e) = result {
            job.last_error = Some(e);
            job.ready_end = 0;
            job.ready_at = None;
            job.claim_view = Value::Null;
        }
        if job.quiescent {
            let _ = self.book.mark_quiescent(id);
        }
        self.jobs.insert(id.into(), job);
    }
    async fn advance(&mut self, job: &mut Job) -> Result<(), String> {
        let cancel = self.book.job(&job.id).is_some_and(|b| b.cancel_requested);
        if !job.submitted && !job.attempted && cancel {
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
            if matches!(
                current["state"].as_str(),
                Some("released" | "expired" | "invalidated")
            ) {
                job.claim_view = current;
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
        if !self.workers().iter().any(|w| *w == job.url) {
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
            self.invalidate_feedback(&job.url, effect);
            job.cache_watermark = snapshot
                .get("cache_watermark")
                .cloned()
                .unwrap_or(Value::Null);
            job.ready_end = confirmed_end(&snapshot, &live, &job.prefix, tier)?;
            job.ready_at = Some(Instant::now());
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
}
fn task_fn(value: &Value, caller: &str) -> Result<SessionIdentity, String> {
    task(value, caller)
}
