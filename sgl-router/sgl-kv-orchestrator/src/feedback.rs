//! Owner-local polling fences and bounded diagnostic facts, never a placement
//! index. Discovery binds source incarnations; event arrival cannot rebind them.
use super::client::Capabilities;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};

const MAX_INTEGER: u64 = (1 << 53) - 1;
fn identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheWatermark {
    source_id: String,
    source_epoch: String,
    generation: u64,
    version: u64,
}
impl CacheWatermark {
    fn parse(value: &Value) -> Result<Self, String> {
        let mark: Self =
            serde_json::from_value(value.clone()).map_err(|_| "cache watermark absent/invalid")?;
        if !identifier(&mark.source_id)
            || !identifier(&mark.source_epoch)
            || mark.generation > MAX_INTEGER
            || mark.version > MAX_INTEGER
        {
            return Err("cache watermark exceeds protocol limits".into());
        }
        Ok(mark)
    }
    fn same_source(&self, other: &Self) -> bool {
        self.source_id == other.source_id && self.source_epoch == other.source_epoch
    }
    fn position(&self) -> (u64, u64) {
        (self.generation, self.version)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Cursor {
    source_id: String,
    source_epoch: String,
    sequence: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Binding {
    engine_id: String,
    engine_epoch: String,
    namespace: String,
    source_id: String,
    source_epoch: String,
    cache: CacheWatermark,
}
impl Binding {
    fn discovered(caps: &Capabilities) -> Result<Option<Self>, String> {
        let Some(feedback) = &caps.feedback else {
            return Ok(None);
        };
        if feedback["query"] != "events"
            || feedback["version"] != "1.0"
            || feedback["recovery"] != "query_known_scopes"
        {
            return Err("unsupported feedback capability".into());
        }
        let source_id = feedback["source_id"]
            .as_str()
            .filter(|v| identifier(v))
            .ok_or("invalid feedback source")?
            .into();
        let source_epoch = feedback["source_epoch"]
            .as_str()
            .filter(|v| identifier(v))
            .ok_or("invalid feedback epoch")?
            .into();
        Ok(Some(Self {
            engine_id: caps.engine_id.clone(),
            engine_epoch: caps.epoch.clone(),
            namespace: caps.compatibility.namespace.clone(),
            source_id,
            source_epoch,
            cache: CacheWatermark::parse(&feedback["cache_watermark"])?,
        }))
    }
    fn same_owner(&self, other: &Self) -> bool {
        self.engine_id == other.engine_id
            && self.engine_epoch == other.engine_epoch
            && self.namespace == other.namespace
            && self.source_id == other.source_id
            && self.source_epoch == other.source_epoch
            && self.cache.same_source(&other.cache)
    }
}
#[derive(Default, Clone, Copy)]
pub struct Effect {
    pub ready: bool,
    pub claims: bool,
    pub storage: bool,
}
impl Effect {
    fn unknown() -> Self {
        Self {
            ready: true,
            claims: true,
            storage: true,
        }
    }
    pub(super) fn merge(&mut self, other: Self) {
        self.ready |= other.ready;
        self.claims |= other.claims;
        self.storage |= other.storage;
    }
}
#[derive(Clone)]
struct Owner {
    binding: Option<Binding>,
    cursor: Option<Cursor>,
    high: Option<CacheWatermark>,
    barrier: u64,
    failure: Option<String>,
    last_recovery: Option<String>,
    ever_supported: bool,
    rejected_queries: u64,
}
/// Owner entries are bounded by max_workers; diagnostic record/byte budgets
/// are explicitly taken from the host's max_operations/max_engine_bytes.
pub struct FeedbackCache {
    owners: BTreeMap<String, Owner>,
    max_owners: usize,
    max_facts: usize,
    max_bytes: usize,
    facts: VecDeque<(Value, usize)>,
    fact_bytes: usize,
    dropped_facts: u64,
}
#[derive(Clone)]
pub struct QueryTicket {
    binding: Option<Binding>,
    barrier: u64,
}
impl FeedbackCache {
    pub fn new(max_owners: usize, max_facts: usize, max_bytes: usize) -> Self {
        Self {
            owners: BTreeMap::new(),
            max_owners,
            max_facts,
            max_bytes,
            facts: VecDeque::new(),
            fact_bytes: 0,
            dropped_facts: 0,
        }
    }
    /// Each worker has one in-flight poll. It carries only that worker's
    /// cursor and fences, with fresh diagnostics merged at completion.
    pub fn for_worker(&self, url: &str) -> Self {
        let mut scoped = Self::new(1, self.max_facts, self.max_bytes);
        if let Some(owner) = self.owners.get(url) {
            scoped.owners.insert(url.into(), owner.clone());
        }
        scoped
    }
    pub fn merge_worker(&mut self, other: Self) {
        self.owners.extend(other.owners);
        self.dropped_facts = self.dropped_facts.saturating_add(other.dropped_facts);
        for (fact, _) in other.facts {
            self.push_fact(fact);
        }
    }
    pub fn forget_worker(&mut self, url: &str) {
        self.owners.remove(url);
    }
    pub fn bind(&mut self, url: &str, caps: &Capabilities) -> Effect {
        if !self.owners.contains_key(url) && self.owners.len() >= self.max_owners {
            return Effect::unknown();
        }
        let owner = self.owners.entry(url.into()).or_insert_with(|| Owner {
            binding: None,
            cursor: None,
            high: None,
            barrier: 0,
            failure: None,
            last_recovery: None,
            ever_supported: false,
            rejected_queries: 0,
        });
        match Binding::discovered(caps) {
            Ok(None) if !owner.ever_supported => {
                owner.failure = None;
                Effect::default()
            }
            Ok(None) => {
                owner.binding = None;
                Self::fail_owner(owner, "feedback capability disappeared")
            }
            Err(reason) => {
                owner.ever_supported = true;
                owner.binding = None;
                Self::fail_owner(owner, &reason)
            }
            Ok(Some(binding)) => {
                owner.ever_supported = true;
                if owner
                    .binding
                    .as_ref()
                    .is_none_or(|old| !old.same_owner(&binding))
                {
                    owner.high = Some(binding.cache.clone());
                    owner.binding = Some(binding);
                    owner.cursor = None;
                    owner.barrier = owner.barrier.saturating_add(1);
                    owner.failure = Some("feedback source requires polling".into());
                    owner.last_recovery = Some("source_changed; query_known_scopes".into());
                    Effect::unknown()
                } else {
                    let mut effect = Effect::default();
                    if owner
                        .high
                        .as_ref()
                        .is_some_and(|high| binding.cache.position() > high.position())
                    {
                        effect.ready = true;
                        effect.claims = owner
                            .high
                            .as_ref()
                            .is_some_and(|high| binding.cache.generation != high.generation);
                        owner.high = Some(binding.cache.clone());
                        if effect.claims {
                            owner.barrier = owner.barrier.saturating_add(1);
                        }
                    }
                    // Repeated discovery cannot reset a cursor, clear a poll
                    // failure, or replace a newer native watermark with an old one.
                    owner.binding = Some(binding);
                    effect
                }
            }
        }
    }
    fn fail_owner(owner: &mut Owner, reason: &str) -> Effect {
        owner.barrier = owner.barrier.saturating_add(1);
        owner.failure = Some(reason.into());
        owner.last_recovery = Some("query_known_scopes".into());
        Effect::unknown()
    }
    pub fn fail(&mut self, url: &str, reason: &str) -> Effect {
        self.owners
            .get_mut(url)
            .map_or_else(Effect::unknown, |owner| Self::fail_owner(owner, reason))
    }
    /// None is legacy query fallback only if feedback has never been advertised.
    pub fn command(
        &self,
        url: &str,
        context: Value,
        limit: usize,
    ) -> Result<Option<Value>, String> {
        let owner = self.owners.get(url).ok_or("undiscovered feedback owner")?;
        if !owner.ever_supported {
            return Ok(None);
        }
        if owner.binding.is_none() {
            return Err("feedback source unavailable".into());
        }
        Ok(Some(
            json!({"kind":"events","context":context,"cursor":owner.cursor,"limit":limit.min(256).max(1)}),
        ))
    }
    pub fn apply(
        &mut self,
        url: &str,
        caller: &str,
        page: &Value,
        limit: usize,
    ) -> Result<Effect, String> {
        let owner = self
            .owners
            .get_mut(url)
            .ok_or("undiscovered feedback owner")?;
        let binding = owner
            .binding
            .as_ref()
            .ok_or("feedback source unavailable")?;
        if page["source_id"] != binding.source_id
            || page["source_epoch"] != binding.source_epoch
            || page["engine_id"] != binding.engine_id
            || page["engine_epoch"] != binding.engine_epoch
            || page["namespace"] != binding.namespace
            || page["recovery"] != "query_known_scopes"
        {
            return Err("feedback owner/source mismatch".into());
        }
        let next: Cursor = serde_json::from_value(page["next_cursor"].clone())
            .map_err(|_| "invalid feedback cursor")?;
        let earliest = page["earliest_sequence"]
            .as_u64()
            .filter(|n| *n <= MAX_INTEGER)
            .ok_or("invalid earliest sequence")?;
        let latest = page["latest_sequence"]
            .as_u64()
            .filter(|n| *n <= MAX_INTEGER)
            .ok_or("invalid latest sequence")?;
        let after = owner.cursor.as_ref().map_or(0, |cursor| cursor.sequence);
        if next.source_id != binding.source_id
            || next.source_epoch != binding.source_epoch
            || next.sequence < after
            || next.sequence > latest
            || earliest > latest.saturating_add(1)
        {
            return Err("feedback cursor regression/range".into());
        }
        let high = CacheWatermark::parse(&page["cache_watermark"])?;
        if !high.same_source(&binding.cache)
            || owner
                .high
                .as_ref()
                .is_some_and(|old| high.position() < old.position())
        {
            return Err("stale/foreign feedback cache watermark".into());
        }
        let items = page["items"]
            .as_array()
            .filter(|items| items.len() <= limit.min(256))
            .ok_or("feedback page exceeds event budget")?;
        let gap = page["gap"].as_bool().ok_or("feedback gap flag absent")?;
        let mut effect = Effect::default();
        let mut facts = Vec::new();
        let mut previous = 0;
        for event in items {
            let seq = event["sequence"]
                .as_u64()
                .filter(|n| *n <= MAX_INTEGER)
                .ok_or("invalid event sequence")?;
            if seq <= previous || seq > next.sequence || seq < earliest {
                return Err("unordered/out-of-range feedback event".into());
            }
            previous = seq;
            if !event["caller_id"].is_null() && event["caller_id"] != caller {
                return Err("foreign feedback caller".into());
            }
            let mark = CacheWatermark::parse(&event["cache_watermark"])?;
            if !mark.same_source(&high) {
                // A journal may retain history from an earlier recorder
                // incarnation. Discovery already fenced that source transition;
                // consuming history cannot rebind it or install an observation.
                effect.merge(Effect::unknown());
                continue;
            }
            if mark.position() > high.position() {
                return Err("invalid event cache watermark".into());
            }
            if seq <= after {
                continue;
            }
            match event["kind"].as_str() {
                Some("invalidation") => {
                    effect.ready = true;
                    let tiers = event["payload"]["tiers"].as_array();
                    effect.claims |= tiers.is_none_or(|tiers| {
                        tiers.iter().any(|tier| tier == "gpu" || tier == "host")
                    });
                    effect.storage |=
                        tiers.is_none_or(|tiers| tiers.iter().any(|tier| tier == "storage"));
                }
                Some("claim") => effect.claims = true,
                Some("operation") => {} // Historical outcomes never install a location.
                Some("association" | "request_use") => {
                    facts.push(diagnostic(url, binding, seq, event))
                }
                _ => return Err("unsupported feedback event kind".into()),
            }
        }
        if gap || after < earliest.saturating_sub(1) {
            effect.merge(Effect::unknown());
            owner.last_recovery = Some("journal_gap; query_known_scopes".into());
        }
        effect.claims |= owner
            .high
            .as_ref()
            .is_some_and(|old| old.generation != high.generation);
        if effect.claims || effect.storage || gap {
            owner.barrier = owner.barrier.saturating_add(1);
        }
        effect.ready |= owner
            .high
            .as_ref()
            .is_none_or(|old| old.position() < high.position());
        owner.high = Some(high);
        owner.cursor = Some(next);
        owner.failure = None;
        for fact in facts {
            self.push_fact(fact);
        }
        Ok(effect)
    }
    pub fn ticket(&self, url: &str) -> Result<QueryTicket, String> {
        let owner = self.owners.get(url).ok_or("undiscovered feedback owner")?;
        if owner.ever_supported {
            if let Some(reason) = &owner.failure {
                return Err(reason.clone());
            }
        }
        Ok(QueryTicket {
            binding: owner.binding.clone(),
            barrier: owner.barrier,
        })
    }
    pub fn accept_query(
        &mut self,
        url: &str,
        ticket: &QueryTicket,
        mark: &Value,
    ) -> Result<Effect, String> {
        let owner = self
            .owners
            .get_mut(url)
            .ok_or("undiscovered feedback owner")?;
        if owner.failure.is_some() || owner.barrier != ticket.barrier {
            return Self::reject_query(owner, "query crossed feedback invalidation/gap");
        }
        let Some(binding) = &ticket.binding else {
            return if !owner.ever_supported {
                Ok(Effect::default())
            } else {
                Self::reject_query(owner, "feedback downgrade during query")
            };
        };
        if owner
            .binding
            .as_ref()
            .is_none_or(|current| !binding.same_owner(current))
        {
            return Self::reject_query(owner, "query feedback incarnation changed");
        }
        let observed = CacheWatermark::parse(mark)?;
        if !observed.same_source(&binding.cache)
            || owner
                .high
                .as_ref()
                .is_some_and(|high| observed.position() < high.position())
        {
            return Self::reject_query(owner, "query older than native cache watermark");
        }
        let effect = Effect {
            ready: owner
                .high
                .as_ref()
                .is_none_or(|high| high.position() < observed.position()),
            claims: false,
            storage: false,
        };
        owner.high = Some(observed);
        Ok(effect)
    }
    fn reject_query(owner: &mut Owner, reason: &str) -> Result<Effect, String> {
        owner.rejected_queries = owner.rejected_queries.saturating_add(1);
        Err(reason.into())
    }
    pub fn accept_claim(&self, url: &str, ticket: &QueryTicket) -> bool {
        self.owners
            .get(url)
            .is_some_and(|owner| owner.failure.is_none() && owner.barrier == ticket.barrier)
    }
    pub fn current(&self, url: &str, mark: &Value) -> bool {
        let Some(owner) = self.owners.get(url) else {
            return false;
        };
        if !owner.ever_supported {
            return true;
        }
        if owner.failure.is_some() {
            return false;
        }
        CacheWatermark::parse(mark).is_ok_and(|mark| {
            owner
                .high
                .as_ref()
                .is_some_and(|high| mark.same_source(high) && mark.position() >= high.position())
        })
    }
    fn push_fact(&mut self, fact: Value) {
        let bytes = serde_json::to_vec(&fact).expect("diagnostic JSON").len();
        if bytes > self.max_bytes || self.max_facts == 0 {
            self.dropped_facts = self.dropped_facts.saturating_add(1);
            return;
        }
        while self.facts.len() >= self.max_facts
            || self.fact_bytes.saturating_add(bytes) > self.max_bytes
        {
            if let Some((_, removed)) = self.facts.pop_front() {
                self.fact_bytes -= removed;
                self.dropped_facts = self.dropped_facts.saturating_add(1);
            } else {
                break;
            }
        }
        self.fact_bytes += bytes;
        self.facts.push_back((fact, bytes));
    }
    pub fn view(&self) -> Value {
        json!({"coverage":"requested_scopes_only","recovery":"query_known_scopes","global_snapshot_complete":false,
            "diagnostic_limits":{"records":self.max_facts,"bytes":self.max_bytes},"diagnostic_bytes":self.fact_bytes,"dropped_diagnostics":self.dropped_facts,
            "owners":self.owners.iter().map(|(url,owner)|json!({"worker_url":url,"mode":if owner.ever_supported{"feedback"}else{"legacy_query_fallback"},
                "cursor":owner.cursor,"cache_watermark":owner.high,"rejected_scope_queries":owner.rejected_queries,"error":owner.failure,"last_recovery":owner.last_recovery})).collect::<Vec<_>>(),
            "facts":self.facts.iter().map(|(fact,_)|fact).collect::<Vec<_>>()})
    }
}
fn diagnostic(url: &str, binding: &Binding, sequence: u64, event: &Value) -> Value {
    let payload = &event["payload"];
    let mut selected = serde_json::Map::new();
    for key in [
        "request_id",
        "association_state",
        "provenance",
        "tier_accounting",
    ] {
        if let Some(value) = payload[key].as_str().filter(|v| identifier(v)) {
            selected.insert(key.into(), json!(value));
        }
    }
    for key in [
        "cached_tokens",
        "device_tokens",
        "host_tokens",
        "storage_tokens",
        "checkpointed_tokens",
    ] {
        selected.insert(
            key.into(),
            payload[key]
                .as_u64()
                .filter(|n| *n <= MAX_INTEGER)
                .map_or(Value::Null, |n| json!(n)),
        );
    }
    for key in ["checkpointed", "aborted"] {
        if let Some(value) = payload[key].as_bool() {
            selected.insert(key.into(), json!(value));
        }
    }
    let mut association = serde_json::Map::new();
    for key in [
        "request_id",
        "session_id",
        "session_incarnation",
        "branch_id",
    ] {
        if let Some(value) = payload["association"][key]
            .as_str()
            .filter(|v| identifier(v))
        {
            association.insert(key.into(), json!(value));
        }
    }
    if let Some(revision) = payload["association"]["branch_revision"]
        .as_u64()
        .filter(|n| *n <= MAX_INTEGER)
    {
        association.insert("branch_revision".into(), json!(revision));
    }
    let provided = payload["association_state"] == "provided" && !association.is_empty();
    json!({"worker_url":url,"source_id":binding.source_id,"source_epoch":binding.source_epoch,"sequence":sequence,"kind":event["kind"],
        "provenance":payload["provenance"].as_str().filter(|v|identifier(v)).unwrap_or("unknown"),
        "association_state":if provided{"provided_diagnostic_only"}else{"unknown"},"task_authority":"none","payload":selected,"association":association})
}
