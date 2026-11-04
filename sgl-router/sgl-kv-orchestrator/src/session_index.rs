//! Bounded business associations, independent of physical cache residency.
//!
//! Only explicit `start_session` creates a task. Trusted Router adapters supply
//! caller identity and content digests; digests are neither event block hashes
//! nor storage keys. A page-aligned node stores only its suffix material, with a
//! shared parent. Snapshots/operation targets hold `Arc`s, so GC cannot remove
//! metadata still needed by an operation. Material may be evicted separately;
//! this leaves logical associations intact and makes materialization fail closed.
use crate::models::{Compatibility, SessionIdentity};
use crate::protocol::MAX_INTEGER;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndexError {
    InvalidIdentity,
    InvalidPrefix,
    Conflict,
    StaleRevision,
    UnknownSession,
    UnknownContent,
    UnknownParent,
    RetiredIncarnation,
    ActiveIncarnation,
    Capacity(&'static str),
}

/// Every variable-size collection has an explicit bound. No runtime defaults.
/// Identifier bounds apply to compatibility strings, request/resource IDs and digests too.
#[derive(Clone, Debug)]
pub struct SessionLimits {
    pub max_sessions: usize,
    pub max_branches: usize,
    pub max_history_revisions: usize,
    pub max_requests: usize,
    pub max_content_nodes: usize,
    pub max_token_material_bytes: usize,
    pub max_tombstones: usize,
    pub max_claims: usize,
    pub max_intents: usize,
    pub max_identifier_bytes: usize,
    pub max_components: usize,
}

/// Compatibility and adapter-supplied digest together identify content. Engine
/// epoch and session identity deliberately do not participate. Components are a set.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentKey {
    compatibility: Compatibility,
    digest: String,
}
impl ContentKey {
    pub fn new(mut compatibility: Compatibility, digest: String) -> Result<Self, IndexError> {
        // P1 salt encoding treats empty and null raw salt alike.
        compatibility.cache_salt = compatibility.cache_salt.filter(|salt| !salt.is_empty());
        if digest.is_empty()
            || compatibility.namespace.is_empty()
            || compatibility.layout_id.is_empty()
            || compatibility.hash_scheme.is_empty()
            || compatibility.page_size == 0
            || compatibility.page_size > 256
            || compatibility.components.is_empty()
            || compatibility.components.len() > 64
            || compatibility.components.iter().any(String::is_empty)
            || [
                &compatibility.namespace,
                &compatibility.layout_id,
                &compatibility.hash_scheme,
            ]
            .iter()
            .any(|value| value.len() > 256)
            || compatibility
                .cache_salt
                .as_ref()
                .is_some_and(|salt| salt.len() > 256)
            || compatibility
                .components
                .iter()
                .any(|component| component.len() > 256)
        {
            return Err(IndexError::InvalidIdentity);
        }
        compatibility.components.sort();
        if compatibility.components.windows(2).any(|v| v[0] == v[1]) {
            return Err(IndexError::InvalidIdentity);
        }
        Ok(Self {
            compatibility,
            digest,
        })
    }
    pub fn compatibility(&self) -> &Compatibility {
        &self.compatibility
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
}

#[derive(Debug)]
pub struct ContentNode {
    key: ContentKey,
    parent: Option<Arc<ContentNode>>,
    start_token: u64,
    end_token: u64,
}
impl ContentNode {
    pub fn key(&self) -> &ContentKey {
        &self.key
    }
    pub fn parent(&self) -> Option<&Arc<Self>> {
        self.parent.as_ref()
    }
    pub fn start_token(&self) -> u64 {
        self.start_token
    }
    pub fn end_token(&self) -> u64 {
        self.end_token
    }
    pub fn contains_ancestor(&self, key: &ContentKey, end: u64) -> bool {
        let mut node = Some(self);
        while let Some(n) = node {
            if &n.key == key && n.end_token == end {
                return true;
            }
            node = n.parent.as_deref();
        }
        false
    }
    fn ancestor_at(&self, end: u64) -> Option<&Self> {
        let mut node = Some(self);
        while let Some(n) = node {
            if n.end_token == end {
                return Some(n);
            }
            node = n.parent.as_deref();
        }
        None
    }
}

/// Tokens cover only [parent.end, end), not a second copy of the entire prefix.
/// Missing tokens are valid metadata, never an executable hash-only target.
pub struct ContentRegistration {
    pub key: ContentKey,
    pub parent: Option<ContentKey>,
    pub end_token: u64,
    pub suffix_tokens: Option<Vec<u32>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SnapshotId {
    pub session: SessionIdentity,
    pub branch_id: String,
    pub revision: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParentFrontier {
    pub snapshot: SnapshotId,
    pub end_token: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RequestIdentity {
    pub request_id: String,
    /// Distinct sampling/fan-out/reuse binding; not an Engine epoch.
    pub incarnation: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssociationProvenance {
    RenderedRequest,
    ConfirmedCacheHit,
    EngineCheckpoint,
    ExplicitApplication,
}
#[derive(Debug)]
pub struct BranchRevision {
    id: SnapshotId,
    session_revision: u64,
    parent_frontier: Option<ParentFrontier>,
    provenance: AssociationProvenance,
    content: Arc<ContentNode>,
}
impl BranchRevision {
    pub fn id(&self) -> &SnapshotId {
        &self.id
    }
    pub fn session_revision(&self) -> u64 {
        self.session_revision
    }
    pub fn parent_frontier(&self) -> Option<&ParentFrontier> {
        self.parent_frontier.as_ref()
    }
    pub fn provenance(&self) -> AssociationProvenance {
        self.provenance
    }
    pub fn content(&self) -> &Arc<ContentNode> {
        &self.content
    }
}

pub struct Association {
    pub session: SessionIdentity,
    pub session_revision: u64,
    pub branch_id: String,
    pub branch_revision: u64,
    pub content: ContentKey,
    pub parent_frontier: Option<ParentFrontier>,
    pub request: Option<RequestIdentity>,
    pub provenance: AssociationProvenance,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct OwnedResourceId {
    pub task: SessionIdentity,
    pub id: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    Claim,
    Intent,
}
#[derive(Debug, Default)]
pub struct Cleanup {
    pub tasks: Vec<SessionIdentity>,
    pub associations: Vec<SnapshotId>,
    pub requests: Vec<(SessionIdentity, RequestIdentity)>,
    pub claims: Vec<OwnedResourceId>,
    pub intents: Vec<OwnedResourceId>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexUsage {
    pub sessions: usize,
    pub branches: usize,
    pub history_revisions: usize,
    pub requests: usize,
    pub content_nodes: usize,
    pub token_material_bytes: usize,
    pub tombstones: usize,
    pub claims: usize,
    pub intents: usize,
}
#[derive(Debug)]
struct SessionRecord {
    revision: u64,
    parent: Option<SessionIdentity>,
    branches: BTreeMap<String, SnapshotId>,
    requests: BTreeMap<RequestIdentity, SnapshotId>,
    claims: BTreeSet<String>,
    intents: BTreeSet<String>,
}

pub struct SessionIndex {
    limits: SessionLimits,
    sessions: BTreeMap<SessionIdentity, SessionRecord>,
    history: BTreeMap<SnapshotId, Arc<BranchRevision>>,
    contents: BTreeMap<ContentKey, Arc<ContentNode>>,
    material: BTreeMap<ContentKey, Box<[u32]>>,
    // Immutable validation checksum survives material eviction; it is neither
    // an authoritative content ID nor a backend key. At most one per node.
    material_checksums: BTreeMap<ContentKey, [u8; 32]>,
    tombstones: BTreeSet<SessionIdentity>,
    material_bytes: usize,
}
impl SessionIndex {
    pub fn new(limits: SessionLimits) -> Self {
        Self {
            limits,
            sessions: BTreeMap::new(),
            history: BTreeMap::new(),
            contents: BTreeMap::new(),
            material: BTreeMap::new(),
            material_checksums: BTreeMap::new(),
            tombstones: BTreeSet::new(),
            material_bytes: 0,
        }
    }
    fn check_id(&self, id: &str) -> Result<(), IndexError> {
        if id.is_empty() || id.len() > self.limits.max_identifier_bytes {
            Err(IndexError::InvalidIdentity)
        } else {
            Ok(())
        }
    }
    fn check_session(&self, id: &SessionIdentity) -> Result<(), IndexError> {
        self.check_id(&id.caller_id)?;
        self.check_id(&id.session_id)?;
        self.check_id(&id.incarnation)
    }
    fn check_content(&self, key: &ContentKey) -> Result<(), IndexError> {
        self.check_id(key.digest())?;
        let c = key.compatibility();
        for v in [&c.namespace, &c.layout_id, &c.hash_scheme] {
            self.check_id(v)?;
        }
        if let Some(salt) = &c.cache_salt {
            if salt.len() > self.limits.max_identifier_bytes {
                return Err(IndexError::InvalidIdentity);
            }
        }
        if c.components.len() > self.limits.max_components {
            return Err(IndexError::Capacity("components"));
        }
        for component in &c.components {
            self.check_id(component)?;
        }
        Ok(())
    }
    pub fn usage(&self) -> IndexUsage {
        let mut u = IndexUsage {
            sessions: self.sessions.len(),
            history_revisions: self.history.len(),
            content_nodes: self.contents.len(),
            token_material_bytes: self.material_bytes,
            tombstones: self.tombstones.len(),
            ..IndexUsage::default()
        };
        for s in self.sessions.values() {
            u.branches += s.branches.len();
            u.requests += s.requests.len();
            u.claims += s.claims.len();
            u.intents += s.intents.len();
        }
        u
    }
    /// No automatic incarnation inference from requests, cache events, or completions.
    pub fn start_session(
        &mut self,
        id: SessionIdentity,
        parent: Option<SessionIdentity>,
    ) -> Result<(), IndexError> {
        self.check_session(&id)?;
        if self.tombstones.contains(&id) {
            return Err(IndexError::RetiredIncarnation);
        }
        if let Some(current) = self.sessions.get(&id) {
            return if current.parent == parent {
                Ok(())
            } else {
                Err(IndexError::Conflict)
            };
        }
        if self
            .sessions
            .keys()
            .any(|s| s.caller_id == id.caller_id && s.session_id == id.session_id)
        {
            return Err(IndexError::ActiveIncarnation);
        }
        if let Some(p) = &parent {
            if p.caller_id != id.caller_id || !self.sessions.contains_key(p) {
                return Err(IndexError::UnknownParent);
            }
        }
        if self.sessions.len() >= self.limits.max_sessions {
            return Err(IndexError::Capacity("sessions"));
        }
        self.sessions.insert(
            id,
            SessionRecord {
                revision: 0,
                parent,
                branches: BTreeMap::new(),
                requests: BTreeMap::new(),
                claims: BTreeSet::new(),
                intents: BTreeSet::new(),
            },
        );
        Ok(())
    }
    /// Registers immutable content independently of any task or replica event.
    pub fn register_content(
        &mut self,
        registration: ContentRegistration,
    ) -> Result<(), IndexError> {
        let ContentRegistration {
            key,
            parent,
            end_token,
            suffix_tokens,
        } = registration;
        self.check_content(&key)?;
        let parent_node = match parent.as_ref() {
            Some(p) => Some(
                self.contents
                    .get(p)
                    .ok_or(IndexError::UnknownContent)?
                    .clone(),
            ),
            None => None,
        };
        let start = parent_node.as_ref().map_or(0, |n| n.end_token);
        if end_token > MAX_INTEGER
            || end_token <= start
            || end_token % key.compatibility.page_size != 0
            || parent_node
                .as_ref()
                .is_some_and(|p| p.key.compatibility != key.compatibility)
            || suffix_tokens
                .as_ref()
                .is_some_and(|t| t.len() as u64 != end_token - start)
        {
            return Err(IndexError::InvalidPrefix);
        }
        if let Some(existing) = self.contents.get(&key) {
            if existing.end_token != end_token
                || existing.parent.as_ref().map(|n| &n.key) != parent.as_ref()
            {
                return Err(IndexError::Conflict);
            }
            if let (Some(old), Some(new)) = (self.material.get(&key), &suffix_tokens) {
                if old.as_ref() != new.as_slice() {
                    return Err(IndexError::Conflict);
                }
            }
            return self.add_material(key, suffix_tokens);
        }
        // Do not evict a supplied parent while making room for its child.
        self.collect_garbage();
        if self.contents.len() >= self.limits.max_content_nodes {
            return Err(IndexError::Capacity("content_nodes"));
        }
        self.add_material(key.clone(), suffix_tokens)?;
        self.contents.insert(
            key.clone(),
            Arc::new(ContentNode {
                key,
                parent: parent_node,
                start_token: start,
                end_token,
            }),
        );
        Ok(())
    }
    fn add_material(
        &mut self,
        key: ContentKey,
        tokens: Option<Vec<u32>>,
    ) -> Result<(), IndexError> {
        if let Some(tokens) = tokens {
            let mut hash = Sha256::new();
            for token in &tokens {
                hash.update(token.to_le_bytes());
            }
            let checksum: [u8; 32] = hash.finalize().into();
            if self
                .material_checksums
                .get(&key)
                .is_some_and(|old| old != &checksum)
            {
                return Err(IndexError::Conflict);
            }
            if self.material.contains_key(&key) {
                return Ok(());
            }
            let bytes = tokens
                .len()
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or(IndexError::Capacity("token_material_bytes"))?;
            let total = self
                .material_bytes
                .checked_add(bytes)
                .ok_or(IndexError::Capacity("token_material_bytes"))?;
            if total > self.limits.max_token_material_bytes {
                return Err(IndexError::Capacity("token_material_bytes"));
            }
            self.material_checksums.insert(key.clone(), checksum);
            // Discard spare Vec capacity so the material-byte budget reflects
            // retained token allocation rather than just the caller's length.
            self.material.insert(key, tokens.into_boxed_slice());
            self.material_bytes = total;
        }
        Ok(())
    }
    /// Eviction never erases a branch or a location. Re-registration may refill it.
    pub fn evict_material(&mut self, key: &ContentKey) -> bool {
        if let Some(tokens) = self.material.remove(key) {
            self.material_bytes -= tokens.len() * std::mem::size_of::<u32>();
            true
        } else {
            false
        }
    }
    pub(crate) fn tokens(&self, key: &ContentKey) -> Option<&[u32]> {
        self.material.get(key).map(AsRef::as_ref)
    }
    pub fn content(&self, key: &ContentKey) -> Option<Arc<ContentNode>> {
        self.contents.get(key).cloned()
    }
    pub fn session_revision(&self, id: &SessionIdentity) -> Option<u64> {
        self.sessions.get(id).map(|s| s.revision)
    }
    pub(crate) fn branch_ids(&self, id: &SessionIdentity) -> Option<Vec<SnapshotId>> {
        self.sessions
            .get(id)
            .map(|s| s.branches.values().cloned().collect())
    }
    pub fn branch(
        &self,
        session: &SessionIdentity,
        branch: &str,
        revision: Option<u64>,
    ) -> Option<Arc<BranchRevision>> {
        let s = self.sessions.get(session)?;
        let id = match revision {
            Some(revision) => SnapshotId {
                session: session.clone(),
                branch_id: branch.into(),
                revision,
            },
            None => s.branches.get(branch)?.clone(),
        };
        self.history.get(&id).cloned()
    }
    pub fn request(
        &self,
        session: &SessionIdentity,
        request: &RequestIdentity,
    ) -> Option<Arc<BranchRevision>> {
        let id = self.sessions.get(session)?.requests.get(request)?;
        self.history.get(id).cloned()
    }
    pub fn associate(
        &mut self,
        association: Association,
    ) -> Result<Arc<BranchRevision>, IndexError> {
        let a = association;
        self.check_id(&a.branch_id)?;
        if a.branch_revision > MAX_INTEGER || a.session_revision > MAX_INTEGER {
            return Err(IndexError::StaleRevision);
        }
        if let Some(r) = &a.request {
            self.check_id(&r.request_id)?;
            self.check_id(&r.incarnation)?;
        }
        let s = self.sessions.get(&a.session).ok_or_else(|| {
            if self.tombstones.contains(&a.session) {
                IndexError::RetiredIncarnation
            } else {
                IndexError::UnknownSession
            }
        })?;
        let id = SnapshotId {
            session: a.session.clone(),
            branch_id: a.branch_id.clone(),
            revision: a.branch_revision,
        };
        if let Some(old) = self.history.get(&id) {
            // Exact retries are idempotent; a lower revision can never rewind latest.
            if old.session_revision == a.session_revision
                && old.content.key == a.content
                && old.parent_frontier == a.parent_frontier
                && old.provenance == a.provenance
                && a.request
                    .as_ref()
                    .is_none_or(|r| s.requests.get(r) == Some(&id))
            {
                return Ok(old.clone());
            }
            return Err(IndexError::Conflict);
        }
        if a.session_revision <= s.revision
            || s.branches
                .get(&a.branch_id)
                .is_some_and(|b| a.branch_revision <= b.revision)
        {
            return Err(IndexError::StaleRevision);
        }
        if a.request
            .as_ref()
            .is_some_and(|r| s.requests.contains_key(r))
        {
            return Err(IndexError::Conflict);
        }
        let content = self
            .contents
            .get(&a.content)
            .ok_or(IndexError::UnknownContent)?
            .clone();
        if let Some(frontier) = &a.parent_frontier {
            let p = self
                .history
                .get(&frontier.snapshot)
                .ok_or(IndexError::UnknownParent)?;
            if frontier.snapshot.session.caller_id != a.session.caller_id
                || p.content
                    .ancestor_at(frontier.end_token)
                    .is_none_or(|ancestor| {
                        !content.contains_ancestor(ancestor.key(), frontier.end_token)
                    })
            {
                return Err(IndexError::InvalidPrefix);
            }
        }
        let is_new_branch = !s.branches.contains_key(&a.branch_id);
        let replaced = s.branches.get(&a.branch_id).cloned();
        let u = self.usage();
        if is_new_branch && u.branches >= self.limits.max_branches {
            return Err(IndexError::Capacity("branches"));
        }
        if a.request.is_some() && u.requests >= self.limits.max_requests {
            return Err(IndexError::Capacity("requests"));
        }
        self.collect_garbage();
        if self.history.len() >= self.limits.max_history_revisions {
            // A replaced latest revision can be reclaimed atomically if no
            // request or frozen operation still needs it.
            let reclaim = replaced.filter(|id| {
                self.history
                    .get(id)
                    .is_some_and(|old| Arc::strong_count(old) == 1)
                    && !self
                        .sessions
                        .values()
                        .any(|s| s.requests.values().any(|request| request == id))
            });
            if let Some(id) = reclaim {
                self.history.remove(&id);
            } else {
                return Err(IndexError::Capacity("history_revisions"));
            }
        }
        let snapshot = Arc::new(BranchRevision {
            id: id.clone(),
            session_revision: a.session_revision,
            parent_frontier: a.parent_frontier,
            provenance: a.provenance,
            content,
        });
        self.history.insert(id.clone(), snapshot.clone());
        let s = self
            .sessions
            .get_mut(&a.session)
            .expect("validated active session");
        s.revision = a.session_revision;
        s.branches.insert(a.branch_id, id.clone());
        if let Some(request) = a.request {
            s.requests.insert(request, id);
        }
        Ok(snapshot)
    }
    /// Request completion releases only the binding; the task and branch remain active.
    pub fn forget_request(
        &mut self,
        task: &SessionIdentity,
        request: &RequestIdentity,
        revision: u64,
    ) -> Result<bool, IndexError> {
        let s = self
            .sessions
            .get_mut(task)
            .ok_or(IndexError::UnknownSession)?;
        if revision <= s.revision || revision > MAX_INTEGER {
            return Err(IndexError::StaleRevision);
        }
        s.revision = revision;
        Ok(s.requests.remove(request).is_some())
    }
    pub fn track_resource(
        &mut self,
        task: &SessionIdentity,
        kind: ResourceKind,
        id: String,
    ) -> Result<OwnedResourceId, IndexError> {
        self.check_id(&id)?;
        let s = self.sessions.get(task).ok_or(IndexError::UnknownSession)?;
        let (set, count, max, name) = match kind {
            ResourceKind::Claim => (
                &s.claims,
                self.usage().claims,
                self.limits.max_claims,
                "claims",
            ),
            ResourceKind::Intent => (
                &s.intents,
                self.usage().intents,
                self.limits.max_intents,
                "intents",
            ),
        };
        if !set.contains(&id) && count >= max {
            return Err(IndexError::Capacity(name));
        }
        let s = self
            .sessions
            .get_mut(task)
            .expect("validated active session");
        match kind {
            ResourceKind::Claim => &mut s.claims,
            ResourceKind::Intent => &mut s.intents,
        }
        .insert(id.clone());
        Ok(OwnedResourceId {
            task: task.clone(),
            id,
        })
    }
    pub fn untrack_resource(
        &mut self,
        task: &SessionIdentity,
        kind: ResourceKind,
        id: &str,
    ) -> Result<bool, IndexError> {
        let s = self
            .sessions
            .get_mut(task)
            .ok_or(IndexError::UnknownSession)?;
        Ok(match kind {
            ResourceKind::Claim => &mut s.claims,
            ResourceKind::Intent => &mut s.intents,
        }
        .remove(id))
    }
    /// Explicit task finish/cancel share cleanup semantics. Default is non-cascading.
    /// This receipt names interests to revoke; it never deletes physical KV or
    /// invokes Engine's legacy recursive finish_task.
    pub fn finish_task(
        &mut self,
        task: &SessionIdentity,
        revision: u64,
    ) -> Result<Cleanup, IndexError> {
        self.finish_task_with_cascade(task, revision, false)
    }
    pub fn finish_task_with_cascade(
        &mut self,
        task: &SessionIdentity,
        revision: u64,
        cascade: bool,
    ) -> Result<Cleanup, IndexError> {
        if self.tombstones.contains(task) {
            return Ok(Cleanup::default());
        }
        let s = self.sessions.get(task).ok_or(IndexError::UnknownSession)?;
        if revision <= s.revision || revision > MAX_INTEGER {
            return Err(IndexError::StaleRevision);
        }
        let mut tasks = BTreeSet::from([task.clone()]);
        if cascade {
            loop {
                let before = tasks.len();
                for (id, s) in &self.sessions {
                    if s.parent.as_ref().is_some_and(|p| tasks.contains(p)) {
                        tasks.insert(id.clone());
                    }
                }
                if tasks.len() == before {
                    break;
                }
            }
        }
        if tasks.len()
            > self
                .limits
                .max_tombstones
                .saturating_sub(self.tombstones.len())
        {
            return Err(IndexError::Capacity("tombstones"));
        }
        let mut cleanup = Cleanup::default();
        for id in tasks {
            let s = self.sessions.remove(&id).expect("collected active task");
            cleanup
                .associations
                .extend(self.history.keys().filter(|key| key.session == id).cloned());
            cleanup
                .requests
                .extend(s.requests.into_keys().map(|request| (id.clone(), request)));
            cleanup
                .claims
                .extend(s.claims.into_iter().map(|resource| OwnedResourceId {
                    task: id.clone(),
                    id: resource,
                }));
            cleanup
                .intents
                .extend(s.intents.into_iter().map(|resource| OwnedResourceId {
                    task: id.clone(),
                    id: resource,
                }));
            self.tombstones.insert(id.clone());
            cleanup.tasks.push(id);
        }
        self.collect_garbage();
        Ok(cleanup)
    }
    /// Tombstones intentionally have no automatic eviction. Exhaustion refuses
    /// new cleanup atomically, rather than allowing retired tasks to resurrect.
    /// A deployment may rotate the entire fenced Router ownership domain.
    pub fn collect_garbage(&mut self) {
        let live: BTreeSet<_> = self
            .sessions
            .values()
            .flat_map(|s| s.branches.values().chain(s.requests.values()))
            .cloned()
            .collect();
        self.history
            .retain(|id, snapshot| live.contains(id) || Arc::strong_count(snapshot) > 1);
        loop {
            let old = self.contents.len();
            self.contents.retain(|key, node| {
                if Arc::strong_count(node) > 1 {
                    true
                } else {
                    self.material_checksums.remove(key);
                    if let Some(tokens) = self.material.remove(key) {
                        self.material_bytes -= tokens.len() * std::mem::size_of::<u32>();
                    }
                    false
                }
            });
            if self.contents.len() == old {
                break;
            }
        }
    }
}
