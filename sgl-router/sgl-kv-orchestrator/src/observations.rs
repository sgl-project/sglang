//! Evidence types and bounded owner/version fences, NOT a placement index.
//!
//! Runtime adapters must query the existing indexer/provider and split pending
//! stakes from confirmed matches. Operation `PublicResult.state_version` belongs
//! to a different sequence domain than KV events and cannot produce a confirmed
//! location here. Reconcile a result with a current scope query or comparable
//! cache watermark. Query tickets expire on every newer event, gap or owner bind.
use crate::{
    models::SessionIdentity, protocol::MAX_INTEGER, session_index::ContentKey,
    target_resolver::ContentScope,
};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CacheTier {
    Gpu,
    Host,
    Storage {
        backend: String,
        backend_namespace: Option<String>,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct OwnerLocation {
    pub owner_id: String,
    pub tier: CacheTier,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CacheEventVersion(pub u64);
/// This is NOT a CacheEventVersion; P1 PublicResult.state_version has this domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct OperationEvidenceVersion(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActualScope {
    content: ContentKey,
    start_token: u64,
    end_token: u64,
    components: Vec<String>,
}
impl ActualScope {
    /// Requires actual coverage within the requested range. It may cover only a
    /// subset of components/pages. A generic operation success is insufficient.
    pub fn new(
        requested: &ContentScope,
        start_token: u64,
        end_token: u64,
        mut components: Vec<String>,
    ) -> Result<Self, ObservationError> {
        components.sort();
        let compatibility = requested.key().compatibility();
        if start_token < requested.start_token()
            || end_token > requested.end_token()
            || start_token >= end_token
            || end_token > MAX_INTEGER
            || start_token % compatibility.page_size != 0
            || end_token % compatibility.page_size != 0
            || components.is_empty()
            || components.windows(2).any(|p| p[0] == p[1])
            || components
                .iter()
                .any(|c| !compatibility.components.contains(c))
        {
            return Err(ObservationError::InvalidScope);
        }
        Ok(Self {
            content: requested.key().clone(),
            start_token,
            end_token,
            components,
        })
    }
    pub fn content(&self) -> &ContentKey {
        &self.content
    }
    pub fn start_token(&self) -> u64 {
        self.start_token
    }
    pub fn end_token(&self) -> u64 {
        self.end_token
    }
    pub fn components(&self) -> &[String] {
        &self.components
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    KvEvent,
    IndexerSnapshot,
    EngineScopeQuery,
}
#[derive(Clone, Debug)]
pub struct ConfirmedLocation {
    owner: OwnerLocation,
    owner_epoch: String,
    scope: ActualScope,
    cache_watermark: Option<CacheEventVersion>,
    observed_at: u64,
    provenance: Provenance,
    fence_revision: u64,
    domain: Arc<()>,
}
impl ConfirmedLocation {
    pub fn owner(&self) -> &OwnerLocation {
        &self.owner
    }
    pub fn owner_epoch(&self) -> &str {
        &self.owner_epoch
    }
    pub fn scope(&self) -> &ActualScope {
        &self.scope
    }
    pub fn cache_watermark(&self) -> Option<CacheEventVersion> {
        self.cache_watermark
    }
    /// Injected adapter time; never used to order different owners or streams.
    pub fn observed_at(&self) -> u64 {
        self.observed_at
    }
    pub fn provenance(&self) -> Provenance {
        self.provenance
    }
}
#[derive(Clone, Debug)]
pub struct ExpectedPlacement {
    pub task: SessionIdentity,
    pub operation_id: String,
    pub destination: OwnerLocation,
    pub scope: ActualScope,
    pub expires_at: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownReason {
    NeverObserved,
    Gap,
    Restart,
    StaleEvidence,
    ProviderUnavailable,
}
#[derive(Clone, Debug)]
pub struct UnknownLocation {
    pub owner: OwnerLocation,
    pub scope: ActualScope,
    pub reason: UnknownReason,
}
/// Deliberately separate fields: a pending match cannot imply residency.
#[derive(Clone, Debug, Default)]
pub struct LocationView {
    pub confirmed: Vec<ConfirmedLocation>,
    pub expected: Vec<ExpectedPlacement>,
    pub unknown: Vec<UnknownLocation>,
}
impl LocationView {
    /// Recheck a retained view before consuming it. New events turn old
    /// confirmations into explicit unknown scopes; expected placement stays separate.
    pub fn revalidate(&mut self, fence: &ObservationFence) {
        self.confirmed.retain(|location| {
            if fence.is_current(location) {
                true
            } else {
                self.unknown.push(UnknownLocation {
                    owner: location.owner.clone(),
                    scope: location.scope.clone(),
                    reason: UnknownReason::StaleEvidence,
                });
                false
            }
        });
    }
}
/// No global placement map is stored in this crate. Implement using the existing
/// indexer/local confirmed tree; merged routing depths are not confirmed evidence.
/// Bounded transport/result sizes are the adapter's responsibility.
pub trait LocationProvider {
    type Error;
    fn locations(
        &self,
        scope: &ContentScope,
        candidates: &[OwnerLocation],
    ) -> Result<LocationView, Self::Error>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObservationError {
    InvalidIdentity,
    InvalidScope,
    UnknownOwner,
    WrongEpoch,
    StaleEvidence,
    IncompleteStream,
    Capacity,
    VersionExhausted,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventDisposition {
    Applied,
    Gap,
    DuplicateOrStale,
}
#[derive(Clone, Debug)]
pub struct QueryTicket {
    owner: OwnerLocation,
    owner_epoch: String,
    fence_revision: u64,
    domain: Arc<()>,
}
impl QueryTicket {
    pub fn owner(&self) -> &OwnerLocation {
        &self.owner
    }
    pub fn owner_epoch(&self) -> &str {
        &self.owner_epoch
    }
}
#[derive(Debug)]
struct OwnerFence {
    epoch: String,
    last_event: Option<CacheEventVersion>,
    complete: bool,
    revision: u64,
}
/// Stores only O(owners × tiers) fencing metadata. It never stores content,
/// replicas, tokens or operation results, and it never auto-rebinds an epoch.
pub struct ObservationFence {
    max_owners: usize,
    max_identifier_bytes: usize,
    owners: BTreeMap<OwnerLocation, OwnerFence>,
    next_revision: u64,
    domain: Arc<()>,
}
impl ObservationFence {
    pub fn new(max_owners: usize, max_identifier_bytes: usize) -> Self {
        Self {
            max_owners,
            max_identifier_bytes,
            owners: BTreeMap::new(),
            next_revision: 0,
            domain: Arc::new(()),
        }
    }
    fn check_id(&self, id: &str) -> Result<(), ObservationError> {
        if id.is_empty() || id.len() > self.max_identifier_bytes {
            Err(ObservationError::InvalidIdentity)
        } else {
            Ok(())
        }
    }
    fn revision(&mut self) -> Result<u64, ObservationError> {
        self.next_revision = self
            .next_revision
            .checked_add(1)
            .ok_or(ObservationError::VersionExhausted)?;
        Ok(self.next_revision)
    }
    /// Only trusted discovery may bind/rebind the current epoch. Cache events
    /// and late operation completions must never call this to fix WrongEpoch.
    /// Calling with the existing epoch is idempotent; `mark_gap` handles reconnects.
    pub fn bind_owner(
        &mut self,
        owner: OwnerLocation,
        epoch: String,
    ) -> Result<(), ObservationError> {
        self.check_id(&owner.owner_id)?;
        self.check_id(&epoch)?;
        if let CacheTier::Storage {
            backend,
            backend_namespace,
        } = &owner.tier
        {
            self.check_id(backend)?;
            if let Some(namespace) = backend_namespace {
                self.check_id(namespace)?;
            }
        }
        if self.owners.get(&owner).is_some_and(|f| f.epoch == epoch) {
            return Ok(());
        }
        if !self.owners.contains_key(&owner) && self.owners.len() >= self.max_owners {
            return Err(ObservationError::Capacity);
        }
        let revision = self.revision()?;
        self.owners.insert(
            owner,
            OwnerFence {
                epoch,
                last_event: None,
                complete: false,
                revision,
            },
        );
        Ok(())
    }
    /// Remove an owner only when discovery removes it. Rebinding uses a fresh
    /// fence revision, so outstanding tickets remain invalid even with epoch reuse.
    pub fn remove_owner(&mut self, owner: &OwnerLocation) -> bool {
        self.owners.remove(owner).is_some()
    }
    pub fn begin_query(&self, owner: &OwnerLocation) -> Result<QueryTicket, ObservationError> {
        let f = self
            .owners
            .get(owner)
            .ok_or(ObservationError::UnknownOwner)?;
        Ok(QueryTicket {
            owner: owner.clone(),
            owner_epoch: f.epoch.clone(),
            fence_revision: f.revision,
            domain: self.domain.clone(),
        })
    }
    fn check_ticket(&self, ticket: &QueryTicket) -> Result<&OwnerFence, ObservationError> {
        if !Arc::ptr_eq(&self.domain, &ticket.domain) {
            return Err(ObservationError::StaleEvidence);
        }
        let f = self
            .owners
            .get(&ticket.owner)
            .ok_or(ObservationError::UnknownOwner)?;
        if f.epoch != ticket.owner_epoch {
            return Err(ObservationError::WrongEpoch);
        }
        if f.revision != ticket.fence_revision {
            return Err(ObservationError::StaleEvidence);
        }
        Ok(f)
    }
    pub fn mark_gap(&mut self, owner: &OwnerLocation, epoch: &str) -> Result<(), ObservationError> {
        let f = self
            .owners
            .get(owner)
            .ok_or(ObservationError::UnknownOwner)?;
        if f.epoch != epoch {
            return Err(ObservationError::WrongEpoch);
        }
        let revision = self.revision()?;
        let f = self.owners.get_mut(owner).expect("validated owner");
        f.complete = false;
        f.revision = revision;
        Ok(())
    }
    /// The adapter calls this for Stored, Removed and Cleared alike. Payloads
    /// remain in the authoritative provider. A gap makes full-stream evidence
    /// unavailable until an explicit complete snapshot is installed.
    pub fn observe_event(
        &mut self,
        owner: &OwnerLocation,
        epoch: &str,
        sequence: CacheEventVersion,
    ) -> Result<EventDisposition, ObservationError> {
        let f = self
            .owners
            .get(owner)
            .ok_or(ObservationError::UnknownOwner)?;
        if f.epoch != epoch {
            return Err(ObservationError::WrongEpoch);
        }
        if f.last_event.is_some_and(|last| sequence <= last) {
            return Ok(EventDisposition::DuplicateOrStale);
        }
        let gap = f
            .last_event
            .is_none_or(|last| last.0.checked_add(1) != Some(sequence.0));
        let revision = self.revision()?;
        let f = self.owners.get_mut(owner).expect("validated owner");
        f.last_event = Some(sequence);
        f.revision = revision;
        if gap {
            f.complete = false;
        }
        Ok(if gap {
            EventDisposition::Gap
        } else {
            EventDisposition::Applied
        })
    }
    /// Only a COMPLETE owner/tier snapshot may repair stream completeness. A
    /// partial match or a scope lookup must use confirm_query and cannot call this.
    /// Snapshot starts after begin_query; any intervening event rejects it.
    pub fn install_complete_snapshot(
        &mut self,
        ticket: &QueryTicket,
        response_epoch: &str,
        watermark: CacheEventVersion,
    ) -> Result<(), ObservationError> {
        if response_epoch != ticket.owner_epoch {
            return Err(ObservationError::WrongEpoch);
        }
        let f = self.check_ticket(ticket)?;
        if f.last_event.is_some_and(|last| watermark < last) {
            return Err(ObservationError::StaleEvidence);
        }
        let revision = self.revision()?;
        let f = self.owners.get_mut(&ticket.owner).expect("validated owner");
        f.last_event = Some(watermark);
        f.complete = true;
        f.revision = revision;
        Ok(())
    }
    /// Confirm only the actual scope a current scope query observed. It does not
    /// repair a missing global stream or assert anything outside this scope.
    pub fn confirm_query(
        &self,
        ticket: &QueryTicket,
        response_epoch: &str,
        actual_scope: ActualScope,
        observed_at: u64,
    ) -> Result<ConfirmedLocation, ObservationError> {
        if response_epoch != ticket.owner_epoch {
            return Err(ObservationError::WrongEpoch);
        }
        self.check_ticket(ticket)?;
        Ok(ConfirmedLocation {
            owner: ticket.owner.clone(),
            owner_epoch: ticket.owner_epoch.clone(),
            scope: actual_scope,
            cache_watermark: None,
            observed_at,
            provenance: Provenance::EngineScopeQuery,
            fence_revision: ticket.fence_revision,
            domain: self.domain.clone(),
        })
    }
    /// Provider must read current committed state at this cache stream watermark.
    /// Operation outcome sequence or PublicResult.state_version is not accepted.
    pub fn confirm_stream(
        &self,
        owner: &OwnerLocation,
        epoch: &str,
        watermark: CacheEventVersion,
        actual_scope: ActualScope,
        observed_at: u64,
        provenance: Provenance,
    ) -> Result<ConfirmedLocation, ObservationError> {
        let f = self
            .owners
            .get(owner)
            .ok_or(ObservationError::UnknownOwner)?;
        if f.epoch != epoch {
            return Err(ObservationError::WrongEpoch);
        }
        if !f.complete {
            return Err(ObservationError::IncompleteStream);
        }
        if f.last_event != Some(watermark) {
            return Err(ObservationError::StaleEvidence);
        }
        if provenance == Provenance::EngineScopeQuery {
            return Err(ObservationError::StaleEvidence);
        }
        Ok(ConfirmedLocation {
            owner: owner.clone(),
            owner_epoch: epoch.into(),
            scope: actual_scope,
            cache_watermark: Some(watermark),
            observed_at,
            provenance,
            fence_revision: f.revision,
            domain: self.domain.clone(),
        })
    }
    /// Must be checked whenever a cached view is consumed. Any intervening cache
    /// event invalidates it conservatively; there is no second placement index.
    pub fn is_current(&self, location: &ConfirmedLocation) -> bool {
        Arc::ptr_eq(&self.domain, &location.domain)
            && self.owners.get(&location.owner).is_some_and(|f| {
                f.epoch == location.owner_epoch && f.revision == location.fence_revision
            })
    }
    pub fn stream_complete(&self, owner: &OwnerLocation) -> bool {
        self.owners.get(owner).is_some_and(|f| f.complete)
    }
    pub fn len(&self) -> usize {
        self.owners.len()
    }
    pub fn is_empty(&self) -> bool {
        self.owners.is_empty()
    }
}

/// Historical operation evidence, deliberately not consumable as a location.
/// Engine result adapters preserve partial actual ranges and keep the operation
/// evidence version in its own domain; then query LocationProvider for residency.
#[derive(Clone, Debug)]
pub struct OperationObservation {
    pub operation_id: String,
    pub owner: OwnerLocation,
    pub owner_epoch: String,
    pub state_version: OperationEvidenceVersion,
    pub ready_scope: Vec<ActualScope>,
    pub reclaimed_scope: Vec<ActualScope>,
}
