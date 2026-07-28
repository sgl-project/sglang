use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::pd::buffer::{BufferError, CapacityLedger, LeaseHandle, NativeSafety, TransitionResult};
use crate::pd::room::PdReason;

pub const QUARANTINE_HARD_DEADLINE_MS: u64 = 300_000;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NativeBatchToken(u64);

impl NativeBatchToken {
    pub fn new(value: u64) -> Result<Self, BufferError> {
        if value == 0 {
            return Err(BufferError::InvalidDescriptor {
                field: "native_batch",
                detail: "logical batch token must be non-zero",
            });
        }
        Ok(Self(value))
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for NativeBatchToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeBatchToken")
            .finish_non_exhaustive()
    }
}

struct QuarantineEntry {
    batch: NativeBatchToken,
    entered_monotonic_ms: u64,
    reason: PdReason,
    last_safety: NativeSafety,
    fatal_emitted: bool,
}

#[derive(Default)]
struct QuarantineState {
    entries: HashMap<LeaseHandle, QuarantineEntry>,
    released: HashSet<LeaseHandle>,
    fatal_effects: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuarantineUpdate {
    Pending,
    Released,
    LocalFatal,
    AlreadyApplied,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuarantineSnapshot {
    pub entries: usize,
    pub unsafe_entries: usize,
    pub fatal_effects: u64,
    pub reasons: BTreeMap<PdReason, usize>,
}

pub struct QuarantineManager {
    ledger: Arc<CapacityLedger>,
    state: Mutex<QuarantineState>,
}

impl QuarantineManager {
    pub fn new(ledger: Arc<CapacityLedger>) -> Self {
        Self {
            ledger,
            state: Mutex::new(QuarantineState::default()),
        }
    }

    pub fn insert(
        &self,
        handle: LeaseHandle,
        batch: NativeBatchToken,
        now_monotonic_ms: u64,
        reason: PdReason,
    ) -> Result<TransitionResult, BufferError> {
        if reason == PdReason::Success {
            return Err(BufferError::InvalidTransition);
        }
        let mut state = self.lock()?;
        if state.entries.contains_key(&handle) || state.released.contains(&handle) {
            return Ok(TransitionResult::AlreadyApplied);
        }
        self.ledger.quarantine(handle)?;
        state.entries.insert(
            handle,
            QuarantineEntry {
                batch,
                entered_monotonic_ms: now_monotonic_ms,
                reason,
                last_safety: NativeSafety::Pending,
                fatal_emitted: false,
            },
        );
        Ok(TransitionResult::Applied)
    }

    pub fn observe(
        &self,
        handle: LeaseHandle,
        batch: NativeBatchToken,
        safety: NativeSafety,
        now_monotonic_ms: u64,
    ) -> Result<QuarantineUpdate, BufferError> {
        let mut state = self.lock()?;
        if state.released.contains(&handle) {
            return Ok(QuarantineUpdate::AlreadyApplied);
        }
        let entry = state
            .entries
            .get_mut(&handle)
            .ok_or(BufferError::StaleHandle)?;
        if entry.batch != batch {
            return Err(BufferError::StaleHandle);
        }
        entry.last_safety = safety;
        if safety.is_safe() {
            self.ledger.resolve_quarantine(handle)?;
            state.entries.remove(&handle);
            state.released.insert(handle);
            return Ok(QuarantineUpdate::Released);
        }
        if now_monotonic_ms.saturating_sub(entry.entered_monotonic_ms)
            >= QUARANTINE_HARD_DEADLINE_MS
            && !entry.fatal_emitted
        {
            entry.fatal_emitted = true;
            state.fatal_effects = state.fatal_effects.saturating_add(1);
            return Ok(QuarantineUpdate::LocalFatal);
        }
        Ok(QuarantineUpdate::Pending)
    }

    pub fn snapshot(&self) -> QuarantineSnapshot {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut reasons = BTreeMap::new();
        for entry in state.entries.values() {
            *reasons.entry(entry.reason).or_insert(0) += 1;
        }
        QuarantineSnapshot {
            entries: state.entries.len(),
            unsafe_entries: state
                .entries
                .values()
                .filter(|entry| !entry.last_safety.is_safe())
                .count(),
            fatal_effects: state.fatal_effects,
            reasons,
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, QuarantineState>, BufferError> {
        self.state
            .lock()
            .map_err(|_| BufferError::InvalidTransition)
    }
}
