use crate::pd::room::PdReason;

use super::RuntimeLifecycle;

impl RuntimeLifecycle {
    /// Returns whether the frozen process FSM contains this directed edge.
    ///
    /// Idempotence is handled by the owning coordinator; self-transitions are
    /// deliberately not state-machine edges.
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Starting, Self::LocalReady | Self::Fatal)
                | (
                    Self::LocalReady,
                    Self::PairReady | Self::Draining | Self::Fatal
                )
                | (
                    Self::PairReady,
                    Self::LocalReady | Self::Draining | Self::Fatal
                )
                | (Self::Draining, Self::Stopped | Self::Fatal)
                | (Self::Fatal, Self::Stopped)
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FatalSource {
    StartupInvariant,
    WorkerExit,
    CommandChannelClosed,
    EngineOwner,
    RegistryInvariant,
    QuarantineHardDeadline,
    ShutdownUnsafe,
    ProtocolInvariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FatalRecord {
    pub generation: u64,
    pub source: FatalSource,
    pub reason: PdReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FatalPublish {
    First(FatalRecord),
    Duplicate(FatalRecord),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FirstFatalSnapshot {
    pub first: Option<FatalRecord>,
    pub duplicate_sources: u64,
}

/// Process-local fatal arbitration. The first source fixes the public reason
/// and generation; later observations are counted without changing either.
pub struct FirstFatal {
    first: Option<FatalRecord>,
    next_generation: u64,
    duplicate_sources: u64,
}

impl FirstFatal {
    pub const fn new() -> Self {
        Self {
            first: None,
            next_generation: 1,
            duplicate_sources: 0,
        }
    }

    pub fn publish(&mut self, source: FatalSource, reason: PdReason) -> FatalPublish {
        if let Some(first) = self.first {
            self.duplicate_sources = self.duplicate_sources.saturating_add(1);
            return FatalPublish::Duplicate(first);
        }
        let record = FatalRecord {
            generation: self.next_generation,
            source,
            reason: if reason == PdReason::Success {
                PdReason::LocalFatal
            } else {
                reason
            },
        };
        self.next_generation = self.next_generation.saturating_add(1);
        self.first = Some(record);
        FatalPublish::First(record)
    }

    pub const fn snapshot(&self) -> FirstFatalSnapshot {
        FirstFatalSnapshot {
            first: self.first,
            duplicate_sources: self.duplicate_sources,
        }
    }
}

impl Default for FirstFatal {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownMode {
    Graceful,
    Fatal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownPhase {
    Idle,
    ReadinessDown,
    GoAway,
    StopAccepting,
    DrainingRooms,
    AbortingRooms,
    NativeSafety,
    SchedulerRelease,
    WorkerJoin,
    EngineQuiesce,
    ConnectionClose,
    RegionUnregister,
    EngineDestroy,
    Stopped,
}

impl ShutdownPhase {
    const fn next(self) -> Option<Self> {
        match self {
            Self::Idle => Some(Self::ReadinessDown),
            Self::ReadinessDown => Some(Self::GoAway),
            Self::GoAway => Some(Self::StopAccepting),
            Self::StopAccepting => Some(Self::DrainingRooms),
            Self::DrainingRooms => Some(Self::AbortingRooms),
            Self::AbortingRooms => Some(Self::NativeSafety),
            Self::NativeSafety => Some(Self::SchedulerRelease),
            Self::SchedulerRelease => Some(Self::WorkerJoin),
            Self::WorkerJoin => Some(Self::EngineQuiesce),
            Self::EngineQuiesce => Some(Self::ConnectionClose),
            Self::ConnectionClose => Some(Self::RegionUnregister),
            Self::RegionUnregister => Some(Self::EngineDestroy),
            Self::EngineDestroy | Self::Stopped => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeShutdownOutcome {
    SafeTerminal,
    FatalUnsafe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerLifecycle {
    Starting,
    Running,
    Quiescing,
    Joined,
    Failed,
}

/// Tracks the explicit reverse-dependency shutdown protocol.
///
/// The first mode and terminal outcome are sticky so repeated shutdown calls
/// observe the same result.
pub struct ShutdownTracker {
    generation: u64,
    mode: Option<ShutdownMode>,
    phase: ShutdownPhase,
    outcome: Option<RuntimeShutdownOutcome>,
    history: Vec<ShutdownPhase>,
}

impl ShutdownTracker {
    pub const fn new() -> Self {
        Self {
            generation: 0,
            mode: None,
            phase: ShutdownPhase::Idle,
            outcome: None,
            history: Vec::new(),
        }
    }

    pub fn begin(&mut self, mode: ShutdownMode) -> u64 {
        if self.mode.is_some() {
            return self.generation;
        }
        self.generation = self.generation.saturating_add(1);
        self.mode = Some(mode);
        self.phase = ShutdownPhase::ReadinessDown;
        self.history.push(self.phase);
        self.generation
    }

    pub fn advance(&mut self, next: ShutdownPhase) -> Result<(), PdReason> {
        if self.outcome.is_some() || self.phase.next() != Some(next) {
            return Err(PdReason::ProtocolMismatch);
        }
        self.phase = next;
        self.history.push(next);
        Ok(())
    }

    pub fn complete(
        &mut self,
        outcome: RuntimeShutdownOutcome,
    ) -> Result<RuntimeShutdownOutcome, PdReason> {
        if let Some(first) = self.outcome {
            return Ok(first);
        }
        if self.phase != ShutdownPhase::EngineDestroy {
            return Err(PdReason::ProtocolMismatch);
        }
        self.outcome = Some(outcome);
        self.phase = ShutdownPhase::Stopped;
        self.history.push(self.phase);
        Ok(outcome)
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn mode(&self) -> Option<ShutdownMode> {
        self.mode
    }

    pub const fn phase(&self) -> ShutdownPhase {
        self.phase
    }

    pub const fn outcome(&self) -> Option<RuntimeShutdownOutcome> {
        self.outcome
    }

    pub fn history(&self) -> &[ShutdownPhase] {
        &self.history
    }
}

impl Default for ShutdownTracker {
    fn default() -> Self {
        Self::new()
    }
}
