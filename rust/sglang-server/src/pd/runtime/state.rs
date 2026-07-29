use std::sync::Arc;

use crate::pd::config::PdProfileV1;
use crate::pd::protocol::{FixedBytes, Role};
use crate::pd::room::{
    Clock, PdReason, ProcessEpoch, RegistrationEpoch, RoomId, RoomSnapshot, RoomTable,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeLifecycle {
    Starting,
    LocalReady,
    PairReady,
    Draining,
    Fatal,
    Stopped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairReadiness {
    pub role: Role,
    pub ready: bool,
    pub local_process_epoch: FixedBytes<16>,
    pub local_registration_epoch: FixedBytes<16>,
    pub peer_process_epoch: FixedBytes<16>,
    pub peer_registration_epoch: Option<FixedBytes<16>>,
    pub profile_digest: FixedBytes<32>,
    pub probe_generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeSnapshot {
    pub role: Role,
    pub lifecycle: RuntimeLifecycle,
    pub local_ready: bool,
    pub pair_ready: bool,
    pub session_count: u64,
    pub process_epoch: ProcessEpoch,
    pub registration_epoch: RegistrationEpoch,
    pub peer_process_epoch: Option<FixedBytes<16>>,
    pub peer_registration_epoch: Option<FixedBytes<16>>,
    pub profile_digest: FixedBytes<32>,
    pub active_rooms: usize,
    pub tombstones: usize,
    pub last_reason: Option<PdReason>,
}

impl RuntimeSnapshot {
    pub fn starting(
        role: Role,
        process_epoch: ProcessEpoch,
        registration_epoch: RegistrationEpoch,
        profile_digest: FixedBytes<32>,
    ) -> Self {
        Self {
            role,
            lifecycle: RuntimeLifecycle::Starting,
            local_ready: false,
            pair_ready: false,
            session_count: 0,
            process_epoch,
            registration_epoch,
            peer_process_epoch: None,
            peer_registration_epoch: None,
            profile_digest,
            active_rooms: 0,
            tombstones: 0,
            last_reason: None,
        }
    }

    pub fn local_ready(
        role: Role,
        process_epoch: ProcessEpoch,
        registration_epoch: RegistrationEpoch,
        profile_digest: FixedBytes<32>,
    ) -> Self {
        Self {
            role,
            lifecycle: RuntimeLifecycle::LocalReady,
            local_ready: true,
            pair_ready: false,
            session_count: 0,
            process_epoch,
            registration_epoch,
            peer_process_epoch: None,
            peer_registration_epoch: None,
            profile_digest,
            active_rooms: 0,
            tombstones: 0,
            last_reason: None,
        }
    }

    pub fn enter_pair_ready(&mut self, readiness: &PairReadiness) {
        self.lifecycle = RuntimeLifecycle::PairReady;
        self.pair_ready = true;
        self.session_count = self.session_count.saturating_add(1);
        self.peer_process_epoch = Some(readiness.peer_process_epoch);
        self.peer_registration_epoch = readiness.peer_registration_epoch;
        self.last_reason = None;
    }

    pub fn leave_pair_ready(&mut self, reason: PdReason, rooms: Option<&RoomSnapshot>) {
        self.lifecycle = RuntimeLifecycle::LocalReady;
        self.pair_ready = false;
        self.last_reason = Some(reason);
        if let Some(rooms) = rooms {
            self.active_rooms = rooms.active_rooms;
            self.tombstones = rooms.tombstones;
        }
    }
}

pub struct PairState {
    snapshot: RuntimeSnapshot,
}

impl PairState {
    pub const fn new(snapshot: RuntimeSnapshot) -> Self {
        Self { snapshot }
    }

    pub fn enter_local_ready(&mut self) -> Result<(), PdReason> {
        if self.snapshot.lifecycle != RuntimeLifecycle::Starting {
            return Err(PdReason::ProtocolMismatch);
        }
        self.snapshot.lifecycle = RuntimeLifecycle::LocalReady;
        self.snapshot.local_ready = true;
        self.snapshot.last_reason = None;
        Ok(())
    }

    pub fn activate(
        &mut self,
        readiness: &PairReadiness,
        rooms: &mut RoomTable,
    ) -> Result<Vec<RoomId>, PdReason> {
        if !self.snapshot.local_ready
            || !readiness.ready
            || readiness.role != self.snapshot.role
            || readiness.local_process_epoch
                != FixedBytes::new(self.snapshot.process_epoch.as_bytes())
            || readiness.local_registration_epoch
                != FixedBytes::new(self.snapshot.registration_epoch.as_bytes())
            || readiness.profile_digest != self.snapshot.profile_digest
        {
            return Err(PdReason::ProtocolMismatch);
        }
        if self.snapshot.pair_ready
            && self.snapshot.peer_process_epoch == Some(readiness.peer_process_epoch)
        {
            return Err(PdReason::ProtocolMismatch);
        }
        let replacing_peer = self.snapshot.pair_ready;
        let terminated = if replacing_peer {
            rooms.fail_all(PdReason::PeerUnavailable)
        } else {
            Vec::new()
        };
        self.snapshot.enter_pair_ready(readiness);
        let room_snapshot = rooms.snapshot();
        self.snapshot.active_rooms = room_snapshot.active_rooms;
        self.snapshot.tombstones = room_snapshot.tombstones;
        tracing::info!(
            role = ?self.snapshot.role,
            state = "pair_ready",
            session_count = self.snapshot.session_count,
            replaced_peer = replacing_peer,
            "PD runtime activated an authenticated peer session"
        );
        Ok(terminated)
    }

    pub fn disconnect(&mut self, rooms: &mut RoomTable) -> Vec<RoomId> {
        let terminated = rooms.fail_all(PdReason::PeerUnavailable);
        let rooms = rooms.snapshot();
        self.snapshot
            .leave_pair_ready(PdReason::PeerUnavailable, Some(&rooms));
        tracing::warn!(
            role = ?self.snapshot.role,
            state = "local_ready",
            reason = PdReason::PeerUnavailable.code(),
            terminated_rooms = terminated.len(),
            "PD runtime left PairReady after peer loss"
        );
        terminated
    }

    pub const fn snapshot(&self) -> &RuntimeSnapshot {
        &self.snapshot
    }

    pub fn update_rooms(&mut self, rooms: &RoomSnapshot) {
        self.snapshot.active_rooms = rooms.active_rooms;
        self.snapshot.tombstones = rooms.tombstones;
    }

    pub fn begin_draining(&mut self, rooms: Option<&RoomSnapshot>) {
        self.snapshot.lifecycle = RuntimeLifecycle::Draining;
        self.snapshot.pair_ready = false;
        if let Some(rooms) = rooms {
            self.update_rooms(rooms);
        }
    }

    pub fn mark_fatal(&mut self, reason: PdReason, rooms: Option<&RoomSnapshot>) {
        self.snapshot.lifecycle = RuntimeLifecycle::Fatal;
        self.snapshot.local_ready = false;
        self.snapshot.pair_ready = false;
        self.snapshot.last_reason = Some(if reason == PdReason::Success {
            PdReason::LocalFatal
        } else {
            reason
        });
        if let Some(rooms) = rooms {
            self.update_rooms(rooms);
        }
    }

    pub fn stop(&mut self, rooms: Option<&RoomSnapshot>) {
        self.snapshot.lifecycle = RuntimeLifecycle::Stopped;
        self.snapshot.local_ready = false;
        self.snapshot.pair_ready = false;
        if let Some(rooms) = rooms {
            self.update_rooms(rooms);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeartbeatAction {
    Wait,
    SendPing(u64),
    PeerLost,
}

pub struct HeartbeatTracker {
    interval_ms: u64,
    miss_limit: u8,
    next_deadline_monotonic_ms: u64,
    next_ping_id: u64,
    outstanding_ping: Option<u64>,
    consecutive_misses: u8,
    clock: Arc<dyn Clock>,
}

impl HeartbeatTracker {
    pub fn new(profile: &PdProfileV1, clock: Arc<dyn Clock>) -> Result<Self, &'static str> {
        if profile.deadline_ms.heartbeat_interval != 5_000
            || profile.deadline_ms.heartbeat_misses != 2
        {
            return Err("heartbeat profile does not match frozen v1");
        }
        let now = clock.now_monotonic_ms();
        Ok(Self {
            interval_ms: profile.deadline_ms.heartbeat_interval,
            miss_limit: profile.deadline_ms.heartbeat_misses,
            next_deadline_monotonic_ms: now.saturating_add(profile.deadline_ms.heartbeat_interval),
            next_ping_id: 1,
            outstanding_ping: None,
            consecutive_misses: 0,
            clock,
        })
    }

    pub fn poll(&mut self) -> HeartbeatAction {
        let now = self.clock.now_monotonic_ms();
        if now < self.next_deadline_monotonic_ms {
            return HeartbeatAction::Wait;
        }
        self.next_deadline_monotonic_ms = now.saturating_add(self.interval_ms);
        if self.outstanding_ping.is_some() {
            self.consecutive_misses = self.consecutive_misses.saturating_add(1);
        }
        if self.consecutive_misses >= self.miss_limit {
            return HeartbeatAction::PeerLost;
        }
        let ping_id = self.next_ping_id;
        self.next_ping_id = self.next_ping_id.saturating_add(1);
        self.outstanding_ping = Some(ping_id);
        HeartbeatAction::SendPing(ping_id)
    }

    pub fn on_pong(&mut self, ping_id: u64) -> Result<(), PdReason> {
        if self.outstanding_ping != Some(ping_id) {
            return Err(PdReason::ProtocolMismatch);
        }
        self.outstanding_ping = None;
        self.consecutive_misses = 0;
        self.next_deadline_monotonic_ms = self
            .clock
            .now_monotonic_ms()
            .saturating_add(self.interval_ms);
        Ok(())
    }

    pub const fn consecutive_misses(&self) -> u8 {
        self.consecutive_misses
    }
}
