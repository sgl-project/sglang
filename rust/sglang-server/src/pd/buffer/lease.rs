use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, MutexGuard};

use crate::pd::buffer::BufferError;
use crate::pd::buffer::descriptor::{TableUseGuard, TableUseTracker};
use crate::pd::config::PdProfileV1;
use crate::pd::room::RoomId;

const MAX_KV_PAGES_PER_ROOM: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservationRequest {
    pub room: RoomId,
    pub handle_generation: u64,
    pub source_pages: Vec<u32>,
    pub destination_pages: Vec<u32>,
    pub aux_slot: u16,
    pub completion_slot: u16,
    pub request_slot: u16,
    pub kv_bytes: u64,
    pub deadline_monotonic_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeaseHandle {
    room: RoomId,
    generation: u64,
}

impl LeaseHandle {
    pub const fn room(self) -> RoomId {
        self.room
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStage {
    Kv,
    Aux,
    Completion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionResult {
    Applied,
    AlreadyApplied,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaseSnapshot {
    pub active_rooms: usize,
    pub source_kv_pages: usize,
    pub destination_kv_pages: usize,
    pub aux_slots: usize,
    pub completion_slots: usize,
    pub request_slots: usize,
    pub in_flight_transfers: usize,
    pub pending_bytes: u64,
    pub quarantined_rooms: usize,
    pub release_actions: u64,
    pub handoff_actions: u64,
}

struct CapacityLimits {
    active_rooms: usize,
    native_transfers: usize,
    leased_kv_pages: usize,
    aux_slots: u16,
    completion_slots: u16,
    request_slots: u16,
    pending_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeaseState {
    Active,
    Released,
    HandedOff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NextStage {
    Kv,
    Aux,
    Completion,
    Done,
}

struct RoomLeases {
    request: ReservationRequest,
    source: LeaseState,
    destination: LeaseState,
    aux_completion: LeaseState,
    request_slot: LeaseState,
    next_stage: NextStage,
    in_flight: Option<TransferStage>,
    quarantined: bool,
    source_table_use: TableUseGuard,
    destination_table_use: TableUseGuard,
}

#[derive(Default)]
struct LedgerState {
    rooms: HashMap<LeaseHandle, RoomLeases>,
    terminal: HashSet<LeaseHandle>,
    latest_generation: HashMap<RoomId, u64>,
    source_pages: HashMap<u32, LeaseHandle>,
    destination_pages: HashMap<u32, LeaseHandle>,
    aux_slots: HashMap<u16, LeaseHandle>,
    completion_slots: HashMap<u16, LeaseHandle>,
    request_slots: HashMap<u16, LeaseHandle>,
    in_flight_transfers: usize,
    pending_bytes: u64,
    release_actions: u64,
    handoff_actions: u64,
}

pub struct CapacityLedger {
    limits: CapacityLimits,
    source_tracker: TableUseTracker,
    destination_tracker: TableUseTracker,
    state: Mutex<LedgerState>,
}

impl CapacityLedger {
    pub fn new(
        profile: &PdProfileV1,
        source_tracker: TableUseTracker,
        destination_tracker: TableUseTracker,
    ) -> Self {
        Self {
            limits: CapacityLimits {
                active_rooms: usize::try_from(profile.capacity.active_rooms_per_pair)
                    .expect("frozen active room capacity fits usize"),
                native_transfers: usize::try_from(profile.capacity.native_transfers_per_pair)
                    .expect("frozen native transfer capacity fits usize"),
                leased_kv_pages: usize::try_from(profile.capacity.leased_kv_pages_per_endpoint)
                    .expect("frozen leased page capacity fits usize"),
                aux_slots: u16::try_from(profile.capacity.aux_slots_per_endpoint)
                    .expect("frozen aux capacity fits u16"),
                completion_slots: u16::try_from(profile.capacity.completion_slots_per_endpoint)
                    .expect("frozen completion capacity fits u16"),
                request_slots: u16::try_from(profile.capacity.request_slots_per_endpoint)
                    .expect("frozen request capacity fits u16"),
                pending_bytes: profile.capacity.pending_transfer_bytes_per_pair,
            },
            source_tracker,
            destination_tracker,
            state: Mutex::new(LedgerState::default()),
        }
    }

    pub fn reserve(&self, request: ReservationRequest) -> Result<LeaseHandle, BufferError> {
        validate_request(&request, &self.limits)?;
        let handle = LeaseHandle {
            room: request.room,
            generation: request.handle_generation,
        };
        let mut state = self.lock()?;
        if state.rooms.keys().any(|active| active.room == handle.room)
            || state
                .latest_generation
                .get(&handle.room)
                .is_some_and(|generation| *generation >= handle.generation)
        {
            return Err(BufferError::StaleHandle);
        }
        if state.rooms.len() >= self.limits.active_rooms {
            return exhausted("active_rooms");
        }
        if state.source_pages.len() + request.source_pages.len() > self.limits.leased_kv_pages
            || state.destination_pages.len() + request.destination_pages.len()
                > self.limits.leased_kv_pages
        {
            return exhausted("kv_pages");
        }
        let pending_bytes = state.pending_bytes.checked_add(request.kv_bytes).ok_or(
            BufferError::CapacityExhausted {
                resource: "pending_bytes",
            },
        )?;
        if pending_bytes > self.limits.pending_bytes {
            return exhausted("pending_bytes");
        }
        if request
            .source_pages
            .iter()
            .any(|page| state.source_pages.contains_key(page))
            || request
                .destination_pages
                .iter()
                .any(|page| state.destination_pages.contains_key(page))
            || state.aux_slots.contains_key(&request.aux_slot)
            || state
                .completion_slots
                .contains_key(&request.completion_slot)
            || state.request_slots.contains_key(&request.request_slot)
        {
            return Err(BufferError::ResourceOwned);
        }

        for page in &request.source_pages {
            state.source_pages.insert(*page, handle);
        }
        for page in &request.destination_pages {
            state.destination_pages.insert(*page, handle);
        }
        state.aux_slots.insert(request.aux_slot, handle);
        state
            .completion_slots
            .insert(request.completion_slot, handle);
        state.request_slots.insert(request.request_slot, handle);
        state.pending_bytes = pending_bytes;
        state
            .latest_generation
            .insert(handle.room, handle.generation);
        state.rooms.insert(
            handle,
            RoomLeases {
                request,
                source: LeaseState::Active,
                destination: LeaseState::Active,
                aux_completion: LeaseState::Active,
                request_slot: LeaseState::Active,
                next_stage: NextStage::Kv,
                in_flight: None,
                quarantined: false,
                source_table_use: self.source_tracker.acquire(),
                destination_table_use: self.destination_tracker.acquire(),
            },
        );
        Ok(handle)
    }

    pub fn begin_stage(
        &self,
        handle: LeaseHandle,
        stage: TransferStage,
    ) -> Result<(), BufferError> {
        let mut state = self.lock()?;
        if state.in_flight_transfers >= self.limits.native_transfers {
            return exhausted("native_transfers");
        }
        let room = room_mut(&mut state, handle)?;
        if room.quarantined || room.in_flight.is_some() || !stage_matches(room.next_stage, stage) {
            return Err(BufferError::InvalidTransition);
        }
        room.in_flight = Some(stage);
        state.in_flight_transfers += 1;
        Ok(())
    }

    pub fn finish_stage(
        &self,
        handle: LeaseHandle,
        stage: TransferStage,
    ) -> Result<(), BufferError> {
        let mut state = self.lock()?;
        let room = room_mut(&mut state, handle)?;
        if room.quarantined || room.in_flight != Some(stage) {
            return Err(BufferError::InvalidTransition);
        }
        room.in_flight = None;
        room.next_stage = match stage {
            TransferStage::Kv => NextStage::Aux,
            TransferStage::Aux => NextStage::Completion,
            TransferStage::Completion => NextStage::Done,
        };
        state.in_flight_transfers = state.in_flight_transfers.saturating_sub(1);
        Ok(())
    }

    pub fn abort_pre_submit(&self, handle: LeaseHandle) -> Result<TransitionResult, BufferError> {
        let mut state = self.lock()?;
        if state.terminal.contains(&handle) {
            return Ok(TransitionResult::AlreadyApplied);
        }
        let room = state.rooms.get(&handle).ok_or(BufferError::StaleHandle)?;
        if room.next_stage != NextStage::Kv || room.in_flight.is_some() || room.quarantined {
            return Err(BufferError::InvalidTransition);
        }
        let room = state.rooms.remove(&handle).expect("room checked above");
        release_room_resources(&mut state, handle, &room);
        state.release_actions = state.release_actions.saturating_add(4);
        state.terminal.insert(handle);
        Ok(TransitionResult::Applied)
    }

    pub fn release_source_safe(
        &self,
        handle: LeaseHandle,
    ) -> Result<TransitionResult, BufferError> {
        let mut state = self.lock()?;
        if state.terminal.contains(&handle) {
            return Ok(TransitionResult::AlreadyApplied);
        }
        let pages = {
            let room = room_mut(&mut state, handle)?;
            if room.quarantined {
                return Err(BufferError::InvalidTransition);
            }
            if room.source == LeaseState::Released {
                return Ok(TransitionResult::AlreadyApplied);
            }
            if room.next_stage != NextStage::Done || room.in_flight.is_some() {
                return Err(BufferError::InvalidTransition);
            }
            room.source = LeaseState::Released;
            room.request.source_pages.clone()
        };
        for page in pages {
            state.source_pages.remove(&page);
        }
        state.release_actions = state.release_actions.saturating_add(1);
        Ok(TransitionResult::Applied)
    }

    pub fn handoff_destination(
        &self,
        handle: LeaseHandle,
    ) -> Result<TransitionResult, BufferError> {
        let mut state = self.lock()?;
        if state.terminal.contains(&handle) {
            return Ok(TransitionResult::AlreadyApplied);
        }
        let pages = {
            let room = room_mut(&mut state, handle)?;
            if room.quarantined {
                return Err(BufferError::InvalidTransition);
            }
            if room.destination == LeaseState::HandedOff {
                return Ok(TransitionResult::AlreadyApplied);
            }
            if room.next_stage != NextStage::Done || room.in_flight.is_some() {
                return Err(BufferError::InvalidTransition);
            }
            room.destination = LeaseState::HandedOff;
            room.request.destination_pages.clone()
        };
        for page in pages {
            state.destination_pages.remove(&page);
        }
        state.handoff_actions = state.handoff_actions.saturating_add(1);
        Ok(TransitionResult::Applied)
    }

    pub fn release_terminal(&self, handle: LeaseHandle) -> Result<TransitionResult, BufferError> {
        let mut state = self.lock()?;
        if state.terminal.contains(&handle) {
            return Ok(TransitionResult::AlreadyApplied);
        }
        let room = state.rooms.get(&handle).ok_or(BufferError::StaleHandle)?;
        if room.quarantined
            || room.in_flight.is_some()
            || room.next_stage != NextStage::Done
            || room.source != LeaseState::Released
            || room.destination != LeaseState::HandedOff
        {
            return Err(BufferError::InvalidTransition);
        }
        let mut room = state.rooms.remove(&handle).expect("room checked above");
        state.aux_slots.remove(&room.request.aux_slot);
        state.completion_slots.remove(&room.request.completion_slot);
        state.request_slots.remove(&room.request.request_slot);
        state.pending_bytes = state.pending_bytes.saturating_sub(room.request.kv_bytes);
        room.aux_completion = LeaseState::Released;
        room.request_slot = LeaseState::Released;
        state.release_actions = state.release_actions.saturating_add(2);
        state.terminal.insert(handle);
        Ok(TransitionResult::Applied)
    }

    pub fn quarantine(&self, handle: LeaseHandle) -> Result<TransitionResult, BufferError> {
        let mut state = self.lock()?;
        if state.terminal.contains(&handle) {
            return Ok(TransitionResult::AlreadyApplied);
        }
        let room = room_mut(&mut state, handle)?;
        if room.quarantined {
            return Ok(TransitionResult::AlreadyApplied);
        }
        if room.next_stage == NextStage::Kv && room.in_flight.is_none() {
            return Err(BufferError::InvalidTransition);
        }
        room.quarantined = true;
        room.source_table_use.quarantine();
        room.destination_table_use.quarantine();
        Ok(TransitionResult::Applied)
    }

    /// Quarantine a destination lease after its descriptor was exposed to a
    /// remote peer. The local stage can still be `Kv` because remote DMA is
    /// not represented by this endpoint's stage tracker.
    pub(crate) fn quarantine_remote_exposed(
        &self,
        handle: LeaseHandle,
    ) -> Result<TransitionResult, BufferError> {
        let mut state = self.lock()?;
        if state.terminal.contains(&handle) {
            return Ok(TransitionResult::AlreadyApplied);
        }
        let room = room_mut(&mut state, handle)?;
        if room.quarantined {
            return Ok(TransitionResult::AlreadyApplied);
        }
        room.quarantined = true;
        room.source_table_use.quarantine();
        room.destination_table_use.quarantine();
        Ok(TransitionResult::Applied)
    }

    pub fn release_failed_safe(
        &self,
        handle: LeaseHandle,
    ) -> Result<TransitionResult, BufferError> {
        let mut state = self.lock()?;
        if state.terminal.contains(&handle) {
            return Ok(TransitionResult::AlreadyApplied);
        }
        if state
            .rooms
            .get(&handle)
            .is_some_and(|room| room.destination == LeaseState::HandedOff || room.quarantined)
        {
            return Err(BufferError::InvalidTransition);
        }
        let room = state
            .rooms
            .remove(&handle)
            .ok_or(BufferError::StaleHandle)?;
        if room.in_flight.is_some() {
            state.in_flight_transfers = state.in_flight_transfers.saturating_sub(1);
        }
        let release_count = [
            room.source,
            room.destination,
            room.aux_completion,
            room.request_slot,
        ]
        .into_iter()
        .filter(|lease| *lease == LeaseState::Active)
        .count() as u64;
        release_room_resources(&mut state, handle, &room);
        state.release_actions = state.release_actions.saturating_add(release_count);
        state.terminal.insert(handle);
        Ok(TransitionResult::Applied)
    }

    pub fn resolve_quarantine(&self, handle: LeaseHandle) -> Result<TransitionResult, BufferError> {
        let mut state = self.lock()?;
        if state.terminal.contains(&handle) {
            return Ok(TransitionResult::AlreadyApplied);
        }
        if !state
            .rooms
            .get(&handle)
            .is_some_and(|room| room.quarantined)
        {
            return Err(BufferError::InvalidTransition);
        }
        let room = state
            .rooms
            .remove(&handle)
            .expect("quarantined room exists");
        if room.in_flight.is_some() {
            state.in_flight_transfers = state.in_flight_transfers.saturating_sub(1);
        }
        let release_count = [
            room.source,
            room.destination,
            room.aux_completion,
            room.request_slot,
        ]
        .into_iter()
        .filter(|lease| *lease == LeaseState::Active)
        .count() as u64;
        release_room_resources(&mut state, handle, &room);
        state.release_actions = state.release_actions.saturating_add(release_count);
        state.terminal.insert(handle);
        Ok(TransitionResult::Applied)
    }

    pub fn snapshot(&self) -> LeaseSnapshot {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        LeaseSnapshot {
            active_rooms: state.rooms.len(),
            source_kv_pages: state.source_pages.len(),
            destination_kv_pages: state.destination_pages.len(),
            aux_slots: state.aux_slots.len(),
            completion_slots: state.completion_slots.len(),
            request_slots: state.request_slots.len(),
            in_flight_transfers: state.in_flight_transfers,
            pending_bytes: state.pending_bytes,
            quarantined_rooms: state.rooms.values().filter(|room| room.quarantined).count(),
            release_actions: state.release_actions,
            handoff_actions: state.handoff_actions,
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, LedgerState>, BufferError> {
        self.state
            .lock()
            .map_err(|_| BufferError::InvalidTransition)
    }
}

fn validate_request(
    request: &ReservationRequest,
    limits: &CapacityLimits,
) -> Result<(), BufferError> {
    if request.handle_generation == 0 || request.deadline_monotonic_ms == 0 || request.kv_bytes == 0
    {
        return Err(BufferError::InvalidDescriptor {
            field: "reservation",
            detail: "generation, deadline, and bytes must be non-zero",
        });
    }
    if request.source_pages.is_empty()
        || request.source_pages.len() > MAX_KV_PAGES_PER_ROOM
        || request.destination_pages.len() != request.source_pages.len()
    {
        return Err(BufferError::PlanLimit { field: "pages" });
    }
    if HashSet::<&u32>::from_iter(&request.source_pages).len() != request.source_pages.len()
        || HashSet::<&u32>::from_iter(&request.destination_pages).len()
            != request.destination_pages.len()
    {
        return Err(BufferError::PlanMismatch {
            field: "page_allocation",
        });
    }
    if request.aux_slot >= limits.aux_slots
        || request.completion_slot >= limits.completion_slots
        || request.request_slot >= limits.request_slots
    {
        return Err(BufferError::PlanLimit { field: "slot" });
    }
    Ok(())
}

fn room_mut(state: &mut LedgerState, handle: LeaseHandle) -> Result<&mut RoomLeases, BufferError> {
    state.rooms.get_mut(&handle).ok_or(BufferError::StaleHandle)
}

fn release_room_resources(state: &mut LedgerState, handle: LeaseHandle, room: &RoomLeases) {
    for page in &room.request.source_pages {
        if state.source_pages.get(page) == Some(&handle) {
            state.source_pages.remove(page);
        }
    }
    for page in &room.request.destination_pages {
        if state.destination_pages.get(page) == Some(&handle) {
            state.destination_pages.remove(page);
        }
    }
    if state.aux_slots.get(&room.request.aux_slot) == Some(&handle) {
        state.aux_slots.remove(&room.request.aux_slot);
    }
    if state.completion_slots.get(&room.request.completion_slot) == Some(&handle) {
        state.completion_slots.remove(&room.request.completion_slot);
    }
    if state.request_slots.get(&room.request.request_slot) == Some(&handle) {
        state.request_slots.remove(&room.request.request_slot);
    }
    state.pending_bytes = state.pending_bytes.saturating_sub(room.request.kv_bytes);
}

fn stage_matches(next: NextStage, stage: TransferStage) -> bool {
    matches!(
        (next, stage),
        (NextStage::Kv, TransferStage::Kv)
            | (NextStage::Aux, TransferStage::Aux)
            | (NextStage::Completion, TransferStage::Completion)
    )
}

fn exhausted<T>(resource: &'static str) -> Result<T, BufferError> {
    Err(BufferError::CapacityExhausted { resource })
}
