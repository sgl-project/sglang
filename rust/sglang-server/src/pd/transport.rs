use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

use crate::pd::protocol::{FixedBytes, Role};
use crate::pd::request::KVPoll;
use crate::pd::room::{
    Clock, PdReason, ProcessEpoch, RoomEvent, RoomId, RoomKey, RoomOutcome, RoomRole, RoomSpec,
    RoomTable,
};
use crate::pd::runtime::{
    FatalPublish, FatalSource, PairReadiness, PairState, RuntimeIdentity, RuntimeLifecycle,
    RuntimeShutdownOutcome, RuntimeSnapshot, ShutdownMode, ShutdownPhase,
};

mod helpers;
mod types;

use helpers::{error_for_reason, expect_applied, owner_tag, validate_batch};
use types::{
    HANDLE_GENERATION_MASK, HANDLE_GENERATION_SHIFT, HANDLE_OWNER_SHIFT, HANDLE_SLOT_BITS,
    HandleEntry, HandleSlot, HandleState, SharedClock, TerminalResult,
};
pub use types::{
    MAX_TRANSPORT_BATCH, MAX_TRANSPORT_HANDLES, OpaqueHandle, PD_REGION_COUNT, PdReadinessHandle,
    ReceiverCreateInput, SenderChunk, SenderCreateInput, TerminalEvent, TransportError,
    TransportHandleRole, TransportPollResult, TransportRoomContext, TransportSnapshot,
};

/// Single Rust owner for PD Room state and opaque Scheduler handles.
///
/// Authenticated bootstrap and the data-plane executor feed typed events into
/// this core. Python-facing code never owns a Room, epoch, deadline, or result
/// slot.
pub struct PdTransportCore {
    identity: RuntimeIdentity,
    gateway_bootstrap_host: String,
    gateway_bootstrap_ports: BTreeSet<u16>,
    pair: PairState,
    rooms: Option<RoomTable>,
    slots: Vec<HandleSlot>,
    next_room_generation: u64,
    owner_tag: u32,
    clock: Arc<SharedClock>,
    shared: Arc<RwLock<TransportSnapshot>>,
    abort_generation: u64,
    last_abort_reason: Option<PdReason>,
}

impl PdTransportCore {
    pub fn new<C>(identity: RuntimeIdentity, clock: Arc<C>) -> Result<Self, TransportError>
    where
        C: Clock + 'static,
    {
        let clock: Arc<dyn Clock> = clock;
        let clock = Arc::new(SharedClock(clock));
        let runtime = RuntimeSnapshot::starting(
            identity.role,
            identity.process_epoch,
            identity.registration_epoch,
            identity.profile_digest(),
        );
        let pair = PairState::new(runtime.clone());
        let owner_tag = owner_tag(&identity);
        let shared = Arc::new(RwLock::new(TransportSnapshot {
            runtime,
            model_manifest_digest: identity.model_manifest_digest,
            tokenizer_manifest_digest: identity.tokenizer_manifest_digest,
            layout_fingerprint: identity.layout_fingerprint,
            expected_bootstrap_host: identity.expected_mooncake_host.clone(),
            allowed_bootstrap_ports: identity.allowed_mooncake_ports.clone(),
            accepting_rooms: false,
            active_handles: 0,
            result_slots: 0,
            abort_generation: 0,
            last_abort_reason: None,
        }));
        Ok(Self {
            gateway_bootstrap_host: identity.expected_mooncake_host.clone(),
            gateway_bootstrap_ports: identity.allowed_mooncake_ports.clone(),
            identity,
            pair,
            rooms: None,
            slots: std::iter::repeat_with(HandleSlot::default)
                .take(MAX_TRANSPORT_HANDLES)
                .collect(),
            next_room_generation: 1,
            owner_tag,
            clock,
            shared,
            abort_generation: 0,
            last_abort_reason: None,
        })
    }

    pub fn readiness(&self) -> PdReadinessHandle {
        PdReadinessHandle {
            shared: Arc::clone(&self.shared),
        }
    }

    pub fn configure_gateway_bootstrap(
        &mut self,
        host: String,
        ports: BTreeSet<u16>,
    ) -> Result<(), TransportError> {
        if self.pair.snapshot().lifecycle != RuntimeLifecycle::Starting {
            return Err(TransportError::InvalidTransition);
        }
        if host.is_empty() || ports.is_empty() || ports.contains(&0) {
            return Err(TransportError::InvalidBatch);
        }
        self.gateway_bootstrap_host = host;
        self.gateway_bootstrap_ports = ports;
        self.sync_snapshot();
        Ok(())
    }

    pub fn room_context(
        &self,
        handle: OpaqueHandle,
    ) -> Result<TransportRoomContext, TransportError> {
        let entry = self.resolve(handle, None)?;
        Ok(TransportRoomContext {
            room: entry.room,
            request_digest: entry.request_digest,
        })
    }

    pub fn start_local(&mut self, registered_region_count: usize) -> Result<(), TransportError> {
        if registered_region_count != PD_REGION_COUNT {
            return self.fail_local(PdReason::ProtocolMismatch);
        }
        self.pair
            .enter_local_ready()
            .map_err(TransportError::LocalFatal)?;
        self.sync_snapshot();
        Ok(())
    }

    pub fn activate_pair(
        &mut self,
        readiness: PairReadiness,
        peer_region_count: usize,
        canary_verified: bool,
    ) -> Result<(), TransportError> {
        if peer_region_count != PD_REGION_COUNT || !canary_verified {
            return self.fail_local(PdReason::ProtocolMismatch);
        }
        let decode_epoch = match self.identity.role {
            Role::Decode => self.identity.process_epoch,
            Role::Prefill => ProcessEpoch::from_bytes(readiness.peer_process_epoch.into_array())
                .map_err(|_| TransportError::LocalFatal(PdReason::ProtocolMismatch))?,
        };
        let room_role = match self.identity.role {
            Role::Prefill => RoomRole::Prefill,
            Role::Decode => RoomRole::Decode,
        };
        let mut rooms = RoomTable::new(
            room_role,
            decode_epoch,
            self.identity.registration_epoch,
            &self.identity.profile,
            Arc::clone(&self.clock),
        )
        .map_err(|_| TransportError::LocalFatal(PdReason::ProtocolMismatch))?;
        self.pair
            .activate(&readiness, &mut rooms)
            .map_err(TransportError::Peer)?;
        self.rooms = Some(rooms);
        // Room generations are session-local wire data. A restarted peer also
        // starts at one, so both sides must reset only after the new
        // authenticated session has been accepted.
        self.next_room_generation = 1;
        self.sync_snapshot();
        Ok(())
    }

    pub fn validate_pair_candidate(&self, readiness: &PairReadiness) -> Result<(), TransportError> {
        self.pair
            .validate_candidate(readiness)
            .map_err(TransportError::Peer)
    }

    pub fn sender_create(
        &mut self,
        input: SenderCreateInput,
    ) -> Result<OpaqueHandle, TransportError> {
        self.require_role(Role::Prefill)?;
        self.require_accepting()?;
        let peer_epoch = self
            .pair
            .snapshot()
            .peer_process_epoch
            .ok_or(TransportError::NotReady)?;
        if peer_epoch.as_bytes() != input.decode_process_epoch.as_bytes() {
            return Err(TransportError::StaleHandle);
        }
        let generation = self.next_room_generation;
        let next_generation = self
            .next_room_generation
            .checked_add(1)
            .ok_or(TransportError::LocalFatal(PdReason::LocalFatal))?;
        let room = RoomId::new(
            RoomKey::new(
                input.decode_process_epoch,
                input.bootstrap_room,
                input.attempt_id,
            )
            .map_err(|_| TransportError::InvalidBatch)?,
            generation,
        )
        .map_err(|_| TransportError::InvalidBatch)?;
        let handle = self.create_handle(TransportHandleRole::Sender, room, input.request_digest)?;
        self.next_room_generation = next_generation;
        Ok(handle)
    }

    pub fn sender_create_many(
        &mut self,
        inputs: &[SenderCreateInput],
    ) -> Result<Vec<Result<OpaqueHandle, TransportError>>, TransportError> {
        self.require_role(Role::Prefill)?;
        validate_batch(inputs.len())?;
        self.require_accepting()?;
        Ok(inputs
            .iter()
            .cloned()
            .map(|input| self.sender_create(input))
            .collect())
    }

    pub fn receiver_create_many(
        &mut self,
        inputs: &[ReceiverCreateInput],
    ) -> Result<Vec<Result<OpaqueHandle, TransportError>>, TransportError> {
        self.require_role(Role::Decode)?;
        validate_batch(inputs.len())?;
        self.require_accepting()?;
        let mut output = Vec::with_capacity(inputs.len());
        for input in inputs {
            let generation = self.next_room_generation;
            let next_generation = self
                .next_room_generation
                .checked_add(1)
                .ok_or(TransportError::LocalFatal(PdReason::LocalFatal))?;
            let room = match RoomKey::new(
                self.identity.process_epoch,
                input.bootstrap_room,
                input.attempt_id,
            )
            .and_then(|key| RoomId::new(key, generation))
            {
                Ok(room) => room,
                Err(_) => {
                    output.push(Err(TransportError::InvalidBatch));
                    continue;
                }
            };
            let result =
                self.create_handle(TransportHandleRole::Receiver, room, input.request_digest);
            if result.is_ok() {
                self.next_room_generation = next_generation;
            }
            output.push(result);
        }
        Ok(output)
    }

    pub fn sender_init_many(
        &mut self,
        handles: &[OpaqueHandle],
    ) -> Result<Vec<Result<(), TransportError>>, TransportError> {
        self.require_role(Role::Prefill)?;
        validate_batch(handles.len())?;
        self.require_accepting()?;
        Ok(handles
            .iter()
            .copied()
            .map(|handle| self.sender_init(handle))
            .collect())
    }

    pub fn sender_send_chunks(
        &mut self,
        chunks: &[SenderChunk],
    ) -> Result<Vec<Result<(), TransportError>>, TransportError> {
        self.require_role(Role::Prefill)?;
        validate_batch(chunks.len())?;
        self.require_accepting()?;
        Ok(chunks
            .iter()
            .copied()
            .map(|chunk| self.sender_send_chunk(chunk))
            .collect())
    }

    pub fn receiver_prepare_many(
        &mut self,
        handles: &[OpaqueHandle],
    ) -> Result<Vec<Result<(), TransportError>>, TransportError> {
        self.require_role(Role::Decode)?;
        validate_batch(handles.len())?;
        self.require_accepting()?;
        Ok(handles
            .iter()
            .copied()
            .map(|handle| self.receiver_prepare(handle))
            .collect())
    }

    pub fn poll_many(
        &mut self,
        handles: &[OpaqueHandle],
    ) -> Result<Vec<Result<TransportPollResult, TransportError>>, TransportError> {
        validate_batch(handles.len())?;
        Ok(handles
            .iter()
            .copied()
            .map(|handle| self.poll_one(handle))
            .collect())
    }

    pub fn abort_many(
        &mut self,
        handles: &[OpaqueHandle],
        reason: PdReason,
    ) -> Result<Vec<Result<(), TransportError>>, TransportError> {
        validate_batch(handles.len())?;
        if reason == PdReason::Success {
            return Err(TransportError::InvalidBatch);
        }
        Ok(handles
            .iter()
            .copied()
            .map(|handle| {
                self.record_terminal(TerminalEvent {
                    handle,
                    reason,
                    first_token_id: None,
                    transfer_bytes: 0,
                })
                .map(|_| ())
            })
            .collect())
    }

    pub fn clear_many(
        &mut self,
        handles: &[OpaqueHandle],
    ) -> Result<Vec<Result<(), TransportError>>, TransportError> {
        validate_batch(handles.len())?;
        let results = handles
            .iter()
            .copied()
            .map(|handle| self.clear_one(handle))
            .collect();
        self.sync_snapshot();
        Ok(results)
    }

    /// Ingests the single typed terminal path used by control and data workers.
    /// Returns `true` for the first terminal transition and `false` for an
    /// idempotent duplicate.
    pub fn record_terminal(&mut self, event: TerminalEvent) -> Result<bool, TransportError> {
        let (role, room, request_digest, state, created_monotonic_ms, terminal_exists) = {
            let entry = self.resolve(event.handle, None)?;
            (
                entry.role,
                entry.room,
                entry.request_digest,
                entry.state,
                entry.created_monotonic_ms,
                entry.terminal.is_some(),
            )
        };
        if terminal_exists {
            return Ok(false);
        }
        if event.reason == PdReason::Success {
            if state != HandleState::Transferring
                || (role == TransportHandleRole::Sender && event.first_token_id.is_some())
            {
                return Err(TransportError::InvalidTransition);
            }
            self.finish_success(role, room, request_digest)?;
        } else {
            self.apply_room(room, RoomEvent::Abort(event.reason))?;
            self.abort_generation = self.abort_generation.saturating_add(1);
            self.last_abort_reason = Some(event.reason);
        }
        let now = self.clock.now_monotonic_ms();
        let entry = self.resolve_mut(event.handle, None)?;
        entry.state = HandleState::Terminal;
        entry.transfer_bytes = event.transfer_bytes;
        entry.terminal = Some(TerminalResult {
            reason: event.reason,
            transfer_bytes: event.transfer_bytes,
            transfer_latency_ms: now.saturating_sub(created_monotonic_ms),
            first_token_id: event.first_token_id,
            first_token_consumed: event.first_token_id.is_none(),
        });
        self.sync_room_counts();
        Ok(true)
    }

    pub fn peer_lost(&mut self) -> Vec<OpaqueHandle> {
        let handles: Vec<_> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, value)| {
                value.entry.as_ref().and_then(|entry| {
                    (entry.state != HandleState::Terminal)
                        .then(|| self.handle_for(slot, value.generation, entry.role))
                })
            })
            .collect();
        for handle in &handles {
            let _ = self.record_terminal(TerminalEvent {
                handle: *handle,
                reason: PdReason::PeerUnavailable,
                first_token_id: None,
                transfer_bytes: 0,
            });
        }
        if let Some(rooms) = self.rooms.as_mut() {
            self.pair.disconnect(rooms);
        }
        self.sync_snapshot();
        handles
    }

    pub fn publish_fatal(&mut self, source: FatalSource, reason: PdReason) -> FatalPublish {
        let published = self.pair.publish_fatal(source, reason, self.rooms.as_mut());
        self.sync_snapshot();
        if let FatalPublish::First(record) = published {
            let handles: Vec<_> = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(slot, value)| {
                    value.entry.as_ref().and_then(|entry| {
                        (entry.state != HandleState::Terminal)
                            .then(|| self.handle_for(slot, value.generation, entry.role))
                    })
                })
                .collect();
            for handle in handles {
                let _ = self.record_terminal(TerminalEvent {
                    handle,
                    reason: record.reason,
                    first_token_id: None,
                    transfer_bytes: 0,
                });
            }
        }
        self.sync_snapshot();
        published
    }

    pub fn begin_shutdown(&mut self, mode: ShutdownMode) -> Result<u64, TransportError> {
        if self.pair.snapshot().lifecycle == RuntimeLifecycle::Starting {
            self.publish_fatal(FatalSource::ShutdownUnsafe, PdReason::LocalFatal);
        }
        let rooms = self.rooms.as_ref().map(RoomTable::snapshot);
        let generation = self
            .pair
            .begin_draining(mode, rooms.as_ref())
            .map_err(TransportError::LocalFatal)?;
        self.sync_snapshot();
        Ok(generation)
    }

    pub fn advance_shutdown(&mut self, phase: ShutdownPhase) -> Result<(), TransportError> {
        self.pair
            .advance_shutdown(phase)
            .map_err(TransportError::LocalFatal)?;
        if phase == ShutdownPhase::AbortingRooms {
            let handles: Vec<_> = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(slot, value)| {
                    value.entry.as_ref().and_then(|entry| {
                        (entry.state != HandleState::Terminal)
                            .then(|| self.handle_for(slot, value.generation, entry.role))
                    })
                })
                .collect();
            for handle in handles {
                let _ = self.record_terminal(TerminalEvent {
                    handle,
                    reason: PdReason::Aborted,
                    first_token_id: None,
                    transfer_bytes: 0,
                });
            }
        }
        self.sync_snapshot();
        Ok(())
    }

    pub fn complete_shutdown(
        &mut self,
        outcome: RuntimeShutdownOutcome,
    ) -> Result<RuntimeShutdownOutcome, TransportError> {
        let rooms = self.rooms.as_ref().map(RoomTable::snapshot);
        let outcome = self
            .pair
            .stop(outcome, rooms.as_ref())
            .map_err(TransportError::LocalFatal)?;
        self.sync_snapshot();
        Ok(outcome)
    }

    pub fn shutdown(&mut self) -> Result<RuntimeShutdownOutcome, TransportError> {
        if let Some(outcome) = self.pair.snapshot().shutdown_outcome {
            return Ok(outcome);
        }
        let mode = if matches!(
            self.pair.snapshot().lifecycle,
            RuntimeLifecycle::Starting | RuntimeLifecycle::Fatal
        ) {
            ShutdownMode::Fatal
        } else {
            ShutdownMode::Graceful
        };
        self.begin_shutdown(mode)?;
        for phase in [
            ShutdownPhase::GoAway,
            ShutdownPhase::StopAccepting,
            ShutdownPhase::DrainingRooms,
            ShutdownPhase::AbortingRooms,
            ShutdownPhase::NativeSafety,
            ShutdownPhase::SchedulerRelease,
            ShutdownPhase::WorkerJoin,
            ShutdownPhase::EngineQuiesce,
            ShutdownPhase::ConnectionClose,
            ShutdownPhase::RegionUnregister,
            ShutdownPhase::EngineDestroy,
        ] {
            self.advance_shutdown(phase)?;
        }
        let outcome = if mode == ShutdownMode::Fatal {
            RuntimeShutdownOutcome::FatalUnsafe
        } else {
            RuntimeShutdownOutcome::SafeTerminal
        };
        self.complete_shutdown(outcome)
    }

    fn create_handle(
        &mut self,
        role: TransportHandleRole,
        room: RoomId,
        request_digest: FixedBytes<32>,
    ) -> Result<OpaqueHandle, TransportError> {
        if request_digest.as_bytes().iter().all(|byte| *byte == 0) {
            return Err(TransportError::InvalidBatch);
        }
        let slot = self
            .slots
            .iter()
            .position(|slot| slot.entry.is_none())
            .ok_or(TransportError::CapacityExhausted)?;
        let spec = RoomSpec::new(room, request_digest, self.identity.registration_epoch)
            .map_err(|_| TransportError::InvalidBatch)?;
        self.apply_room_spec(spec)?;
        let generation = self.slots[slot]
            .generation
            .checked_add(1)
            .filter(|generation| *generation <= HANDLE_GENERATION_MASK)
            .ok_or(TransportError::LocalFatal(PdReason::LocalFatal))?;
        self.slots[slot].generation = generation;
        self.slots[slot].entry = Some(HandleEntry {
            role,
            room,
            request_digest,
            process_epoch: self.identity.process_epoch,
            registration_epoch: self.identity.registration_epoch,
            state: HandleState::Created,
            created_monotonic_ms: self.clock.now_monotonic_ms(),
            transfer_bytes: 0,
            terminal: None,
        });
        let handle = self.handle_for(slot, generation, role);
        self.sync_snapshot();
        Ok(handle)
    }

    fn sender_init(&mut self, handle: OpaqueHandle) -> Result<(), TransportError> {
        let (room, request_digest, state) = {
            let entry = self.resolve(handle, Some(TransportHandleRole::Sender))?;
            (entry.room, entry.request_digest, entry.state)
        };
        if state == HandleState::WaitingForInput {
            return Ok(());
        }
        if state != HandleState::Created {
            return Err(TransportError::InvalidTransition);
        }
        self.apply_peer_spec(room, request_digest)?;
        self.apply_room(room, RoomEvent::SourceReady)?;
        self.resolve_mut(handle, Some(TransportHandleRole::Sender))?
            .state = HandleState::WaitingForInput;
        self.sync_room_counts();
        Ok(())
    }

    fn sender_send_chunk(&mut self, chunk: SenderChunk) -> Result<(), TransportError> {
        let (room, request_digest, state) = {
            let entry = self.resolve(chunk.handle, Some(TransportHandleRole::Sender))?;
            (entry.room, entry.request_digest, entry.state)
        };
        if state == HandleState::Transferring {
            return Ok(());
        }
        if state != HandleState::WaitingForInput || chunk.transfer_bytes == 0 {
            return Err(TransportError::InvalidTransition);
        }
        self.apply_room(
            room,
            RoomEvent::TransferSubmitted {
                plan_digest: request_digest,
            },
        )?;
        self.apply_room(room, RoomEvent::TransferTerminal)?;
        let entry = self.resolve_mut(chunk.handle, Some(TransportHandleRole::Sender))?;
        entry.state = HandleState::Transferring;
        entry.transfer_bytes = chunk.transfer_bytes;
        self.sync_room_counts();
        Ok(())
    }

    fn receiver_prepare(&mut self, handle: OpaqueHandle) -> Result<(), TransportError> {
        let (room, request_digest, state) = {
            let entry = self.resolve(handle, Some(TransportHandleRole::Receiver))?;
            (entry.room, entry.request_digest, entry.state)
        };
        if state == HandleState::Transferring {
            return Ok(());
        }
        if state != HandleState::Created {
            return Err(TransportError::InvalidTransition);
        }
        self.apply_peer_spec(room, request_digest)?;
        self.apply_room(
            room,
            RoomEvent::PrepareAccepted {
                plan_digest: request_digest,
            },
        )?;
        self.resolve_mut(handle, Some(TransportHandleRole::Receiver))?
            .state = HandleState::Transferring;
        self.sync_room_counts();
        Ok(())
    }

    fn finish_success(
        &mut self,
        role: TransportHandleRole,
        room: RoomId,
        plan_digest: FixedBytes<32>,
    ) -> Result<(), TransportError> {
        match role {
            TransportHandleRole::Sender => {
                self.apply_room(room, RoomEvent::TransferComplete { plan_digest })?;
            }
            TransportHandleRole::Receiver => {
                self.apply_room(room, RoomEvent::DataReady { plan_digest })?;
                self.apply_room(room, RoomEvent::TransferCompleteAck { plan_digest })?;
            }
        }
        Ok(())
    }

    fn poll_one(&mut self, handle: OpaqueHandle) -> Result<TransportPollResult, TransportError> {
        let entry = self.resolve_mut(handle, None)?;
        let status = match entry.state {
            HandleState::Created => KVPoll::Bootstrapping,
            HandleState::WaitingForInput => KVPoll::WaitingForInput,
            HandleState::Transferring => KVPoll::Transferring,
            HandleState::Terminal => {
                if entry
                    .terminal
                    .as_ref()
                    .is_some_and(|terminal| terminal.reason == PdReason::Success)
                {
                    KVPoll::Success
                } else {
                    KVPoll::Failed
                }
            }
        };
        let Some(terminal) = entry.terminal.as_mut() else {
            return Ok(TransportPollResult {
                handle,
                status,
                reason: PdReason::Success,
                retryable: false,
                transfer_bytes: entry.transfer_bytes,
                transfer_latency_ms: 0,
                terminal_generation: entry.room.generation,
                first_token_id: None,
                first_token_consumed: true,
            });
        };
        let consumed_before = terminal.first_token_consumed;
        let first_token_id = (!consumed_before)
            .then_some(terminal.first_token_id)
            .flatten();
        if first_token_id.is_some() {
            terminal.first_token_consumed = true;
        }
        Ok(TransportPollResult {
            handle,
            status,
            reason: terminal.reason,
            retryable: terminal.reason.retryable(),
            transfer_bytes: terminal.transfer_bytes,
            transfer_latency_ms: terminal.transfer_latency_ms,
            terminal_generation: entry.room.generation,
            first_token_id,
            first_token_consumed: consumed_before,
        })
    }

    fn clear_one(&mut self, handle: OpaqueHandle) -> Result<(), TransportError> {
        let slot = handle.slot();
        let entry = self.resolve(handle, None)?;
        if entry.state != HandleState::Terminal
            || entry
                .terminal
                .as_ref()
                .is_some_and(|result| !result.first_token_consumed)
        {
            return Err(TransportError::InvalidTransition);
        }
        self.slots[slot].entry = None;
        Ok(())
    }

    fn resolve(
        &self,
        handle: OpaqueHandle,
        expected_role: Option<TransportHandleRole>,
    ) -> Result<&HandleEntry, TransportError> {
        if handle.owner_tag() != self.owner_tag
            || handle.slot() >= self.slots.len()
            || handle.generation() == 0
        {
            return Err(TransportError::StaleHandle);
        }
        if expected_role.is_some_and(|expected| expected != handle.role()) {
            return Err(TransportError::WrongRole);
        }
        let slot = &self.slots[handle.slot()];
        let entry = slot
            .entry
            .as_ref()
            .filter(|_| slot.generation == handle.generation())
            .ok_or(TransportError::StaleHandle)?;
        if entry.role != handle.role()
            || expected_role.is_some_and(|expected| entry.role != expected)
            || entry.process_epoch != self.identity.process_epoch
            || entry.registration_epoch != self.identity.registration_epoch
        {
            return Err(TransportError::StaleHandle);
        }
        Ok(entry)
    }

    fn resolve_mut(
        &mut self,
        handle: OpaqueHandle,
        expected_role: Option<TransportHandleRole>,
    ) -> Result<&mut HandleEntry, TransportError> {
        if handle.owner_tag() != self.owner_tag
            || handle.slot() >= self.slots.len()
            || handle.generation() == 0
        {
            return Err(TransportError::StaleHandle);
        }
        if expected_role.is_some_and(|expected| expected != handle.role()) {
            return Err(TransportError::WrongRole);
        }
        let slot = &mut self.slots[handle.slot()];
        let entry = slot
            .entry
            .as_mut()
            .filter(|_| slot.generation == handle.generation())
            .ok_or(TransportError::StaleHandle)?;
        if entry.role != handle.role()
            || expected_role.is_some_and(|expected| entry.role != expected)
            || entry.process_epoch != self.identity.process_epoch
            || entry.registration_epoch != self.identity.registration_epoch
        {
            return Err(TransportError::StaleHandle);
        }
        Ok(entry)
    }

    fn handle_for(&self, slot: usize, generation: u64, role: TransportHandleRole) -> OpaqueHandle {
        OpaqueHandle(
            (u64::from(self.owner_tag) << HANDLE_OWNER_SHIFT)
                | (generation << HANDLE_GENERATION_SHIFT)
                | (role.bit() << HANDLE_SLOT_BITS)
                | slot as u64,
        )
    }

    fn apply_room_spec(&mut self, spec: RoomSpec) -> Result<(), TransportError> {
        let outcome = self
            .rooms
            .as_mut()
            .ok_or(TransportError::NotReady)?
            .observe_local(spec);
        expect_applied(outcome)
    }

    fn apply_peer_spec(
        &mut self,
        room: RoomId,
        request_digest: FixedBytes<32>,
    ) -> Result<(), TransportError> {
        let spec = RoomSpec::new(room, request_digest, self.identity.registration_epoch)
            .map_err(|_| TransportError::InvalidBatch)?;
        let outcome = self
            .rooms
            .as_mut()
            .ok_or(TransportError::NotReady)?
            .observe_peer(spec);
        expect_applied(outcome)
    }

    fn apply_room(&mut self, room: RoomId, event: RoomEvent) -> Result<(), TransportError> {
        let outcome = self
            .rooms
            .as_mut()
            .ok_or(TransportError::NotReady)?
            .apply(room, event);
        match outcome {
            RoomOutcome::Applied(_) | RoomOutcome::Terminal { .. } => Ok(()),
            RoomOutcome::Rejected(reason) => Err(error_for_reason(reason)),
        }
    }

    fn require_role(&self, expected: Role) -> Result<(), TransportError> {
        if self.identity.role == expected {
            Ok(())
        } else {
            Err(TransportError::WrongRole)
        }
    }

    fn require_accepting(&self) -> Result<(), TransportError> {
        let snapshot = self.readiness().snapshot();
        if snapshot.accepting_rooms {
            Ok(())
        } else if snapshot.runtime.lifecycle == RuntimeLifecycle::Fatal {
            Err(TransportError::LocalFatal(
                snapshot.runtime.last_reason.unwrap_or(PdReason::LocalFatal),
            ))
        } else {
            Err(TransportError::NotReady)
        }
    }

    fn fail_local<T>(&mut self, reason: PdReason) -> Result<T, TransportError> {
        self.publish_fatal(FatalSource::ProtocolInvariant, reason);
        Err(TransportError::LocalFatal(reason))
    }

    fn sync_room_counts(&mut self) {
        if let Some(rooms) = self.rooms.as_ref() {
            self.pair.update_rooms(&rooms.snapshot());
        }
        self.sync_snapshot();
    }

    fn sync_snapshot(&self) {
        let active_handles = self
            .slots
            .iter()
            .filter(|slot| slot.entry.is_some())
            .count();
        let result_slots = self
            .slots
            .iter()
            .filter(|slot| {
                slot.entry
                    .as_ref()
                    .is_some_and(|entry| entry.terminal.is_some())
            })
            .count();
        let runtime = self.pair.snapshot().clone();
        let accepting_rooms =
            runtime.lifecycle == RuntimeLifecycle::PairReady && active_handles < self.slots.len();
        *self
            .shared
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = TransportSnapshot {
            runtime,
            model_manifest_digest: self.identity.model_manifest_digest,
            tokenizer_manifest_digest: self.identity.tokenizer_manifest_digest,
            layout_fingerprint: self.identity.layout_fingerprint,
            expected_bootstrap_host: self.gateway_bootstrap_host.clone(),
            allowed_bootstrap_ports: self.gateway_bootstrap_ports.clone(),
            accepting_rooms,
            active_handles,
            result_slots,
            abort_generation: self.abort_generation,
            last_abort_reason: self.last_abort_reason,
        };
    }
}
