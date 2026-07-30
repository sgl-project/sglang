use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::mooncake::{MemoryBuffer, Region as MooncakeRegion};
use crate::pd::buffer::{
    AUX_BYTES, BufferError, CapacityLedger, CompletionRecordInput, CompletionWrites,
    CudaEventSourceFence, CudaHostFlushPort, DataPlaneEffect, DataPlaneIdentity, DataPlaneWorker,
    DataPlaneWorkerState, DestinationExecutor, DestinationRecordPort, DestinationVisibilityFence,
    LeaseHandle, NativeBatchToken, NativeObservationTicket, QUARANTINE_HARD_DEADLINE_MS,
    QuarantineManager, QuarantineUpdate, ReservationRequest, SourceExecutor, SourceWorkRequest,
    TableUseTracker, TransferPlan, TransferStage, ValidatedCompletion,
};
use crate::pd::config::PdProfileV1;
use crate::pd::room::{Clock, PdReason, SystemClock};
use crate::pd::runtime::NativeBootstrapPort;
use crate::pd::transport::TransportError;

use super::PyPdResourceSnapshot;

pub(super) struct NativeSender {
    worker: DataPlaneWorker,
    ledger: Arc<CapacityLedger>,
    quarantine: Arc<QuarantineManager>,
    clock: Arc<SystemClock>,
    deadline_ms: u64,
    leases: HashMap<u64, LeaseHandle>,
    quarantined: HashMap<u64, QuarantinedLease>,
}

struct QuarantinedLease {
    lease: LeaseHandle,
    batch: NativeBatchToken,
    expected_lengths: Vec<u64>,
    observation: Option<NativeObservationTicket>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeLifecycleEffect {
    Idle,
    Released { raw_handle: u64 },
    HardDeadline,
}

impl NativeSender {
    pub(super) fn new(
        port: &NativeBootstrapPort,
        profile: &PdProfileV1,
    ) -> Result<Self, TransportError> {
        let table = port.table().map_err(TransportError::LocalFatal)?;
        let ledger = Arc::new(CapacityLedger::new(
            profile,
            table.tracker(),
            TableUseTracker::new(),
        ));
        let quarantine = Arc::new(QuarantineManager::new(Arc::clone(&ledger)));
        let clock = Arc::new(SystemClock::default());
        let executor = Arc::new(SourceExecutor::new(
            Arc::clone(&ledger),
            Arc::clone(&quarantine),
            Arc::clone(&clock),
        ));
        let stage = port.take_stage_port().map_err(TransportError::LocalFatal)?;
        let worker = DataPlaneWorker::start(4, executor, stage).map_err(buffer_error)?;
        Ok(Self {
            worker,
            ledger,
            quarantine,
            clock,
            deadline_ms: profile.deadline_ms.native_transfer,
            leases: HashMap::new(),
            quarantined: HashMap::new(),
        })
    }

    pub(super) fn execute(
        &mut self,
        raw_handle: u64,
        plan: TransferPlan,
        aux: [u8; AUX_BYTES],
        completion: CompletionWrites,
        cuda_stream: u64,
    ) -> Result<(), TransportError> {
        if self.leases.contains_key(&raw_handle) {
            return Err(TransportError::StaleHandle);
        }
        let deadline = self
            .clock
            .now_monotonic_ms()
            .checked_add(self.deadline_ms)
            .ok_or(TransportError::LocalFatal(PdReason::LocalFatal))?;
        let lease = self
            .ledger
            .reserve(reservation(&plan, deadline))
            .map_err(buffer_error)?;
        let fence = match CudaEventSourceFence::production(4, cuda_stream, Arc::clone(&self.clock))
        {
            Ok(fence) => fence,
            Err(error) => {
                let _ = self.ledger.abort_pre_submit(lease);
                return Err(buffer_error(error));
            }
        };
        let ticket = match self.worker.try_execute_source(SourceWorkRequest {
            plan,
            handle: lease,
            source_fence: fence,
            aux,
            completion,
            deadline_monotonic_ms: deadline,
        }) {
            Ok(ticket) => ticket,
            Err(error) => {
                let _ = self.ledger.abort_pre_submit(lease);
                return Err(buffer_error(error));
            }
        };
        match ticket.wait().map_err(buffer_error)? {
            DataPlaneEffect::DataReady { .. } => {
                self.leases.insert(raw_handle, lease);
                Ok(())
            }
            DataPlaneEffect::TransferFailed { reason, .. } => Err(TransportError::Room(reason)),
            DataPlaneEffect::Quarantined {
                batch,
                expected_lengths,
                reason,
                ..
            } => {
                self.quarantined.insert(
                    raw_handle,
                    QuarantinedLease {
                        lease,
                        batch,
                        expected_lengths,
                        observation: None,
                    },
                );
                Err(TransportError::Room(reason))
            }
            DataPlaneEffect::TransferComplete { .. } => Err(TransportError::InvalidTransition),
        }
    }

    pub(super) fn finish_after_ack(&mut self, raw_handle: u64) -> Result<(), TransportError> {
        let lease = self
            .leases
            .remove(&raw_handle)
            .ok_or(TransportError::StaleHandle)?;
        self.ledger
            .release_source_safe(lease)
            .map_err(buffer_error)?;
        self.ledger
            .handoff_destination(lease)
            .map_err(buffer_error)?;
        self.ledger.release_terminal(lease).map_err(buffer_error)?;
        Ok(())
    }

    pub(super) fn abort(&mut self, raw_handle: u64) -> Result<(), TransportError> {
        if self.quarantined.contains_key(&raw_handle) {
            return Err(TransportError::Room(PdReason::TransferTimeout));
        }
        if self.leases.contains_key(&raw_handle) {
            self.finish_after_ack(raw_handle)?;
        }
        Ok(())
    }

    pub(super) fn lifecycle_tick(&mut self) -> Result<NativeLifecycleEffect, TransportError> {
        if self.worker.lifecycle() == DataPlaneWorkerState::Failed {
            return Err(TransportError::LocalFatal(PdReason::LocalFatal));
        }
        let Some(raw_handle) = self.quarantined.keys().next().copied() else {
            return Ok(NativeLifecycleEffect::Idle);
        };
        let entry = self
            .quarantined
            .get_mut(&raw_handle)
            .ok_or(TransportError::StaleHandle)?;
        if entry.observation.is_none() {
            match self
                .worker
                .try_observe_native(entry.batch, &entry.expected_lengths)
            {
                Ok(observation) => entry.observation = Some(observation),
                Err(BufferError::WorkerFull) => return Ok(NativeLifecycleEffect::Idle),
                Err(error) => return Err(buffer_error(error)),
            }
            return Ok(NativeLifecycleEffect::Idle);
        }
        let Some(safety) = entry
            .observation
            .as_ref()
            .ok_or(TransportError::InvalidTransition)?
            .try_wait()
            .map_err(buffer_error)?
        else {
            return Ok(NativeLifecycleEffect::Idle);
        };
        entry.observation.take();
        match self
            .quarantine
            .observe(
                entry.lease,
                entry.batch,
                safety,
                self.clock.now_monotonic_ms(),
            )
            .map_err(buffer_error)?
        {
            QuarantineUpdate::Pending => Ok(NativeLifecycleEffect::Idle),
            QuarantineUpdate::Released | QuarantineUpdate::AlreadyApplied => {
                self.quarantined.remove(&raw_handle);
                Ok(NativeLifecycleEffect::Released { raw_handle })
            }
            QuarantineUpdate::LocalFatal => Ok(NativeLifecycleEffect::HardDeadline),
        }
    }

    pub(super) fn has_unsafe_leases(&self) -> bool {
        !self.quarantined.is_empty()
    }

    pub(super) fn isolate_peer(&mut self) -> Result<bool, TransportError> {
        for raw_handle in self.leases.keys().copied().collect::<Vec<_>>() {
            self.abort(raw_handle)?;
        }
        Ok(self.has_unsafe_leases())
    }

    pub(super) fn shutdown_worker(
        &mut self,
        timeout: std::time::Duration,
    ) -> Result<DataPlaneWorkerState, TransportError> {
        if self.has_unsafe_leases() {
            return Err(TransportError::LocalFatal(PdReason::LocalFatal));
        }
        for raw_handle in self.leases.keys().copied().collect::<Vec<_>>() {
            self.abort(raw_handle)?;
        }
        self.worker.shutdown(timeout).map_err(buffer_error)
    }

    pub(super) fn shutdown_worker_unsafe(
        &mut self,
        timeout: std::time::Duration,
    ) -> Result<DataPlaneWorkerState, TransportError> {
        self.worker.shutdown(timeout).map_err(buffer_error)
    }

    pub(super) fn resource_snapshot(&self) -> PyPdResourceSnapshot {
        resource_snapshot(
            &self.ledger,
            self.leases.len() + self.quarantined.len(),
            self.worker.pending_count() + self.quarantined.len(),
        )
    }
}

pub(super) struct NativeReceiver {
    device: u32,
    table: Arc<crate::pd::buffer::RegisteredRegionTable<MooncakeRegion>>,
    buffers: BTreeMap<u16, MemoryBuffer>,
    ledger: Arc<CapacityLedger>,
    executor: DestinationExecutor,
    clock: Arc<SystemClock>,
    leases: HashMap<u64, DestinationLease>,
    quarantined: HashMap<u64, QuarantinedDestinationLease>,
}

#[derive(Clone, Copy)]
struct DestinationLease {
    lease: LeaseHandle,
}

struct QuarantinedDestinationLease {
    _lease: LeaseHandle,
    entered_monotonic_ms: u64,
    fatal_emitted: bool,
}

impl NativeReceiver {
    pub(super) fn new(
        port: &NativeBootstrapPort,
        profile: &PdProfileV1,
    ) -> Result<Self, TransportError> {
        let table = port.table().map_err(TransportError::LocalFatal)?;
        let ledger = Arc::new(CapacityLedger::new(
            profile,
            TableUseTracker::new(),
            table.tracker(),
        ));
        Ok(Self {
            device: 5,
            table,
            buffers: port.buffers(),
            executor: DestinationExecutor::new(Arc::clone(&ledger)),
            ledger,
            clock: Arc::new(SystemClock::default()),
            leases: HashMap::new(),
            quarantined: HashMap::new(),
        })
    }

    pub(super) fn reserve(
        &mut self,
        raw_handle: u64,
        plan: &TransferPlan,
        deadline_monotonic_ms: u64,
    ) -> Result<(), TransportError> {
        if self.leases.contains_key(&raw_handle) || self.quarantined.contains_key(&raw_handle) {
            return Err(TransportError::StaleHandle);
        }
        for block in plan.kv_blocks() {
            self.table
                .resolve_kv_range(
                    plan.destination_registration_epoch(),
                    block.region_id,
                    block.destination_page,
                    block.byte_offset,
                    block.byte_length,
                )
                .map_err(buffer_error)?;
        }
        let lease = self
            .ledger
            .reserve(reservation(plan, deadline_monotonic_ms))
            .map_err(buffer_error)?;
        self.leases.insert(raw_handle, DestinationLease { lease });
        Ok(())
    }

    pub(super) fn validate(
        &self,
        raw_handle: u64,
        plan: &TransferPlan,
        expected: &CompletionRecordInput,
    ) -> Result<(), TransportError> {
        let lease = self
            .leases
            .get(&raw_handle)
            .ok_or(TransportError::StaleHandle)?
            .lease;
        for stage in [
            TransferStage::Kv,
            TransferStage::Aux,
            TransferStage::Completion,
        ] {
            self.ledger
                .begin_stage(lease, stage)
                .map_err(|error| destination_validation_error("lease_begin", error))?;
            self.ledger
                .finish_stage(lease, stage)
                .map_err(|error| destination_validation_error("lease_finish", error))?;
        }
        let identity = DataPlaneIdentity::from_plan(plan);
        let flush = CudaHostFlushPort::production()
            .map_err(|error| destination_validation_error("visibility_port", error))?;
        let mut visibility = DestinationVisibilityFence::new(self.device, flush)
            .map_err(|error| destination_validation_error("visibility_fence", error))?;
        let mut records = ExternalRecords {
            buffers: self.buffers.clone(),
        };
        match self
            .executor
            .validate_ready(
                plan,
                lease,
                identity,
                &mut visibility,
                &mut records,
                expected,
            )
            .map_err(|error| destination_validation_error("record_validation", error))?
        {
            DataPlaneEffect::TransferComplete { .. } => Ok(()),
            _ => Err(TransportError::InvalidTransition),
        }
    }

    pub(super) fn finish_after_ack(
        &mut self,
        raw_handle: u64,
        plan: &TransferPlan,
    ) -> Result<ValidatedCompletion, TransportError> {
        let lease = self
            .leases
            .remove(&raw_handle)
            .ok_or(TransportError::StaleHandle)?
            .lease;
        let identity = DataPlaneIdentity::from_plan(plan);
        self.executor
            .commit_after_ack(lease, identity)
            .map_err(buffer_error)?;
        let completion = self
            .executor
            .consume_after_ack(identity)
            .map_err(buffer_error)?
            .ok_or(TransportError::InvalidTransition)?;
        self.ledger
            .release_source_safe(lease)
            .map_err(buffer_error)?;
        self.ledger.release_terminal(lease).map_err(buffer_error)?;
        Ok(completion)
    }

    pub(super) fn abort_after_peer_ack(&mut self, raw_handle: u64) -> Result<(), TransportError> {
        if self.quarantined.contains_key(&raw_handle) {
            return Err(TransportError::Room(PdReason::TransferTimeout));
        }
        let Some(lease) = self.leases.remove(&raw_handle).map(|entry| entry.lease) else {
            return Ok(());
        };
        self.ledger
            .release_failed_safe(lease)
            .map(|_| ())
            .map_err(buffer_error)
    }

    pub(super) fn isolate_peer(&mut self) -> Result<(), TransportError> {
        let entered_monotonic_ms = self.clock.now_monotonic_ms();
        for (raw_handle, entry) in self.leases.drain() {
            self.ledger
                .quarantine_remote_exposed(entry.lease)
                .map_err(buffer_error)?;
            self.quarantined.insert(
                raw_handle,
                QuarantinedDestinationLease {
                    _lease: entry.lease,
                    entered_monotonic_ms,
                    fatal_emitted: false,
                },
            );
        }
        Ok(())
    }

    pub(super) fn lifecycle_tick(&mut self) -> Result<NativeLifecycleEffect, TransportError> {
        let now = self.clock.now_monotonic_ms();
        for entry in self.quarantined.values_mut() {
            if !entry.fatal_emitted
                && now.saturating_sub(entry.entered_monotonic_ms) >= QUARANTINE_HARD_DEADLINE_MS
            {
                entry.fatal_emitted = true;
                return Ok(NativeLifecycleEffect::HardDeadline);
            }
        }
        Ok(NativeLifecycleEffect::Idle)
    }

    pub(super) fn has_unsafe_leases(&self) -> bool {
        !self.quarantined.is_empty()
    }

    pub(super) fn resource_snapshot(&self) -> PyPdResourceSnapshot {
        resource_snapshot(
            &self.ledger,
            self.leases.len() + self.quarantined.len(),
            self.quarantined.len(),
        )
    }
}

fn resource_snapshot(
    ledger: &CapacityLedger,
    native_leases: usize,
    native_batches: usize,
) -> PyPdResourceSnapshot {
    let lease = ledger.snapshot();
    PyPdResourceSnapshot {
        active_rooms: 0,
        active_handles: 0,
        result_slots: 0,
        pending_prepares: 0,
        wire_plans: 0,
        native_leases,
        source_kv_pages: lease.source_kv_pages,
        destination_kv_pages: lease.destination_kv_pages,
        aux_slots: lease.aux_slots,
        completion_slots: lease.completion_slots,
        request_slots: lease.request_slots,
        in_flight_transfers: lease.in_flight_transfers,
        native_batches,
        pending_bytes: lease.pending_bytes,
        quarantined_rooms: lease.quarantined_rooms,
    }
}

struct ExternalRecords {
    buffers: BTreeMap<u16, MemoryBuffer>,
}

impl DestinationRecordPort for ExternalRecords {
    fn read_completion(
        &mut self,
        slot: u16,
    ) -> Result<[u8; crate::pd::buffer::COMPLETION_BYTES], BufferError> {
        let offset = usize::from(slot)
            .checked_mul(crate::pd::buffer::COMPLETION_BYTES)
            .ok_or(BufferError::NativeTransfer)?;
        self.buffers
            .get(&57)
            .ok_or(BufferError::NativeTransfer)?
            .read(offset, crate::pd::buffer::COMPLETION_BYTES)
            .map_err(|_| BufferError::NativeTransfer)?
            .try_into()
            .map_err(|_| BufferError::NativeTransfer)
    }

    fn read_aux(&mut self, slot: u16) -> Result<[u8; AUX_BYTES], BufferError> {
        let offset = usize::from(slot)
            .checked_mul(AUX_BYTES)
            .ok_or(BufferError::NativeTransfer)?;
        self.buffers
            .get(&56)
            .ok_or(BufferError::NativeTransfer)?
            .read(offset, AUX_BYTES)
            .map_err(|_| BufferError::NativeTransfer)?
            .try_into()
            .map_err(|_| BufferError::NativeTransfer)
    }

    fn clear_final_kv_page_tail(
        &mut self,
        region_id: u16,
        page: u32,
        valid_token_count: u32,
    ) -> Result<(), BufferError> {
        let valid_rows = valid_token_count % 64;
        if valid_rows == 0 {
            return Ok(());
        }
        let clear_rows = 64_u32
            .checked_sub(valid_rows)
            .ok_or(BufferError::NativeTransfer)?;
        let offset = u64::from(page)
            .checked_mul(131_072)
            .and_then(|value| value.checked_add(u64::from(valid_rows) * 2_048))
            .ok_or(BufferError::NativeTransfer)?;
        let length = usize::try_from(u64::from(clear_rows) * 2_048)
            .map_err(|_| BufferError::NativeTransfer)?;
        self.buffers
            .get(&region_id)
            .ok_or(BufferError::NativeTransfer)?
            .write(
                usize::try_from(offset).map_err(|_| BufferError::NativeTransfer)?,
                &vec![0; length],
            )
            .map_err(|_| BufferError::NativeTransfer)
    }
}

fn reservation(plan: &TransferPlan, deadline_monotonic_ms: u64) -> ReservationRequest {
    ReservationRequest {
        room: plan.room(),
        handle_generation: plan.transfer_generation(),
        source_pages: plan.source_pages().to_vec(),
        destination_pages: plan.destination_pages().to_vec(),
        aux_slot: plan.source_aux_slot(),
        completion_slot: plan.source_completion_slot(),
        request_slot: plan.source_completion_slot(),
        kv_bytes: plan.expected_kv_bytes(),
        deadline_monotonic_ms,
    }
}

fn destination_validation_error(stage: &'static str, error: BufferError) -> TransportError {
    tracing::warn!(
        stage,
        error = %error,
        "Rust PD destination validation failed closed"
    );
    buffer_error(error)
}

fn buffer_error(error: BufferError) -> TransportError {
    let class = crate::pd::runtime::FailureClass::for_buffer(&error);
    match class.scope {
        crate::pd::runtime::FailureScope::Request => match class.reason {
            PdReason::CapacityExhausted => TransportError::CapacityExhausted,
            PdReason::StaleEpoch => TransportError::StaleHandle,
            _ => TransportError::InvalidBatch,
        },
        crate::pd::runtime::FailureScope::Room => TransportError::Room(class.reason),
        crate::pd::runtime::FailureScope::PeerSession => TransportError::Peer(class.reason),
        crate::pd::runtime::FailureScope::LocalFatal => TransportError::LocalFatal(class.reason),
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use crate::mooncake::PinnedMemory;
    use crate::pd::buffer::TransferPlanInput;
    use crate::pd::protocol::{FixedBytes, RegisterRegions, Role};
    use crate::pd::room::{AttemptId, ProcessEpoch, RegistrationEpoch, RoomId, RoomKey};
    use crate::pd::runtime::{BootstrapPort, NativeRegionDescriptor};

    use super::*;

    fn descriptors(
        role: Role,
        generation: u64,
    ) -> (Vec<NativeRegionDescriptor>, [PinnedMemory; 2]) {
        let aux = PinnedMemory::new(32 * 64).expect("aux owner");
        let completion = PinnedMemory::new(32 * 192).expect("completion owner");
        let device: u32 = if role == Role::Prefill { 4 } else { 5 };
        let mut result = (0_u16..56)
            .map(|region_id| NativeRegionDescriptor {
                region_id,
                address: 0x1_0000_0000
                    + u64::from(device) * 0x1_0000_0000
                    + u64::from(region_id) * 0x40_000,
                length_bytes: 2 * 131_072,
                device: format!("cuda:{device}"),
                dtype: "torch.bfloat16".to_string(),
                shape: vec![2, 64, 8, 128],
                stride_bytes: vec![131_072, 2_048, 256, 2],
                generation,
            })
            .collect::<Vec<_>>();
        for (region_id, owner, slot_bytes) in [(56, &aux, 64_u64), (57, &completion, 192_u64)] {
            result.push(NativeRegionDescriptor {
                region_id,
                address: owner.address(),
                length_bytes: 32 * slot_bytes,
                device: "cpu:0".to_string(),
                dtype: "torch.uint8".to_string(),
                shape: vec![32, slot_bytes],
                stride_bytes: vec![slot_bytes, 1],
                generation,
            });
        }
        (result, [aux, completion])
    }

    fn native_port(role: Role, port: u16) -> (NativeBootstrapPort, [PinnedMemory; 2]) {
        let (descriptors, owners) = descriptors(role, 9);
        let native = NativeBootstrapPort::new_mock(
            role,
            SocketAddr::from(([127, 0, 0, 1], port)),
            FixedBytes::new([0x55; 32]),
            descriptors,
        )
        .expect("mock native port");
        (native, owners)
    }

    fn room() -> RoomId {
        RoomId::new(
            RoomKey::new(ProcessEpoch::random(), 0, AttemptId::random()).expect("room key"),
            1,
        )
        .expect("room")
    }

    fn plan(source: RegistrationEpoch, destination: RegistrationEpoch) -> TransferPlan {
        TransferPlan::new(TransferPlanInput {
            room: room(),
            transfer_generation: 1,
            source_registration_epoch: source,
            destination_registration_epoch: destination,
            source_pages: vec![0],
            destination_pages: vec![0],
            source_aux_slot: 0,
            destination_aux_slot: 0,
            source_completion_slot: 0,
            destination_completion_slot: 0,
            valid_token_count: 1,
            chunk_sequence: 0,
            chunk_count: 1,
            is_last_chunk: true,
        })
        .expect("plan")
    }

    fn connect_prefill(prefill: &NativeBootstrapPort, decode: &NativeBootstrapPort) {
        let registration = decode.registration().expect("decode registration");
        prefill
            .open_peer(&RegisterRegions {
                registration_epoch: registration.registration_epoch,
                layout_fingerprint: registration.layout_fingerprint,
                mooncake_host: registration.mooncake_host,
                mooncake_port: registration.mooncake_port,
                regions: registration.regions,
            })
            .expect("mock peer");
    }

    #[test]
    fn native_sender_and_receiver_account_leases_without_hardware_submission() {
        let profile = PdProfileV1::load_embedded().expect("profile");
        let (prefill, _prefill_owners) = native_port(Role::Prefill, 19100);
        let (decode, _decode_owners) = native_port(Role::Decode, 19101);
        connect_prefill(&prefill, &decode);
        let transfer = plan(
            prefill.registration_epoch().expect("prefill epoch"),
            decode.registration_epoch().expect("decode epoch"),
        );

        let mut receiver = NativeReceiver::new(&decode, &profile).expect("receiver");
        assert_eq!(
            receiver.lifecycle_tick().expect("idle receiver"),
            NativeLifecycleEffect::Idle
        );
        receiver
            .reserve(11, &transfer, u64::MAX)
            .expect("receiver reserve");
        assert_eq!(
            receiver.reserve(11, &transfer, u64::MAX),
            Err(TransportError::StaleHandle)
        );
        let snapshot = receiver.resource_snapshot();
        assert_eq!(snapshot.native_leases, 1);
        assert_eq!(snapshot.destination_kv_pages, 1);
        receiver
            .abort_after_peer_ack(11)
            .expect("receiver abort after safe peer ack");
        receiver
            .abort_after_peer_ack(11)
            .expect("receiver duplicate abort");
        assert_eq!(receiver.resource_snapshot().native_leases, 0);

        let exposed_transfer = plan(
            prefill.registration_epoch().expect("prefill epoch"),
            decode.registration_epoch().expect("decode epoch"),
        );
        receiver
            .reserve(12, &exposed_transfer, u64::MAX)
            .expect("receiver exposed reserve");
        receiver.isolate_peer().expect("isolate exposed peer");
        assert!(receiver.has_unsafe_leases());
        assert_eq!(
            receiver
                .lifecycle_tick()
                .expect("pending receiver quarantine"),
            NativeLifecycleEffect::Idle
        );
        assert_eq!(
            receiver.abort_after_peer_ack(12),
            Err(TransportError::Room(PdReason::TransferTimeout))
        );
        let snapshot = receiver.resource_snapshot();
        assert_eq!(snapshot.native_leases, 1);
        assert_eq!(snapshot.destination_kv_pages, 1);
        assert_eq!(snapshot.quarantined_rooms, 1);

        let mut sender = NativeSender::new(&prefill, &profile).expect("sender");
        assert_eq!(sender.resource_snapshot().native_leases, 0);
        assert_eq!(
            sender.lifecycle_tick().expect("idle sender"),
            NativeLifecycleEffect::Idle
        );
        sender.abort(77).expect("unknown sender abort");
        let deadline = u64::MAX;
        let lease = sender
            .ledger
            .reserve(reservation(&transfer, deadline))
            .expect("sender reserve");
        sender.leases.insert(77, lease);
        for stage in [
            TransferStage::Kv,
            TransferStage::Aux,
            TransferStage::Completion,
        ] {
            sender
                .ledger
                .begin_stage(lease, stage)
                .expect("begin stage");
            sender
                .ledger
                .finish_stage(lease, stage)
                .expect("finish stage");
        }
        assert!(!sender.isolate_peer().expect("safe sender isolation"));
        assert_eq!(sender.resource_snapshot().native_leases, 0);
        assert_eq!(
            sender
                .shutdown_worker(std::time::Duration::from_secs(1))
                .expect("safe sender worker shutdown"),
            DataPlaneWorkerState::Joined
        );
        assert_eq!(
            sender
                .shutdown_worker(std::time::Duration::from_secs(1))
                .expect("duplicate sender worker shutdown"),
            DataPlaneWorkerState::Joined
        );

        let (unsafe_prefill, _unsafe_prefill_owners) = native_port(Role::Prefill, 19102);
        let (unsafe_decode, _unsafe_decode_owners) = native_port(Role::Decode, 19103);
        connect_prefill(&unsafe_prefill, &unsafe_decode);
        let unsafe_transfer = plan(
            unsafe_prefill
                .registration_epoch()
                .expect("unsafe prefill epoch"),
            unsafe_decode
                .registration_epoch()
                .expect("unsafe decode epoch"),
        );
        let mut unsafe_sender =
            NativeSender::new(&unsafe_prefill, &profile).expect("unsafe sender");
        let unsafe_lease = unsafe_sender
            .ledger
            .reserve(reservation(&unsafe_transfer, u64::MAX))
            .expect("unsafe lease");
        unsafe_sender
            .ledger
            .begin_stage(unsafe_lease, TransferStage::Kv)
            .expect("submitted stage");
        let batch = NativeBatchToken::new(99).expect("batch token");
        unsafe_sender
            .quarantine
            .insert(
                unsafe_lease,
                batch,
                unsafe_sender.clock.now_monotonic_ms(),
                PdReason::TransferTimeout,
            )
            .expect("quarantine lease");
        unsafe_sender.quarantined.insert(
            88,
            QuarantinedLease {
                lease: unsafe_lease,
                batch,
                expected_lengths: vec![131_072],
                observation: None,
            },
        );
        assert_eq!(
            unsafe_sender
                .lifecycle_tick()
                .expect("schedule native observation"),
            NativeLifecycleEffect::Idle
        );
        assert!(
            unsafe_sender
                .quarantined
                .get(&88)
                .expect("quarantined handle")
                .observation
                .is_some()
        );
        for _ in 0..100 {
            std::thread::sleep(std::time::Duration::from_millis(1));
            assert_eq!(
                unsafe_sender
                    .lifecycle_tick()
                    .expect("observe pending native batch"),
                NativeLifecycleEffect::Idle
            );
            if unsafe_sender
                .quarantined
                .get(&88)
                .expect("quarantined handle")
                .observation
                .is_none()
            {
                break;
            }
        }
        assert!(
            unsafe_sender
                .quarantined
                .get(&88)
                .expect("quarantined handle")
                .observation
                .is_none()
        );
        assert!(unsafe_sender.has_unsafe_leases());
        assert_eq!(
            unsafe_sender.shutdown_worker(std::time::Duration::from_secs(1)),
            Err(TransportError::LocalFatal(PdReason::LocalFatal))
        );
        assert_eq!(
            unsafe_sender
                .shutdown_worker_unsafe(std::time::Duration::from_secs(1))
                .expect("unsafe worker shutdown"),
            DataPlaneWorkerState::Joined
        );
    }

    #[test]
    fn external_records_use_exact_slots_and_clear_only_the_final_tail() {
        let aux = PinnedMemory::new(2 * AUX_BYTES).expect("aux");
        let completion =
            PinnedMemory::new(2 * crate::pd::buffer::COMPLETION_BYTES).expect("completion");
        let kv = PinnedMemory::new(131_072).expect("kv");
        aux.write(AUX_BYTES, &[0xa5; AUX_BYTES]).expect("write aux");
        completion
            .write(
                crate::pd::buffer::COMPLETION_BYTES,
                &[0x5a; crate::pd::buffer::COMPLETION_BYTES],
            )
            .expect("write completion");
        kv.fill(0xff).expect("fill kv");
        let mut records = ExternalRecords {
            buffers: BTreeMap::from([
                (0, MemoryBuffer::Pinned(kv.clone())),
                (56, MemoryBuffer::Pinned(aux)),
                (57, MemoryBuffer::Pinned(completion)),
            ]),
        };
        assert_eq!(records.read_aux(1).expect("aux slot"), [0xa5; AUX_BYTES]);
        assert_eq!(
            records.read_completion(1).expect("completion slot"),
            [0x5a; crate::pd::buffer::COMPLETION_BYTES]
        );
        records
            .clear_final_kv_page_tail(0, 0, 1)
            .expect("clear tail");
        assert_eq!(kv.read(0, 2_048).expect("valid row"), vec![0xff; 2_048]);
        assert!(
            kv.read(2_048, 131_072 - 2_048)
                .expect("tail")
                .iter()
                .all(|byte| *byte == 0)
        );
        records
            .clear_final_kv_page_tail(0, 0, 64)
            .expect("full page has no tail");
        assert!(records.read_aux(2).is_err());
        assert!(records.clear_final_kv_page_tail(1, 0, 1).is_err());
    }

    #[test]
    fn buffer_errors_preserve_stable_transport_classes() {
        assert_eq!(
            buffer_error(BufferError::WorkerFull),
            TransportError::CapacityExhausted
        );
        assert_eq!(
            buffer_error(BufferError::StaleRegistration),
            TransportError::StaleHandle
        );
        assert_eq!(
            buffer_error(BufferError::PlanMismatch { field: "test" }),
            TransportError::InvalidBatch
        );
        assert_eq!(
            buffer_error(BufferError::SourceFence),
            TransportError::Room(PdReason::TransferFailed)
        );
        assert_eq!(
            buffer_error(BufferError::InvalidTransition),
            TransportError::LocalFatal(PdReason::LocalFatal)
        );
    }

    #[test]
    fn destination_validation_errors_keep_a_safe_internal_check_and_stable_public_reason() {
        let error = BufferError::DataRecord {
            check: "completion_crc",
        };
        assert_eq!(
            error.to_string(),
            "PD data record failed the frozen completion_crc check"
        );
        assert_eq!(
            destination_validation_error("completion", error),
            TransportError::Room(PdReason::TransferFailed)
        );
    }
}
