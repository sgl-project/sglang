use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::mooncake::{
    Batch, EngineOwner, MemoryBuffer, MemoryLocation, Peer, Region as MooncakeRegion,
    RemoteRegionDescriptor, TransferOperation,
};
use crate::pd::buffer::descriptor::{
    AUX_SLOT_BYTES, COMPLETION_SLOT_BYTES, KV_PAGE_BYTES, KV_REGION_COUNT, RegisteredRegionTable,
};
use crate::pd::buffer::{
    BufferError, NativeBatchToken, NativePhase, NativeStageCommand, NativeStagePort,
};
use crate::pd::protocol::{RegionRecord, RegisterRegions};
use crate::pd::room::RegistrationEpoch;

#[derive(Clone)]
pub struct AuthenticatedRemoteRegionTable {
    epoch: RegistrationEpoch,
    regions: Vec<RemoteRegionDescriptor>,
}

impl AuthenticatedRemoteRegionTable {
    pub(crate) fn from_authenticated_register(
        registration: &RegisterRegions,
    ) -> Result<Self, BufferError> {
        let epoch = RegistrationEpoch::from_bytes(registration.registration_epoch.into_array())
            .map_err(|_| BufferError::StaleRegistration)?;
        let regions = validate_authenticated_destination(&registration.regions)?;
        Ok(Self { epoch, regions })
    }

    pub const fn epoch(&self) -> RegistrationEpoch {
        self.epoch
    }

    pub(crate) fn region(&self, region_id: u16) -> Result<&RemoteRegionDescriptor, BufferError> {
        self.regions
            .get(usize::from(region_id))
            .ok_or(BufferError::StaleRegistration)
    }
}

/// Mooncake-backed production adapter for the typed PD data-plane stages.
///
/// Construction is crate-scoped because remote addresses must come directly
/// from an authenticated `RegisterRegions` frame, never from an external
/// request or an unauthenticated public API.
pub struct MooncakeNativeStagePort {
    owner: Arc<EngineOwner>,
    source: Arc<RegisteredRegionTable<MooncakeRegion>>,
    peer: Peer,
    source_buffers: BTreeMap<u16, MemoryBuffer>,
    destination_epoch: RegistrationEpoch,
    destination_regions: Vec<RemoteRegionDescriptor>,
    batches: HashMap<NativeBatchToken, Batch>,
    next_batch: u64,
}

impl MooncakeNativeStagePort {
    pub fn new(
        owner: Arc<EngineOwner>,
        source: Arc<RegisteredRegionTable<MooncakeRegion>>,
        peer: Peer,
        source_buffers: BTreeMap<u16, MemoryBuffer>,
        destination: AuthenticatedRemoteRegionTable,
    ) -> Result<Self, BufferError> {
        validate_source_buffers(&source, &source_buffers)?;
        Ok(Self {
            owner,
            source,
            peer,
            source_buffers,
            destination_epoch: destination.epoch,
            destination_regions: destination.regions,
            batches: HashMap::new(),
            next_batch: 1,
        })
    }

    #[cfg(test)]
    fn new_cpu_mock(
        owner: Arc<EngineOwner>,
        source: Arc<RegisteredRegionTable<MooncakeRegion>>,
        peer: Peer,
        source_buffers: BTreeMap<u16, MemoryBuffer>,
        destination: AuthenticatedRemoteRegionTable,
    ) -> Self {
        Self {
            owner,
            source,
            peer,
            source_buffers,
            destination_epoch: destination.epoch,
            destination_regions: destination.regions,
            batches: HashMap::new(),
            next_batch: 1,
        }
    }

    fn build_operations(
        &self,
        command: &NativeStageCommand,
    ) -> Result<Vec<TransferOperation>, BufferError> {
        if command.source_registration_epoch() != self.source.epoch()
            || command.destination_registration_epoch() != self.destination_epoch
        {
            return Err(BufferError::StaleRegistration);
        }
        match command.phase() {
            NativePhase::Kv => self.build_kv_operations(command),
            NativePhase::Aux => self.build_slot_operation(command, 56, AUX_SLOT_BYTES, 0),
            NativePhase::CompletionBody => {
                self.build_slot_operation(command, 57, COMPLETION_SLOT_BYTES, 0)
            }
            NativePhase::CompletionMarker => {
                self.build_slot_operation(command, 57, COMPLETION_SLOT_BYTES, 188)
            }
        }
    }

    fn build_kv_operations(
        &self,
        command: &NativeStageCommand,
    ) -> Result<Vec<TransferOperation>, BufferError> {
        if command.kv_blocks().is_empty()
            || !command.payload().is_empty()
            || command.kv_blocks().len() != command.expected_lengths().len()
        {
            return Err(BufferError::NativeTransfer);
        }
        command
            .kv_blocks()
            .iter()
            .zip(command.expected_lengths())
            .map(|(block, expected_length)| {
                if block.region_id >= KV_REGION_COUNT as u16
                    || block.byte_length != *expected_length
                {
                    return Err(BufferError::NativeTransfer);
                }
                let local_offset =
                    page_range_offset(block.source_page, block.byte_offset, block.byte_length)?;
                let remote_offset = page_range_offset(
                    block.destination_page,
                    block.byte_offset,
                    block.byte_length,
                )?;
                TransferOperation::write(
                    self.source.registered_handle(block.region_id)?,
                    local_offset,
                    &self.peer,
                    self.destination_region(block.region_id)?,
                    remote_offset,
                    block.byte_length,
                )
                .map_err(|_| BufferError::NativeTransfer)
            })
            .collect()
    }

    fn build_slot_operation(
        &self,
        command: &NativeStageCommand,
        region_id: u16,
        slot_bytes: u64,
        field_offset: u64,
    ) -> Result<Vec<TransferOperation>, BufferError> {
        if !command.kv_blocks().is_empty()
            || command.expected_lengths() != [command.payload().len() as u64]
        {
            return Err(BufferError::NativeTransfer);
        }
        let source_slot = command.source_slot().ok_or(BufferError::NativeTransfer)?;
        let destination_slot = command
            .destination_slot()
            .ok_or(BufferError::NativeTransfer)?;
        let local_offset = u64::from(source_slot)
            .checked_mul(slot_bytes)
            .and_then(|offset| offset.checked_add(field_offset))
            .ok_or(BufferError::NativeTransfer)?;
        let remote_offset = u64::from(destination_slot)
            .checked_mul(slot_bytes)
            .and_then(|offset| offset.checked_add(field_offset))
            .ok_or(BufferError::NativeTransfer)?;
        self.write_source_payload(region_id, local_offset, command.payload())?;
        let operation = TransferOperation::write(
            self.source.registered_handle(region_id)?,
            local_offset,
            &self.peer,
            self.destination_region(region_id)?,
            remote_offset,
            command.payload().len() as u64,
        )
        .map_err(|_| BufferError::NativeTransfer)?;
        Ok(vec![operation])
    }

    fn write_source_payload(
        &self,
        region_id: u16,
        offset: u64,
        payload: &[u8],
    ) -> Result<(), BufferError> {
        let offset = usize::try_from(offset).map_err(|_| BufferError::NativeTransfer)?;
        let buffer = self
            .source_buffers
            .get(&region_id)
            .ok_or(BufferError::NativeTransfer)?;
        match buffer {
            MemoryBuffer::Host(memory) => memory.write(offset, payload),
            MemoryBuffer::Pinned(memory) => memory.write(offset, payload),
            MemoryBuffer::Cuda(memory) => memory.write(offset, payload),
        }
        .map_err(|_| BufferError::NativeTransfer)
    }

    fn destination_region(&self, region_id: u16) -> Result<&RemoteRegionDescriptor, BufferError> {
        self.destination_regions
            .get(usize::from(region_id))
            .ok_or(BufferError::StaleRegistration)
    }
}

impl NativeStagePort for MooncakeNativeStagePort {
    fn submit(&mut self, command: &NativeStageCommand) -> Result<NativeBatchToken, BufferError> {
        let operations = self.build_operations(command)?;
        let batch = self
            .owner
            .submit(operations)
            .map_err(|_| BufferError::NativeTransfer)?;
        let token = NativeBatchToken::new(self.next_batch)?;
        self.next_batch = self.next_batch.saturating_add(1);
        self.batches.insert(token, batch);
        Ok(token)
    }

    fn poll(
        &mut self,
        batch: NativeBatchToken,
    ) -> Result<crate::mooncake::BatchSnapshot, BufferError> {
        self.batches
            .get(&batch)
            .ok_or(BufferError::StaleHandle)?
            .status()
            .map_err(|_| BufferError::NativeTransfer)
    }

    fn free_safe(&mut self, batch: NativeBatchToken) -> Result<(), BufferError> {
        let safe = self
            .batches
            .get(&batch)
            .ok_or(BufferError::StaleHandle)?
            .status()
            .map_err(|_| BufferError::NativeTransfer)?
            .safe_terminal;
        if !safe {
            return Err(BufferError::NativeTransfer);
        }
        self.batches
            .remove(&batch)
            .ok_or(BufferError::StaleHandle)?;
        Ok(())
    }
}

fn page_range_offset(page: u32, offset: u64, length: u64) -> Result<u64, BufferError> {
    let end = offset
        .checked_add(length)
        .ok_or(BufferError::NativeTransfer)?;
    if length == 0 || end > KV_PAGE_BYTES {
        return Err(BufferError::NativeTransfer);
    }
    u64::from(page)
        .checked_mul(KV_PAGE_BYTES)
        .and_then(|page_offset| page_offset.checked_add(offset))
        .ok_or(BufferError::NativeTransfer)
}

fn validate_source_buffers(
    source: &RegisteredRegionTable<MooncakeRegion>,
    buffers: &BTreeMap<u16, MemoryBuffer>,
) -> Result<(), BufferError> {
    if buffers.len() != 58 {
        return Err(BufferError::NativeTransfer);
    }
    for region_id in 0_u16..58 {
        let spec = source.region(region_id)?;
        let buffer = buffers.get(&region_id).ok_or(BufferError::NativeTransfer)?;
        if buffer.address() != spec.base_address || buffer.len() as u64 != spec.length_bytes {
            return Err(BufferError::NativeTransfer);
        }
        let compatible = matches!(
            (region_id, buffer),
            (0..=55, MemoryBuffer::Cuda(_)) | (56 | 57, MemoryBuffer::Pinned(_))
        );
        if !compatible {
            return Err(BufferError::NativeTransfer);
        }
    }
    Ok(())
}

fn validate_authenticated_destination(
    records: &[RegionRecord],
) -> Result<Vec<RemoteRegionDescriptor>, BufferError> {
    if records.len() != 58 {
        return Err(BufferError::StaleRegistration);
    }
    let mut regions = Vec::with_capacity(58);
    let mut kv_length = None;
    for (expected_id, record) in (0_u16..58).zip(records) {
        if record.region_id != expected_id
            || record.remote_base_addr == 0
            || !record.remote_base_addr.is_multiple_of(64)
        {
            return Err(BufferError::StaleRegistration);
        }
        let location = match (record.region_id, record.location.as_str()) {
            (0..=55, "cuda:5") => MemoryLocation::Cuda5,
            (56 | 57, "cpu:0") => MemoryLocation::Cpu0,
            (56 | 57, "cpu:1") => MemoryLocation::Cpu1,
            _ => return Err(BufferError::StaleRegistration),
        };
        match record.region_id {
            0..=55 => {
                if record.length_bytes == 0
                    || !record.length_bytes.is_multiple_of(KV_PAGE_BYTES)
                    || kv_length.is_some_and(|length| length != record.length_bytes)
                {
                    return Err(BufferError::StaleRegistration);
                }
                kv_length = Some(record.length_bytes);
            }
            56 if record.length_bytes != 32 * AUX_SLOT_BYTES => {
                return Err(BufferError::StaleRegistration);
            }
            57 if record.length_bytes != 32 * COMPLETION_SLOT_BYTES => {
                return Err(BufferError::StaleRegistration);
            }
            _ => {}
        }
        regions.push(
            RemoteRegionDescriptor::from_authenticated_record(
                record.remote_base_addr,
                record.length_bytes,
                location,
            )
            .map_err(|_| BufferError::StaleRegistration)?,
        );
    }
    Ok(regions)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::mooncake::{
        HostMemory, MockEngineFactory, MockEvent, MockPlan, OwnerConfig, PeerDescriptor,
    };
    use crate::pd::buffer::descriptor::{
        BufferDType, BufferRegionSpec, BufferTable, RegionKind, RegionLayout, RegionLocation,
        RegistrationPort, RegistrationPortError,
    };
    use crate::pd::buffer::{
        AuxRecord, AuxRecordInput, CapacityLedger, CompletionRecordInput, CompletionWrites,
        DataPlaneEffect, QuarantineManager, ReservationRequest, SourceComputeFence,
        SourceExecutionRequest, SourceExecutor, TableUseTracker, TransferPlan, TransferPlanInput,
    };
    use crate::pd::config::PdProfileV1;
    use crate::pd::protocol::{FixedBytes, RegisterRegions};
    use crate::pd::room::{
        AttemptId, ManualClock, ProcessEpoch, RegistrationEpoch, RoomId, RoomKey,
    };

    use super::*;

    struct CpuRegistrationPort {
        owner: Arc<EngineOwner>,
        buffers: BTreeMap<u16, MemoryBuffer>,
    }

    impl RegistrationPort for CpuRegistrationPort {
        type Handle = MooncakeRegion;

        fn register(
            &mut self,
            region: &BufferRegionSpec,
        ) -> Result<Self::Handle, RegistrationPortError> {
            self.owner
                .register_region(
                    self.buffers
                        .get(&region.region_id)
                        .expect("CPU buffer")
                        .clone(),
                    MemoryLocation::Cpu0,
                )
                .map_err(|_| RegistrationPortError::Register)
        }

        fn unregister(&mut self, handle: Self::Handle) -> Result<(), RegistrationPortError> {
            handle
                .close()
                .map_err(|_| RegistrationPortError::Unregister)
        }
    }

    struct Ready;

    impl SourceComputeFence for Ready {
        fn wait_ready(&mut self, _deadline_monotonic_ms: u64) -> Result<(), BufferError> {
            Ok(())
        }
    }

    #[test]
    fn mooncake_adapter_submits_four_safe_batches_without_exposing_native_ids() {
        let factory = MockEngineFactory::new(MockPlan::default());
        let events = factory.events();
        let owner = Arc::new(
            EngineOwner::start(OwnerConfig::default(), factory).expect("mock engine owner"),
        );
        let buffers = cpu_buffers();
        let mut registration = CpuRegistrationPort {
            owner: Arc::clone(&owner),
            buffers: buffers.clone(),
        };
        let source = Arc::new(
            buffer_table()
                .register(&mut registration)
                .expect("registered source table"),
        );
        let destination_epoch = RegistrationEpoch::random();
        let destination =
            AuthenticatedRemoteRegionTable::from_authenticated_register(&RegisterRegions {
                registration_epoch: FixedBytes::new(destination_epoch.as_bytes()),
                layout_fingerprint: FixedBytes::new([0x33; 32]),
                mooncake_host: "127.0.0.1".into(),
                mooncake_port: 19000,
                regions: remote_records(),
            })
            .expect("authenticated remote table");
        let peer = owner
            .open_peer(PeerDescriptor::new("127.0.0.1:19001").expect("peer descriptor"))
            .expect("mock peer");
        assert!(matches!(
            MooncakeNativeStagePort::new(
                Arc::clone(&owner),
                Arc::clone(&source),
                owner
                    .open_peer(
                        PeerDescriptor::new("127.0.0.1:19002").expect("negative peer descriptor")
                    )
                    .expect("negative peer"),
                buffers.clone(),
                destination.clone(),
            ),
            Err(BufferError::NativeTransfer)
        ));
        let mut native = MooncakeNativeStagePort::new_cpu_mock(
            Arc::clone(&owner),
            Arc::clone(&source),
            peer,
            buffers,
            destination,
        );
        let room = RoomId::new(
            RoomKey::new(ProcessEpoch::random(), 0, AttemptId::random()).expect("RoomKey"),
            1,
        )
        .expect("RoomId");
        let plan = TransferPlan::new(TransferPlanInput {
            room,
            transfer_generation: 7,
            source_registration_epoch: source.epoch(),
            destination_registration_epoch: destination_epoch,
            source_pages: vec![0],
            destination_pages: vec![0],
            source_aux_slot: 1,
            destination_aux_slot: 2,
            source_completion_slot: 1,
            destination_completion_slot: 2,
            valid_token_count: 1,
            chunk_sequence: 0,
            chunk_count: 1,
            is_last_chunk: true,
        })
        .expect("plan");
        let profile = PdProfileV1::load_embedded().expect("profile");
        let ledger = Arc::new(CapacityLedger::new(
            &profile,
            source.tracker(),
            TableUseTracker::new(),
        ));
        let handle = ledger
            .reserve(ReservationRequest {
                room,
                handle_generation: 1,
                source_pages: vec![0],
                destination_pages: vec![0],
                aux_slot: 1,
                completion_slot: 1,
                request_slot: 1,
                kv_bytes: plan.expected_kv_bytes(),
                deadline_monotonic_ms: 10_000,
            })
            .expect("leases");
        let request_digest = FixedBytes::new([0xa1; 32]);
        let aux = AuxRecord::encode(AuxRecordInput {
            first_token_valid: true,
            first_token_id: 42,
            prompt_token_count: 1,
            prefill_output_count: 1,
            request_digest,
        })
        .expect("aux");
        let completion_input = CompletionRecordInput {
            decode_process_epoch: room.key.decode_process_epoch,
            attempt_id: room.key.attempt_id,
            source_registration_epoch: plan.source_registration_epoch(),
            destination_registration_epoch: plan.destination_registration_epoch(),
            bootstrap_room: 0,
            transfer_generation: 7,
            chunk_sequence: 0,
            chunk_count: 1,
            page_count: 1,
            valid_token_count: 1,
            request_digest,
            transfer_plan_digest: plan.digest(),
        };
        let completion =
            CompletionWrites::encode(&completion_input, &aux).expect("completion writes");
        let quarantine = Arc::new(QuarantineManager::new(Arc::clone(&ledger)));
        let executor = SourceExecutor::new(
            Arc::clone(&ledger),
            quarantine,
            Arc::new(ManualClock::new(100)),
        );
        let effect = executor
            .execute(
                SourceExecutionRequest {
                    plan: &plan,
                    handle,
                    source_fence: &mut Ready,
                    aux: &aux,
                    completion: &completion,
                    deadline_monotonic_ms: 1_000,
                },
                &mut native,
            )
            .expect("native execution");
        assert!(matches!(effect, DataPlaneEffect::DataReady { .. }));

        let events = events.lock().expect("mock events");
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, MockEvent::AllocateBatch { .. }))
                .count(),
            4
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, MockEvent::SubmitBatch { .. }))
                .count(),
            4
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, MockEvent::FreeBatch { .. }))
                .count(),
            4
        );
        drop(events);
        ledger.release_source_safe(handle).expect("source release");
        ledger
            .handoff_destination(handle)
            .expect("destination handoff");
        ledger.release_terminal(handle).expect("terminal release");
        drop(native);
        let mut source = Arc::try_unwrap(source).ok().expect("unique source table");
        source
            .unregister(&mut registration)
            .expect("unregister source table");
        owner.shutdown().expect("safe mock shutdown");
    }

    fn cpu_buffers() -> BTreeMap<u16, MemoryBuffer> {
        (0_u16..58)
            .map(|region_id| {
                let length = match region_id {
                    0..=55 => KV_PAGE_BYTES as usize,
                    56 => (32 * AUX_SLOT_BYTES) as usize,
                    57 => (32 * COMPLETION_SLOT_BYTES) as usize,
                    _ => unreachable!(),
                };
                (
                    region_id,
                    MemoryBuffer::Host(HostMemory::new(length).expect("CPU mock memory")),
                )
            })
            .collect()
    }

    fn buffer_table() -> BufferTable {
        let fingerprint = FixedBytes::new([0x33; 32]);
        let regions = (0_u16..58)
            .map(|region_id| {
                let (kind, location, dtype, layout, length_bytes) = match region_id {
                    0..=27 => (
                        RegionKind::Key { layer: region_id },
                        RegionLocation::Device { device: 4 },
                        BufferDType::BFloat16,
                        RegionLayout::kv(1).expect("KV layout"),
                        KV_PAGE_BYTES,
                    ),
                    28..=55 => (
                        RegionKind::Value {
                            layer: region_id - 28,
                        },
                        RegionLocation::Device { device: 4 },
                        BufferDType::BFloat16,
                        RegionLayout::kv(1).expect("KV layout"),
                        KV_PAGE_BYTES,
                    ),
                    56 => (
                        RegionKind::Aux,
                        RegionLocation::PinnedHost { numa_node: 0 },
                        BufferDType::Bytes,
                        RegionLayout::aux(),
                        32 * AUX_SLOT_BYTES,
                    ),
                    57 => (
                        RegionKind::Completion,
                        RegionLocation::PinnedHost { numa_node: 0 },
                        BufferDType::Bytes,
                        RegionLayout::completion(),
                        32 * COMPLETION_SLOT_BYTES,
                    ),
                    _ => unreachable!(),
                };
                BufferRegionSpec {
                    region_id,
                    kind,
                    base_address: 0x1000_0000 + u64::from(region_id) * 0x0100_0000,
                    length_bytes,
                    location,
                    dtype,
                    layout,
                    owner_generation: 1,
                    layout_fingerprint: fingerprint,
                }
            })
            .collect();
        BufferTable::new(regions, 1, 4, fingerprint).expect("source table")
    }

    fn remote_records() -> Vec<RegionRecord> {
        (0_u16..58)
            .map(|region_id| {
                let (length_bytes, location) = match region_id {
                    0..=55 => (KV_PAGE_BYTES, "cuda:5"),
                    56 => (32 * AUX_SLOT_BYTES, "cpu:0"),
                    57 => (32 * COMPLETION_SLOT_BYTES, "cpu:0"),
                    _ => unreachable!(),
                };
                RegionRecord {
                    region_id,
                    remote_base_addr: 0x8000_0000 + u64::from(region_id) * 0x0100_0000,
                    length_bytes,
                    location: location.into(),
                }
            })
            .collect()
    }
}
