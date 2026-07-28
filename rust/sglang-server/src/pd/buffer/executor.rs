use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::mooncake::BatchSnapshot;
use crate::pd::buffer::{
    AUX_BYTES, BufferError, CapacityLedger, CompletionRecordInput, CompletionWrites,
    DestinationVisibilityFence, GpuDirectFlushPort, LeaseHandle, NativeBatchToken, NativeSafety,
    QuarantineManager, SourceComputeFence, TransferPlan, TransferPlanDigest, TransferStage,
    TransitionResult, evaluate_native_fence, validate_completion,
};
use crate::pd::protocol::KvBlock;
use crate::pd::room::{Clock, PdReason, RoomId};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataPlaneIdentity {
    pub room: RoomId,
    pub transfer_generation: u64,
    pub transfer_plan_digest: TransferPlanDigest,
}

impl DataPlaneIdentity {
    pub fn from_plan(plan: &TransferPlan) -> Self {
        Self {
            room: plan.room(),
            transfer_generation: plan.transfer_generation(),
            transfer_plan_digest: plan.digest(),
        }
    }

    fn matches_plan(self, plan: &TransferPlan) -> bool {
        self == Self::from_plan(plan)
    }
}

impl fmt::Debug for DataPlaneIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DataPlaneIdentity")
            .field("transfer_generation", &self.transfer_generation)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativePhase {
    Kv,
    Aux,
    CompletionBody,
    CompletionMarker,
}

#[derive(Clone)]
pub struct NativeStageCommand {
    identity: DataPlaneIdentity,
    source_registration_epoch: crate::pd::room::RegistrationEpoch,
    destination_registration_epoch: crate::pd::room::RegistrationEpoch,
    phase: NativePhase,
    expected_lengths: Vec<u64>,
    kv_blocks: Vec<KvBlock>,
    source_slot: Option<u16>,
    destination_slot: Option<u16>,
    payload: Vec<u8>,
}

impl fmt::Debug for NativeStageCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeStageCommand")
            .field("identity", &self.identity)
            .field("phase", &self.phase)
            .field("operation_count", &self.expected_lengths.len())
            .finish_non_exhaustive()
    }
}

impl NativeStageCommand {
    pub const fn identity(&self) -> DataPlaneIdentity {
        self.identity
    }

    pub const fn phase(&self) -> NativePhase {
        self.phase
    }

    pub const fn source_registration_epoch(&self) -> crate::pd::room::RegistrationEpoch {
        self.source_registration_epoch
    }

    pub const fn destination_registration_epoch(&self) -> crate::pd::room::RegistrationEpoch {
        self.destination_registration_epoch
    }

    pub fn expected_lengths(&self) -> &[u64] {
        &self.expected_lengths
    }

    pub fn kv_blocks(&self) -> &[KvBlock] {
        &self.kv_blocks
    }

    pub const fn source_slot(&self) -> Option<u16> {
        self.source_slot
    }

    pub const fn destination_slot(&self) -> Option<u16> {
        self.destination_slot
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

pub trait NativeStagePort: Send {
    fn submit(&mut self, command: &NativeStageCommand) -> Result<NativeBatchToken, BufferError>;

    fn poll(&mut self, batch: NativeBatchToken) -> Result<BatchSnapshot, BufferError>;

    fn free_safe(&mut self, batch: NativeBatchToken) -> Result<(), BufferError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataPlaneEffect {
    DataReady {
        identity: DataPlaneIdentity,
    },
    TransferComplete {
        identity: DataPlaneIdentity,
    },
    TransferFailed {
        identity: DataPlaneIdentity,
        reason: PdReason,
    },
    Quarantined {
        identity: DataPlaneIdentity,
        batch: NativeBatchToken,
        reason: PdReason,
    },
}

pub struct SourceExecutionRequest<'a, F>
where
    F: SourceComputeFence + ?Sized,
{
    pub plan: &'a TransferPlan,
    pub handle: LeaseHandle,
    pub source_fence: &'a mut F,
    pub aux: &'a [u8; AUX_BYTES],
    pub completion: &'a CompletionWrites,
    pub deadline_monotonic_ms: u64,
}

pub struct SourceExecutor {
    ledger: Arc<CapacityLedger>,
    quarantine: Arc<QuarantineManager>,
    clock: Arc<dyn Clock>,
}

impl SourceExecutor {
    pub fn new<C>(
        ledger: Arc<CapacityLedger>,
        quarantine: Arc<QuarantineManager>,
        clock: Arc<C>,
    ) -> Self
    where
        C: Clock + 'static,
    {
        Self {
            ledger,
            quarantine,
            clock,
        }
    }

    pub fn execute<P, F>(
        &self,
        request: SourceExecutionRequest<'_, F>,
        port: &mut P,
    ) -> Result<DataPlaneEffect, BufferError>
    where
        P: NativeStagePort,
        F: SourceComputeFence + ?Sized,
    {
        let SourceExecutionRequest {
            plan,
            handle,
            source_fence,
            aux,
            completion,
            deadline_monotonic_ms,
        } = request;
        let identity = DataPlaneIdentity::from_plan(plan);
        if handle.room() != plan.room() {
            return Err(BufferError::StaleHandle);
        }
        if deadline_monotonic_ms <= self.clock.now_monotonic_ms() {
            self.ledger.abort_pre_submit(handle)?;
            return Ok(DataPlaneEffect::TransferFailed {
                identity,
                reason: PdReason::TransferTimeout,
            });
        }
        if source_fence.wait_ready(deadline_monotonic_ms).is_err() {
            self.ledger.abort_pre_submit(handle)?;
            return Ok(DataPlaneEffect::TransferFailed {
                identity,
                reason: PdReason::TransferFailed,
            });
        }
        if deadline_monotonic_ms <= self.clock.now_monotonic_ms() {
            self.ledger.abort_pre_submit(handle)?;
            return Ok(DataPlaneEffect::TransferFailed {
                identity,
                reason: PdReason::TransferTimeout,
            });
        }

        let kv = NativeStageCommand {
            identity,
            source_registration_epoch: plan.source_registration_epoch(),
            destination_registration_epoch: plan.destination_registration_epoch(),
            phase: NativePhase::Kv,
            expected_lengths: plan
                .kv_blocks()
                .iter()
                .map(|block| block.byte_length)
                .collect(),
            kv_blocks: plan.kv_blocks().to_vec(),
            source_slot: None,
            destination_slot: None,
            payload: Vec::new(),
        };
        if let Some(effect) = self.execute_lease_stage(
            handle,
            TransferStage::Kv,
            &[kv],
            port,
            deadline_monotonic_ms,
        )? {
            return Ok(effect);
        }

        let aux = NativeStageCommand {
            identity,
            source_registration_epoch: plan.source_registration_epoch(),
            destination_registration_epoch: plan.destination_registration_epoch(),
            phase: NativePhase::Aux,
            expected_lengths: vec![AUX_BYTES as u64],
            kv_blocks: Vec::new(),
            source_slot: Some(plan.source_aux_slot()),
            destination_slot: Some(plan.destination_aux_slot()),
            payload: aux.to_vec(),
        };
        if let Some(effect) = self.execute_lease_stage(
            handle,
            TransferStage::Aux,
            &[aux],
            port,
            deadline_monotonic_ms,
        )? {
            return Ok(effect);
        }

        let completion_body = NativeStageCommand {
            identity,
            source_registration_epoch: plan.source_registration_epoch(),
            destination_registration_epoch: plan.destination_registration_epoch(),
            phase: NativePhase::CompletionBody,
            expected_lengths: vec![completion.body_and_crc().len() as u64],
            kv_blocks: Vec::new(),
            source_slot: Some(plan.source_completion_slot()),
            destination_slot: Some(plan.destination_completion_slot()),
            payload: completion.body_and_crc().to_vec(),
        };
        let completion_marker = NativeStageCommand {
            identity,
            source_registration_epoch: plan.source_registration_epoch(),
            destination_registration_epoch: plan.destination_registration_epoch(),
            phase: NativePhase::CompletionMarker,
            expected_lengths: vec![completion.commit_marker().len() as u64],
            kv_blocks: Vec::new(),
            source_slot: Some(plan.source_completion_slot()),
            destination_slot: Some(plan.destination_completion_slot()),
            payload: completion.commit_marker().to_vec(),
        };
        if let Some(effect) = self.execute_lease_stage(
            handle,
            TransferStage::Completion,
            &[completion_body, completion_marker],
            port,
            deadline_monotonic_ms,
        )? {
            return Ok(effect);
        }

        Ok(DataPlaneEffect::DataReady { identity })
    }

    fn execute_lease_stage<P>(
        &self,
        handle: LeaseHandle,
        stage: TransferStage,
        commands: &[NativeStageCommand],
        port: &mut P,
        deadline_monotonic_ms: u64,
    ) -> Result<Option<DataPlaneEffect>, BufferError>
    where
        P: NativeStagePort,
    {
        self.ledger.begin_stage(handle, stage)?;
        for command in commands {
            match self.execute_batch(command, port, deadline_monotonic_ms)? {
                BatchOutcome::Success => {}
                BatchOutcome::SafeFailure | BatchOutcome::NotSubmitted => {
                    self.ledger.release_failed_safe(handle)?;
                    return Ok(Some(DataPlaneEffect::TransferFailed {
                        identity: command.identity,
                        reason: PdReason::TransferFailed,
                    }));
                }
                BatchOutcome::Quarantined(batch) => {
                    self.quarantine.insert(
                        handle,
                        batch,
                        self.clock.now_monotonic_ms(),
                        PdReason::TransferTimeout,
                    )?;
                    return Ok(Some(DataPlaneEffect::Quarantined {
                        identity: command.identity,
                        batch,
                        reason: PdReason::TransferTimeout,
                    }));
                }
            }
        }
        self.ledger.finish_stage(handle, stage)?;
        Ok(None)
    }

    fn execute_batch<P>(
        &self,
        command: &NativeStageCommand,
        port: &mut P,
        deadline_monotonic_ms: u64,
    ) -> Result<BatchOutcome, BufferError>
    where
        P: NativeStagePort,
    {
        let batch = match port.submit(command) {
            Ok(batch) => batch,
            Err(_) => return Ok(BatchOutcome::NotSubmitted),
        };
        loop {
            let snapshot = match port.poll(batch) {
                Ok(snapshot) => snapshot,
                Err(_) => return Ok(BatchOutcome::Quarantined(batch)),
            };
            match evaluate_native_fence(&snapshot, command.expected_lengths()) {
                NativeSafety::SafeSuccess => {
                    if port.free_safe(batch).is_err() {
                        return Ok(BatchOutcome::SafeFailure);
                    }
                    return Ok(BatchOutcome::Success);
                }
                NativeSafety::SafeFailure => {
                    let _ = port.free_safe(batch);
                    return Ok(BatchOutcome::SafeFailure);
                }
                NativeSafety::Pending => {
                    if self.clock.now_monotonic_ms() >= deadline_monotonic_ms {
                        return Ok(BatchOutcome::Quarantined(batch));
                    }
                }
            }
        }
    }
}

enum BatchOutcome {
    Success,
    SafeFailure,
    Quarantined(NativeBatchToken),
    NotSubmitted,
}

pub trait DestinationRecordPort: Send {
    fn read_completion(
        &mut self,
        slot: u16,
    ) -> Result<[u8; crate::pd::buffer::COMPLETION_BYTES], BufferError>;

    fn read_aux(&mut self, slot: u16) -> Result<[u8; AUX_BYTES], BufferError>;

    fn clear_final_kv_page_tail(
        &mut self,
        region_id: u16,
        page: u32,
        valid_token_count: u32,
    ) -> Result<(), BufferError>;
}

pub struct DestinationExecutor {
    ledger: Arc<CapacityLedger>,
    validated: Mutex<HashSet<DataPlaneIdentity>>,
    committed: Mutex<HashSet<DataPlaneIdentity>>,
}

impl DestinationExecutor {
    pub fn new(ledger: Arc<CapacityLedger>) -> Self {
        Self {
            ledger,
            validated: Mutex::new(HashSet::new()),
            committed: Mutex::new(HashSet::new()),
        }
    }

    pub fn validate_ready<P, V>(
        &self,
        plan: &TransferPlan,
        handle: LeaseHandle,
        identity: DataPlaneIdentity,
        visibility: &mut DestinationVisibilityFence<V>,
        records: &mut P,
        expected: &CompletionRecordInput,
    ) -> Result<DataPlaneEffect, BufferError>
    where
        P: DestinationRecordPort + ?Sized,
        V: GpuDirectFlushPort,
    {
        if handle.room() != plan.room()
            || !identity.matches_plan(plan)
            || expected.transfer_generation != plan.transfer_generation()
            || expected.transfer_plan_digest != plan.digest()
        {
            return Err(BufferError::StaleHandle);
        }
        if self
            .validated
            .lock()
            .map_err(|_| BufferError::InvalidTransition)?
            .contains(&identity)
        {
            return Ok(DataPlaneEffect::TransferComplete { identity });
        }

        if let Err(error) = self.validate_visible_records(plan, visibility, records, expected) {
            let _ = self.ledger.release_failed_safe(handle);
            return Err(error);
        }
        self.validated
            .lock()
            .map_err(|_| BufferError::InvalidTransition)?
            .insert(identity);
        Ok(DataPlaneEffect::TransferComplete { identity })
    }

    pub fn commit_after_ack(
        &self,
        handle: LeaseHandle,
        identity: DataPlaneIdentity,
    ) -> Result<TransitionResult, BufferError> {
        if handle.room() != identity.room {
            return Err(BufferError::StaleHandle);
        }
        if !self
            .validated
            .lock()
            .map_err(|_| BufferError::InvalidTransition)?
            .contains(&identity)
        {
            return Err(BufferError::InvalidTransition);
        }
        let mut committed = self
            .committed
            .lock()
            .map_err(|_| BufferError::InvalidTransition)?;
        if !committed.insert(identity) {
            return Ok(TransitionResult::AlreadyApplied);
        }
        match self.ledger.handoff_destination(handle) {
            Ok(TransitionResult::Applied) => Ok(TransitionResult::Applied),
            Ok(TransitionResult::AlreadyApplied) | Err(_) => {
                committed.remove(&identity);
                Err(BufferError::InvalidTransition)
            }
        }
    }

    fn validate_visible_records<P, V>(
        &self,
        plan: &TransferPlan,
        visibility: &mut DestinationVisibilityFence<V>,
        records: &mut P,
        expected: &CompletionRecordInput,
    ) -> Result<(), BufferError>
    where
        P: DestinationRecordPort + ?Sized,
        V: GpuDirectFlushPort,
    {
        visibility.flush()?;
        let completion = records.read_completion(plan.destination_completion_slot())?;
        let aux = records.read_aux(plan.destination_aux_slot())?;
        validate_completion(&completion, &aux, expected)?;
        let final_page = *plan
            .destination_pages()
            .last()
            .ok_or(BufferError::PlanMismatch {
                field: "destination_pages",
            })?;
        for region_id in 0_u16..56 {
            records.clear_final_kv_page_tail(region_id, final_page, plan.valid_token_count())?;
        }
        Ok(())
    }
}
