//! Typed ownership boundary for the frozen PD v1 buffer data plane.
#![deny(unsafe_code)]

mod data;
mod descriptor;
mod executor;
mod fence;
mod integration;
mod lease;
mod native;
mod plan;
mod quarantine;
mod worker;

use thiserror::Error;

pub use data::{
    AUX_BYTES, AuxRecord, AuxRecordInput, COMPLETION_BYTES, CompletionRecordInput,
    CompletionWrites, ValidatedCompletion, clear_partial_page_tail, crc32c, validate_completion,
};
pub use descriptor::{
    BufferDType, BufferRegionSpec, BufferTable, MooncakeRegistrationPort, RegionKind, RegionLayout,
    RegionLocation, RegisteredRegionTable, RegistrationPort, RegistrationPortError, TableUseGuard,
    TableUseSnapshot, TableUseTracker,
};
pub use executor::{
    DataPlaneEffect, DataPlaneIdentity, DestinationExecutor, DestinationRecordPort, NativePhase,
    NativeStageCommand, NativeStagePort, SourceExecutionRequest, SourceExecutor,
};
pub use fence::{
    CudaEventQuery, CudaEventRuntime, CudaEventRuntimePort, CudaEventSourceFence,
    CudaHostFlushPort, DestinationVisibilityFence, GpuDirectFlushPort, NativeSafety,
    SourceComputeFence, evaluate_native_fence,
};
pub use integration::{
    apply_decode_ack, apply_decode_data_effect, apply_prefill_data_effect, apply_prepare_accepted,
};
pub use lease::{
    CapacityLedger, LeaseHandle, LeaseSnapshot, ReservationRequest, TransferStage, TransitionResult,
};
pub use native::{AuthenticatedRemoteRegionTable, MooncakeNativeStagePort};
pub use plan::{TransferPlan, TransferPlanDigest, TransferPlanInput};
pub use quarantine::{
    NativeBatchToken, QUARANTINE_HARD_DEADLINE_MS, QuarantineManager, QuarantineSnapshot,
    QuarantineUpdate,
};
pub use worker::{
    DataPlaneWorker, DataPlaneWorkerState, DataPlaneWorkerTicket, DestinationWorkRequest,
    NativeObservationTicket, SourceWorkRequest,
};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BufferError {
    #[error("invalid PD buffer descriptor field {field}: {detail}")]
    InvalidDescriptor {
        field: &'static str,
        detail: &'static str,
    },
    #[error("PD region {region_id} registration failed after {rollback_failures} rollback errors")]
    Registration {
        region_id: u16,
        rollback_failures: usize,
    },
    #[error("PD region unregistration completed with {failures} errors")]
    Unregistration { failures: usize },
    #[error(
        "PD registered table is in use by {active} active and {quarantined} quarantined leases"
    )]
    TableInUse { active: usize, quarantined: usize },
    #[error("PD registration epoch is stale")]
    StaleRegistration,
    #[error("PD transfer plan exceeds frozen {field} limit")]
    PlanLimit { field: &'static str },
    #[error("PD transfer plan field {field} does not match the frozen contract")]
    PlanMismatch { field: &'static str },
    #[error("PD capacity is exhausted for {resource}")]
    CapacityExhausted { resource: &'static str },
    #[error("PD resource is already leased by another owner")]
    ResourceOwned,
    #[error("PD lease handle is stale or belongs to another owner")]
    StaleHandle,
    #[error("PD lease transition is invalid for the current stage")]
    InvalidTransition,
    #[error("PD data record failed the frozen {check} check")]
    DataRecord { check: &'static str },
    #[error("PD source compute fence did not become ready")]
    SourceFence,
    #[error("PD destination visibility fence failed closed")]
    VisibilityFence,
    #[error("PD native transfer did not reach the required safe result")]
    NativeTransfer,
    #[error("PD data-plane deadline expired")]
    Deadline,
    #[error("PD bounded data-plane worker queue is full")]
    WorkerFull,
}
