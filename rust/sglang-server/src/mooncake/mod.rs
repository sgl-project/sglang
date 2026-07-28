//! Safe Rust boundary for the frozen Mooncake transfer-engine C ABI.
#![deny(unsafe_code)]

mod engine;
mod error;
mod mock;
mod native;
mod owner;
mod profile;
mod types;
mod worker;

pub use engine::{EngineFactory, EngineOperation, TransferEngine};
pub use error::{EngineError, NativeCode, NativeOperation};
pub use mock::{MockEngineFactory, MockEvent, MockFailurePoint, MockPlan};
pub use native::{NativeEngineConfig, NativeEngineFactory};
pub use owner::{Batch, EngineOwner, OwnerConfig, Peer, Region, ShutdownOutcome};
pub use profile::PdNicProfile;
pub use types::{
    BatchId, BatchSnapshot, CudaMemory, HostMemory, MemoryBuffer, MemoryLocation,
    OperationProgress, OperationState, PeerDescriptor, PeerId, PinnedMemory, RegionDescriptor,
    RegionId, RemoteRegionDescriptor, TransferOperation,
};

#[cfg(test)]
pub(crate) use native::{
    load_library_for_test, validate_and_load_artifact_for_test, validate_artifact_for_test,
};

#[cfg(test)]
mod tests;
