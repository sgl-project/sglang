use crate::mooncake::types::TransferOpcode;
use crate::mooncake::{
    BatchId, EngineError, OperationProgress, PeerDescriptor, PeerId, RegionDescriptor, RegionId,
};

#[derive(Debug, Clone)]
pub struct EngineOperation {
    opcode: TransferOpcode,
    region_id: RegionId,
    peer_id: PeerId,
    local_address: u64,
    remote_address: u64,
    length: u64,
}

impl EngineOperation {
    pub(crate) fn new(
        opcode: TransferOpcode,
        region_id: RegionId,
        peer_id: PeerId,
        local_address: u64,
        remote_address: u64,
        length: u64,
    ) -> Self {
        Self {
            opcode,
            region_id,
            peer_id,
            local_address,
            remote_address,
            length,
        }
    }

    pub fn region_id(&self) -> RegionId {
        self.region_id
    }

    pub fn peer_id(&self) -> PeerId {
        self.peer_id
    }

    pub fn length(&self) -> u64 {
        self.length
    }

    pub(crate) fn opcode(&self) -> TransferOpcode {
        self.opcode
    }

    pub(crate) fn local_address(&self) -> u64 {
        self.local_address
    }

    pub(crate) fn remote_address(&self) -> u64 {
        self.remote_address
    }
}

/// Safe, logical-handle contract shared by the CPU mock and native adapter.
///
/// Implementations keep any native pointer or handle internal. A factory creates
/// the implementation on the owner worker, and the object never leaves that
/// thread.
pub trait TransferEngine {
    fn local_peer_descriptor(&mut self) -> Result<PeerDescriptor, EngineError>;

    fn register_region(
        &mut self,
        id: RegionId,
        descriptor: &RegionDescriptor,
    ) -> Result<(), EngineError>;

    fn unregister_region(&mut self, id: RegionId) -> Result<(), EngineError>;

    fn open_peer(&mut self, id: PeerId, descriptor: &PeerDescriptor) -> Result<(), EngineError>;

    fn close_peer(&mut self, id: PeerId) -> Result<(), EngineError>;

    fn allocate_batch(&mut self, id: BatchId, operation_count: usize) -> Result<(), EngineError>;

    fn submit_batch(
        &mut self,
        id: BatchId,
        operations: &[EngineOperation],
    ) -> Result<(), EngineError>;

    fn poll(
        &mut self,
        id: BatchId,
        operation_index: usize,
    ) -> Result<OperationProgress, EngineError>;

    fn free_batch(&mut self, id: BatchId) -> Result<(), EngineError>;

    fn shutdown(&mut self) -> Result<(), EngineError>;
}

pub trait EngineFactory: Send + 'static {
    fn create(&self) -> Result<Box<dyn TransferEngine>, EngineError>;
}
