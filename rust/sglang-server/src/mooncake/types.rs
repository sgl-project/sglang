use std::fmt;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::mooncake::EngineError;
use crate::mooncake::native::ffi::NativeMemory;

macro_rules! logical_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u64);

        impl $name {
            pub(crate) fn new(value: u64) -> Self {
                Self(value)
            }

            pub fn get(self) -> u64 {
                self.0
            }
        }
    };
}

logical_id!(RegionId);
logical_id!(PeerId);
logical_id!(BatchId);

#[derive(Clone)]
pub struct HostMemory {
    inner: Arc<Mutex<Box<[u8]>>>,
}

impl fmt::Debug for HostMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostMemory")
            .field("length", &self.len())
            .finish_non_exhaustive()
    }
}

impl HostMemory {
    pub fn new(length: usize) -> Result<Self, EngineError> {
        if length == 0 {
            return Err(EngineError::InvalidDescriptor {
                field: "memory.length",
                detail: "must be non-zero".into(),
            });
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(vec![0; length].into_boxed_slice())),
        })
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, EngineError> {
        if bytes.is_empty() {
            return Err(EngineError::InvalidDescriptor {
                field: "memory.length",
                detail: "must be non-zero".into(),
            });
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(bytes.into_boxed_slice())),
        })
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|memory| memory.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), EngineError> {
        let mut memory = self.inner.lock().map_err(|_| EngineError::LockPoisoned)?;
        let end = offset
            .checked_add(bytes.len())
            .ok_or(EngineError::RangeOverflow {
                field: "host_memory",
            })?;
        if end > memory.len() {
            return Err(EngineError::RangeOutOfBounds {
                field: "host_memory",
                offset: offset as u64,
                end: end as u64,
                registered_length: memory.len() as u64,
            });
        }
        memory[offset..end].copy_from_slice(bytes);
        Ok(())
    }

    pub fn read(&self, offset: usize, length: usize) -> Result<Vec<u8>, EngineError> {
        let memory = self.inner.lock().map_err(|_| EngineError::LockPoisoned)?;
        let end = offset
            .checked_add(length)
            .ok_or(EngineError::RangeOverflow {
                field: "host_memory",
            })?;
        if end > memory.len() {
            return Err(EngineError::RangeOutOfBounds {
                field: "host_memory",
                offset: offset as u64,
                end: end as u64,
                registered_length: memory.len() as u64,
            });
        }
        Ok(memory[offset..end].to_vec())
    }

    pub fn fill(&self, value: u8) -> Result<(), EngineError> {
        self.inner
            .lock()
            .map_err(|_| EngineError::LockPoisoned)?
            .fill(value);
        Ok(())
    }

    pub(crate) fn address(&self) -> u64 {
        self.inner
            .lock()
            .map(|memory| memory.as_ptr() as usize as u64)
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct PinnedMemory {
    inner: NativeMemory,
}

impl PinnedMemory {
    pub fn new(length: usize) -> Result<Self, EngineError> {
        Ok(Self {
            inner: NativeMemory::pinned(length)?,
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, EngineError> {
        let memory = Self::new(bytes.len())?;
        memory.write(0, bytes)?;
        Ok(memory)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), EngineError> {
        self.inner.write(offset, bytes)
    }

    pub fn read(&self, offset: usize, length: usize) -> Result<Vec<u8>, EngineError> {
        self.inner.read(offset, length)
    }

    pub fn fill(&self, value: u8) -> Result<(), EngineError> {
        self.inner.fill(value)
    }

    pub(crate) fn address(&self) -> u64 {
        self.inner.address()
    }
}

#[derive(Debug, Clone)]
pub struct CudaMemory {
    inner: NativeMemory,
}

impl CudaMemory {
    pub fn new(device: u32, length: usize) -> Result<Self, EngineError> {
        if !matches!(device, 4 | 5) {
            return Err(EngineError::UnsupportedGpu { device });
        }
        Ok(Self {
            inner: NativeMemory::cuda(device, length)?,
        })
    }

    pub fn from_bytes(device: u32, bytes: &[u8]) -> Result<Self, EngineError> {
        let memory = Self::new(device, bytes.len())?;
        memory.write(0, bytes)?;
        Ok(memory)
    }

    pub fn device(&self) -> u32 {
        self.inner
            .cuda_device()
            .expect("CUDA allocation has a device")
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), EngineError> {
        self.inner.write(offset, bytes)
    }

    pub fn read(&self, offset: usize, length: usize) -> Result<Vec<u8>, EngineError> {
        self.inner.read(offset, length)
    }

    pub fn fill(&self, value: u8) -> Result<(), EngineError> {
        self.inner.fill(value)
    }

    pub(crate) fn address(&self) -> u64 {
        self.inner.address()
    }
}

#[derive(Debug, Clone)]
pub enum MemoryBuffer {
    Host(HostMemory),
    Pinned(PinnedMemory),
    Cuda(CudaMemory),
}

impl MemoryBuffer {
    pub fn len(&self) -> usize {
        match self {
            Self::Host(memory) => memory.len(),
            Self::Pinned(memory) => memory.len(),
            Self::Cuda(memory) => memory.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn address(&self) -> u64 {
        match self {
            Self::Host(memory) => memory.address(),
            Self::Pinned(memory) => memory.address(),
            Self::Cuda(memory) => memory.address(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLocation {
    Cpu0,
    Cpu1,
    Cuda4,
    Cuda5,
}

impl MemoryLocation {
    pub fn as_native_str(self) -> &'static str {
        match self {
            Self::Cpu0 => "cpu:0",
            Self::Cpu1 => "cpu:1",
            Self::Cuda4 => "cuda:4",
            Self::Cuda5 => "cuda:5",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegionDescriptor {
    pub(crate) buffer: MemoryBuffer,
    pub(crate) location: MemoryLocation,
}

impl RegionDescriptor {
    pub fn new(buffer: MemoryBuffer, location: MemoryLocation) -> Result<Self, EngineError> {
        if buffer.is_empty() || buffer.address() == 0 {
            return Err(EngineError::InvalidDescriptor {
                field: "memory",
                detail: "buffer must have a stable non-null address and non-zero length".into(),
            });
        }
        let location_matches = match (&buffer, location) {
            (
                MemoryBuffer::Host(_) | MemoryBuffer::Pinned(_),
                MemoryLocation::Cpu0 | MemoryLocation::Cpu1,
            ) => true,
            (MemoryBuffer::Cuda(memory), MemoryLocation::Cuda4) => memory.device() == 4,
            (MemoryBuffer::Cuda(memory), MemoryLocation::Cuda5) => memory.device() == 5,
            _ => false,
        };
        if !location_matches {
            return Err(EngineError::InvalidDescriptor {
                field: "memory.location",
                detail: "host/pinned memory requires cpu:0 or cpu:1; CUDA memory must match cuda:4 or cuda:5".into(),
            });
        }
        let length = buffer.len() as u64;
        buffer
            .address()
            .checked_add(length)
            .ok_or(EngineError::RangeOverflow {
                field: "memory.address",
            })?;
        Ok(Self { buffer, location })
    }

    pub fn length(&self) -> u64 {
        self.buffer.len() as u64
    }

    pub(crate) fn address(&self) -> u64 {
        self.buffer.address()
    }

    pub fn location(&self) -> MemoryLocation {
        self.location
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteRegionDescriptor {
    base_address: u64,
    length: u64,
    location: MemoryLocation,
}

impl RemoteRegionDescriptor {
    pub(crate) fn from_local(descriptor: &RegionDescriptor) -> Self {
        Self {
            base_address: descriptor.address(),
            length: descriptor.length(),
            location: descriptor.location,
        }
    }

    pub fn base_address(&self) -> u64 {
        self.base_address
    }

    pub fn length(&self) -> u64 {
        self.length
    }

    pub fn location(&self) -> MemoryLocation {
        self.location
    }

    pub(crate) fn from_authenticated_record(
        base_address: u64,
        length: u64,
        location: MemoryLocation,
    ) -> Result<Self, EngineError> {
        if base_address == 0 || length == 0 {
            return Err(EngineError::InvalidDescriptor {
                field: "remote_region",
                detail: "authenticated address and length must be non-zero".into(),
            });
        }
        base_address
            .checked_add(length)
            .ok_or(EngineError::RangeOverflow {
                field: "remote_region",
            })?;
        Ok(Self {
            base_address,
            length,
            location,
        })
    }

    fn checked_address(&self, offset: u64, length: u64) -> Result<u64, EngineError> {
        let end = offset
            .checked_add(length)
            .ok_or(EngineError::RangeOverflow {
                field: "remote_region",
            })?;
        if end > self.length {
            return Err(EngineError::RangeOutOfBounds {
                field: "remote_region",
                offset,
                end,
                registered_length: self.length,
            });
        }
        self.base_address
            .checked_add(offset)
            .ok_or(EngineError::RangeOverflow {
                field: "remote_address",
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerDescriptor {
    endpoint: SocketAddr,
}

impl PeerDescriptor {
    pub fn new(endpoint: &str) -> Result<Self, EngineError> {
        let endpoint = endpoint
            .parse()
            .map_err(|error| EngineError::InvalidDescriptor {
                field: "peer.endpoint",
                detail: format!("{error}"),
            })?;
        Ok(Self { endpoint })
    }

    pub fn endpoint(&self) -> SocketAddr {
        self.endpoint
    }

    pub(crate) fn segment_name(&self) -> String {
        self.endpoint.to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationState {
    Waiting,
    Pending,
    Invalid,
    Canceled,
    Completed,
    Timeout,
    Failed,
    Unknown(i32),
}

impl OperationState {
    pub fn from_raw(raw: i32) -> Self {
        match raw {
            0 => Self::Waiting,
            1 => Self::Pending,
            2 => Self::Invalid,
            3 => Self::Canceled,
            4 => Self::Completed,
            5 => Self::Timeout,
            6 => Self::Failed,
            value => Self::Unknown(value),
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Invalid | Self::Canceled | Self::Completed | Self::Timeout | Self::Failed
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationProgress {
    pub state: OperationState,
    pub transferred_bytes: u64,
}

impl Default for OperationProgress {
    fn default() -> Self {
        Self {
            state: OperationState::Waiting,
            transferred_bytes: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchSnapshot {
    pub operations: Vec<OperationProgress>,
    pub logical_aborted: bool,
    pub safe_terminal: bool,
}

#[derive(Debug, Clone)]
pub struct TransferOperation {
    pub(crate) owner_id: u64,
    pub(crate) opcode: TransferOpcode,
    pub(crate) region_id: RegionId,
    pub(crate) peer_id: PeerId,
    pub(crate) local_offset: u64,
    pub(crate) remote_address: u64,
    pub(crate) length: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum TransferOpcode {
    Read,
    Write,
}

impl TransferOperation {
    pub fn write(
        region: &crate::mooncake::Region,
        local_offset: u64,
        peer: &crate::mooncake::Peer,
        remote: &RemoteRegionDescriptor,
        remote_offset: u64,
        length: u64,
    ) -> Result<Self, EngineError> {
        Self::new(
            TransferOpcode::Write,
            region,
            local_offset,
            peer,
            remote,
            remote_offset,
            length,
        )
    }

    pub fn read(
        region: &crate::mooncake::Region,
        local_offset: u64,
        peer: &crate::mooncake::Peer,
        remote: &RemoteRegionDescriptor,
        remote_offset: u64,
        length: u64,
    ) -> Result<Self, EngineError> {
        Self::new(
            TransferOpcode::Read,
            region,
            local_offset,
            peer,
            remote,
            remote_offset,
            length,
        )
    }

    fn new(
        opcode: TransferOpcode,
        region: &crate::mooncake::Region,
        local_offset: u64,
        peer: &crate::mooncake::Peer,
        remote: &RemoteRegionDescriptor,
        remote_offset: u64,
        length: u64,
    ) -> Result<Self, EngineError> {
        if region.owner_id() != peer.owner_id() {
            return Err(EngineError::InvalidDescriptor {
                field: "operation.owner",
                detail: "region and peer must belong to the same engine owner".into(),
            });
        }
        if length == 0 {
            return Err(EngineError::InvalidDescriptor {
                field: "operation.length",
                detail: "must be non-zero".into(),
            });
        }
        let local_end = local_offset
            .checked_add(length)
            .ok_or(EngineError::RangeOverflow {
                field: "local_region",
            })?;
        if local_end > region.length() {
            return Err(EngineError::RangeOutOfBounds {
                field: "local_region",
                offset: local_offset,
                end: local_end,
                registered_length: region.length(),
            });
        }
        let remote_address = remote.checked_address(remote_offset, length)?;
        Ok(Self {
            owner_id: region.owner_id(),
            opcode,
            region_id: region.id(),
            peer_id: peer.id(),
            local_offset,
            remote_address,
            length,
        })
    }

    pub(crate) fn owner_id(&self) -> u64 {
        self.owner_id
    }
}
