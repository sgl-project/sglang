use crate::mooncake::{BatchSnapshot, OperationState};
use crate::pd::buffer::BufferError;

#[path = "fence_native.rs"]
mod native;

pub use native::CudaHostFlushPort;

pub trait SourceComputeFence: Send {
    fn wait_ready(&mut self, deadline_monotonic_ms: u64) -> Result<(), BufferError>;
}

pub trait GpuDirectFlushPort: Send {
    fn supports_flush_to_owner(&self, device: u32) -> bool;
    fn flush_to_owner(&mut self, device: u32) -> Result<(), BufferError>;
}

impl<T> GpuDirectFlushPort for Box<T>
where
    T: GpuDirectFlushPort + ?Sized,
{
    fn supports_flush_to_owner(&self, device: u32) -> bool {
        (**self).supports_flush_to_owner(device)
    }

    fn flush_to_owner(&mut self, device: u32) -> Result<(), BufferError> {
        (**self).flush_to_owner(device)
    }
}

pub struct DestinationVisibilityFence<P> {
    device: u32,
    port: P,
}

impl<P> DestinationVisibilityFence<P>
where
    P: GpuDirectFlushPort,
{
    pub fn new(device: u32, port: P) -> Result<Self, BufferError> {
        if !matches!(device, 4 | 5) || !port.supports_flush_to_owner(device) {
            return Err(BufferError::VisibilityFence);
        }
        Ok(Self { device, port })
    }

    pub fn flush(&mut self) -> Result<(), BufferError> {
        if !self.port.supports_flush_to_owner(self.device) {
            return Err(BufferError::VisibilityFence);
        }
        self.port
            .flush_to_owner(self.device)
            .map_err(|_| BufferError::VisibilityFence)
    }

    pub const fn device(&self) -> u32 {
        self.device
    }

    pub fn into_port(self) -> P {
        self.port
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeSafety {
    Pending,
    SafeSuccess,
    SafeFailure,
}

impl NativeSafety {
    pub const fn is_safe(self) -> bool {
        matches!(self, Self::SafeSuccess | Self::SafeFailure)
    }
}

pub fn evaluate_native_fence(snapshot: &BatchSnapshot, expected_lengths: &[u64]) -> NativeSafety {
    if snapshot.operations.len() != expected_lengths.len()
        || snapshot.operations.is_empty()
        || !snapshot.safe_terminal
    {
        return NativeSafety::Pending;
    }
    if snapshot
        .operations
        .iter()
        .any(|operation| !operation.state.is_terminal())
    {
        return NativeSafety::Pending;
    }
    if snapshot
        .operations
        .iter()
        .zip(expected_lengths)
        .all(|(operation, expected_length)| {
            operation.state == OperationState::Completed
                && operation.transferred_bytes == *expected_length
        })
    {
        NativeSafety::SafeSuccess
    } else {
        NativeSafety::SafeFailure
    }
}
