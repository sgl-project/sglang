use std::sync::Arc;

use crate::mooncake::{BatchSnapshot, OperationState};
use crate::pd::buffer::BufferError;
use crate::pd::room::Clock;

#[path = "fence_native.rs"]
mod native;

pub use native::{CudaEventRuntimePort, CudaHostFlushPort};

pub trait SourceComputeFence: Send {
    fn wait_ready(&mut self, deadline_monotonic_ms: u64) -> Result<(), BufferError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaEventQuery {
    Pending,
    Ready,
}

/// Injectable boundary around the five CUDA Runtime calls needed by the
/// source-compute fence. Implementations own no Python objects.
pub trait CudaEventRuntime: Send {
    type Event: Copy + Send;

    fn set_device(&mut self, device: u32) -> Result<(), BufferError>;
    fn create_event(&mut self) -> Result<Self::Event, BufferError>;
    fn record_event(&mut self, event: Self::Event, stream: u64) -> Result<(), BufferError>;
    fn query_event(&mut self, event: Self::Event) -> Result<CudaEventQuery, BufferError>;
    fn destroy_event(&mut self, event: Self::Event);
}

/// Rust-owned CUDA event recorded on the Scheduler's current compute stream.
///
/// The opaque stream value is copied at the boundary. The event itself is
/// created, recorded, queried, and destroyed entirely in Rust.
pub struct CudaEventSourceFence<P>
where
    P: CudaEventRuntime,
{
    port: P,
    event: Option<P::Event>,
    clock: Arc<dyn Clock>,
}

impl<P> CudaEventSourceFence<P>
where
    P: CudaEventRuntime,
{
    pub fn new<C>(device: u32, stream: u64, mut port: P, clock: Arc<C>) -> Result<Self, BufferError>
    where
        C: Clock + 'static,
    {
        if !matches!(device, 4 | 5) {
            return Err(BufferError::SourceFence);
        }
        port.set_device(device)
            .map_err(|_| BufferError::SourceFence)?;
        let event = port.create_event().map_err(|_| BufferError::SourceFence)?;
        if port.record_event(event, stream).is_err() {
            port.destroy_event(event);
            return Err(BufferError::SourceFence);
        }
        Ok(Self {
            port,
            event: Some(event),
            clock,
        })
    }
}

impl CudaEventSourceFence<CudaEventRuntimePort> {
    pub fn production<C>(device: u32, stream: u64, clock: Arc<C>) -> Result<Self, BufferError>
    where
        C: Clock + 'static,
    {
        Self::new(device, stream, CudaEventRuntimePort::production()?, clock)
    }
}

impl<P> SourceComputeFence for CudaEventSourceFence<P>
where
    P: CudaEventRuntime,
{
    fn wait_ready(&mut self, deadline_monotonic_ms: u64) -> Result<(), BufferError> {
        let event = self.event.ok_or(BufferError::SourceFence)?;
        loop {
            if self.clock.now_monotonic_ms() >= deadline_monotonic_ms {
                return Err(BufferError::SourceFence);
            }
            match self
                .port
                .query_event(event)
                .map_err(|_| BufferError::SourceFence)?
            {
                CudaEventQuery::Ready => return Ok(()),
                CudaEventQuery::Pending => std::thread::yield_now(),
            }
        }
    }
}

impl<P> Drop for CudaEventSourceFence<P>
where
    P: CudaEventRuntime,
{
    fn drop(&mut self) {
        if let Some(event) = self.event.take() {
            self.port.destroy_event(event);
        }
    }
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
