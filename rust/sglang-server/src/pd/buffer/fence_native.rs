// The CUDA Runtime API requires a small, localized dynamic-FFI boundary.
#![allow(unsafe_code)]

use std::ffi::{c_int, c_uint, c_void};
use std::path::Path;

use libloading::Library;

use crate::pd::buffer::{BufferError, CudaEventQuery, CudaEventRuntime, GpuDirectFlushPort};

const CUDA_RUNTIME_PATH: &str = "/usr/local/cuda/lib64/libcudart.so.13";
const CUDA_SUCCESS: c_int = 0;
const CUDA_ERROR_NOT_READY: c_int = 600;
const CUDA_EVENT_DISABLE_TIMING: c_uint = 2;
const CUDA_DEVICE_ATTRIBUTE_FLUSH_WRITES_OPTIONS: c_int = 117;
const CUDA_DEVICE_ATTRIBUTE_WRITES_ORDERING: c_int = 118;
const CUDA_FLUSH_OPTION_HOST: c_int = 1;
const CUDA_WRITES_ORDERING_OWNER: c_int = 100;
const CUDA_FLUSH_TARGET_CURRENT_DEVICE: c_int = 0;
const CUDA_FLUSH_SCOPE_TO_OWNER: c_int = 100;

type CudaSetDevice = unsafe extern "C" fn(c_int) -> c_int;
type CudaDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
type CudaDeviceFlushGpudirectRdmaWrites = unsafe extern "C" fn(c_int, c_int) -> c_int;
type CudaEventCreateWithFlags = unsafe extern "C" fn(*mut *mut c_void, c_uint) -> c_int;
type CudaEventRecord = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type CudaEventQueryFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type CudaEventDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;

/// Production host-side visibility adapter for GPUDirect RDMA writes.
///
/// The library is retained for at least as long as every resolved function
/// pointer. Capability and call failures are deliberately collapsed into the
/// stable fail-closed buffer error by `DestinationVisibilityFence`.
pub struct CudaHostFlushPort {
    _library: Library,
    set_device: CudaSetDevice,
    get_attribute: CudaDeviceGetAttribute,
    flush_writes: CudaDeviceFlushGpudirectRdmaWrites,
}

impl CudaHostFlushPort {
    pub fn production() -> Result<Self, BufferError> {
        Self::load(CUDA_RUNTIME_PATH)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, BufferError> {
        let library =
            unsafe { Library::new(path.as_ref()) }.map_err(|_| BufferError::VisibilityFence)?;
        let set_device = unsafe { load_symbol(&library, b"cudaSetDevice\0")? };
        let get_attribute = unsafe { load_symbol(&library, b"cudaDeviceGetAttribute\0")? };
        let flush_writes =
            unsafe { load_symbol(&library, b"cudaDeviceFlushGPUDirectRDMAWrites\0")? };
        Ok(Self {
            _library: library,
            set_device,
            get_attribute,
            flush_writes,
        })
    }

    fn attribute(&self, device: u32, attribute: c_int) -> Option<c_int> {
        let device = c_int::try_from(device).ok()?;
        let mut value = 0;
        let result = unsafe { (self.get_attribute)(&mut value, attribute, device) };
        (result == CUDA_SUCCESS).then_some(value)
    }
}

impl GpuDirectFlushPort for CudaHostFlushPort {
    fn supports_flush_to_owner(&self, device: u32) -> bool {
        if !matches!(device, 4 | 5) {
            return false;
        }
        self.attribute(device, CUDA_DEVICE_ATTRIBUTE_FLUSH_WRITES_OPTIONS)
            .is_some_and(|options| options & CUDA_FLUSH_OPTION_HOST != 0)
            && self
                .attribute(device, CUDA_DEVICE_ATTRIBUTE_WRITES_ORDERING)
                .is_some_and(|ordering| ordering >= CUDA_WRITES_ORDERING_OWNER)
    }

    fn flush_to_owner(&mut self, device: u32) -> Result<(), BufferError> {
        if !self.supports_flush_to_owner(device) {
            return Err(BufferError::VisibilityFence);
        }
        let device = c_int::try_from(device).map_err(|_| BufferError::VisibilityFence)?;
        if unsafe { (self.set_device)(device) } != CUDA_SUCCESS {
            return Err(BufferError::VisibilityFence);
        }
        if unsafe {
            (self.flush_writes)(CUDA_FLUSH_TARGET_CURRENT_DEVICE, CUDA_FLUSH_SCOPE_TO_OWNER)
        } != CUDA_SUCCESS
        {
            return Err(BufferError::VisibilityFence);
        }
        Ok(())
    }
}

/// Production dynamic CUDA Runtime adapter for the source-compute event.
pub struct CudaEventRuntimePort {
    _library: Library,
    set_device: CudaSetDevice,
    create_event: CudaEventCreateWithFlags,
    record_event: CudaEventRecord,
    query_event: CudaEventQueryFn,
    destroy_event: CudaEventDestroy,
}

impl CudaEventRuntimePort {
    pub fn production() -> Result<Self, BufferError> {
        Self::load(CUDA_RUNTIME_PATH)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, BufferError> {
        let library =
            unsafe { Library::new(path.as_ref()) }.map_err(|_| BufferError::SourceFence)?;
        let set_device = unsafe { load_source_symbol(&library, b"cudaSetDevice\0")? };
        let create_event = unsafe { load_source_symbol(&library, b"cudaEventCreateWithFlags\0")? };
        let record_event = unsafe { load_source_symbol(&library, b"cudaEventRecord\0")? };
        let query_event = unsafe { load_source_symbol(&library, b"cudaEventQuery\0")? };
        let destroy_event = unsafe { load_source_symbol(&library, b"cudaEventDestroy\0")? };
        Ok(Self {
            _library: library,
            set_device,
            create_event,
            record_event,
            query_event,
            destroy_event,
        })
    }
}

impl CudaEventRuntime for CudaEventRuntimePort {
    type Event = usize;

    fn set_device(&mut self, device: u32) -> Result<(), BufferError> {
        let device = c_int::try_from(device).map_err(|_| BufferError::SourceFence)?;
        if unsafe { (self.set_device)(device) } == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(BufferError::SourceFence)
        }
    }

    fn create_event(&mut self) -> Result<Self::Event, BufferError> {
        let mut event = std::ptr::null_mut();
        if unsafe { (self.create_event)(&mut event, CUDA_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS
            || event.is_null()
        {
            return Err(BufferError::SourceFence);
        }
        Ok(event as usize)
    }

    fn record_event(&mut self, event: Self::Event, stream: u64) -> Result<(), BufferError> {
        let event = event as *mut c_void;
        let stream = usize::try_from(stream).map_err(|_| BufferError::SourceFence)? as *mut c_void;
        if unsafe { (self.record_event)(event, stream) } == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(BufferError::SourceFence)
        }
    }

    fn query_event(&mut self, event: Self::Event) -> Result<CudaEventQuery, BufferError> {
        match unsafe { (self.query_event)(event as *mut c_void) } {
            CUDA_SUCCESS => Ok(CudaEventQuery::Ready),
            CUDA_ERROR_NOT_READY => Ok(CudaEventQuery::Pending),
            _ => Err(BufferError::SourceFence),
        }
    }

    fn destroy_event(&mut self, event: Self::Event) {
        let _ = unsafe { (self.destroy_event)(event as *mut c_void) };
    }
}

unsafe fn load_symbol<T>(library: &Library, symbol: &[u8]) -> Result<T, BufferError>
where
    T: Copy,
{
    unsafe { library.get::<T>(symbol) }
        .map(|loaded| *loaded)
        .map_err(|_| BufferError::VisibilityFence)
}

unsafe fn load_source_symbol<T>(library: &Library, symbol: &[u8]) -> Result<T, BufferError>
where
    T: Copy,
{
    unsafe { library.get::<T>(symbol) }
        .map(|loaded| *loaded)
        .map_err(|_| BufferError::SourceFence)
}
