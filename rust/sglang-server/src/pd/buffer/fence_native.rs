// The CUDA Runtime API requires a small, localized dynamic-FFI boundary.
#![allow(unsafe_code)]

use std::ffi::c_int;
use std::path::Path;

use libloading::Library;

use crate::pd::buffer::{BufferError, GpuDirectFlushPort};

const CUDA_RUNTIME_PATH: &str = "/usr/local/cuda/lib64/libcudart.so.13";
const CUDA_SUCCESS: c_int = 0;
const CUDA_DEVICE_ATTRIBUTE_FLUSH_WRITES_OPTIONS: c_int = 117;
const CUDA_DEVICE_ATTRIBUTE_WRITES_ORDERING: c_int = 118;
const CUDA_FLUSH_OPTION_HOST: c_int = 1;
const CUDA_WRITES_ORDERING_OWNER: c_int = 100;
const CUDA_FLUSH_TARGET_CURRENT_DEVICE: c_int = 0;
const CUDA_FLUSH_SCOPE_TO_OWNER: c_int = 100;

type CudaSetDevice = unsafe extern "C" fn(c_int) -> c_int;
type CudaDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
type CudaDeviceFlushGpudirectRdmaWrites = unsafe extern "C" fn(c_int, c_int) -> c_int;

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

unsafe fn load_symbol<T>(library: &Library, symbol: &[u8]) -> Result<T, BufferError>
where
    T: Copy,
{
    unsafe { library.get::<T>(symbol) }
        .map(|loaded| *loaded)
        .map_err(|_| BufferError::VisibilityFence)
}
