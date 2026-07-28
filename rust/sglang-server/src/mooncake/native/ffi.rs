// Native FFI bindings require localized unsafe code.
#![allow(unsafe_code)]

use std::ffi::{CString, c_char, c_int, c_void};
use std::fmt;
use std::mem::{align_of, size_of};
use std::net::SocketAddr;
use std::path::Path;
use std::ptr;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use libloading::Library;

use crate::mooncake::types::TransferOpcode;
use crate::mooncake::{EngineError, NativeOperation, OperationProgress, OperationState};

const CUDA_RUNTIME_PATH: &str = "/usr/local/cuda/lib64/libcudart.so.13";
const INVALID_BATCH: u64 = u64::MAX;
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

type CudaMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type CudaFree = unsafe extern "C" fn(*mut c_void) -> c_int;
type CudaMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type CudaSetDevice = unsafe extern "C" fn(c_int) -> c_int;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AllocationKind {
    Pinned,
    Cuda { device: u32 },
}

struct NativeAllocation {
    _library: Library,
    address: usize,
    length: usize,
    kind: AllocationKind,
    free: CudaFree,
    memcpy: CudaMemcpy,
    set_device: CudaSetDevice,
    access: Mutex<()>,
}

impl Drop for NativeAllocation {
    fn drop(&mut self) {
        if let AllocationKind::Cuda { device } = self.kind {
            let Ok(device) = c_int::try_from(device) else {
                return;
            };
            if unsafe { (self.set_device)(device) } != 0 {
                return;
            }
        }
        let _ = unsafe { (self.free)(self.address as *mut c_void) };
    }
}

#[derive(Clone)]
pub(crate) struct NativeMemory {
    inner: Arc<NativeAllocation>,
}

impl fmt::Debug for NativeMemory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeMemory")
            .field("length", &self.inner.length)
            .field("kind", &self.inner.kind)
            .finish_non_exhaustive()
    }
}

impl NativeMemory {
    pub(crate) fn pinned(length: usize) -> Result<Self, EngineError> {
        Self::allocate(length, AllocationKind::Pinned)
    }

    pub(crate) fn cuda(device: u32, length: usize) -> Result<Self, EngineError> {
        Self::allocate(length, AllocationKind::Cuda { device })
    }

    fn allocate(length: usize, kind: AllocationKind) -> Result<Self, EngineError> {
        if length == 0 {
            return Err(EngineError::InvalidDescriptor {
                field: "memory.length",
                detail: "must be non-zero".into(),
            });
        }
        let library = load_cudart()?;
        let malloc: CudaMalloc = unsafe {
            match kind {
                AllocationKind::Pinned => {
                    load_symbol(&library, b"cudaMallocHost\0", "cudaMallocHost")?
                }
                AllocationKind::Cuda { .. } => {
                    load_symbol(&library, b"cudaMalloc\0", "cudaMalloc")?
                }
            }
        };
        let free: CudaFree = unsafe {
            match kind {
                AllocationKind::Pinned => load_symbol(&library, b"cudaFreeHost\0", "cudaFreeHost")?,
                AllocationKind::Cuda { .. } => load_symbol(&library, b"cudaFree\0", "cudaFree")?,
            }
        };
        let memcpy: CudaMemcpy = unsafe { load_symbol(&library, b"cudaMemcpy\0", "cudaMemcpy")? };
        let set_device: CudaSetDevice =
            unsafe { load_symbol(&library, b"cudaSetDevice\0", "cudaSetDevice")? };
        if let AllocationKind::Cuda { device } = kind {
            let raw_device =
                c_int::try_from(device).map_err(|_| EngineError::UnsupportedGpu { device })?;
            check_code(NativeOperation::SetCudaDevice, unsafe {
                set_device(raw_device)
            })?;
        }
        let mut address = ptr::null_mut();
        let operation = match kind {
            AllocationKind::Pinned => NativeOperation::AllocatePinnedMemory,
            AllocationKind::Cuda { .. } => NativeOperation::AllocateCudaMemory,
        };
        check_code(operation, unsafe { malloc(&mut address, length) })?;
        if address.is_null() {
            return Err(EngineError::NativeHandle { operation });
        }
        Ok(Self {
            inner: Arc::new(NativeAllocation {
                _library: library,
                address: address as usize,
                length,
                kind,
                free,
                memcpy,
                set_device,
                access: Mutex::new(()),
            }),
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.length
    }

    pub(crate) fn address(&self) -> u64 {
        self.inner.address as u64
    }

    pub(crate) fn cuda_device(&self) -> Option<u32> {
        match self.inner.kind {
            AllocationKind::Pinned => None,
            AllocationKind::Cuda { device } => Some(device),
        }
    }

    pub(crate) fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), EngineError> {
        let end = checked_memory_range(self.inner.length, offset, bytes.len())?;
        let _access = self
            .inner
            .access
            .lock()
            .map_err(|_| EngineError::LockPoisoned)?;
        let destination =
            self.inner
                .address
                .checked_add(offset)
                .ok_or(EngineError::RangeOverflow {
                    field: "memory.address",
                })? as *mut c_void;
        match self.inner.kind {
            AllocationKind::Pinned => unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), destination.cast::<u8>(), end - offset);
            },
            AllocationKind::Cuda { device } => {
                self.set_device(device)?;
                check_code(NativeOperation::CopyMemory, unsafe {
                    (self.inner.memcpy)(
                        destination,
                        bytes.as_ptr().cast(),
                        end - offset,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                    )
                })?;
            }
        }
        Ok(())
    }

    pub(crate) fn read(&self, offset: usize, length: usize) -> Result<Vec<u8>, EngineError> {
        checked_memory_range(self.inner.length, offset, length)?;
        let _access = self
            .inner
            .access
            .lock()
            .map_err(|_| EngineError::LockPoisoned)?;
        let source = self
            .inner
            .address
            .checked_add(offset)
            .ok_or(EngineError::RangeOverflow {
                field: "memory.address",
            })? as *const c_void;
        let mut bytes = vec![0; length];
        match self.inner.kind {
            AllocationKind::Pinned => unsafe {
                ptr::copy_nonoverlapping(source.cast::<u8>(), bytes.as_mut_ptr(), length);
            },
            AllocationKind::Cuda { device } => {
                self.set_device(device)?;
                check_code(NativeOperation::CopyMemory, unsafe {
                    (self.inner.memcpy)(
                        bytes.as_mut_ptr().cast(),
                        source,
                        length,
                        CUDA_MEMCPY_DEVICE_TO_HOST,
                    )
                })?;
            }
        }
        Ok(bytes)
    }

    pub(crate) fn fill(&self, value: u8) -> Result<(), EngineError> {
        self.write(0, &vec![value; self.len()])
    }

    fn set_device(&self, device: u32) -> Result<(), EngineError> {
        let raw_device =
            c_int::try_from(device).map_err(|_| EngineError::UnsupportedGpu { device })?;
        check_code(NativeOperation::SetCudaDevice, unsafe {
            (self.inner.set_device)(raw_device)
        })
    }
}

fn checked_memory_range(
    registered_length: usize,
    offset: usize,
    length: usize,
) -> Result<usize, EngineError> {
    let end = offset
        .checked_add(length)
        .ok_or(EngineError::RangeOverflow { field: "memory" })?;
    if end > registered_length {
        return Err(EngineError::RangeOutOfBounds {
            field: "memory",
            offset: offset as u64,
            end: end as u64,
            registered_length: registered_length as u64,
        });
    }
    Ok(end)
}

fn load_cudart() -> Result<Library, EngineError> {
    unsafe { Library::new(CUDA_RUNTIME_PATH) }.map_err(|error| EngineError::LoaderFailure {
        path: Path::new(CUDA_RUNTIME_PATH).to_path_buf(),
        detail: error.to_string(),
    })
}

type CreateTransferEngine =
    unsafe extern "C" fn(*const c_char, *const c_char, *const c_char, u64, c_int) -> *mut c_void;
type DestroyTransferEngine = unsafe extern "C" fn(*mut c_void);
type GetLocalIpAndPort = unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> c_int;
type InstallTransport =
    unsafe extern "C" fn(*mut c_void, *const c_char, *mut *mut c_void) -> *mut c_void;
type UninstallTransport = unsafe extern "C" fn(*mut c_void, *const c_char) -> c_int;
type RegisterLocalMemory =
    unsafe extern "C" fn(*mut c_void, *mut c_void, usize, *const c_char, c_int) -> c_int;
type UnregisterLocalMemory = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type OpenSegment = unsafe extern "C" fn(*mut c_void, *const c_char) -> i32;
type CloseSegment = unsafe extern "C" fn(*mut c_void, i32) -> c_int;
type AllocateBatchId = unsafe extern "C" fn(*mut c_void, usize) -> u64;
type SubmitTransfer =
    unsafe extern "C" fn(*mut c_void, u64, *mut RawTransferRequest, usize) -> c_int;
type GetTransferStatus =
    unsafe extern "C" fn(*mut c_void, u64, usize, *mut RawTransferStatus) -> c_int;
type FreeBatchId = unsafe extern "C" fn(*mut c_void, u64) -> c_int;

#[repr(C)]
struct RawTransferRequest {
    opcode: c_int,
    source: *mut c_void,
    target_id: i32,
    target_offset: u64,
    length: u64,
}

#[repr(C)]
struct RawTransferStatus {
    status: c_int,
    transferred_bytes: u64,
}

#[derive(Debug)]
pub struct AbiLayout {
    pub pointer_width_bits: usize,
    pub transfer_request_size: usize,
    pub transfer_request_align: usize,
    pub transfer_status_size: usize,
    pub transfer_status_align: usize,
}

pub fn abi_layout() -> AbiLayout {
    AbiLayout {
        pointer_width_bits: usize::BITS as usize,
        transfer_request_size: size_of::<RawTransferRequest>(),
        transfer_request_align: align_of::<RawTransferRequest>(),
        transfer_status_size: size_of::<RawTransferStatus>(),
        transfer_status_align: align_of::<RawTransferStatus>(),
    }
}

struct Functions {
    create_transfer_engine: CreateTransferEngine,
    destroy_transfer_engine: DestroyTransferEngine,
    get_local_ip_and_port: GetLocalIpAndPort,
    install_transport: InstallTransport,
    uninstall_transport: UninstallTransport,
    register_local_memory: RegisterLocalMemory,
    unregister_local_memory: UnregisterLocalMemory,
    open_segment: OpenSegment,
    close_segment: CloseSegment,
    allocate_batch_id: AllocateBatchId,
    submit_transfer: SubmitTransfer,
    get_transfer_status: GetTransferStatus,
    free_batch_id: FreeBatchId,
}

pub struct FfiLibrary {
    library: Option<Library>,
    functions: Functions,
}

impl FfiLibrary {
    pub fn load(path: &Path) -> Result<Self, EngineError> {
        let library =
            unsafe { Library::new(path) }.map_err(|error| EngineError::LoaderFailure {
                path: path.to_path_buf(),
                detail: error.to_string(),
            })?;
        let functions = unsafe {
            Functions {
                create_transfer_engine: load_symbol(
                    &library,
                    b"createTransferEngine\0",
                    "createTransferEngine",
                )?,
                destroy_transfer_engine: load_symbol(
                    &library,
                    b"destroyTransferEngine\0",
                    "destroyTransferEngine",
                )?,
                get_local_ip_and_port: load_symbol(
                    &library,
                    b"getLocalIpAndPort\0",
                    "getLocalIpAndPort",
                )?,
                install_transport: load_symbol(
                    &library,
                    b"installTransport\0",
                    "installTransport",
                )?,
                uninstall_transport: load_symbol(
                    &library,
                    b"uninstallTransport\0",
                    "uninstallTransport",
                )?,
                register_local_memory: load_symbol(
                    &library,
                    b"registerLocalMemory\0",
                    "registerLocalMemory",
                )?,
                unregister_local_memory: load_symbol(
                    &library,
                    b"unregisterLocalMemory\0",
                    "unregisterLocalMemory",
                )?,
                open_segment: load_symbol(&library, b"openSegment\0", "openSegment")?,
                close_segment: load_symbol(&library, b"closeSegment\0", "closeSegment")?,
                allocate_batch_id: load_symbol(&library, b"allocateBatchID\0", "allocateBatchID")?,
                submit_transfer: load_symbol(&library, b"submitTransfer\0", "submitTransfer")?,
                get_transfer_status: load_symbol(
                    &library,
                    b"getTransferStatus\0",
                    "getTransferStatus",
                )?,
                free_batch_id: load_symbol(&library, b"freeBatchID\0", "freeBatchID")?,
            }
        };
        Ok(Self {
            library: Some(library),
            functions,
        })
    }

    pub fn set_cuda_device(&self, device: u32) -> Result<(), EngineError> {
        let library = unsafe { Library::new(CUDA_RUNTIME_PATH) }.map_err(|error| {
            EngineError::LoaderFailure {
                path: Path::new(CUDA_RUNTIME_PATH).to_path_buf(),
                detail: error.to_string(),
            }
        })?;
        let set_device: unsafe extern "C" fn(c_int) -> c_int =
            unsafe { load_symbol(&library, b"cudaSetDevice\0", "cudaSetDevice")? };
        let raw_device =
            c_int::try_from(device).map_err(|_| EngineError::UnsupportedGpu { device })?;
        let result = unsafe { set_device(raw_device) };
        if result == 0 {
            Ok(())
        } else {
            Err(EngineError::native(NativeOperation::SetCudaDevice, result))
        }
    }

    pub fn create_engine(self, endpoint: SocketAddr) -> Result<FfiEngine, EngineError> {
        let metadata = c_string("P2PHANDSHAKE", "metadata")?;
        let local_name = c_string(&endpoint.to_string(), "local_server_name")?;
        let host = c_string(&endpoint.ip().to_string(), "ip_or_host_name")?;
        let handle = unsafe {
            (self.functions.create_transfer_engine)(
                metadata.as_ptr(),
                local_name.as_ptr(),
                host.as_ptr(),
                u64::from(endpoint.port()),
                0,
            )
        };
        if handle.is_null() {
            return Err(EngineError::NativeHandle {
                operation: NativeOperation::CreateEngine,
            });
        }
        Ok(FfiEngine {
            library: self,
            handle,
            creator_pid: std::process::id(),
            _not_send_or_sync: std::marker::PhantomData,
        })
    }
}

pub struct FfiRequest {
    pub opcode: TransferOpcode,
    pub local_address: u64,
    pub target_id: i32,
    pub remote_address: u64,
    pub length: u64,
}

pub struct FfiEngine {
    library: FfiLibrary,
    handle: *mut c_void,
    creator_pid: u32,
    _not_send_or_sync: std::marker::PhantomData<Rc<()>>,
}

impl FfiEngine {
    fn check_pid(&self) -> Result<(), EngineError> {
        let current_pid = std::process::id();
        if current_pid == self.creator_pid {
            Ok(())
        } else {
            Err(EngineError::ForkDetected {
                creator_pid: self.creator_pid,
                current_pid,
            })
        }
    }

    pub fn local_endpoint(&mut self) -> Result<String, EngineError> {
        self.check_pid()?;
        let mut output = [0_i8; 256];
        let result = unsafe {
            (self.library.functions.get_local_ip_and_port)(
                self.handle,
                output.as_mut_ptr(),
                output.len(),
            )
        };
        check_code(NativeOperation::GetLocalEndpoint, result)?;
        let length = output.iter().position(|value| *value == 0).ok_or_else(|| {
            EngineError::AbiMismatch {
                detail: "getLocalIpAndPort returned an unterminated value".into(),
            }
        })?;
        let bytes: Vec<_> = output[..length].iter().map(|value| *value as u8).collect();
        String::from_utf8(bytes).map_err(|error| EngineError::AbiMismatch {
            detail: format!("getLocalIpAndPort returned non-UTF-8 data: {error}"),
        })
    }

    pub fn install_rdma(&mut self, matrix: &str) -> Result<(), EngineError> {
        self.check_pid()?;
        let protocol = c_string("rdma", "transport")?;
        let matrix = c_string(matrix, "nic_priority_matrix")?;
        let mut args = [matrix.as_ptr() as *mut c_void, ptr::null_mut()];
        let transport = unsafe {
            (self.library.functions.install_transport)(
                self.handle,
                protocol.as_ptr(),
                args.as_mut_ptr(),
            )
        };
        if transport.is_null() {
            Err(EngineError::NativeHandle {
                operation: NativeOperation::InstallTransport,
            })
        } else {
            Ok(())
        }
    }

    pub fn uninstall_rdma(&mut self) -> Result<(), EngineError> {
        self.check_pid()?;
        let protocol = c_string("rdma", "transport")?;
        let result =
            unsafe { (self.library.functions.uninstall_transport)(self.handle, protocol.as_ptr()) };
        check_code(NativeOperation::UninstallTransport, result)
    }

    pub fn register_region(
        &mut self,
        address: u64,
        length: u64,
        location: &str,
    ) -> Result<(), EngineError> {
        self.check_pid()?;
        let length = usize::try_from(length).map_err(|_| EngineError::RangeOverflow {
            field: "region.length",
        })?;
        let location = c_string(location, "region.location")?;
        let result = unsafe {
            (self.library.functions.register_local_memory)(
                self.handle,
                address as usize as *mut c_void,
                length,
                location.as_ptr(),
                1,
            )
        };
        check_code(NativeOperation::RegisterRegion, result)
    }

    pub fn unregister_region(&mut self, address: u64) -> Result<(), EngineError> {
        self.check_pid()?;
        let result = unsafe {
            (self.library.functions.unregister_local_memory)(
                self.handle,
                address as usize as *mut c_void,
            )
        };
        check_code(NativeOperation::UnregisterRegion, result)
    }

    pub fn open_peer(&mut self, segment_name: &str) -> Result<i32, EngineError> {
        self.check_pid()?;
        let segment_name = c_string(segment_name, "peer.segment_name")?;
        let result =
            unsafe { (self.library.functions.open_segment)(self.handle, segment_name.as_ptr()) };
        if result < 0 {
            Err(EngineError::native(NativeOperation::OpenPeer, result))
        } else {
            Ok(result)
        }
    }

    pub fn close_peer(&mut self, segment_id: i32) -> Result<(), EngineError> {
        self.check_pid()?;
        let result = unsafe { (self.library.functions.close_segment)(self.handle, segment_id) };
        check_code(NativeOperation::ClosePeer, result)
    }

    pub fn allocate_batch(&mut self, operation_count: usize) -> Result<u64, EngineError> {
        self.check_pid()?;
        let result =
            unsafe { (self.library.functions.allocate_batch_id)(self.handle, operation_count) };
        if result == INVALID_BATCH {
            Err(EngineError::NativeHandle {
                operation: NativeOperation::AllocateBatch,
            })
        } else {
            Ok(result)
        }
    }

    pub fn submit(&mut self, batch_id: u64, requests: &[FfiRequest]) -> Result<(), EngineError> {
        self.check_pid()?;
        let mut raw_requests: Vec<_> = requests
            .iter()
            .map(|request| RawTransferRequest {
                opcode: match request.opcode {
                    TransferOpcode::Read => 0,
                    TransferOpcode::Write => 1,
                },
                source: request.local_address as usize as *mut c_void,
                target_id: request.target_id,
                target_offset: request.remote_address,
                length: request.length,
            })
            .collect();
        let result = unsafe {
            (self.library.functions.submit_transfer)(
                self.handle,
                batch_id,
                raw_requests.as_mut_ptr(),
                raw_requests.len(),
            )
        };
        check_code(NativeOperation::SubmitBatch, result)
    }

    pub fn poll(
        &mut self,
        batch_id: u64,
        operation_index: usize,
    ) -> Result<OperationProgress, EngineError> {
        self.check_pid()?;
        let mut status = RawTransferStatus {
            status: i32::MIN,
            transferred_bytes: u64::MAX,
        };
        let result = unsafe {
            (self.library.functions.get_transfer_status)(
                self.handle,
                batch_id,
                operation_index,
                &mut status,
            )
        };
        check_code(NativeOperation::Poll, result)?;
        if status.status == i32::MIN || status.transferred_bytes == u64::MAX {
            return Err(EngineError::AbiMismatch {
                detail: format!(
                    "getTransferStatus left output incomplete for operation {operation_index}"
                ),
            });
        }
        Ok(OperationProgress {
            state: OperationState::from_raw(status.status),
            transferred_bytes: status.transferred_bytes,
        })
    }

    pub fn free_batch(&mut self, batch_id: u64) -> Result<(), EngineError> {
        self.check_pid()?;
        let result = unsafe { (self.library.functions.free_batch_id)(self.handle, batch_id) };
        check_code(NativeOperation::FreeBatch, result)
    }

    pub fn destroy(&mut self) -> Result<(), EngineError> {
        self.check_pid()?;
        if self.handle.is_null() {
            return Ok(());
        }
        unsafe { (self.library.functions.destroy_transfer_engine)(self.handle) };
        self.handle = ptr::null_mut();
        Ok(())
    }
}

impl Drop for FfiEngine {
    fn drop(&mut self) {
        if !self.handle.is_null()
            && let Some(library) = self.library.library.take()
        {
            std::mem::forget(library);
        }
    }
}

fn check_code(operation: NativeOperation, raw_code: i32) -> Result<(), EngineError> {
    if raw_code == 0 {
        Ok(())
    } else {
        Err(EngineError::native(operation, raw_code))
    }
}

fn c_string(value: &str, field: &'static str) -> Result<CString, EngineError> {
    CString::new(value).map_err(|_| EngineError::InvalidDescriptor {
        field,
        detail: "contains an interior NUL byte".into(),
    })
}

unsafe fn load_symbol<T: Copy>(
    library: &Library,
    bytes: &[u8],
    name: &str,
) -> Result<T, EngineError> {
    let symbol = unsafe { library.get::<T>(bytes) }.map_err(|_| EngineError::SymbolMissing {
        symbol: name.into(),
    })?;
    Ok(*symbol)
}
