pub(crate) mod ffi;
mod manifest;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::rc::Rc;

use crate::mooncake::{
    BatchId, EngineError, EngineFactory, EngineOperation, OperationProgress, PdNicProfile,
    PeerDescriptor, PeerId, RegionDescriptor, RegionId, TransferEngine,
};

pub const PRODUCTION_ARTIFACT_DIR: &str = "/opt/sglang/mooncake/v0.3.11.post1";

#[derive(Debug, Clone)]
pub struct NativeEngineConfig {
    endpoint: SocketAddr,
    gpu_device: u32,
    profile: PdNicProfile,
}

impl NativeEngineConfig {
    pub fn new(endpoint: SocketAddr, gpu_device: u32) -> Result<Self, EngineError> {
        if endpoint.ip().is_unspecified() {
            return Err(EngineError::InvalidDescriptor {
                field: "native.endpoint",
                detail: "must use a concrete local IP".into(),
            });
        }
        if endpoint.port() == 0 {
            return Err(EngineError::InvalidDescriptor {
                field: "native.endpoint.port",
                detail: "must be non-zero".into(),
            });
        }
        Ok(Self {
            endpoint,
            gpu_device,
            profile: PdNicProfile::for_gpu(gpu_device)?,
        })
    }

    pub fn endpoint(&self) -> SocketAddr {
        self.endpoint
    }

    pub fn gpu_device(&self) -> u32 {
        self.gpu_device
    }
}

#[derive(Debug, Clone)]
pub struct NativeEngineFactory {
    config: NativeEngineConfig,
    artifact_dir: PathBuf,
}

impl NativeEngineFactory {
    pub fn production(config: NativeEngineConfig) -> Self {
        Self {
            config,
            artifact_dir: PathBuf::from(PRODUCTION_ARTIFACT_DIR),
        }
    }
}

impl EngineFactory for NativeEngineFactory {
    fn create(&self) -> Result<Box<dyn TransferEngine>, EngineError> {
        let artifact = manifest::validate_artifact(&self.artifact_dir)?;
        let library = ffi::FfiLibrary::load(&artifact.library_path)?;
        library.set_cuda_device(self.config.gpu_device)?;
        let mut engine = library.create_engine(self.config.endpoint)?;
        if let Err(error) = engine.install_rdma(self.config.profile.canonical_json()) {
            let _ = engine.destroy();
            return Err(error);
        }
        Ok(Box::new(NativeEngine {
            ffi: Some(engine),
            regions: HashMap::new(),
            peers: HashMap::new(),
            batches: HashMap::new(),
            transport_installed: true,
            shutdown: false,
            _not_send_or_sync: PhantomData,
        }))
    }
}

struct NativeBatch {
    native_id: u64,
    submitted: bool,
    progress: Vec<OperationProgress>,
}

struct NativeEngine {
    ffi: Option<ffi::FfiEngine>,
    regions: HashMap<RegionId, u64>,
    peers: HashMap<PeerId, i32>,
    batches: HashMap<BatchId, NativeBatch>,
    transport_installed: bool,
    shutdown: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl NativeEngine {
    fn ffi(&mut self) -> Result<&mut ffi::FfiEngine, EngineError> {
        self.ffi.as_mut().ok_or(EngineError::WorkerClosed)
    }
}

impl TransferEngine for NativeEngine {
    fn local_peer_descriptor(&mut self) -> Result<PeerDescriptor, EngineError> {
        PeerDescriptor::new(&self.ffi()?.local_endpoint()?)
    }

    fn register_region(
        &mut self,
        id: RegionId,
        descriptor: &RegionDescriptor,
    ) -> Result<(), EngineError> {
        self.ffi()?.register_region(
            descriptor.address(),
            descriptor.length(),
            descriptor.location().as_native_str(),
        )?;
        self.regions.insert(id, descriptor.address());
        Ok(())
    }

    fn unregister_region(&mut self, id: RegionId) -> Result<(), EngineError> {
        let address = *self.regions.get(&id).ok_or(EngineError::ResourceClosed {
            kind: "region",
            id: id.get(),
        })?;
        self.ffi()?.unregister_region(address)?;
        self.regions.remove(&id);
        Ok(())
    }

    fn open_peer(&mut self, id: PeerId, descriptor: &PeerDescriptor) -> Result<(), EngineError> {
        let native_id = self.ffi()?.open_peer(&descriptor.segment_name())?;
        self.peers.insert(id, native_id);
        Ok(())
    }

    fn close_peer(&mut self, id: PeerId) -> Result<(), EngineError> {
        let native_id = *self.peers.get(&id).ok_or(EngineError::ResourceClosed {
            kind: "peer",
            id: id.get(),
        })?;
        self.ffi()?.close_peer(native_id)?;
        self.peers.remove(&id);
        Ok(())
    }

    fn allocate_batch(&mut self, id: BatchId, operation_count: usize) -> Result<(), EngineError> {
        let native_id = self.ffi()?.allocate_batch(operation_count)?;
        self.batches.insert(
            id,
            NativeBatch {
                native_id,
                submitted: false,
                progress: vec![OperationProgress::default(); operation_count],
            },
        );
        Ok(())
    }

    fn submit_batch(
        &mut self,
        id: BatchId,
        operations: &[EngineOperation],
    ) -> Result<(), EngineError> {
        let native_id = self
            .batches
            .get(&id)
            .ok_or(EngineError::ResourceClosed {
                kind: "batch",
                id: id.get(),
            })?
            .native_id;
        let mut requests = Vec::with_capacity(operations.len());
        for operation in operations {
            let target_id =
                *self
                    .peers
                    .get(&operation.peer_id())
                    .ok_or(EngineError::ResourceClosed {
                        kind: "peer",
                        id: operation.peer_id().get(),
                    })?;
            requests.push(ffi::FfiRequest {
                opcode: operation.opcode(),
                local_address: operation.local_address(),
                target_id,
                remote_address: operation.remote_address(),
                length: operation.length(),
            });
        }
        self.ffi()?.submit(native_id, &requests)?;
        self.batches.get_mut(&id).expect("batch exists").submitted = true;
        Ok(())
    }

    fn poll(
        &mut self,
        id: BatchId,
        operation_index: usize,
    ) -> Result<OperationProgress, EngineError> {
        let native_id = self
            .batches
            .get(&id)
            .ok_or(EngineError::ResourceClosed {
                kind: "batch",
                id: id.get(),
            })?
            .native_id;
        let progress = self.ffi()?.poll(native_id, operation_index)?;
        self.batches.get_mut(&id).expect("batch exists").progress[operation_index] = progress;
        Ok(progress)
    }

    fn free_batch(&mut self, id: BatchId) -> Result<(), EngineError> {
        let batch = self.batches.get(&id).ok_or(EngineError::ResourceClosed {
            kind: "batch",
            id: id.get(),
        })?;
        if batch.submitted && !batch.progress.iter().all(|value| value.state.is_terminal()) {
            return Err(EngineError::BatchNotTerminal { id: id.get() });
        }
        let native_id = batch.native_id;
        self.ffi()?.free_batch(native_id)?;
        self.batches.remove(&id);
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), EngineError> {
        if self.shutdown {
            return Ok(());
        }
        if let Some((id, _)) = self.batches.iter().next() {
            return Err(EngineError::BatchNotTerminal { id: id.get() });
        }
        if !self.peers.is_empty() || !self.regions.is_empty() {
            return Err(EngineError::InvalidDescriptor {
                field: "native_shutdown",
                detail: "peer/region resources remain registered".into(),
            });
        }
        let mut ffi = self.ffi.take().ok_or(EngineError::WorkerClosed)?;
        if self.transport_installed {
            ffi.uninstall_rdma()?;
            self.transport_installed = false;
        }
        ffi.destroy()?;
        self.shutdown = true;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn validate_artifact_for_test(path: &std::path::Path) -> Result<(), EngineError> {
    manifest::validate_artifact(path).map(|_| ())
}

#[cfg(test)]
pub(crate) fn load_library_for_test(path: &std::path::Path) -> Result<(), EngineError> {
    ffi::FfiLibrary::load(path).map(|_| ())
}

#[cfg(test)]
pub(crate) fn validate_and_load_artifact_for_test(
    path: &std::path::Path,
) -> Result<(), EngineError> {
    let artifact = manifest::validate_artifact(path)?;
    ffi::FfiLibrary::load(&artifact.library_path).map(|_| ())
}
