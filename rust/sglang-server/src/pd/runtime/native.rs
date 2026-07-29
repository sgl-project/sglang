use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Deserialize;

use crate::mooncake::{
    CudaMemory, EngineOwner, MemoryBuffer, NativeEngineConfig, NativeEngineFactory, OwnerConfig,
    Peer, PeerDescriptor, PinnedMemory, Region as MooncakeRegion, ShutdownOutcome,
    TransferOperation,
};
#[cfg(test)]
use crate::mooncake::{MockEngineFactory, MockPlan};
use crate::pd::buffer::{
    AuthenticatedRemoteRegionTable, BufferDType, BufferRegionSpec, BufferTable,
    MooncakeNativeStagePort, MooncakeRegistrationPort, RegionKind, RegionLayout, RegionLocation,
    RegisteredRegionTable,
};
use crate::pd::protocol::{FixedBytes, RegisterRegions, Role};
use crate::pd::room::{PdReason, RegistrationEpoch};
use crate::pd::runtime::RuntimeShutdownOutcome;
use crate::pd::runtime::bootstrap::{BootstrapPort, BootstrapRegistration};

const KV_PAGE_BYTES: u64 = 131_072;
const KV_ROW_BYTES: u64 = 2_048;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRegionDescriptor {
    pub region_id: u16,
    pub address: u64,
    pub length_bytes: u64,
    pub device: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub stride_bytes: Vec<u64>,
    pub generation: u64,
}

pub struct NativeBootstrapPort {
    role: Role,
    endpoint: SocketAddr,
    owner: Arc<EngineOwner>,
    table: Mutex<Option<Arc<RegisteredRegionTable<MooncakeRegion>>>>,
    buffers: BTreeMap<u16, MemoryBuffer>,
    peer: Mutex<Option<Peer>>,
    remote: Mutex<Option<AuthenticatedRemoteRegionTable>>,
    shutdown_outcome: Mutex<Option<RuntimeShutdownOutcome>>,
}

impl NativeBootstrapPort {
    pub fn new(
        role: Role,
        endpoint: SocketAddr,
        layout_fingerprint: FixedBytes<32>,
        descriptors: Vec<NativeRegionDescriptor>,
    ) -> Result<Self, PdReason> {
        let device = role_device(role);
        let (table, buffers) = build_table(descriptors, device, layout_fingerprint)
            .map_err(|_| PdReason::Unsupported)?;
        let config = NativeEngineConfig::new(endpoint, device).map_err(|_| PdReason::LocalFatal)?;
        let owner = Arc::new(
            EngineOwner::start(
                OwnerConfig::default(),
                NativeEngineFactory::production(config),
            )
            .map_err(|_| PdReason::LocalFatal)?,
        );
        let mut registration = MooncakeRegistrationPort::new(&owner, buffers.clone());
        let table = Arc::new(
            table
                .register(&mut registration)
                .map_err(|_| PdReason::LocalFatal)?,
        );
        Ok(Self {
            role,
            endpoint,
            owner,
            table: Mutex::new(Some(table)),
            buffers,
            peer: Mutex::new(None),
            remote: Mutex::new(None),
            shutdown_outcome: Mutex::new(None),
        })
    }

    #[cfg(test)]
    pub(crate) fn new_mock(
        role: Role,
        endpoint: SocketAddr,
        layout_fingerprint: FixedBytes<32>,
        descriptors: Vec<NativeRegionDescriptor>,
    ) -> Result<Self, PdReason> {
        let device = role_device(role);
        let (table, buffers) = build_table(descriptors, device, layout_fingerprint)
            .map_err(|_| PdReason::Unsupported)?;
        let owner = Arc::new(
            EngineOwner::start(
                OwnerConfig::default(),
                MockEngineFactory::new(MockPlan::default()),
            )
            .map_err(|_| PdReason::LocalFatal)?,
        );
        let mut registration = MooncakeRegistrationPort::new(&owner, buffers.clone());
        let table = Arc::new(
            table
                .register(&mut registration)
                .map_err(|_| PdReason::LocalFatal)?,
        );
        Ok(Self {
            role,
            endpoint,
            owner,
            table: Mutex::new(Some(table)),
            buffers,
            peer: Mutex::new(None),
            remote: Mutex::new(None),
            shutdown_outcome: Mutex::new(None),
        })
    }

    pub fn registration_epoch(&self) -> Result<RegistrationEpoch, PdReason> {
        Ok(self.table()?.epoch())
    }

    pub fn table(&self) -> Result<Arc<RegisteredRegionTable<MooncakeRegion>>, PdReason> {
        self.table
            .lock()
            .map_err(|_| PdReason::LocalFatal)?
            .as_ref()
            .cloned()
            .ok_or(PdReason::LocalFatal)
    }

    pub fn buffers(&self) -> BTreeMap<u16, MemoryBuffer> {
        self.buffers.clone()
    }

    pub fn take_stage_port(&self) -> Result<MooncakeNativeStagePort, PdReason> {
        if self.role != Role::Prefill {
            return Err(PdReason::ProtocolMismatch);
        }
        let peer = self
            .peer
            .lock()
            .map_err(|_| PdReason::LocalFatal)?
            .take()
            .ok_or(PdReason::ProtocolMismatch)?;
        let remote = self
            .remote
            .lock()
            .map_err(|_| PdReason::LocalFatal)?
            .take()
            .ok_or(PdReason::ProtocolMismatch)?;
        MooncakeNativeStagePort::new(
            Arc::clone(&self.owner),
            self.table()?,
            peer,
            self.buffers.clone(),
            remote,
        )
        .map_err(|_| PdReason::LocalFatal)
    }

    pub fn reset_peer(&self) -> Result<(), PdReason> {
        if let Some(peer) = self.peer.lock().map_err(|_| PdReason::LocalFatal)?.take() {
            peer.close().map_err(|_| PdReason::LocalFatal)?;
        }
        self.remote.lock().map_err(|_| PdReason::LocalFatal)?.take();
        Ok(())
    }

    pub fn shutdown(&self) -> Result<RuntimeShutdownOutcome, PdReason> {
        if let Some(outcome) = *self
            .shutdown_outcome
            .lock()
            .map_err(|_| PdReason::LocalFatal)?
        {
            return Ok(outcome);
        }
        let mut safe = self.reset_peer().is_ok();
        let table = self.table.lock().map_err(|_| PdReason::LocalFatal)?.take();
        if let Some(table) = table {
            match Arc::try_unwrap(table) {
                Ok(mut table) => {
                    let mut registration =
                        MooncakeRegistrationPort::new(&self.owner, self.buffers.clone());
                    safe &= table.unregister(&mut registration).is_ok();
                }
                Err(table) => {
                    *self.table.lock().map_err(|_| PdReason::LocalFatal)? = Some(table);
                    safe = false;
                }
            }
        }
        safe &= matches!(self.owner.shutdown(), Ok(ShutdownOutcome::SafeTerminal));
        let outcome = if safe {
            RuntimeShutdownOutcome::SafeTerminal
        } else {
            RuntimeShutdownOutcome::FatalUnsafe
        };
        *self
            .shutdown_outcome
            .lock()
            .map_err(|_| PdReason::LocalFatal)? = Some(outcome);
        Ok(outcome)
    }
}

impl BootstrapPort for NativeBootstrapPort {
    fn registration(&self) -> Result<BootstrapRegistration, PdReason> {
        if self.role != Role::Decode {
            return Err(PdReason::ProtocolMismatch);
        }
        let table = self.table()?;
        BootstrapRegistration::from_registered_table(
            &table,
            self.endpoint.ip().to_string(),
            self.endpoint.port(),
        )
    }

    fn open_peer(&self, registration: &RegisterRegions) -> Result<(), PdReason> {
        if self.role != Role::Prefill {
            return Err(PdReason::ProtocolMismatch);
        }
        let remote = AuthenticatedRemoteRegionTable::from_authenticated_register(registration)
            .map_err(|_| PdReason::ProtocolMismatch)?;
        let descriptor = PeerDescriptor::new(&format!(
            "{}:{}",
            registration.mooncake_host, registration.mooncake_port
        ))
        .map_err(|_| PdReason::ProtocolMismatch)?;
        let peer = self
            .owner
            .open_peer(descriptor)
            .map_err(|_| PdReason::PeerUnavailable)?;
        let mut current_peer = self.peer.lock().map_err(|_| PdReason::LocalFatal)?;
        let mut current_remote = self.remote.lock().map_err(|_| PdReason::LocalFatal)?;
        if current_peer.is_some() || current_remote.is_some() {
            return Err(PdReason::ProtocolMismatch);
        }
        *current_peer = Some(peer);
        *current_remote = Some(remote);
        Ok(())
    }

    fn produce_canary(&self, generation: u64) -> Result<FixedBytes<64>, PdReason> {
        if self.role != Role::Prefill || generation == 0 {
            return Err(PdReason::ProtocolMismatch);
        }
        let mut bytes = [0_u8; 64];
        getrandom::fill(&mut bytes).map_err(|_| PdReason::LocalFatal)?;
        if bytes.iter().all(|byte| *byte == 0) {
            return Err(PdReason::LocalFatal);
        }
        self.buffers
            .get(&56)
            .ok_or(PdReason::LocalFatal)?
            .write(0, &bytes)
            .map_err(|_| PdReason::TransferFailed)?;
        let peer = self.peer.lock().map_err(|_| PdReason::LocalFatal)?;
        let remote = self.remote.lock().map_err(|_| PdReason::LocalFatal)?;
        let table = self.table()?;
        let operation = TransferOperation::write(
            table
                .registered_handle(56)
                .map_err(|_| PdReason::LocalFatal)?,
            0,
            peer.as_ref().ok_or(PdReason::ProtocolMismatch)?,
            remote
                .as_ref()
                .ok_or(PdReason::ProtocolMismatch)?
                .region(56)
                .map_err(|_| PdReason::ProtocolMismatch)?,
            0,
            64,
        )
        .map_err(|_| PdReason::TransferFailed)?;
        let batch = self
            .owner
            .submit(vec![operation])
            .map_err(|_| PdReason::TransferFailed)?;
        let terminal = batch
            .wait_terminal(Duration::from_secs(30))
            .map_err(|_| PdReason::TransferFailed)?;
        if terminal.operations.len() != 1
            || terminal.operations[0].transferred_bytes != 64
            || !matches!(
                terminal.operations[0].state,
                crate::mooncake::OperationState::Completed
            )
        {
            return Err(PdReason::TransferFailed);
        }
        Ok(FixedBytes::new(bytes))
    }

    fn verify_and_clear_canary(
        &self,
        generation: u64,
        data: FixedBytes<64>,
    ) -> Result<(), PdReason> {
        if self.role != Role::Decode || generation == 0 {
            return Err(PdReason::ProtocolMismatch);
        }
        let memory = self.buffers.get(&56).ok_or(PdReason::LocalFatal)?;
        if memory.read(0, 64).map_err(|_| PdReason::TransferFailed)? != data.as_bytes() {
            return Err(PdReason::TransferFailed);
        }
        memory.fill(0).map_err(|_| PdReason::LocalFatal)?;
        if memory
            .read(0, 64)
            .map_err(|_| PdReason::LocalFatal)?
            .iter()
            .any(|byte| *byte != 0)
        {
            return Err(PdReason::LocalFatal);
        }
        Ok(())
    }
}

fn build_table(
    descriptors: Vec<NativeRegionDescriptor>,
    device: u32,
    layout_fingerprint: FixedBytes<32>,
) -> Result<(BufferTable, BTreeMap<u16, MemoryBuffer>), ()> {
    if descriptors.len() != 58 {
        return Err(());
    }
    let generation = descriptors.first().ok_or(())?.generation;
    if generation == 0 {
        return Err(());
    }
    let mut regions = Vec::with_capacity(58);
    let mut buffers = BTreeMap::new();
    for (expected_id, descriptor) in (0_u16..58).zip(descriptors) {
        if descriptor.region_id != expected_id || descriptor.generation != generation {
            return Err(());
        }
        let (kind, location, dtype, layout, buffer) = match descriptor.region_id {
            0..=55 => kv_region(&descriptor, device)?,
            56 => host_region(&descriptor, 32, 64, RegionKind::Aux)?,
            57 => host_region(&descriptor, 32, 192, RegionKind::Completion)?,
            _ => return Err(()),
        };
        buffers.insert(descriptor.region_id, buffer);
        regions.push(BufferRegionSpec {
            region_id: descriptor.region_id,
            kind,
            base_address: descriptor.address,
            length_bytes: descriptor.length_bytes,
            location,
            dtype,
            layout,
            owner_generation: generation,
            layout_fingerprint,
        });
    }
    Ok((
        BufferTable::new(regions, generation, device, layout_fingerprint).map_err(|_| ())?,
        buffers,
    ))
}

fn kv_region(
    descriptor: &NativeRegionDescriptor,
    device: u32,
) -> Result<
    (
        RegionKind,
        RegionLocation,
        BufferDType,
        RegionLayout,
        MemoryBuffer,
    ),
    (),
> {
    let page_capacity = *descriptor.shape.first().ok_or(())?;
    if descriptor.device != format!("cuda:{device}")
        || descriptor.dtype != "torch.bfloat16"
        || descriptor.shape != [page_capacity, 64, 8, 128]
        || descriptor.stride_bytes != [KV_PAGE_BYTES, KV_ROW_BYTES, 256, 2]
        || page_capacity == 0
        || descriptor.length_bytes != page_capacity.checked_mul(KV_PAGE_BYTES).ok_or(())?
    {
        return Err(());
    }
    let length = usize::try_from(descriptor.length_bytes).map_err(|_| ())?;
    let kind = if descriptor.region_id < 28 {
        RegionKind::Key {
            layer: descriptor.region_id,
        }
    } else {
        RegionKind::Value {
            layer: descriptor.region_id - 28,
        }
    };
    Ok((
        kind,
        RegionLocation::Device { device },
        BufferDType::BFloat16,
        RegionLayout::kv(page_capacity).map_err(|_| ())?,
        MemoryBuffer::Cuda(
            CudaMemory::borrowed(device, descriptor.address, length).map_err(|_| ())?,
        ),
    ))
}

fn host_region(
    descriptor: &NativeRegionDescriptor,
    slots: u64,
    slot_bytes: u64,
    kind: RegionKind,
) -> Result<
    (
        RegionKind,
        RegionLocation,
        BufferDType,
        RegionLayout,
        MemoryBuffer,
    ),
    (),
> {
    if descriptor.device != "cpu:0"
        || descriptor.dtype != "torch.uint8"
        || descriptor.shape != [slots, slot_bytes]
        || descriptor.stride_bytes != [slot_bytes, 1]
        || descriptor.length_bytes != slots.checked_mul(slot_bytes).ok_or(())?
    {
        return Err(());
    }
    let layout = match kind {
        RegionKind::Aux => RegionLayout::aux(),
        RegionKind::Completion => RegionLayout::completion(),
        _ => return Err(()),
    };
    Ok((
        kind,
        RegionLocation::PinnedHost { numa_node: 0 },
        BufferDType::Bytes,
        layout,
        MemoryBuffer::Pinned(
            PinnedMemory::borrowed(
                descriptor.address,
                usize::try_from(descriptor.length_bytes).map_err(|_| ())?,
            )
            .map_err(|_| ())?,
        ),
    ))
}

const fn role_device(role: Role) -> u32 {
    match role {
        Role::Prefill => 4,
        Role::Decode => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptors(
        role: Role,
        generation: u64,
    ) -> (Vec<NativeRegionDescriptor>, [PinnedMemory; 2]) {
        let aux = PinnedMemory::new(32 * 64).expect("aux owner");
        let completion = PinnedMemory::new(32 * 192).expect("completion owner");
        let device = role_device(role);
        let mut result = (0_u16..56)
            .map(|region_id| NativeRegionDescriptor {
                region_id,
                address: 0x1_0000_0000
                    + u64::from(device) * 0x1_0000_0000
                    + u64::from(region_id) * 0x40_000,
                length_bytes: 2 * KV_PAGE_BYTES,
                device: format!("cuda:{device}"),
                dtype: "torch.bfloat16".to_string(),
                shape: vec![2, 64, 8, 128],
                stride_bytes: vec![KV_PAGE_BYTES, KV_ROW_BYTES, 256, 2],
                generation,
            })
            .collect::<Vec<_>>();
        result.push(NativeRegionDescriptor {
            region_id: 56,
            address: aux.address(),
            length_bytes: 32 * 64,
            device: "cpu:0".to_string(),
            dtype: "torch.uint8".to_string(),
            shape: vec![32, 64],
            stride_bytes: vec![64, 1],
            generation,
        });
        result.push(NativeRegionDescriptor {
            region_id: 57,
            address: completion.address(),
            length_bytes: 32 * 192,
            device: "cpu:0".to_string(),
            dtype: "torch.uint8".to_string(),
            shape: vec![32, 192],
            stride_bytes: vec![192, 1],
            generation,
        });
        (result, [aux, completion])
    }

    fn mock_port(role: Role, port: u16) -> (NativeBootstrapPort, [PinnedMemory; 2]) {
        let (descriptors, owners) = descriptors(role, 7);
        let native = NativeBootstrapPort::new_mock(
            role,
            SocketAddr::from(([127, 0, 0, 1], port)),
            FixedBytes::new([0x55; 32]),
            descriptors,
        )
        .expect("mock native port");
        (native, owners)
    }

    #[test]
    fn native_descriptor_table_rejects_layout_and_generation_mutations() {
        let (valid, _owners) = descriptors(Role::Prefill, 7);
        let (table, buffers) =
            build_table(valid.clone(), 4, FixedBytes::new([0x55; 32])).expect("valid table");
        assert_eq!(table.regions().len(), 58);
        assert_eq!(buffers.len(), 58);
        assert!(
            NativeBootstrapPort::new(
                Role::Prefill,
                SocketAddr::from(([127, 0, 0, 1], 19000)),
                FixedBytes::new([0x55; 32]),
                Vec::new(),
            )
            .is_err()
        );

        let mut cases = Vec::new();
        cases.push(valid[..57].to_vec());
        let mut generation_zero = valid.clone();
        generation_zero[0].generation = 0;
        cases.push(generation_zero);
        let mut wrong_id = valid.clone();
        wrong_id[1].region_id = 0;
        cases.push(wrong_id);
        let mut wrong_generation = valid.clone();
        wrong_generation[1].generation = 8;
        cases.push(wrong_generation);
        let mut wrong_device = valid.clone();
        wrong_device[0].device = "cuda:5".to_string();
        cases.push(wrong_device);
        let mut wrong_kv_layout = valid.clone();
        wrong_kv_layout[0].shape = vec![2, 63, 8, 128];
        cases.push(wrong_kv_layout);
        let mut wrong_host_layout = valid;
        wrong_host_layout[56].stride_bytes = vec![63, 1];
        cases.push(wrong_host_layout);
        assert!(
            cases
                .into_iter()
                .all(|case| { build_table(case, 4, FixedBytes::new([0x55; 32])).is_err() })
        );
    }

    #[test]
    fn mock_native_ports_cover_registration_canary_and_stage_handoff() {
        let (prefill, _prefill_owners) = mock_port(Role::Prefill, 19000);
        let (decode, _decode_owners) = mock_port(Role::Decode, 19001);
        assert_ne!(
            prefill.registration_epoch().expect("prefill epoch"),
            decode.registration_epoch().expect("decode epoch")
        );
        assert!(prefill.table().expect("prefill table").region(57).is_ok());
        assert_eq!(decode.buffers().len(), 58);
        assert_eq!(prefill.registration(), Err(PdReason::ProtocolMismatch));
        assert_eq!(
            decode.open_peer(&RegisterRegions {
                registration_epoch: FixedBytes::new([0; 16]),
                layout_fingerprint: FixedBytes::new([0; 32]),
                mooncake_host: String::new(),
                mooncake_port: 0,
                regions: Vec::new(),
            }),
            Err(PdReason::ProtocolMismatch)
        );

        let registration = decode.registration().expect("decode registration");
        let payload = RegisterRegions {
            registration_epoch: registration.registration_epoch,
            layout_fingerprint: registration.layout_fingerprint,
            mooncake_host: registration.mooncake_host,
            mooncake_port: registration.mooncake_port,
            regions: registration.regions,
        };
        prefill.open_peer(&payload).expect("open mock peer");
        assert_eq!(prefill.open_peer(&payload), Err(PdReason::ProtocolMismatch));
        assert_eq!(decode.produce_canary(1), Err(PdReason::ProtocolMismatch));
        assert_eq!(prefill.produce_canary(0), Err(PdReason::ProtocolMismatch));
        let canary = prefill.produce_canary(1).expect("mock canary");
        decode
            .buffers
            .get(&56)
            .expect("decode aux")
            .write(0, canary.as_bytes())
            .expect("simulate mock copy");
        decode
            .verify_and_clear_canary(1, canary)
            .expect("decode canary");
        assert_eq!(
            prefill.verify_and_clear_canary(1, canary),
            Err(PdReason::ProtocolMismatch)
        );
        assert!(prefill.take_stage_port().is_ok());
        assert_eq!(
            decode.take_stage_port().err(),
            Some(PdReason::ProtocolMismatch)
        );
        assert_eq!(
            prefill.shutdown().expect("prefill native shutdown"),
            RuntimeShutdownOutcome::SafeTerminal
        );
        assert_eq!(
            decode.shutdown().expect("decode native shutdown"),
            RuntimeShutdownOutcome::SafeTerminal
        );
        assert_eq!(
            prefill.shutdown().expect("idempotent prefill shutdown"),
            RuntimeShutdownOutcome::SafeTerminal
        );
        assert!(prefill.table().is_err());
    }
}
