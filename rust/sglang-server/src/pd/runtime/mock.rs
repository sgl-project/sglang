use std::sync::{Arc, Mutex};

use crate::mooncake::{
    EngineOwner, HostMemory, MemoryBuffer, MemoryLocation, MockEngineFactory, MockEvent, MockPlan,
    OwnerConfig, Peer, PeerDescriptor, Region, ShutdownOutcome,
};
use crate::pd::protocol::{FixedBytes, RegionRecord, RegisterRegions, Role};
use crate::pd::room::PdReason;
use crate::pd::runtime::bootstrap::{BootstrapPort, BootstrapRegistration, RuntimeIdentity};

pub struct CpuMockBootstrapPort {
    role: Role,
    owner: EngineOwner,
    registration: Option<BootstrapRegistration>,
    canary_memory: Option<HostMemory>,
    region: Mutex<Option<Region>>,
    peer: Mutex<Option<Peer>>,
    events: Arc<Mutex<Vec<MockEvent>>>,
}

impl CpuMockBootstrapPort {
    pub fn new(identity: &RuntimeIdentity) -> Result<Self, PdReason> {
        let factory = MockEngineFactory::new(MockPlan::default());
        let events = factory.events();
        let owner = EngineOwner::start(OwnerConfig::default(), factory)
            .map_err(|_| PdReason::LocalFatal)?;

        let (registration, canary_memory, region) = if identity.role == Role::Decode {
            let memory = HostMemory::new(131_072).map_err(|_| PdReason::LocalFatal)?;
            let region = owner
                .register_region(MemoryBuffer::Host(memory.clone()), MemoryLocation::Cpu1)
                .map_err(|_| PdReason::LocalFatal)?;
            let remote = region.remote_descriptor();
            let region_id = u16::try_from(region.id().get()).map_err(|_| PdReason::LocalFatal)?;
            let endpoint = owner
                .local_peer_descriptor()
                .map_err(|_| PdReason::LocalFatal)?
                .endpoint();
            (
                Some(BootstrapRegistration {
                    registration_epoch: FixedBytes::new(identity.registration_epoch.as_bytes()),
                    layout_fingerprint: identity.layout_fingerprint,
                    mooncake_host: endpoint.ip().to_string(),
                    mooncake_port: endpoint.port(),
                    regions: frozen_mock_regions(region_id, remote.base_address(), remote.length()),
                }),
                Some(memory),
                Some(region),
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            role: identity.role,
            owner,
            registration,
            canary_memory,
            region: Mutex::new(region),
            peer: Mutex::new(None),
            events,
        })
    }

    pub fn event_count(&self) -> usize {
        self.events.lock().map(|events| events.len()).unwrap_or(0)
    }

    pub fn shutdown(&self) -> Result<(), PdReason> {
        self.reset_peer()?;
        if let Some(region) = self.region.lock().map_err(|_| PdReason::LocalFatal)?.take() {
            region.close().map_err(|_| PdReason::LocalFatal)?;
        }
        let outcome = self.owner.shutdown().map_err(|_| PdReason::LocalFatal)?;
        match outcome {
            ShutdownOutcome::SafeTerminal => Ok(()),
            ShutdownOutcome::NotSafe { .. } => Err(PdReason::LocalFatal),
        }
    }

    pub fn reset_peer(&self) -> Result<(), PdReason> {
        if let Some(peer) = self.peer.lock().map_err(|_| PdReason::LocalFatal)?.take() {
            peer.close().map_err(|_| PdReason::LocalFatal)?;
        }
        Ok(())
    }
}

fn frozen_mock_regions(
    canary_region_id: u16,
    canary_address: u64,
    canary_length: u64,
) -> Vec<RegionRecord> {
    (0_u16..58)
        .map(|region_id| {
            let (length_bytes, location) = match region_id {
                0..=55 => (64 * 131_072, "cuda:5"),
                56 => (32 * 64, "cpu:1"),
                57 => (32 * 192, "cpu:1"),
                _ => unreachable!(),
            };
            RegionRecord {
                region_id,
                remote_base_addr: if region_id == canary_region_id
                    && canary_address.is_multiple_of(64)
                    && canary_length == length_bytes
                {
                    canary_address
                } else {
                    0x1000_0000 + u64::from(region_id) * 0x0100_0000
                },
                length_bytes,
                location: location.into(),
            }
        })
        .collect()
}

impl BootstrapPort for CpuMockBootstrapPort {
    fn registration(&self) -> Result<BootstrapRegistration, PdReason> {
        self.registration.clone().ok_or(PdReason::ProtocolMismatch)
    }

    fn open_peer(&self, registration: &RegisterRegions) -> Result<(), PdReason> {
        if self.role != Role::Prefill {
            return Err(PdReason::ProtocolMismatch);
        }
        let descriptor = PeerDescriptor::new(&format!(
            "{}:{}",
            registration.mooncake_host, registration.mooncake_port
        ))
        .map_err(|_| PdReason::ProtocolMismatch)?;
        let peer = self
            .owner
            .open_peer(descriptor)
            .map_err(|_| PdReason::PeerUnavailable)?;
        let mut current = self.peer.lock().map_err(|_| PdReason::LocalFatal)?;
        if current.is_some() {
            return Err(PdReason::ProtocolMismatch);
        }
        *current = Some(peer);
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
        let memory = self.canary_memory.as_ref().ok_or(PdReason::LocalFatal)?;
        memory
            .write(0, data.as_bytes())
            .map_err(|_| PdReason::TransferFailed)?;
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
