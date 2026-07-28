use std::env;
use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::net::{IpAddr, SocketAddr};
use std::process::{Command, Stdio};
use std::time::Duration;

use sglang_server::mooncake::{
    BatchSnapshot, CudaMemory, EngineError, EngineOwner, HostMemory, MemoryBuffer, MemoryLocation,
    NativeEngineConfig, NativeEngineFactory, OperationState, OwnerConfig, PeerDescriptor,
    PinnedMemory, Region, RemoteRegionDescriptor, ShutdownOutcome, TransferOperation,
};

const BUFFER_LENGTH: usize = 64 * 1024;
const PEER_KILL_LENGTH: usize = 256 * 1024 * 1024;

#[derive(Clone)]
enum GateMemory {
    Host(HostMemory),
    Pinned(PinnedMemory),
    Cuda(CudaMemory),
}

impl GateMemory {
    fn allocate(kind: &str, device: u32, length: usize) -> Result<Self, EngineError> {
        match kind {
            "host" => Ok(Self::Host(HostMemory::new(length)?)),
            "pinned" => Ok(Self::Pinned(PinnedMemory::new(length)?)),
            "cuda" => Ok(Self::Cuda(CudaMemory::new(device, length)?)),
            _ => Err(EngineError::InvalidDescriptor {
                field: "gate.memory_kind",
                detail: kind.into(),
            }),
        }
    }

    fn buffer(&self) -> MemoryBuffer {
        match self {
            Self::Host(memory) => MemoryBuffer::Host(memory.clone()),
            Self::Pinned(memory) => MemoryBuffer::Pinned(memory.clone()),
            Self::Cuda(memory) => MemoryBuffer::Cuda(memory.clone()),
        }
    }

    fn location(&self, device: u32) -> MemoryLocation {
        match self {
            Self::Host(_) | Self::Pinned(_) => {
                if device == 4 {
                    MemoryLocation::Cpu0
                } else {
                    MemoryLocation::Cpu1
                }
            }
            Self::Cuda(_) if device == 4 => MemoryLocation::Cuda4,
            Self::Cuda(_) => MemoryLocation::Cuda5,
        }
    }

    fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), EngineError> {
        match self {
            Self::Host(memory) => memory.write(offset, bytes),
            Self::Pinned(memory) => memory.write(offset, bytes),
            Self::Cuda(memory) => memory.write(offset, bytes),
        }
    }

    fn read(&self, offset: usize, length: usize) -> Result<Vec<u8>, EngineError> {
        match self {
            Self::Host(memory) => memory.read(offset, length),
            Self::Pinned(memory) => memory.read(offset, length),
            Self::Cuda(memory) => memory.read(offset, length),
        }
    }

    fn fill(&self, value: u8) -> Result<(), EngineError> {
        match self {
            Self::Host(memory) => memory.fill(value),
            Self::Pinned(memory) => memory.fill(value),
            Self::Cuda(memory) => memory.fill(value),
        }
    }
}

fn owner_config() -> OwnerConfig {
    OwnerConfig::new(64, Duration::from_millis(1), Duration::from_secs(30))
        .expect("static owner config")
}

fn start_owner(endpoint: SocketAddr, gpu_device: u32) -> Result<EngineOwner, EngineError> {
    let config = NativeEngineConfig::new(endpoint, gpu_device)?;
    EngineOwner::start(owner_config(), NativeEngineFactory::production(config))
}

fn require_completed(snapshot: &BatchSnapshot, expected_bytes: u64) -> Result<(), Box<dyn Error>> {
    if !snapshot.safe_terminal
        || snapshot.operations.len() != 1
        || snapshot.operations[0].state != OperationState::Completed
        || snapshot.operations[0].transferred_bytes != expected_bytes
    {
        return Err(format!(
            "unexpected native terminal result: expected Completed/{expected_bytes}, got {snapshot:?}"
        )
        .into());
    }
    Ok(())
}

fn exact_byte_case(
    kind: &str,
    sender: &EngineOwner,
    receiver: &EngineOwner,
    peer: &sglang_server::mooncake::Peer,
) -> Result<(Region, Region), Box<dyn Error>> {
    let source = GateMemory::allocate(kind, 4, BUFFER_LENGTH)?;
    let destination = GateMemory::allocate(kind, 5, BUFFER_LENGTH)?;
    let source_offset = 113;
    let destination_offset = 257;
    let transfer_length = 8192;
    let payload: Vec<_> = (0..transfer_length)
        .map(|index| ((index * 37 + kind.len()) % 251) as u8)
        .collect();
    source.write(source_offset, &payload)?;
    destination.fill(0xa5)?;

    let source_region = sender.register_region(source.buffer(), source.location(4))?;
    let destination_region =
        receiver.register_region(destination.buffer(), destination.location(5))?;
    let remote = destination_region.remote_descriptor();
    let operation = TransferOperation::write(
        &source_region,
        source_offset as u64,
        peer,
        &remote,
        destination_offset as u64,
        transfer_length as u64,
    )?;
    let batch = sender.submit(vec![operation])?;
    let snapshot = batch.wait_terminal(Duration::from_secs(30))?;
    require_completed(&snapshot, transfer_length as u64)?;

    let actual = destination.read(0, BUFFER_LENGTH)?;
    let mut expected = vec![0xa5; BUFFER_LENGTH];
    expected[destination_offset..destination_offset + transfer_length].copy_from_slice(&payload);
    if actual != expected {
        return Err(format!("{kind} exact-byte comparison failed").into());
    }
    drop(batch);
    println!(
        "PD02_GATE exact kind={kind} bytes={transfer_length} source_offset={source_offset} destination_offset={destination_offset} status=passed"
    );
    Ok((source_region, destination_region))
}

fn concurrent_case(
    sender: &EngineOwner,
    receiver: &EngineOwner,
    peer: &sglang_server::mooncake::Peer,
) -> Result<(Region, Region), Box<dyn Error>> {
    let source = HostMemory::new(BUFFER_LENGTH)?;
    let destination = HostMemory::new(BUFFER_LENGTH)?;
    destination.fill(0x5a)?;
    let source_region =
        sender.register_region(MemoryBuffer::Host(source.clone()), MemoryLocation::Cpu0)?;
    let destination_region = receiver.register_region(
        MemoryBuffer::Host(destination.clone()),
        MemoryLocation::Cpu1,
    )?;
    let remote = destination_region.remote_descriptor();

    let mut expected = vec![0x5a; BUFFER_LENGTH];
    let mut batches = Vec::new();
    for index in 0..4 {
        let local_offset = 1024 + index * 8192;
        let remote_offset = 2048 + index * 8192;
        let length = 4096;
        let payload: Vec<_> = (0..length)
            .map(|byte| ((byte + index * 41) % 253) as u8)
            .collect();
        source.write(local_offset, &payload)?;
        expected[remote_offset..remote_offset + length].copy_from_slice(&payload);
        let operation = TransferOperation::write(
            &source_region,
            local_offset as u64,
            peer,
            &remote,
            remote_offset as u64,
            length as u64,
        )?;
        batches.push(sender.submit(vec![operation])?);
    }
    for batch in &batches {
        require_completed(&batch.wait_terminal(Duration::from_secs(30))?, 4096)?;
    }
    if destination.read(0, BUFFER_LENGTH)? != expected {
        return Err("four-batch concurrent exact-byte comparison failed".into());
    }
    drop(batches);
    println!("PD02_GATE concurrent batches=4 bytes_each=4096 status=passed");
    Ok((source_region, destination_region))
}

fn abort_case(
    sender: &EngineOwner,
    receiver: &EngineOwner,
    peer: &sglang_server::mooncake::Peer,
) -> Result<(Region, Region), Box<dyn Error>> {
    let source = PinnedMemory::new(BUFFER_LENGTH)?;
    let destination = PinnedMemory::new(BUFFER_LENGTH)?;
    let payload = vec![0x3c; 32768];
    source.write(0, &payload)?;
    destination.fill(0)?;
    let source_region =
        sender.register_region(MemoryBuffer::Pinned(source), MemoryLocation::Cpu0)?;
    let destination_region = receiver.register_region(
        MemoryBuffer::Pinned(destination.clone()),
        MemoryLocation::Cpu1,
    )?;
    let operation = TransferOperation::write(
        &source_region,
        0,
        peer,
        &destination_region.remote_descriptor(),
        0,
        payload.len() as u64,
    )?;
    let batch = sender.submit(vec![operation])?;
    batch.abort()?;
    let snapshot = batch.wait_terminal(Duration::from_secs(30))?;
    require_completed(&snapshot, payload.len() as u64)?;
    if !snapshot.logical_aborted || destination.read(0, payload.len())? != payload {
        return Err("logical abort did not drain safely to the exact native result".into());
    }
    drop(batch);
    println!("PD02_GATE logical_abort drain=safe-terminal status=passed");
    Ok((source_region, destination_region))
}

fn peer_receiver(endpoint: SocketAddr) -> Result<(), Box<dyn Error>> {
    let owner = start_owner(endpoint, 5)?;
    let memory = HostMemory::new(PEER_KILL_LENGTH)?;
    let region = owner.register_region(MemoryBuffer::Host(memory), MemoryLocation::Cpu1)?;
    let descriptor = serde_json::to_string(&region.remote_descriptor())?;
    println!(
        "PD02_PEER_READY {} {descriptor}",
        owner.local_peer_descriptor()?.endpoint()
    );
    std::io::stdout().flush()?;
    loop {
        std::thread::park_timeout(Duration::from_secs(60));
    }
}

fn peer_kill_case(
    bind_ip: IpAddr,
    sender_port: u16,
    receiver_port: u16,
) -> Result<(), Box<dyn Error>> {
    let executable = env::current_exe()?;
    let receiver_endpoint = SocketAddr::new(bind_ip, receiver_port);
    let mut child = Command::new(executable)
        .arg("peer-receiver")
        .arg(receiver_endpoint.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or("peer receiver stdout is absent")?;
    let mut ready = None;
    for line in BufReader::new(stdout).lines() {
        let line = line?;
        if let Some(value) = line.strip_prefix("PD02_PEER_READY ") {
            let (endpoint, json) = value
                .split_once(' ')
                .ok_or("peer receiver readiness is malformed")?;
            ready = Some((
                PeerDescriptor::new(endpoint)?,
                serde_json::from_str::<RemoteRegionDescriptor>(json)?,
            ));
            break;
        }
    }
    let (peer_descriptor, remote) = ready.ok_or("peer receiver exited before registration")?;

    let owner = start_owner(SocketAddr::new(bind_ip, sender_port), 4)?;
    let source = HostMemory::new(PEER_KILL_LENGTH)?;
    let source_region = owner.register_region(MemoryBuffer::Host(source), MemoryLocation::Cpu0)?;
    let peer = owner.open_peer(peer_descriptor)?;
    let operation = TransferOperation::write(
        &source_region,
        0,
        &peer,
        &remote,
        0,
        PEER_KILL_LENGTH as u64,
    )?;
    let batch = owner.submit(vec![operation])?;
    if !matches!(
        batch.wait_terminal(Duration::ZERO),
        Err(EngineError::BatchNotTerminal { .. })
    ) {
        let _ = child.kill();
        let _ = child.wait();
        return Err("peer-kill batch reached terminal before the timeout/kill gate".into());
    }
    child.kill()?;
    let _ = child.wait();
    batch.abort()?;

    let gate_result = match batch.wait_terminal(Duration::from_secs(5)) {
        Ok(snapshot) => {
            if !snapshot.safe_terminal
                || snapshot
                    .operations
                    .iter()
                    .any(|progress| !progress.state.is_terminal())
            {
                return Err(format!("peer-kill produced a false safe result: {snapshot:?}").into());
            }
            drop(batch);
            drop(peer);
            drop(source_region);
            if owner.shutdown()? != ShutdownOutcome::SafeTerminal {
                return Err("terminal peer-kill did not cleanly shut down".into());
            }
            "safe-terminal"
        }
        Err(_) => {
            if !matches!(
                owner.shutdown()?,
                ShutdownOutcome::NotSafe { ref batches } if !batches.is_empty()
            ) {
                return Err("non-terminal peer-kill was not reported as NotSafe".into());
            }
            "not-safe"
        }
    };
    println!(
        "PD02_GATE peer_kill submitted_before_kill=true local_wait_timeout=true outcome={gate_result} status=passed"
    );
    Ok(())
}

fn run_gate(bind_ip: IpAddr) -> Result<(), Box<dyn Error>> {
    let sender_endpoint = SocketAddr::new(bind_ip, 19440);
    let receiver_endpoint = SocketAddr::new(bind_ip, 19441);
    let sender = start_owner(sender_endpoint, 4)?;
    let receiver = start_owner(receiver_endpoint, 5)?;
    let peer = sender.open_peer(receiver.local_peer_descriptor()?)?;

    let mut sender_regions = Vec::new();
    let mut receiver_regions = Vec::new();
    for kind in ["host", "pinned", "cuda"] {
        let (source, destination) = exact_byte_case(kind, &sender, &receiver, &peer)?;
        sender_regions.push(source);
        receiver_regions.push(destination);
    }
    let (source, destination) = concurrent_case(&sender, &receiver, &peer)?;
    sender_regions.push(source);
    receiver_regions.push(destination);
    let (source, destination) = abort_case(&sender, &receiver, &peer)?;
    sender_regions.push(source);
    receiver_regions.push(destination);

    drop(peer);
    drop(sender_regions);
    drop(receiver_regions);
    if sender.shutdown()? != ShutdownOutcome::SafeTerminal
        || receiver.shutdown()? != ShutdownOutcome::SafeTerminal
    {
        return Err("normal native owners did not reach safe terminal shutdown".into());
    }
    println!(
        "PD02_GATE transport=rdma metadata=P2PHANDSHAKE auto_discover=false gpu=4,5 hca=mlx5_1,mlx5_2,mlx5_3,mlx5_4 status=passed"
    );
    peer_kill_case(bind_ip, 19442, 19443)
}

fn expect_rdma_init_failure(endpoint: SocketAddr) -> Result<(), Box<dyn Error>> {
    match start_owner(endpoint, 4) {
        Ok(_) => Err("RDMA initialization unexpectedly succeeded without devices".into()),
        Err(error) => {
            println!("PD02_GATE no_rdma fail_closed=true error={error}");
            Ok(())
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    match arguments.next().as_deref() {
        Some("peer-receiver") => {
            let endpoint = arguments
                .next()
                .ok_or("peer-receiver requires an endpoint")?
                .parse()?;
            peer_receiver(endpoint)
        }
        Some("expect-rdma-init-failure") => {
            let endpoint = arguments
                .next()
                .unwrap_or_else(|| "127.0.0.1:19450".into())
                .parse()?;
            expect_rdma_init_failure(endpoint)
        }
        Some(value) => run_gate(value.parse()?),
        None => {
            let bind_ip = env::var("SGLANG_PD02_BIND_IP")
                .map_err(|_| "pass the bind IP or set SGLANG_PD02_BIND_IP")?
                .parse()?;
            run_gate(bind_ip)
        }
    }
}
