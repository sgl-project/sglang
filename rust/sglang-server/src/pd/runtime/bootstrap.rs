use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::TcpStream;
use tokio::time::timeout;

use crate::pd::config::{PdProfileV1, ProfileError};
use crate::pd::protocol::{
    ClientHello, ControlPayload, DecodedFrame, Direction, DirectionalSession, FixedBytes,
    FrameCodec, FrameError, MessageKind, ProbeAck, ProbeReady, Psk, RegisterRegions,
    RegistrationDecision, Role, ServerHello, SessionError, TranscriptConfirmation,
    derive_session_keys, frame_hash, random_nonce, read_raw_frame, transcript_hash,
    write_raw_frame,
};
use crate::pd::room::{Clock, PdReason, ProcessEpoch, RegistrationEpoch};
use crate::pd::runtime::state::PairReadiness;

#[derive(Clone)]
pub struct RuntimeIdentity {
    pub role: Role,
    pub process_epoch: ProcessEpoch,
    pub registration_epoch: RegistrationEpoch,
    pub model_manifest_digest: FixedBytes<32>,
    pub tokenizer_manifest_digest: FixedBytes<32>,
    pub layout_fingerprint: FixedBytes<32>,
    pub native_abi_digest: FixedBytes<32>,
    pub expected_mooncake_host: String,
    pub allowed_mooncake_ports: BTreeSet<u16>,
    pub profile: Arc<PdProfileV1>,
    profile_digest: FixedBytes<32>,
}

impl RuntimeIdentity {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        role: Role,
        process_epoch: ProcessEpoch,
        registration_epoch: RegistrationEpoch,
        model_manifest_digest: FixedBytes<32>,
        tokenizer_manifest_digest: FixedBytes<32>,
        layout_fingerprint: FixedBytes<32>,
        native_abi_digest: FixedBytes<32>,
        expected_mooncake_host: String,
        allowed_mooncake_ports: BTreeSet<u16>,
        profile: Arc<PdProfileV1>,
    ) -> Result<Self, RuntimeError> {
        if expected_mooncake_host.is_empty()
            || allowed_mooncake_ports.is_empty()
            || allowed_mooncake_ports.contains(&0)
        {
            return Err(RuntimeError::Configuration);
        }
        for digest in [
            model_manifest_digest,
            tokenizer_manifest_digest,
            layout_fingerprint,
            native_abi_digest,
        ] {
            if digest.as_bytes().iter().all(|byte| *byte == 0) {
                return Err(RuntimeError::Configuration);
            }
        }
        let profile_digest = FixedBytes::new(profile.digest()?);
        Ok(Self {
            role,
            process_epoch,
            registration_epoch,
            model_manifest_digest,
            tokenizer_manifest_digest,
            layout_fingerprint,
            native_abi_digest,
            expected_mooncake_host,
            allowed_mooncake_ports,
            profile,
            profile_digest,
        })
    }

    pub const fn profile_digest(&self) -> FixedBytes<32> {
        self.profile_digest
    }

    fn process_epoch_bytes(&self) -> FixedBytes<16> {
        FixedBytes::new(self.process_epoch.as_bytes())
    }

    fn registration_epoch_bytes(&self) -> FixedBytes<16> {
        FixedBytes::new(self.registration_epoch.as_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapRegistration {
    pub registration_epoch: FixedBytes<16>,
    pub layout_fingerprint: FixedBytes<32>,
    pub mooncake_host: String,
    pub mooncake_port: u16,
    pub regions: Vec<crate::pd::protocol::RegionRecord>,
}

impl BootstrapRegistration {
    fn into_payload(self) -> RegisterRegions {
        RegisterRegions {
            registration_epoch: self.registration_epoch,
            layout_fingerprint: self.layout_fingerprint,
            mooncake_host: self.mooncake_host,
            mooncake_port: self.mooncake_port,
            regions: self.regions,
        }
    }
}

pub trait BootstrapPort: Send + Sync + 'static {
    fn registration(&self) -> Result<BootstrapRegistration, PdReason>;
    fn open_peer(&self, registration: &RegisterRegions) -> Result<(), PdReason>;
    fn produce_canary(&self, generation: u64) -> Result<FixedBytes<64>, PdReason>;
    fn verify_and_clear_canary(
        &self,
        generation: u64,
        data: FixedBytes<64>,
    ) -> Result<(), PdReason>;
}

pub struct PairConnection {
    stream: TcpStream,
    session: DirectionalSession,
    readiness: PairReadiness,
    profile: Arc<PdProfileV1>,
    clock: Arc<dyn Clock>,
}

impl PairConnection {
    pub const fn readiness(&self) -> &PairReadiness {
        &self.readiness
    }

    pub async fn send(&mut self, payload: &ControlPayload) -> Result<(), RuntimeError> {
        let now = self.clock.now_unix_ms();
        let deadline = now.saturating_add(deadline_for(payload.kind(), &self.profile));
        let frame = self.session.encode(payload, deadline)?;
        timeout(
            Duration::from_millis(self.profile.deadline_ms.connect_and_hello),
            write_raw_frame(&mut self.stream, &frame),
        )
        .await
        .map_err(|_| RuntimeError::Timeout)??;
        Ok(())
    }

    pub async fn receive_expected(
        &mut self,
        expected: MessageKind,
    ) -> Result<DecodedFrame, RuntimeError> {
        let wait = deadline_for(expected, &self.profile);
        let frame = timeout(
            Duration::from_millis(wait),
            read_raw_frame(&mut self.stream),
        )
        .await
        .map_err(|_| RuntimeError::Timeout)??;
        let decoded = self.session.decode(&frame, self.clock.now_unix_ms())?;
        if decoded.header.kind != expected {
            return Err(RuntimeError::UnexpectedMessage);
        }
        Ok(decoded)
    }
}

pub async fn bootstrap_decode(
    stream: TcpStream,
    identity: RuntimeIdentity,
    psk: &Psk,
    port: Arc<dyn BootstrapPort>,
    clock: Arc<dyn Clock>,
) -> Result<PairConnection, RuntimeError> {
    if identity.role != Role::Decode {
        return Err(RuntimeError::Configuration);
    }
    let (stream, session, peer_process_epoch) =
        client_handshake(stream, &identity, psk, Arc::clone(&clock)).await?;
    let registration = run_blocking({
        let port = Arc::clone(&port);
        move || port.registration()
    })
    .await?;
    if registration.registration_epoch != identity.registration_epoch_bytes()
        || registration.layout_fingerprint != identity.layout_fingerprint
        || registration.mooncake_host != identity.expected_mooncake_host
        || !identity
            .allowed_mooncake_ports
            .contains(&registration.mooncake_port)
    {
        return Err(RuntimeError::Compatibility);
    }
    let destination_epoch = registration.registration_epoch;
    let mut connection = PairConnection {
        stream,
        session,
        readiness: PairReadiness {
            role: identity.role,
            ready: false,
            local_process_epoch: identity.process_epoch_bytes(),
            local_registration_epoch: identity.registration_epoch_bytes(),
            peer_process_epoch,
            peer_registration_epoch: None,
            profile_digest: identity.profile_digest,
            probe_generation: 0,
        },
        profile: Arc::clone(&identity.profile),
        clock,
    };
    connection
        .send(&ControlPayload::RegisterRegions(
            registration.into_payload(),
        ))
        .await?;
    let ack = connection
        .receive_expected(MessageKind::RegisterRegionsAck)
        .await?;
    let ControlPayload::RegisterRegionsAck(ack) = ack.payload else {
        return Err(RuntimeError::UnexpectedMessage);
    };
    if !ack.accepted || ack.registration_epoch != destination_epoch || !ack.reason.is_empty() {
        return Err(RuntimeError::PeerRejected);
    }

    let probe = connection.receive_expected(MessageKind::ProbeReady).await?;
    let ControlPayload::ProbeReady(probe) = probe.payload else {
        return Err(RuntimeError::UnexpectedMessage);
    };
    if probe.registration_epoch != destination_epoch
        || probe.probe_generation == 0
        || probe.aux_slot != 0
    {
        return Err(RuntimeError::Compatibility);
    }
    run_blocking({
        let port = Arc::clone(&port);
        let data = probe.probe_data;
        move || port.verify_and_clear_canary(probe.probe_generation, data)
    })
    .await?;
    connection
        .send(&ControlPayload::ProbeAck(ProbeAck {
            registration_epoch: destination_epoch,
            probe_generation: probe.probe_generation,
            aux_slot: 0,
            accepted: true,
            reason: String::new(),
        }))
        .await?;
    connection.readiness.ready = true;
    connection.readiness.probe_generation = probe.probe_generation;
    tracing::info!(
        role = "decode",
        state = "pair_ready",
        probe_generation = probe.probe_generation,
        "authenticated PD pair completed registration and canary"
    );
    Ok(connection)
}

pub async fn bootstrap_prefill(
    stream: TcpStream,
    identity: RuntimeIdentity,
    psk: &Psk,
    port: Arc<dyn BootstrapPort>,
    clock: Arc<dyn Clock>,
) -> Result<PairConnection, RuntimeError> {
    if identity.role != Role::Prefill {
        return Err(RuntimeError::Configuration);
    }
    let (stream, session, peer_process_epoch) =
        server_handshake(stream, &identity, psk, Arc::clone(&clock)).await?;
    let mut connection = PairConnection {
        stream,
        session,
        readiness: PairReadiness {
            role: identity.role,
            ready: false,
            local_process_epoch: identity.process_epoch_bytes(),
            local_registration_epoch: identity.registration_epoch_bytes(),
            peer_process_epoch,
            peer_registration_epoch: None,
            profile_digest: identity.profile_digest,
            probe_generation: 0,
        },
        profile: Arc::clone(&identity.profile),
        clock,
    };
    let registration = connection
        .receive_expected(MessageKind::RegisterRegions)
        .await?;
    let ControlPayload::RegisterRegions(registration) = registration.payload else {
        return Err(RuntimeError::UnexpectedMessage);
    };
    if registration.layout_fingerprint != identity.layout_fingerprint
        || registration.mooncake_host != identity.expected_mooncake_host
        || !identity
            .allowed_mooncake_ports
            .contains(&registration.mooncake_port)
    {
        return Err(RuntimeError::Compatibility);
    }
    let destination_epoch = registration.registration_epoch;
    run_blocking({
        let port = Arc::clone(&port);
        let registration = registration.clone();
        move || port.open_peer(&registration)
    })
    .await?;
    connection
        .send(&ControlPayload::RegisterRegionsAck(RegistrationDecision {
            registration_epoch: destination_epoch,
            accepted: true,
            reason: String::new(),
        }))
        .await?;

    let probe_generation = 1;
    let probe_data = run_blocking({
        let port = Arc::clone(&port);
        move || port.produce_canary(probe_generation)
    })
    .await?;
    connection
        .send(&ControlPayload::ProbeReady(ProbeReady {
            registration_epoch: destination_epoch,
            probe_generation,
            aux_slot: 0,
            probe_data,
        }))
        .await?;
    let ack = connection.receive_expected(MessageKind::ProbeAck).await?;
    let ControlPayload::ProbeAck(ack) = ack.payload else {
        return Err(RuntimeError::UnexpectedMessage);
    };
    if !ack.accepted
        || !ack.reason.is_empty()
        || ack.registration_epoch != destination_epoch
        || ack.probe_generation != probe_generation
        || ack.aux_slot != 0
    {
        return Err(RuntimeError::PeerRejected);
    }
    connection.readiness.ready = true;
    connection.readiness.peer_registration_epoch = Some(destination_epoch);
    connection.readiness.probe_generation = probe_generation;
    tracing::info!(
        role = "prefill",
        state = "pair_ready",
        probe_generation,
        "authenticated PD pair completed registration and canary"
    );
    Ok(connection)
}

async fn client_handshake(
    mut stream: TcpStream,
    identity: &RuntimeIdentity,
    psk: &Psk,
    clock: Arc<dyn Clock>,
) -> Result<(TcpStream, DirectionalSession, FixedBytes<16>), RuntimeError> {
    let decode_nonce = random_nonce()?;
    let client_payload = ControlPayload::ClientHello(ClientHello {
        role: Role::Decode,
        rank: 0,
        process_epoch: identity.process_epoch_bytes(),
        gpu: 5,
        tp: 1,
        pp: 1,
        dp: 1,
        capabilities: 0,
        profile_digest: identity.profile_digest,
        model_manifest_digest: identity.model_manifest_digest,
        tokenizer_manifest_digest: identity.tokenizer_manifest_digest,
        layout_fingerprint: identity.layout_fingerprint,
        native_abi_digest: identity.native_abi_digest,
        psk_id: FixedBytes::new(psk.id()),
        nonce: decode_nonce,
    });
    let deadline = clock
        .now_unix_ms()
        .saturating_add(identity.profile.deadline_ms.connect_and_hello);
    let client_frame = FrameCodec::encode(
        MessageKind::ClientHello,
        Direction::DecodeToPrefill,
        1,
        deadline,
        &client_payload,
        psk.as_bytes(),
    )?;
    timed_write(&mut stream, &client_frame, &identity.profile).await?;
    let server_frame = timed_read(&mut stream, &identity.profile).await?;
    let decoded = FrameCodec::decode(
        &server_frame,
        Direction::PrefillToDecode,
        1,
        clock.now_unix_ms(),
        psk.as_bytes(),
    )?;
    let ControlPayload::ServerHello(server_hello) = decoded.payload else {
        return Err(RuntimeError::UnexpectedMessage);
    };
    if !server_hello.accepted
        || !server_hello.reason.is_empty()
        || server_hello.client_hello_hash != FixedBytes::new(frame_hash(&client_frame))
        || !compatible_server(identity, psk, &server_hello)
    {
        return Err(RuntimeError::Compatibility);
    }
    let transcript = FixedBytes::new(transcript_hash(&client_frame, &server_frame));
    let keys = derive_session_keys(
        psk,
        decode_nonce,
        server_hello.nonce,
        transcript,
        identity.process_epoch_bytes(),
        server_hello.process_epoch,
    )?;
    let ready = ControlPayload::SessionReady(TranscriptConfirmation {
        transcript_hash: transcript,
    });
    let ready_frame = FrameCodec::encode(
        MessageKind::SessionReady,
        Direction::DecodeToPrefill,
        1,
        deadline,
        &ready,
        &keys.decode_to_prefill,
    )?;
    timed_write(&mut stream, &ready_frame, &identity.profile).await?;
    let ack_frame = timed_read(&mut stream, &identity.profile).await?;
    let ack = FrameCodec::decode(
        &ack_frame,
        Direction::PrefillToDecode,
        1,
        clock.now_unix_ms(),
        &keys.prefill_to_decode,
    )?;
    let ControlPayload::SessionReadyAck(ack) = ack.payload else {
        return Err(RuntimeError::UnexpectedMessage);
    };
    if ack.transcript_hash != transcript {
        return Err(RuntimeError::Compatibility);
    }
    let session = DirectionalSession::decode_side(&keys);
    Ok((stream, session, server_hello.process_epoch))
}

async fn server_handshake(
    mut stream: TcpStream,
    identity: &RuntimeIdentity,
    psk: &Psk,
    clock: Arc<dyn Clock>,
) -> Result<(TcpStream, DirectionalSession, FixedBytes<16>), RuntimeError> {
    let client_frame = timed_read(&mut stream, &identity.profile).await?;
    let decoded = FrameCodec::decode(
        &client_frame,
        Direction::DecodeToPrefill,
        1,
        clock.now_unix_ms(),
        psk.as_bytes(),
    )?;
    let ControlPayload::ClientHello(client_hello) = decoded.payload else {
        return Err(RuntimeError::UnexpectedMessage);
    };
    let accepted = compatible_client(identity, psk, &client_hello);
    let prefill_nonce = random_nonce()?;
    let server_payload = ControlPayload::ServerHello(ServerHello {
        role: Role::Prefill,
        rank: 0,
        process_epoch: identity.process_epoch_bytes(),
        gpu: 4,
        tp: 1,
        pp: 1,
        dp: 1,
        capabilities: 0,
        profile_digest: identity.profile_digest,
        model_manifest_digest: identity.model_manifest_digest,
        tokenizer_manifest_digest: identity.tokenizer_manifest_digest,
        layout_fingerprint: identity.layout_fingerprint,
        native_abi_digest: identity.native_abi_digest,
        psk_id: FixedBytes::new(psk.id()),
        nonce: prefill_nonce,
        client_hello_hash: FixedBytes::new(frame_hash(&client_frame)),
        accepted,
        reason: if accepted {
            String::new()
        } else {
            "PD_PROTOCOL_MISMATCH".into()
        },
    });
    let deadline = clock
        .now_unix_ms()
        .saturating_add(identity.profile.deadline_ms.connect_and_hello);
    let server_frame = FrameCodec::encode(
        MessageKind::ServerHello,
        Direction::PrefillToDecode,
        1,
        deadline,
        &server_payload,
        psk.as_bytes(),
    )?;
    timed_write(&mut stream, &server_frame, &identity.profile).await?;
    if !accepted {
        return Err(RuntimeError::Compatibility);
    }

    let transcript = FixedBytes::new(transcript_hash(&client_frame, &server_frame));
    let keys = derive_session_keys(
        psk,
        client_hello.nonce,
        prefill_nonce,
        transcript,
        client_hello.process_epoch,
        identity.process_epoch_bytes(),
    )?;
    let ready_frame = timed_read(&mut stream, &identity.profile).await?;
    let ready = FrameCodec::decode(
        &ready_frame,
        Direction::DecodeToPrefill,
        1,
        clock.now_unix_ms(),
        &keys.decode_to_prefill,
    )?;
    let ControlPayload::SessionReady(ready) = ready.payload else {
        return Err(RuntimeError::UnexpectedMessage);
    };
    if ready.transcript_hash != transcript {
        return Err(RuntimeError::Compatibility);
    }
    let ack = ControlPayload::SessionReadyAck(TranscriptConfirmation {
        transcript_hash: transcript,
    });
    let ack_frame = FrameCodec::encode(
        MessageKind::SessionReadyAck,
        Direction::PrefillToDecode,
        1,
        deadline,
        &ack,
        &keys.prefill_to_decode,
    )?;
    timed_write(&mut stream, &ack_frame, &identity.profile).await?;
    let session = DirectionalSession::prefill_side(&keys);
    Ok((stream, session, client_hello.process_epoch))
}

fn compatible_client(identity: &RuntimeIdentity, psk: &Psk, hello: &ClientHello) -> bool {
    hello.role == Role::Decode
        && hello.profile_digest == identity.profile_digest
        && hello.model_manifest_digest == identity.model_manifest_digest
        && hello.tokenizer_manifest_digest == identity.tokenizer_manifest_digest
        && hello.layout_fingerprint == identity.layout_fingerprint
        && hello.native_abi_digest == identity.native_abi_digest
        && hello.psk_id == FixedBytes::new(psk.id())
}

fn compatible_server(identity: &RuntimeIdentity, psk: &Psk, hello: &ServerHello) -> bool {
    hello.role == Role::Prefill
        && hello.profile_digest == identity.profile_digest
        && hello.model_manifest_digest == identity.model_manifest_digest
        && hello.tokenizer_manifest_digest == identity.tokenizer_manifest_digest
        && hello.layout_fingerprint == identity.layout_fingerprint
        && hello.native_abi_digest == identity.native_abi_digest
        && hello.psk_id == FixedBytes::new(psk.id())
}

async fn timed_read(
    stream: &mut TcpStream,
    profile: &PdProfileV1,
) -> Result<Vec<u8>, RuntimeError> {
    timeout(
        Duration::from_millis(profile.deadline_ms.connect_and_hello),
        read_raw_frame(stream),
    )
    .await
    .map_err(|_| RuntimeError::Timeout)?
    .map_err(RuntimeError::Session)
}

async fn timed_write(
    stream: &mut TcpStream,
    frame: &[u8],
    profile: &PdProfileV1,
) -> Result<(), RuntimeError> {
    timeout(
        Duration::from_millis(profile.deadline_ms.connect_and_hello),
        write_raw_frame(stream, frame),
    )
    .await
    .map_err(|_| RuntimeError::Timeout)?
    .map_err(RuntimeError::Session)
}

async fn run_blocking<T, F>(operation: F) -> Result<T, RuntimeError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, PdReason> + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|_| RuntimeError::Worker)?
        .map_err(RuntimeError::Bootstrap)
}

fn deadline_for(kind: MessageKind, profile: &PdProfileV1) -> u64 {
    match kind {
        MessageKind::PrepareRoom | MessageKind::PrepareAccepted | MessageKind::PrepareRejected => {
            profile.deadline_ms.room_rendezvous
        }
        MessageKind::DataReady | MessageKind::TransferFailed => profile.deadline_ms.native_transfer,
        MessageKind::TransferComplete | MessageKind::TransferCompleteAck => {
            profile.deadline_ms.completion_ack
        }
        MessageKind::Abort | MessageKind::AbortAck => profile.deadline_ms.abort_ack,
        MessageKind::Ping | MessageKind::Pong => profile.deadline_ms.heartbeat_interval,
        MessageKind::GoAway | MessageKind::GoAwayAck => profile.deadline_ms.room_rendezvous,
        _ => profile.deadline_ms.connect_and_hello,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("PD runtime configuration does not match the frozen profile")]
    Configuration,
    #[error("PD peer compatibility check failed")]
    Compatibility,
    #[error("PD peer rejected bootstrap")]
    PeerRejected,
    #[error("PD peer sent a message outside the required state")]
    UnexpectedMessage,
    #[error("PD control operation exceeded its frozen deadline")]
    Timeout,
    #[error("PD bootstrap worker exited unexpectedly")]
    Worker,
    #[error("PD bootstrap port failed with {0:?}")]
    Bootstrap(PdReason),
    #[error(transparent)]
    Profile(#[from] ProfileError),
    #[error(transparent)]
    Frame(#[from] FrameError),
    #[error(transparent)]
    Session(#[from] SessionError),
    #[error(transparent)]
    Crypto(#[from] crate::pd::protocol::CryptoError),
}
