use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::Arc;

use sglang_server::pd::config::PdProfileV1;
use sglang_server::pd::protocol::{
    ClientHello, ControlPayload, Direction, FixedBytes, FrameCodec, MessageKind, Psk, RegionRecord,
    RegisterRegions, Role, ServerHello, TranscriptConfirmation, derive_session_keys, random_nonce,
    read_raw_frame, transcript_hash, write_raw_frame,
};
use sglang_server::pd::room::{
    AttemptId, ManualClock, PdReason, ProcessEpoch, RegistrationEpoch, RoomEvent, RoomId, RoomKey,
    RoomOutcome, RoomRole, RoomSpec, RoomTable,
};
use sglang_server::pd::runtime::{
    BootstrapPort, BootstrapRegistration, ConnectionLifecycle, CpuMockBootstrapPort,
    HeartbeatAction, HeartbeatTracker, PairReadiness, PairState, RuntimeIdentity, RuntimeSnapshot,
    bootstrap_decode, bootstrap_prefill,
};
use tokio::net::{TcpListener, TcpStream};

fn identity(role: Role, model_byte: u8) -> RuntimeIdentity {
    let profile = Arc::new(PdProfileV1::load_embedded().expect("profile"));
    let (process, registration) = match role {
        Role::Prefill => (
            ProcessEpoch::parse("22222222-2222-4222-8222-222222222222").expect("P epoch"),
            RegistrationEpoch::parse("88888888-8888-4888-8888-888888888888")
                .expect("P registration"),
        ),
        Role::Decode => (
            ProcessEpoch::parse("11111111-1111-4111-8111-111111111111").expect("D epoch"),
            RegistrationEpoch::parse("33333333-3333-4333-8333-333333333333")
                .expect("D registration"),
        ),
    };
    RuntimeIdentity::new(
        role,
        process,
        registration,
        FixedBytes::new([model_byte; 32]),
        FixedBytes::new([0x22; 32]),
        FixedBytes::new([0x33; 32]),
        FixedBytes::new([0x44; 32]),
        "127.0.0.1".into(),
        BTreeSet::from([19000]),
        profile,
    )
    .expect("runtime identity")
}

#[tokio::test]
async fn authenticated_handshake_register_canary_reaches_pair_ready_on_both_sides() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("control listener");
    let address = listener.local_addr().expect("listener address");
    let psk = Arc::new(test_psk(7));
    let prefill_identity = identity(Role::Prefill, 0x11);
    let decode_identity = identity(Role::Decode, 0x11);
    let prefill_port =
        Arc::new(CpuMockBootstrapPort::new(&prefill_identity).expect("prefill mock port"));
    let decode_port =
        Arc::new(CpuMockBootstrapPort::new(&decode_identity).expect("decode mock port"));
    let prefill_clock = Arc::new(ManualClock::new(1_700_000_000_000));
    let decode_clock = Arc::new(ManualClock::new(1_700_000_000_000));

    let server = tokio::spawn({
        let psk = Arc::clone(&psk);
        let port = Arc::clone(&prefill_port);
        async move {
            let (stream, _) = listener.accept().await.expect("accept decode");
            bootstrap_prefill(stream, prefill_identity, &psk, port, prefill_clock).await
        }
    });
    let stream = TcpStream::connect(address).await.expect("connect prefill");
    let decode = bootstrap_decode(
        stream,
        decode_identity,
        &psk,
        decode_port.clone(),
        decode_clock,
    )
    .await
    .expect("decode PairReady");
    let prefill = server
        .await
        .expect("prefill task")
        .expect("prefill PairReady");

    assert!(decode.readiness().ready);
    assert!(prefill.readiness().ready);
    assert_eq!(
        decode.readiness().profile_digest,
        prefill.readiness().profile_digest
    );
    assert_eq!(decode.readiness().probe_generation, 1);
    assert!(decode.peer_regions().is_none());
    assert_eq!(
        prefill
            .peer_regions()
            .expect("authenticated destination region table")
            .epoch()
            .as_bytes(),
        decode.readiness().local_registration_epoch.into_array()
    );
    assert!(prefill_port.event_count() >= 2);
    assert!(decode_port.event_count() >= 3);

    drop(decode);
    drop(prefill);
    prefill_port.shutdown().expect("prefill shutdown");
    decode_port.shutdown().expect("decode shutdown");
}

#[tokio::test]
async fn hello_digest_mismatch_matrix_and_wrong_psk_never_reach_pair_ready() {
    let server_psk = Arc::new(test_psk(7));
    for mismatch in ["model", "tokenizer", "layout", "native_abi"] {
        let prefill_identity = identity(Role::Prefill, 0x11);
        let mut decode_identity = identity(Role::Decode, 0x11);
        match mismatch {
            "model" => decode_identity.model_manifest_digest = FixedBytes::new([0x99; 32]),
            "tokenizer" => decode_identity.tokenizer_manifest_digest = FixedBytes::new([0x99; 32]),
            "layout" => decode_identity.layout_fingerprint = FixedBytes::new([0x99; 32]),
            "native_abi" => decode_identity.native_abi_digest = FixedBytes::new([0x99; 32]),
            _ => unreachable!("fixed mismatch matrix"),
        }
        let prefill_port =
            Arc::new(CpuMockBootstrapPort::new(&prefill_identity).expect("prefill mock port"));
        let decode_port =
            Arc::new(CpuMockBootstrapPort::new(&decode_identity).expect("decode mock port"));
        let prefill_boundary: Arc<dyn BootstrapPort> = prefill_port.clone();
        let decode_boundary: Arc<dyn BootstrapPort> = decode_port.clone();
        let (prefill_result, decode_result) = run_test_pair(
            prefill_identity,
            decode_identity,
            prefill_boundary,
            decode_boundary,
        )
        .await;
        assert!(
            prefill_result.is_err(),
            "{mismatch} mismatch reached prefill PairReady"
        );
        assert!(
            decode_result.is_err(),
            "{mismatch} mismatch reached decode PairReady"
        );
        prefill_port.shutdown().expect("prefill shutdown");
        decode_port.shutdown().expect("decode shutdown");
    }

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("control listener");
    let address = listener.local_addr().expect("listener address");
    let prefill_identity = identity(Role::Prefill, 0x11);
    let decode_identity = identity(Role::Decode, 0x11);
    let prefill_port =
        Arc::new(CpuMockBootstrapPort::new(&prefill_identity).expect("prefill mock port"));
    let decode_port =
        Arc::new(CpuMockBootstrapPort::new(&decode_identity).expect("decode mock port"));
    let server = tokio::spawn({
        let psk = Arc::clone(&server_psk);
        let port = Arc::clone(&prefill_port);
        async move {
            let (stream, _) = listener.accept().await.expect("accept decode");
            bootstrap_prefill(
                stream,
                prefill_identity,
                &psk,
                port,
                Arc::new(ManualClock::new(1_700_000_000_000)),
            )
            .await
        }
    });
    let stream = TcpStream::connect(address).await.expect("connect prefill");
    assert!(
        bootstrap_decode(
            stream,
            decode_identity,
            &test_psk(8),
            decode_port.clone(),
            Arc::new(ManualClock::new(1_700_000_000_000)),
        )
        .await
        .is_err()
    );
    assert!(server.await.expect("prefill task").is_err());
    prefill_port.shutdown().expect("prefill shutdown");
    decode_port.shutdown().expect("decode shutdown");
}

#[tokio::test]
async fn bad_registration_open_peer_and_canary_results_keep_readiness_false() {
    for failure in [
        BootstrapFailure::WrongEpoch,
        BootstrapFailure::WrongHost,
        BootstrapFailure::WrongPort,
        BootstrapFailure::WrongLayout,
        BootstrapFailure::BadRegion,
        BootstrapFailure::OpenPeer,
        BootstrapFailure::Canary,
    ] {
        let prefill_identity = identity(Role::Prefill, 0x11);
        let decode_identity = identity(Role::Decode, 0x11);
        let mut registration = BootstrapRegistration {
            registration_epoch: FixedBytes::new(decode_identity.registration_epoch.as_bytes()),
            layout_fingerprint: decode_identity.layout_fingerprint,
            mooncake_host: "127.0.0.1".into(),
            mooncake_port: 19000,
            regions: frozen_regions(),
        };
        if failure == BootstrapFailure::WrongEpoch {
            registration.registration_epoch = FixedBytes::new(
                RegistrationEpoch::parse("99999999-9999-4999-8999-999999999999")
                    .expect("wrong epoch")
                    .as_bytes(),
            );
        }
        if failure == BootstrapFailure::WrongHost {
            registration.mooncake_host = "127.0.0.2".into();
        }
        if failure == BootstrapFailure::WrongPort {
            registration.mooncake_port = 19001;
        }
        if failure == BootstrapFailure::WrongLayout {
            registration.layout_fingerprint = FixedBytes::new([0x99; 32]);
        }
        if failure == BootstrapFailure::BadRegion {
            registration.regions[0].length_bytes = 0;
        }
        let prefill_port: Arc<dyn BootstrapPort> = Arc::new(TestBootstrapPort {
            registration: None,
            fail_open: failure == BootstrapFailure::OpenPeer,
            fail_verify: false,
        });
        let decode_port: Arc<dyn BootstrapPort> = Arc::new(TestBootstrapPort {
            registration: Some(registration),
            fail_open: false,
            fail_verify: failure == BootstrapFailure::Canary,
        });
        let (prefill_result, decode_result) =
            run_test_pair(prefill_identity, decode_identity, prefill_port, decode_port).await;
        assert!(
            prefill_result.is_err(),
            "{failure:?} reached prefill PairReady"
        );
        assert!(
            decode_result.is_err(),
            "{failure:?} reached decode PairReady"
        );
    }
}

#[tokio::test]
async fn transcript_mismatch_fails_before_registration() {
    let prefill_identity = identity(Role::Prefill, 0x11);
    let decode_identity = identity(Role::Decode, 0x11);
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("transcript listener");
    let address = listener.local_addr().expect("transcript address");
    let psk = Arc::new(test_psk(7));
    let server = tokio::spawn({
        let psk = Arc::clone(&psk);
        let identity = prefill_identity;
        async move {
            let (stream, _) = listener.accept().await.expect("transcript accept");
            bootstrap_prefill(
                stream,
                identity,
                &psk,
                Arc::new(TestBootstrapPort {
                    registration: None,
                    fail_open: false,
                    fail_verify: false,
                }),
                Arc::new(ManualClock::new(1_700_000_000_000)),
            )
            .await
        }
    });

    let mut stream = TcpStream::connect(address)
        .await
        .expect("transcript connect");
    let nonce = random_nonce().expect("decode nonce");
    let client_payload = ControlPayload::ClientHello(ClientHello {
        role: Role::Decode,
        rank: 0,
        process_epoch: FixedBytes::new(decode_identity.process_epoch.as_bytes()),
        gpu: 5,
        tp: 1,
        pp: 1,
        dp: 1,
        capabilities: 0,
        profile_digest: decode_identity.profile_digest(),
        model_manifest_digest: decode_identity.model_manifest_digest,
        tokenizer_manifest_digest: decode_identity.tokenizer_manifest_digest,
        layout_fingerprint: decode_identity.layout_fingerprint,
        native_abi_digest: decode_identity.native_abi_digest,
        psk_id: FixedBytes::new(psk.id()),
        nonce,
    });
    let deadline = 1_700_000_030_000;
    let client_frame = FrameCodec::encode(
        MessageKind::ClientHello,
        Direction::DecodeToPrefill,
        1,
        deadline,
        &client_payload,
        psk.as_bytes(),
    )
    .expect("ClientHello frame");
    write_raw_frame(&mut stream, &client_frame)
        .await
        .expect("write ClientHello");
    let server_frame = read_raw_frame(&mut stream).await.expect("read ServerHello");
    let server_hello = FrameCodec::decode(
        &server_frame,
        Direction::PrefillToDecode,
        1,
        1_700_000_000_000,
        psk.as_bytes(),
    )
    .expect("decode ServerHello");
    let ControlPayload::ServerHello(ServerHello {
        process_epoch: prefill_epoch,
        nonce: prefill_nonce,
        ..
    }) = server_hello.payload
    else {
        panic!("expected ServerHello");
    };
    let transcript = FixedBytes::new(transcript_hash(&client_frame, &server_frame));
    let keys = derive_session_keys(
        &psk,
        nonce,
        prefill_nonce,
        transcript,
        FixedBytes::new(decode_identity.process_epoch.as_bytes()),
        prefill_epoch,
    )
    .expect("session keys");
    let wrong_ready = ControlPayload::SessionReady(TranscriptConfirmation {
        transcript_hash: FixedBytes::new([0xee; 32]),
    });
    let wrong_frame = FrameCodec::encode(
        MessageKind::SessionReady,
        Direction::DecodeToPrefill,
        1,
        deadline,
        &wrong_ready,
        &keys.decode_to_prefill,
    )
    .expect("wrong SessionReady frame");
    write_raw_frame(&mut stream, &wrong_frame)
        .await
        .expect("write wrong SessionReady");
    drop(stream);

    assert!(server.await.expect("transcript server task").is_err());
}

#[test]
fn heartbeat_requires_valid_pong_and_closes_after_two_missed_periods() {
    let profile = PdProfileV1::load_embedded().expect("profile");
    let clock = Arc::new(ManualClock::new(0));
    let clock_for_tracker: Arc<dyn sglang_server::pd::room::Clock> = clock.clone();
    let mut tracker =
        HeartbeatTracker::new(&profile, clock_for_tracker).expect("heartbeat tracker");

    assert_eq!(tracker.poll(), HeartbeatAction::Wait);
    clock.advance_unix(5_000);
    assert_eq!(tracker.poll(), HeartbeatAction::Wait);
    clock.advance_monotonic(5_000);
    assert_eq!(tracker.poll(), HeartbeatAction::SendPing(1));
    assert!(tracker.on_pong(999).is_err());
    clock.advance(5_000);
    assert_eq!(tracker.poll(), HeartbeatAction::SendPing(2));
    assert_eq!(tracker.consecutive_misses(), 1);
    clock.advance(5_000);
    assert_eq!(tracker.poll(), HeartbeatAction::PeerLost);

    let clock = Arc::new(ManualClock::new(0));
    let clock_for_tracker: Arc<dyn sglang_server::pd::room::Clock> = clock.clone();
    let mut tracker =
        HeartbeatTracker::new(&profile, clock_for_tracker).expect("heartbeat tracker");
    clock.advance(5_000);
    assert_eq!(tracker.poll(), HeartbeatAction::SendPing(1));
    tracker.on_pong(1).expect("valid pong");
    clock.advance(5_000);
    assert_eq!(tracker.poll(), HeartbeatAction::SendPing(2));
    assert_eq!(tracker.consecutive_misses(), 0);
}

#[tokio::test]
async fn pair_connection_drives_heartbeat_goaway_and_two_miss_peer_loss() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("lifecycle listener");
    let address = listener.local_addr().expect("lifecycle address");
    let psk = Arc::new(test_psk(7));
    let prefill_identity = identity(Role::Prefill, 0x11);
    let decode_identity = identity(Role::Decode, 0x11);
    let prefill_port =
        Arc::new(CpuMockBootstrapPort::new(&prefill_identity).expect("prefill port"));
    let decode_port = Arc::new(CpuMockBootstrapPort::new(&decode_identity).expect("decode port"));
    let prefill_clock = Arc::new(ManualClock::new(1_700_000_000_000));
    let decode_clock = Arc::new(ManualClock::new(1_700_000_000_000));

    let server = tokio::spawn({
        let psk = Arc::clone(&psk);
        let port = Arc::clone(&prefill_port);
        let clock = Arc::clone(&prefill_clock);
        async move {
            let (stream, _) = listener.accept().await.expect("accept decode");
            bootstrap_prefill(stream, prefill_identity, &psk, port, clock).await
        }
    });
    let stream = TcpStream::connect(address).await.expect("connect prefill");
    let mut decode = bootstrap_decode(
        stream,
        decode_identity,
        &psk,
        decode_port.clone(),
        decode_clock.clone(),
    )
    .await
    .expect("decode bootstrap");
    let mut prefill = server
        .await
        .expect("prefill bootstrap task")
        .expect("prefill bootstrap");

    prefill_clock.advance(5_000);
    decode_clock.advance(5_000);
    let (prefill_action, decode_action) =
        tokio::join!(prefill.lifecycle_tick(), decode.lifecycle_tick());
    assert_eq!(
        prefill_action.expect("prefill heartbeat"),
        ConnectionLifecycle::PingSent(1)
    );
    assert_eq!(
        decode_action.expect("decode heartbeat"),
        ConnectionLifecycle::PingSent(1)
    );
    for _ in 0..8 {
        let _ = tokio::join!(prefill.lifecycle_tick(), decode.lifecycle_tick());
        if prefill.heartbeat_snapshot().outstanding_ping.is_none()
            && decode.heartbeat_snapshot().outstanding_ping.is_none()
        {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(prefill.heartbeat_snapshot().last_pong_id, Some(1));
    assert_eq!(decode.heartbeat_snapshot().last_pong_id, Some(1));

    prefill.send_goaway(9).await.expect("send GoAway");
    for _ in 0..8 {
        let _ = decode.lifecycle_tick().await;
        let _ = prefill.lifecycle_tick().await;
        if prefill.goaway_acked(9) {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(decode.peer_draining_generation(), Some(9));
    assert!(prefill.goaway_acked(9));

    prefill.send_goaway(10).await.expect("send final GoAway");
    drop(prefill);
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    assert_eq!(
        decode
            .lifecycle_tick()
            .await
            .expect("decode consumes GoAway before EOF"),
        ConnectionLifecycle::PeerDraining(10)
    );
    assert_eq!(decode.peer_draining_generation(), Some(10));
    drop(decode);
    prefill_port.shutdown().expect("prefill shutdown");
    decode_port.shutdown().expect("decode shutdown");

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("miss listener");
    let address = listener.local_addr().expect("miss address");
    let prefill_identity = identity(Role::Prefill, 0x11);
    let decode_identity = identity(Role::Decode, 0x11);
    let prefill_port =
        Arc::new(CpuMockBootstrapPort::new(&prefill_identity).expect("prefill miss port"));
    let decode_port =
        Arc::new(CpuMockBootstrapPort::new(&decode_identity).expect("decode miss port"));
    let prefill_clock = Arc::new(ManualClock::new(1_700_000_000_000));
    let server = tokio::spawn({
        let psk = Arc::clone(&psk);
        let port = Arc::clone(&prefill_port);
        let clock = Arc::clone(&prefill_clock);
        async move {
            let (stream, _) = listener.accept().await.expect("accept miss peer");
            bootstrap_prefill(stream, prefill_identity, &psk, port, clock).await
        }
    });
    let stream = TcpStream::connect(address)
        .await
        .expect("connect miss peer");
    let silent_decode = bootstrap_decode(
        stream,
        decode_identity,
        &psk,
        decode_port.clone(),
        Arc::new(ManualClock::new(1_700_000_000_000)),
    )
    .await
    .expect("silent decode bootstrap");
    let mut prefill = server
        .await
        .expect("miss server task")
        .expect("prefill miss bootstrap");
    prefill_clock.advance(5_000);
    assert_eq!(
        prefill.lifecycle_tick().await.expect("first ping"),
        ConnectionLifecycle::PingSent(1)
    );
    prefill_clock.advance(5_000);
    assert_eq!(
        prefill.lifecycle_tick().await.expect("second ping"),
        ConnectionLifecycle::PingSent(2)
    );
    prefill_clock.advance(5_000);
    assert_eq!(
        prefill.lifecycle_tick().await.expect("peer loss"),
        ConnectionLifecycle::PeerLost
    );

    drop(prefill);
    drop(silent_decode);
    prefill_port.shutdown().expect("prefill miss shutdown");
    decode_port.shutdown().expect("decode miss shutdown");
}

#[test]
fn new_peer_epoch_and_disconnect_terminate_old_rooms_without_overwriting_reason() {
    let identity = identity(Role::Decode, 0x11);
    let clock = Arc::new(ManualClock::new(0));
    let mut rooms = RoomTable::new(
        RoomRole::Decode,
        identity.process_epoch,
        identity.registration_epoch,
        &identity.profile,
        clock,
    )
    .expect("room table");
    let snapshot = RuntimeSnapshot::local_ready(
        Role::Decode,
        identity.process_epoch,
        identity.registration_epoch,
        identity.profile_digest(),
    );
    let mut pair = PairState::new(snapshot);
    let first_peer = FixedBytes::new(
        ProcessEpoch::parse("22222222-2222-4222-8222-222222222222")
            .expect("first peer")
            .as_bytes(),
    );
    let first = PairReadiness {
        role: Role::Decode,
        ready: true,
        local_process_epoch: FixedBytes::new(identity.process_epoch.as_bytes()),
        local_registration_epoch: FixedBytes::new(identity.registration_epoch.as_bytes()),
        peer_process_epoch: first_peer,
        peer_registration_epoch: None,
        profile_digest: identity.profile_digest(),
        probe_generation: 1,
    };
    assert!(
        pair.activate(&first, &mut rooms)
            .expect("first session")
            .is_empty()
    );
    assert_eq!(
        pair.activate(&first, &mut rooms),
        Err(PdReason::ProtocolMismatch)
    );

    let mut attempt_bytes = 77_u128.to_be_bytes();
    attempt_bytes[6] = (attempt_bytes[6] & 0x0f) | 0x40;
    attempt_bytes[8] = (attempt_bytes[8] & 0x3f) | 0x80;
    let attempt = AttemptId::from_bytes(attempt_bytes).expect("attempt");
    let id = RoomId::new(
        RoomKey::new(identity.process_epoch, 0, attempt).expect("key"),
        1,
    )
    .expect("id");
    let spec = RoomSpec::new(id, FixedBytes::new([0xa1; 32]), identity.registration_epoch)
        .expect("room spec");
    rooms.observe_local(spec);

    let second = PairReadiness {
        peer_process_epoch: FixedBytes::new(
            ProcessEpoch::parse("99999999-9999-4999-8999-999999999999")
                .expect("second peer")
                .as_bytes(),
        ),
        probe_generation: 2,
        ..first
    };
    assert_eq!(
        pair.activate(&second, &mut rooms).expect("new peer epoch"),
        vec![id]
    );
    assert!(matches!(
        rooms.apply(id, RoomEvent::PeerLost),
        RoomOutcome::Terminal {
            reason: PdReason::PeerUnavailable,
            duplicate: true,
            ..
        }
    ));
    assert!(pair.disconnect(&mut rooms).is_empty());
    assert_eq!(pair.snapshot().last_reason, Some(PdReason::PeerUnavailable));
    assert_eq!(pair.snapshot().session_count, 2);
    assert_eq!(
        pair.activate(&second, &mut rooms),
        Err(PdReason::ProtocolMismatch),
        "a disconnected peer epoch must not reclaim the current session"
    );
}

fn test_psk(byte: u8) -> Psk {
    let path = std::env::temp_dir().join(format!("sglang-pd-runtime-{}.psk", uuid::Uuid::new_v4()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true).mode(0o400);
    let mut file = options.open(&path).expect("create runtime test PSK");
    file.write_all(&[byte; 32]).expect("write runtime test PSK");
    drop(file);
    let psk = Psk::load(&path).expect("load runtime test PSK");
    fs::remove_file(path).expect("remove runtime test PSK");
    psk
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BootstrapFailure {
    WrongEpoch,
    WrongHost,
    WrongPort,
    WrongLayout,
    BadRegion,
    OpenPeer,
    Canary,
}

struct TestBootstrapPort {
    registration: Option<BootstrapRegistration>,
    fail_open: bool,
    fail_verify: bool,
}

fn frozen_regions() -> Vec<RegionRecord> {
    (0_u16..58)
        .map(|region_id| {
            let (length_bytes, location) = match region_id {
                0..=55 => (131_072, "cuda:5"),
                56 => (32 * 64, "cpu:1"),
                57 => (32 * 192, "cpu:1"),
                _ => unreachable!(),
            };
            RegionRecord {
                region_id,
                remote_base_addr: 0x1000_0000 + u64::from(region_id) * 0x0100_0000,
                length_bytes,
                location: location.into(),
            }
        })
        .collect()
}

impl BootstrapPort for TestBootstrapPort {
    fn registration(&self) -> Result<BootstrapRegistration, PdReason> {
        self.registration.clone().ok_or(PdReason::ProtocolMismatch)
    }

    fn open_peer(&self, _registration: &RegisterRegions) -> Result<(), PdReason> {
        if self.fail_open {
            Err(PdReason::PeerUnavailable)
        } else {
            Ok(())
        }
    }

    fn produce_canary(&self, _generation: u64) -> Result<FixedBytes<64>, PdReason> {
        Ok(FixedBytes::new([0x80; 64]))
    }

    fn verify_and_clear_canary(
        &self,
        _generation: u64,
        _data: FixedBytes<64>,
    ) -> Result<(), PdReason> {
        if self.fail_verify {
            Err(PdReason::TransferFailed)
        } else {
            Ok(())
        }
    }
}

async fn run_test_pair(
    prefill_identity: RuntimeIdentity,
    decode_identity: RuntimeIdentity,
    prefill_port: Arc<dyn BootstrapPort>,
    decode_port: Arc<dyn BootstrapPort>,
) -> (
    Result<sglang_server::pd::runtime::PairConnection, sglang_server::pd::runtime::RuntimeError>,
    Result<sglang_server::pd::runtime::PairConnection, sglang_server::pd::runtime::RuntimeError>,
) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener");
    let address = listener.local_addr().expect("test address");
    let psk = Arc::new(test_psk(7));
    let server = tokio::spawn({
        let psk = Arc::clone(&psk);
        async move {
            let (stream, _) = listener.accept().await.expect("test accept");
            bootstrap_prefill(
                stream,
                prefill_identity,
                &psk,
                prefill_port,
                Arc::new(ManualClock::new(1_700_000_000_000)),
            )
            .await
        }
    });
    let stream = TcpStream::connect(address).await.expect("test connect");
    let decode = bootstrap_decode(
        stream,
        decode_identity,
        &psk,
        decode_port,
        Arc::new(ManualClock::new(1_700_000_000_000)),
    )
    .await;
    let prefill = server.await.expect("test server task");
    (prefill, decode)
}
