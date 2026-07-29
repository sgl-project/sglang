use std::collections::BTreeSet;
use std::sync::Arc;

use sglang_server::pd::config::PdProfileV1;
use sglang_server::pd::protocol::{FixedBytes, Role};
use sglang_server::pd::room::{AttemptId, ManualClock, PdReason, ProcessEpoch, RegistrationEpoch};
use sglang_server::pd::runtime::{
    FatalPublish, FatalSource, PairReadiness, RuntimeIdentity, RuntimeLifecycle,
    RuntimeShutdownOutcome, ShutdownPhase,
};
use sglang_server::pd::transport::{
    PdTransportCore, ReceiverCreateInput, SenderChunk, SenderCreateInput, TerminalEvent,
    TransportError,
};

fn epoch(fill: u8) -> ProcessEpoch {
    let mut bytes = [fill; 16];
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    ProcessEpoch::from_bytes(bytes).expect("process epoch")
}

fn registration(fill: u8) -> RegistrationEpoch {
    let mut bytes = [fill; 16];
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    RegistrationEpoch::from_bytes(bytes).expect("registration epoch")
}

fn identity(role: Role, fill: u8) -> RuntimeIdentity {
    RuntimeIdentity::new(
        role,
        epoch(fill),
        registration(fill.wrapping_add(1)),
        FixedBytes::new([0x11; 32]),
        FixedBytes::new([0x22; 32]),
        FixedBytes::new([0x33; 32]),
        FixedBytes::new([0x44; 32]),
        "127.0.0.1".to_string(),
        BTreeSet::from([8998]),
        Arc::new(PdProfileV1::load_embedded().expect("profile")),
    )
    .expect("identity")
}

fn pair_ready(local: &RuntimeIdentity, peer: &RuntimeIdentity, generation: u64) -> PairReadiness {
    PairReadiness {
        role: local.role,
        ready: true,
        local_process_epoch: FixedBytes::new(local.process_epoch.as_bytes()),
        local_registration_epoch: FixedBytes::new(local.registration_epoch.as_bytes()),
        peer_process_epoch: FixedBytes::new(peer.process_epoch.as_bytes()),
        peer_registration_epoch: Some(FixedBytes::new(peer.registration_epoch.as_bytes())),
        profile_digest: local.profile_digest(),
        probe_generation: generation,
    }
}

fn started(local: RuntimeIdentity, peer: &RuntimeIdentity) -> (PdTransportCore, Arc<ManualClock>) {
    let clock = Arc::new(ManualClock::new(1_000));
    let readiness = pair_ready(&local, peer, 1);
    let mut core =
        PdTransportCore::new(local, Arc::clone(&clock)).expect("construct transport core");
    core.start_local(58).expect("local registration");
    core.activate_pair(readiness, 58, true)
        .expect("authenticated pair");
    (core, clock)
}

#[test]
fn pair_ready_requires_complete_registration_and_canary() {
    let decode = identity(Role::Decode, 0x20);
    let prefill = identity(Role::Prefill, 0x30);
    let clock = Arc::new(ManualClock::new(1_000));
    let mut missing =
        PdTransportCore::new(decode.clone(), Arc::clone(&clock)).expect("transport core");
    assert_eq!(
        missing.start_local(57),
        Err(TransportError::LocalFatal(PdReason::ProtocolMismatch))
    );
    let snapshot = missing.readiness().snapshot();
    assert_eq!(snapshot.runtime.lifecycle, RuntimeLifecycle::Fatal);
    assert!(!snapshot.accepting_rooms);

    let mut failed_canary =
        PdTransportCore::new(decode.clone(), Arc::clone(&clock)).expect("transport core");
    failed_canary.start_local(58).expect("local ready");
    assert_eq!(
        failed_canary.activate_pair(pair_ready(&decode, &prefill, 1), 58, false),
        Err(TransportError::LocalFatal(PdReason::ProtocolMismatch))
    );
    assert!(!failed_canary.readiness().snapshot().accepting_rooms);

    let (ready, _) = started(decode, &prefill);
    let snapshot = ready.readiness().snapshot();
    assert_eq!(snapshot.runtime.lifecycle, RuntimeLifecycle::PairReady);
    assert!(snapshot.accepting_rooms);
}

#[test]
fn gateway_bootstrap_allowlist_is_distinct_from_mooncake_endpoint() {
    let decode = identity(Role::Decode, 0x20);
    let clock = Arc::new(ManualClock::new(1_000));
    let mut core = PdTransportCore::new(decode, clock).expect("transport core");
    core.configure_gateway_bootstrap("prefill.internal".to_string(), BTreeSet::from([8998]))
        .expect("gateway bootstrap allowlist");
    let snapshot = core.readiness().snapshot();
    assert_eq!(snapshot.expected_bootstrap_host, "prefill.internal");
    assert_eq!(snapshot.allowed_bootstrap_ports, BTreeSet::from([8998]));
    core.start_local(58).expect("local ready");
    assert_eq!(
        core.configure_gateway_bootstrap("attacker.invalid".to_string(), BTreeSet::from([65535]),),
        Err(TransportError::InvalidTransition)
    );
    let snapshot = core.readiness().snapshot();
    assert_eq!(snapshot.expected_bootstrap_host, "prefill.internal");
    assert_eq!(snapshot.allowed_bootstrap_ports, BTreeSet::from([8998]));
}

#[test]
fn generation_role_and_epoch_bound_handles_fail_closed_after_reuse() {
    let decode = identity(Role::Decode, 0x20);
    let prefill = identity(Role::Prefill, 0x30);
    let (mut core, _) = started(prefill.clone(), &decode);
    let request = SenderCreateInput {
        decode_process_epoch: decode.process_epoch,
        bootstrap_room: 0,
        attempt_id: AttemptId::random(),
        request_digest: FixedBytes::new([0xa1; 32]),
    };
    let first = core.sender_create(request.clone()).expect("first handle");
    core.abort_many(&[first], PdReason::Aborted)
        .expect("batch shape")[0]
        .as_ref()
        .expect("abort");
    core.clear_many(&[first]).expect("batch shape")[0]
        .as_ref()
        .expect("clear");
    let second = core
        .sender_create(SenderCreateInput {
            attempt_id: AttemptId::random(),
            ..request
        })
        .expect("reused slot");
    assert_ne!(first.raw(), second.raw());
    assert_eq!(
        core.poll_many(&[first]).expect("batch shape")[0],
        Err(TransportError::StaleHandle)
    );

    assert_eq!(
        core.receiver_create_many(&[ReceiverCreateInput {
            bootstrap_room: 1,
            attempt_id: AttemptId::random(),
            request_digest: FixedBytes::new([0xa2; 32]),
        }]),
        Err(TransportError::WrongRole)
    );

    let other_prefill = identity(Role::Prefill, 0x40);
    let (mut other, _) = started(other_prefill, &decode);
    assert_eq!(
        other.poll_many(&[second]).expect("batch shape")[0],
        Err(TransportError::StaleHandle)
    );
}

#[test]
fn room_generation_restarts_with_each_authenticated_peer_session() {
    let first_decode = identity(Role::Decode, 0x20);
    let restarted_decode = identity(Role::Decode, 0x40);
    let prefill = identity(Role::Prefill, 0x30);
    let (mut core, _) = started(prefill.clone(), &first_decode);
    let first = core
        .sender_create(SenderCreateInput {
            decode_process_epoch: first_decode.process_epoch,
            bootstrap_room: 0,
            attempt_id: AttemptId::random(),
            request_digest: FixedBytes::new([0xa1; 32]),
        })
        .expect("first-session sender");
    assert_eq!(
        core.room_context(first)
            .expect("first-session room")
            .room
            .generation,
        1
    );

    core.peer_lost();
    core.activate_pair(pair_ready(&prefill, &restarted_decode, 2), 58, true)
        .expect("replacement authenticated pair");
    let after_restart = core
        .sender_create(SenderCreateInput {
            decode_process_epoch: restarted_decode.process_epoch,
            bootstrap_room: 1,
            attempt_id: AttemptId::random(),
            request_digest: FixedBytes::new([0xa2; 32]),
        })
        .expect("replacement-session sender");
    assert_eq!(
        core.room_context(after_restart)
            .expect("replacement-session room")
            .room
            .generation,
        1
    );
}

#[test]
fn destination_terminal_first_token_is_consumed_exactly_once() {
    let decode = identity(Role::Decode, 0x20);
    let prefill = identity(Role::Prefill, 0x30);
    let (mut core, clock) = started(decode.clone(), &prefill);
    let handle = core
        .receiver_create_many(&[ReceiverCreateInput {
            bootstrap_room: i64::MAX as u64,
            attempt_id: AttemptId::random(),
            request_digest: FixedBytes::new([0xa1; 32]),
        }])
        .expect("batch shape")
        .remove(0)
        .expect("receiver handle");
    core.receiver_prepare_many(&[handle]).expect("batch shape")[0]
        .as_ref()
        .expect("prepare");
    clock.advance_monotonic(9);
    core.record_terminal(TerminalEvent {
        handle,
        reason: PdReason::Success,
        first_token_id: Some(42),
        transfer_bytes: 131_072,
    })
    .expect("terminal");
    assert_eq!(core.readiness().snapshot().runtime.active_rooms, 0);

    let first = core
        .poll_many(&[handle])
        .expect("batch shape")
        .remove(0)
        .expect("first poll");
    assert_eq!(first.status as u8, 4);
    assert_eq!(first.reason, PdReason::Success);
    assert_eq!(first.terminal_generation, handle.generation());
    assert_eq!(first.first_token_id, Some(42));
    assert!(!first.first_token_consumed);
    assert_eq!(first.transfer_bytes, 131_072);
    assert_eq!(first.transfer_latency_ms, 9);

    let second = core
        .poll_many(&[handle])
        .expect("batch shape")
        .remove(0)
        .expect("second poll");
    assert_eq!(second.first_token_id, None);
    assert!(second.first_token_consumed);

    core.abort_many(&[handle], PdReason::Aborted)
        .expect("batch shape")
        .remove(0)
        .expect("late abort only observes the first terminal");
    let after_abort = core
        .poll_many(&[handle])
        .expect("batch shape")
        .remove(0)
        .expect("terminal remains readable");
    assert_eq!(after_abort.reason, PdReason::Success);
    assert_eq!(after_abort.first_token_id, None);
}

#[test]
fn peer_loss_and_fatal_converge_handles_without_overwriting_first_terminal() {
    let decode = identity(Role::Decode, 0x20);
    let prefill = identity(Role::Prefill, 0x30);
    let (mut core, _) = started(decode, &prefill);
    let create = |index: u8| ReceiverCreateInput {
        bootstrap_room: u64::from(index),
        attempt_id: AttemptId::random(),
        request_digest: FixedBytes::new([index; 32]),
    };
    let mut handles = core
        .receiver_create_many(&[create(1), create(2)])
        .expect("batch shape")
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("receiver handles");
    core.receiver_prepare_many(&handles)
        .expect("batch shape")
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("prepared");
    let already_terminal = handles.remove(0);
    core.record_terminal(TerminalEvent {
        handle: already_terminal,
        reason: PdReason::TransferFailed,
        first_token_id: None,
        transfer_bytes: 0,
    })
    .expect("first terminal");

    let terminated = core.peer_lost();
    assert_eq!(terminated, handles);
    let snapshot = core.readiness().snapshot();
    assert_eq!(snapshot.runtime.lifecycle, RuntimeLifecycle::LocalReady);
    assert_eq!(snapshot.runtime.reconnect_generation, 1);
    assert!(!snapshot.accepting_rooms);
    assert_eq!(snapshot.runtime.active_rooms, 0);
    assert_eq!(
        core.poll_many(&[already_terminal])
            .expect("batch shape")
            .remove(0)
            .expect("old terminal")
            .reason,
        PdReason::TransferFailed
    );
    assert_eq!(
        core.poll_many(&handles)
            .expect("batch shape")
            .remove(0)
            .expect("peer terminal")
            .reason,
        PdReason::PeerUnavailable
    );

    let FatalPublish::First(first) =
        core.publish_fatal(FatalSource::WorkerExit, PdReason::LocalFatal)
    else {
        panic!("worker fatal did not win");
    };
    assert_eq!(first.generation, 1);
    let FatalPublish::Duplicate(duplicate) = core.publish_fatal(
        FatalSource::QuarantineHardDeadline,
        PdReason::TransferTimeout,
    ) else {
        panic!("later fatal source overwrote the first");
    };
    assert_eq!(duplicate, first);
    let snapshot = core.readiness().snapshot();
    assert_eq!(snapshot.runtime.lifecycle, RuntimeLifecycle::Fatal);
    assert_eq!(snapshot.runtime.fatal, Some(first));
    assert_eq!(snapshot.runtime.fatal_duplicate_sources, 1);
    assert!(!snapshot.accepting_rooms);
}

#[test]
fn sender_batch_path_is_bounded_and_tracks_typed_terminal_results() {
    let decode = identity(Role::Decode, 0x20);
    let prefill = identity(Role::Prefill, 0x30);
    let (mut core, _) = started(prefill, &decode);
    let mut handles = Vec::new();
    for room in 0..8 {
        handles.push(
            core.sender_create(SenderCreateInput {
                decode_process_epoch: decode.process_epoch,
                bootstrap_room: room,
                attempt_id: AttemptId::random(),
                request_digest: FixedBytes::new([room as u8 + 1; 32]),
            })
            .expect("sender"),
        );
    }
    assert!(
        core.sender_init_many(&handles)
            .expect("batch shape")
            .iter()
            .all(Result::is_ok)
    );
    let chunks: Vec<_> = handles
        .iter()
        .copied()
        .map(|handle| SenderChunk {
            handle,
            transfer_bytes: 64,
        })
        .collect();
    assert!(
        core.sender_send_chunks(&chunks)
            .expect("batch shape")
            .iter()
            .all(Result::is_ok)
    );
    assert_eq!(
        core.poll_many(&[handles[0]; 9]),
        Err(TransportError::InvalidBatch)
    );
    core.abort_many(&handles, PdReason::PeerUnavailable)
        .expect("batch shape");
    let terminal = core.poll_many(&handles).expect("poll");
    assert!(terminal.iter().all(|result| {
        result.as_ref().is_ok_and(|result| {
            result.status as u8 == 0
                && result.reason == PdReason::PeerUnavailable
                && result.retryable
        })
    }));
}

#[test]
fn graceful_shutdown_aborts_only_active_handles_and_is_idempotent() {
    let decode = identity(Role::Decode, 0x20);
    let prefill = identity(Role::Prefill, 0x30);
    let (mut core, _) = started(prefill, &decode);
    let create = |room: u64, digest: u8| SenderCreateInput {
        decode_process_epoch: decode.process_epoch,
        bootstrap_room: room,
        attempt_id: AttemptId::random(),
        request_digest: FixedBytes::new([digest; 32]),
    };
    let completed = core
        .sender_create(create(1, 0xa1))
        .expect("completed handle");
    let active = core.sender_create(create(2, 0xa2)).expect("active handle");
    core.abort_many(&[completed], PdReason::TransferFailed)
        .expect("completed batch")
        .remove(0)
        .expect("completed terminal");

    assert_eq!(
        core.shutdown().expect("graceful shutdown"),
        RuntimeShutdownOutcome::SafeTerminal
    );
    let snapshot = core.readiness().snapshot();
    assert_eq!(snapshot.runtime.lifecycle, RuntimeLifecycle::Stopped);
    assert_eq!(snapshot.runtime.shutdown_phase, ShutdownPhase::Stopped);
    assert_eq!(
        snapshot.runtime.shutdown_outcome,
        Some(RuntimeShutdownOutcome::SafeTerminal)
    );
    assert!(!snapshot.accepting_rooms);
    assert_eq!(
        core.poll_many(&[completed])
            .expect("completed batch")
            .remove(0)
            .expect("completed result")
            .reason,
        PdReason::TransferFailed
    );
    assert_eq!(
        core.poll_many(&[active])
            .expect("active batch")
            .remove(0)
            .expect("aborted result")
            .reason,
        PdReason::Aborted
    );
    assert_eq!(
        core.shutdown().expect("duplicate shutdown"),
        RuntimeShutdownOutcome::SafeTerminal
    );
}

#[test]
fn shutdown_before_local_registration_is_fatal_unsafe_and_idempotent() {
    let decode = identity(Role::Decode, 0x20);
    let clock = Arc::new(ManualClock::new(1_000));
    let mut core = PdTransportCore::new(decode, clock).expect("transport core");

    assert_eq!(
        core.shutdown().expect("fatal shutdown"),
        RuntimeShutdownOutcome::FatalUnsafe
    );
    let snapshot = core.readiness().snapshot();
    assert_eq!(snapshot.runtime.lifecycle, RuntimeLifecycle::Stopped);
    assert_eq!(
        snapshot.runtime.shutdown_outcome,
        Some(RuntimeShutdownOutcome::FatalUnsafe)
    );
    assert_eq!(
        snapshot.runtime.fatal.map(|record| record.source),
        Some(FatalSource::ShutdownUnsafe)
    );
    assert_eq!(
        core.shutdown().expect("duplicate fatal shutdown"),
        RuntimeShutdownOutcome::FatalUnsafe
    );
}
