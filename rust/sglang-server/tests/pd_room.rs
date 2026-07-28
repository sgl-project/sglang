use std::sync::Arc;

use sglang_server::pd::config::PdProfileV1;
use sglang_server::pd::protocol::FixedBytes;
use sglang_server::pd::room::{
    AttemptId, ManualClock, PdReason, ProcessEpoch, RegistrationEpoch, RoomEffect, RoomEvent,
    RoomId, RoomKey, RoomOutcome, RoomRole, RoomSpec, RoomTable,
};

fn process(value: &str) -> ProcessEpoch {
    ProcessEpoch::parse(value).expect("process epoch")
}

fn registration(value: &str) -> RegistrationEpoch {
    RegistrationEpoch::parse(value).expect("registration epoch")
}

fn attempt(index: u128) -> AttemptId {
    let mut bytes = index.to_be_bytes();
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    AttemptId::from_bytes(bytes).expect("attempt UUIDv4")
}

fn digest(byte: u8) -> FixedBytes<32> {
    FixedBytes::new([byte; 32])
}

fn spec(
    process_epoch: ProcessEpoch,
    registration_epoch: RegistrationEpoch,
    room: u64,
    attempt_index: u128,
    generation: u64,
) -> RoomSpec {
    RoomSpec::new(
        RoomId::new(
            RoomKey::new(process_epoch, room, attempt(attempt_index)).expect("room key"),
            generation,
        )
        .expect("room id"),
        digest(0xa1),
        registration_epoch,
    )
    .expect("room spec")
}

fn table(role: RoomRole) -> (RoomTable, Arc<ManualClock>, ProcessEpoch, RegistrationEpoch) {
    let profile = PdProfileV1::load_embedded().expect("profile");
    let clock = Arc::new(ManualClock::new(1_700_000_000_000));
    let process_epoch = process("11111111-1111-4111-8111-111111111111");
    let registration_epoch = registration("33333333-3333-4333-8333-333333333333");
    let table = RoomTable::new(
        role,
        process_epoch,
        registration_epoch,
        &profile,
        clock.clone(),
    )
    .expect("room table");
    (table, clock, process_epoch, registration_epoch)
}

#[test]
fn local_and_peer_arrivals_rendezvous_in_both_orders_and_room_zero_is_valid() {
    let (mut table, _clock, process_epoch, registration_epoch) = table(RoomRole::Prefill);
    let room_zero = spec(process_epoch, registration_epoch, 0, 1, 1);
    let peer_first = spec(process_epoch, registration_epoch, 7, 2, 1);

    assert!(matches!(
        table.observe_local(room_zero.clone()),
        RoomOutcome::Applied(_)
    ));
    assert_eq!(
        table.observe_peer(room_zero),
        RoomOutcome::Applied(vec![RoomEffect::SendPrepareAccepted])
    );

    assert!(matches!(
        table.observe_peer(peer_first.clone()),
        RoomOutcome::Applied(_)
    ));
    assert_eq!(
        table.observe_local(peer_first),
        RoomOutcome::Applied(vec![RoomEffect::SendPrepareAccepted])
    );

    let snapshot = table.snapshot();
    assert_eq!(snapshot.active_rooms, 2);
    assert_eq!(snapshot.tombstones, 0);
}

#[test]
fn prefill_and_decode_success_paths_emit_one_terminal_notification() {
    let (mut prefill, _clock, process_epoch, registration_epoch) = table(RoomRole::Prefill);
    let room = spec(process_epoch, registration_epoch, 0, 3, 1);
    prefill.observe_local(room.clone());
    prefill.observe_peer(room.clone());
    assert_eq!(
        prefill.apply(room.id, RoomEvent::SourceReady),
        RoomOutcome::Applied(vec![RoomEffect::SubmitTransfer])
    );
    prefill.apply(
        room.id,
        RoomEvent::TransferSubmitted {
            plan_digest: digest(0xb1),
        },
    );
    assert_eq!(
        prefill.apply(room.id, RoomEvent::TransferTerminal),
        RoomOutcome::Applied(vec![RoomEffect::SendDataReady])
    );
    let terminal = prefill.apply(
        room.id,
        RoomEvent::TransferComplete {
            plan_digest: digest(0xb1),
        },
    );
    assert!(matches!(
        terminal,
        RoomOutcome::Terminal {
            reason: PdReason::Success,
            duplicate: false,
            ..
        }
    ));
    assert!(matches!(
        prefill.apply(
            room.id,
            RoomEvent::TransferComplete {
                plan_digest: digest(0xb1),
            },
        ),
        RoomOutcome::Terminal {
            reason: PdReason::Success,
            duplicate: true,
            ..
        }
    ));
    assert_eq!(prefill.snapshot().terminal_notifications, 1);

    let (mut decode, _clock, process_epoch, registration_epoch) = table(RoomRole::Decode);
    let room = spec(process_epoch, registration_epoch, 9, 4, 1);
    decode.observe_peer(room.clone());
    assert_eq!(
        decode.observe_local(room.clone()),
        RoomOutcome::Applied(vec![RoomEffect::SendPrepare])
    );
    decode.apply(
        room.id,
        RoomEvent::PrepareAccepted {
            plan_digest: digest(0xb2),
        },
    );
    assert_eq!(
        decode.apply(
            room.id,
            RoomEvent::DataReady {
                plan_digest: digest(0xb2),
            },
        ),
        RoomOutcome::Applied(vec![RoomEffect::SendTransferComplete])
    );
    assert!(matches!(
        decode.apply(
            room.id,
            RoomEvent::TransferCompleteAck {
                plan_digest: digest(0xb2),
            },
        ),
        RoomOutcome::Terminal {
            reason: PdReason::Success,
            duplicate: false,
            ..
        }
    ));
    assert_eq!(decode.snapshot().terminal_notifications, 1);
}

#[test]
fn local_and_peer_abort_paths_emit_the_expected_ack_and_one_terminal_notification() {
    let (mut local_abort, _clock, process_epoch, registration_epoch) = table(RoomRole::Prefill);
    let room = spec(process_epoch, registration_epoch, 10, 40, 1);
    local_abort.observe_local(room.clone());
    assert_eq!(
        local_abort.apply(room.id, RoomEvent::Abort(PdReason::Aborted)),
        RoomOutcome::Terminal {
            reason: PdReason::Aborted,
            duplicate: false,
            effects: vec![
                RoomEffect::SendAbort(PdReason::Aborted),
                RoomEffect::NotifyTerminal(PdReason::Aborted),
            ],
        }
    );

    let (mut peer_abort, _clock, process_epoch, registration_epoch) = table(RoomRole::Decode);
    let room = spec(process_epoch, registration_epoch, 11, 41, 1);
    peer_abort.observe_peer(room.clone());
    assert_eq!(
        peer_abort.apply(room.id, RoomEvent::AbortReceived(PdReason::TransferFailed),),
        RoomOutcome::Terminal {
            reason: PdReason::TransferFailed,
            duplicate: false,
            effects: vec![
                RoomEffect::SendAbortAck(PdReason::TransferFailed),
                RoomEffect::NotifyTerminal(PdReason::TransferFailed),
            ],
        }
    );
    assert!(matches!(
        peer_abort.apply(room.id, RoomEvent::AbortReceived(PdReason::TransferFailed),),
        RoomOutcome::Terminal {
            reason: PdReason::TransferFailed,
            duplicate: true,
            ..
        }
    ));
    assert_eq!(peer_abort.snapshot().terminal_notifications, 1);
}

#[test]
fn stale_identity_digest_mismatch_and_out_of_order_events_fail_closed() {
    let (mut table, _clock, process_epoch, registration_epoch) = table(RoomRole::Prefill);
    let stale_process = process("22222222-2222-4222-8222-222222222222");
    assert_eq!(
        table.observe_local(spec(stale_process, registration_epoch, 1, 5, 1)),
        RoomOutcome::Rejected(PdReason::StaleEpoch)
    );
    let stale_registration = registration("88888888-8888-4888-8888-888888888888");
    assert_eq!(
        table.observe_local(spec(process_epoch, stale_registration, 1, 6, 1)),
        RoomOutcome::Rejected(PdReason::StaleEpoch)
    );

    let room = spec(process_epoch, registration_epoch, 2, 7, 1);
    table.observe_local(room.clone());
    let mut mismatch = room.clone();
    mismatch.request_digest = digest(0xff);
    assert!(matches!(
        table.observe_peer(mismatch),
        RoomOutcome::Terminal {
            reason: PdReason::ProtocolMismatch,
            duplicate: false,
            ..
        }
    ));
    assert!(matches!(
        table.observe_peer(room.clone()),
        RoomOutcome::Terminal {
            reason: PdReason::ProtocolMismatch,
            duplicate: true,
            ..
        }
    ));

    let out_of_order = spec(process_epoch, registration_epoch, 3, 8, 1);
    table.observe_local(out_of_order.clone());
    table.observe_peer(out_of_order.clone());
    assert!(matches!(
        table.apply(
            out_of_order.id,
            RoomEvent::TransferComplete {
                plan_digest: digest(0xb1),
            },
        ),
        RoomOutcome::Terminal {
            reason: PdReason::ProtocolMismatch,
            duplicate: false,
            ..
        }
    ));

    let stale_generation = RoomId::new(out_of_order.id.key, 2).expect("generation");
    assert_eq!(
        table.apply(stale_generation, RoomEvent::PeerLost),
        RoomOutcome::Rejected(PdReason::StaleEpoch)
    );
}

#[test]
fn active_room_capacity_is_exactly_32() {
    let (mut table, _clock, process_epoch, registration_epoch) = table(RoomRole::Decode);

    for index in 0..32_u64 {
        assert!(matches!(
            table.observe_local(spec(
                process_epoch,
                registration_epoch,
                index,
                100 + index as u128,
                1,
            )),
            RoomOutcome::Applied(_)
        ));
    }
    assert_eq!(table.snapshot().active_rooms, 32);
    assert_eq!(
        table.observe_local(spec(process_epoch, registration_epoch, 32, 132, 1)),
        RoomOutcome::Rejected(PdReason::CapacityExhausted)
    );
}

#[test]
fn tombstones_backpressure_at_4096_until_the_full_300_second_retention() {
    let (mut table, clock, process_epoch, registration_epoch) = table(RoomRole::Decode);

    for index in 0..4096_u64 {
        let room = spec(
            process_epoch,
            registration_epoch,
            index,
            1_000 + index as u128,
            1,
        );
        table.observe_local(room.clone());
        table.apply(room.id, RoomEvent::Abort(PdReason::Aborted));
    }
    assert_eq!(table.snapshot().tombstones, 4096);
    let next = spec(process_epoch, registration_epoch, 4096, 9_999, 1);
    assert_eq!(
        table.observe_local(next.clone()),
        RoomOutcome::Rejected(PdReason::CapacityExhausted)
    );

    clock.advance(299_999);
    assert_eq!(
        table.observe_local(next.clone()),
        RoomOutcome::Rejected(PdReason::CapacityExhausted)
    );
    clock.advance(1);
    assert!(matches!(table.observe_local(next), RoomOutcome::Applied(_)));
    assert_eq!(table.snapshot().tombstones, 0);
}

#[test]
fn injected_clock_uses_frozen_rendezvous_and_ack_deadlines() {
    let (mut table, clock, process_epoch, registration_epoch) = table(RoomRole::Decode);
    let waiting = spec(process_epoch, registration_epoch, 1, 10_000, 1);
    table.observe_local(waiting.clone());
    clock.advance_unix(300_000);
    assert!(table.expire_due().is_empty());
    clock.advance(299_999);
    assert!(table.expire_due().is_empty());
    clock.advance(1);
    let expired = table.expire_due();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].1, PdReason::RendezvousTimeout);

    let awaiting_ack = spec(process_epoch, registration_epoch, 2, 10_001, 1);
    table.observe_local(awaiting_ack.clone());
    table.observe_peer(awaiting_ack.clone());
    table.apply(
        awaiting_ack.id,
        RoomEvent::PrepareAccepted {
            plan_digest: digest(0xb1),
        },
    );
    table.apply(
        awaiting_ack.id,
        RoomEvent::DataReady {
            plan_digest: digest(0xb1),
        },
    );
    clock.advance(9_999);
    assert!(table.expire_due().is_empty());
    clock.advance(1);
    let expired = table.expire_due();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].1, PdReason::AckTimeout);
}
