use sglang_server::mooncake::EngineError;
use sglang_server::pd::buffer::BufferError;
use sglang_server::pd::room::PdReason;
use sglang_server::pd::runtime::{
    FailureClass, FailureScope, FatalPublish, FatalSource, FirstFatal, RuntimeError,
    RuntimeLifecycle, RuntimeShutdownOutcome, ShutdownMode, ShutdownPhase, ShutdownTracker,
};
use sglang_server::pd::transport::TransportError;

#[test]
fn lifecycle_transition_table_matches_the_frozen_process_fsm() {
    use RuntimeLifecycle::{Draining, Fatal, LocalReady, PairReady, Starting, Stopped};

    let states = [Starting, LocalReady, PairReady, Draining, Fatal, Stopped];
    let legal = [
        (Starting, LocalReady),
        (Starting, Fatal),
        (LocalReady, PairReady),
        (LocalReady, Draining),
        (LocalReady, Fatal),
        (PairReady, LocalReady),
        (PairReady, Draining),
        (PairReady, Fatal),
        (Draining, Stopped),
        (Draining, Fatal),
        (Fatal, Stopped),
    ];

    for from in states {
        for to in states {
            assert_eq!(
                from.can_transition_to(to),
                legal.contains(&(from, to)),
                "unexpected lifecycle edge {from:?} -> {to:?}"
            );
        }
    }
}

#[test]
fn fatal_channel_is_first_wins_and_never_overwrites_the_public_reason() {
    let mut fatal = FirstFatal::new();

    let first = fatal.publish(FatalSource::WorkerExit, PdReason::LocalFatal);
    let FatalPublish::First(record) = first else {
        panic!("first fatal source did not win");
    };
    assert_eq!(record.generation, 1);
    assert_eq!(record.source, FatalSource::WorkerExit);
    assert_eq!(record.reason, PdReason::LocalFatal);

    let duplicate = fatal.publish(
        FatalSource::QuarantineHardDeadline,
        PdReason::TransferTimeout,
    );
    let FatalPublish::Duplicate(record) = duplicate else {
        panic!("later fatal source was not classified as a duplicate");
    };
    assert_eq!(record.generation, 1);
    assert_eq!(record.source, FatalSource::WorkerExit);
    assert_eq!(record.reason, PdReason::LocalFatal);

    let snapshot = fatal.snapshot();
    assert_eq!(snapshot.first, Some(record));
    assert_eq!(snapshot.duplicate_sources, 1);
}

#[test]
fn shutdown_tracker_enforces_reverse_dependency_order_and_is_idempotent() {
    let mut shutdown = ShutdownTracker::new();
    let generation = shutdown.begin(ShutdownMode::Graceful);
    assert_eq!(generation, 1);
    assert_eq!(shutdown.begin(ShutdownMode::Fatal), generation);
    assert_eq!(shutdown.mode(), Some(ShutdownMode::Graceful));
    assert_eq!(shutdown.phase(), ShutdownPhase::ReadinessDown);

    assert!(
        shutdown.advance(ShutdownPhase::NativeSafety).is_err(),
        "shutdown skipped GoAway, admission stop, drain, and Abort"
    );

    for phase in [
        ShutdownPhase::GoAway,
        ShutdownPhase::StopAccepting,
        ShutdownPhase::DrainingRooms,
        ShutdownPhase::AbortingRooms,
        ShutdownPhase::NativeSafety,
        ShutdownPhase::SchedulerRelease,
        ShutdownPhase::WorkerJoin,
        ShutdownPhase::EngineQuiesce,
        ShutdownPhase::ConnectionClose,
        ShutdownPhase::RegionUnregister,
        ShutdownPhase::EngineDestroy,
    ] {
        shutdown.advance(phase).expect("ordered shutdown phase");
    }
    let outcome = shutdown
        .complete(RuntimeShutdownOutcome::SafeTerminal)
        .expect("safe shutdown terminal");
    assert_eq!(outcome, RuntimeShutdownOutcome::SafeTerminal);
    assert_eq!(
        shutdown
            .complete(RuntimeShutdownOutcome::FatalUnsafe)
            .expect("repeated shutdown returns the first terminal"),
        RuntimeShutdownOutcome::SafeTerminal
    );
    assert_eq!(shutdown.phase(), ShutdownPhase::Stopped);
    assert_eq!(
        shutdown.history(),
        &[
            ShutdownPhase::ReadinessDown,
            ShutdownPhase::GoAway,
            ShutdownPhase::StopAccepting,
            ShutdownPhase::DrainingRooms,
            ShutdownPhase::AbortingRooms,
            ShutdownPhase::NativeSafety,
            ShutdownPhase::SchedulerRelease,
            ShutdownPhase::WorkerJoin,
            ShutdownPhase::EngineQuiesce,
            ShutdownPhase::ConnectionClose,
            ShutdownPhase::RegionUnregister,
            ShutdownPhase::EngineDestroy,
            ShutdownPhase::Stopped,
        ]
    );
}

#[test]
fn unsafe_shutdown_terminal_is_sticky() {
    let mut shutdown = ShutdownTracker::new();
    shutdown.begin(ShutdownMode::Fatal);
    for phase in [
        ShutdownPhase::GoAway,
        ShutdownPhase::StopAccepting,
        ShutdownPhase::DrainingRooms,
        ShutdownPhase::AbortingRooms,
        ShutdownPhase::NativeSafety,
        ShutdownPhase::SchedulerRelease,
        ShutdownPhase::WorkerJoin,
        ShutdownPhase::EngineQuiesce,
        ShutdownPhase::ConnectionClose,
        ShutdownPhase::RegionUnregister,
        ShutdownPhase::EngineDestroy,
    ] {
        shutdown.advance(phase).expect("ordered shutdown phase");
    }
    assert_eq!(
        shutdown
            .complete(RuntimeShutdownOutcome::FatalUnsafe)
            .expect("unsafe terminal"),
        RuntimeShutdownOutcome::FatalUnsafe
    );
    assert_eq!(
        shutdown
            .complete(RuntimeShutdownOutcome::SafeTerminal)
            .expect("terminal remains sticky"),
        RuntimeShutdownOutcome::FatalUnsafe
    );
}

#[test]
fn current_error_families_map_to_exact_failure_scopes() {
    assert_eq!(
        FailureClass::for_runtime(&RuntimeError::Worker),
        FailureClass::new(FailureScope::LocalFatal, PdReason::LocalFatal)
    );
    assert_eq!(
        FailureClass::for_runtime(&RuntimeError::Timeout),
        FailureClass::new(FailureScope::PeerSession, PdReason::PeerUnavailable)
    );
    assert_eq!(
        FailureClass::for_transport(&TransportError::InvalidBatch),
        FailureClass::new(FailureScope::Request, PdReason::RequestInvalid)
    );
    assert_eq!(
        FailureClass::for_transport(&TransportError::InvalidTransition),
        FailureClass::new(FailureScope::Room, PdReason::ProtocolMismatch)
    );
    assert_eq!(
        FailureClass::for_transport(&TransportError::LocalFatal(PdReason::LocalFatal)),
        FailureClass::new(FailureScope::LocalFatal, PdReason::LocalFatal)
    );
    assert_eq!(
        FailureClass::for_transport(&TransportError::Room(PdReason::TransferTimeout)),
        FailureClass::new(FailureScope::Room, PdReason::TransferTimeout)
    );
    assert_eq!(
        FailureClass::for_transport(&TransportError::Peer(PdReason::PeerUnavailable)),
        FailureClass::new(FailureScope::PeerSession, PdReason::PeerUnavailable)
    );
    assert_eq!(
        FailureClass::for_buffer(&BufferError::CapacityExhausted { resource: "rooms" }),
        FailureClass::new(FailureScope::Request, PdReason::CapacityExhausted)
    );
    assert_eq!(
        FailureClass::for_buffer(&BufferError::Deadline),
        FailureClass::new(FailureScope::Room, PdReason::TransferTimeout)
    );
    assert_eq!(
        FailureClass::for_buffer(&BufferError::TableInUse {
            active: 1,
            quarantined: 0,
        }),
        FailureClass::new(FailureScope::LocalFatal, PdReason::LocalFatal)
    );
    assert_eq!(
        FailureClass::for_engine(&EngineError::QueueFull),
        FailureClass::new(FailureScope::Request, PdReason::CapacityExhausted)
    );
    assert_eq!(
        FailureClass::for_engine(&EngineError::BatchNotTerminal { id: 1 }),
        FailureClass::new(FailureScope::Room, PdReason::TransferTimeout)
    );
    assert_eq!(
        FailureClass::for_engine(&EngineError::WorkerClosed),
        FailureClass::new(FailureScope::LocalFatal, PdReason::LocalFatal)
    );
    assert_eq!(
        FailureClass::for_quarantine(false),
        FailureClass::new(FailureScope::Room, PdReason::TransferTimeout)
    );
    assert_eq!(
        FailureClass::for_quarantine(true),
        FailureClass::new(FailureScope::LocalFatal, PdReason::LocalFatal)
    );
}
