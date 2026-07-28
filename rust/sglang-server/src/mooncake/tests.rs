use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use super::{
    BatchSnapshot, EngineError, EngineOwner, HostMemory, MemoryBuffer, MemoryLocation,
    MockEngineFactory, MockEvent, MockFailurePoint, MockPlan, NativeCode, NativeOperation,
    OperationState, OwnerConfig, PdNicProfile, PeerDescriptor, ShutdownOutcome, TransferOperation,
    load_library_for_test, validate_and_load_artifact_for_test, validate_artifact_for_test,
};

fn owner_config() -> OwnerConfig {
    OwnerConfig::new(8, Duration::from_millis(2), Duration::from_secs(1)).unwrap()
}

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("sglang-pd02-{name}-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn registered_pair(
    owner: &EngineOwner,
) -> (super::Region, super::Peer, super::RemoteRegionDescriptor) {
    let memory = HostMemory::new(128).unwrap();
    let region = owner
        .register_region(MemoryBuffer::Host(memory), MemoryLocation::Cpu0)
        .unwrap();
    let peer = owner
        .open_peer(PeerDescriptor::new("127.0.0.1:19001").unwrap())
        .unwrap();
    let remote = region.remote_descriptor();
    (region, peer, remote)
}

fn assert_native_error(error: EngineError, operation: NativeOperation, raw_code: i32) {
    assert!(matches!(
        error,
        EngineError::Native {
            operation: actual_operation,
            raw_code: actual_code,
            ..
        } if actual_operation == operation && actual_code == raw_code
    ));
}

#[test]
fn frozen_profile_is_canonical_and_rejects_other_devices() {
    let profile = PdNicProfile::for_gpu(4).unwrap();
    assert_eq!(
        profile.canonical_json(),
        r#"{"cpu:0":[["mlx5_1","mlx5_2","mlx5_3","mlx5_4"],[]],"cpu:1":[["mlx5_1","mlx5_2","mlx5_3","mlx5_4"],[]],"cuda:4":[["mlx5_1","mlx5_2","mlx5_3","mlx5_4"],[]],"cuda:5":[["mlx5_1","mlx5_2","mlx5_3","mlx5_4"],[]]}"#
    );
    assert_eq!(PdNicProfile::for_gpu(5).unwrap(), profile);
    assert!(matches!(
        PdNicProfile::for_gpu(0),
        Err(EngineError::UnsupportedGpu { device: 0 })
    ));
}

#[test]
fn native_status_mapping_preserves_unknown_values_and_fails_closed() {
    let expected = [
        (0, OperationState::Waiting, false),
        (1, OperationState::Pending, false),
        (2, OperationState::Invalid, true),
        (3, OperationState::Canceled, true),
        (4, OperationState::Completed, true),
        (5, OperationState::Timeout, true),
        (6, OperationState::Failed, true),
        (77, OperationState::Unknown(77), false),
    ];
    for (raw, state, terminal) in expected {
        let mapped = OperationState::from_raw(raw);
        assert_eq!(mapped, state);
        assert_eq!(mapped.is_terminal(), terminal);
    }

    let known_codes = [
        (1, NativeCode::InvalidArgument),
        (2, NativeCode::TooManyRequests),
        (3, NativeCode::AddressNotRegistered),
        (4, NativeCode::BatchBusy),
        (6, NativeCode::DeviceNotFound),
        (7, NativeCode::AddressOverlapped),
        (8, NativeCode::NotSupportedTransport),
        (101, NativeCode::Dns),
        (102, NativeCode::Socket),
        (103, NativeCode::MalformedJson),
        (104, NativeCode::RejectHandshake),
        (200, NativeCode::Metadata),
        (201, NativeCode::Endpoint),
        (202, NativeCode::Context),
        (300, NativeCode::Numa),
        (301, NativeCode::Clock),
        (302, NativeCode::Memory),
        (303, NativeCode::NotImplemented),
        (999, NativeCode::NotImplemented),
    ];
    for (raw, expected) in known_codes {
        assert_eq!(NativeCode::from_raw(raw), expected);
        assert_eq!(NativeCode::from_raw(-raw), expected);
    }
    assert_eq!(NativeCode::from_raw(4242), NativeCode::Unknown(4242));
    assert_eq!(NativeCode::from_raw(-4242), NativeCode::Unknown(-4242));
    assert!(matches!(
        EngineError::native(NativeOperation::SetCudaDevice, 101),
        EngineError::Native {
            code: NativeCode::Unknown(101),
            ..
        }
    ));
}

#[test]
fn loader_errors_distinguish_missing_library_manifest_and_bad_identity() {
    let missing = TestDirectory::new("missing-library");
    assert!(matches!(
        validate_artifact_for_test(&missing.0),
        Err(EngineError::LibraryMissing { .. })
    ));

    let missing_manifest = TestDirectory::new("missing-manifest");
    fs::write(
        missing_manifest.0.join("libtransfer_engine.so"),
        b"not-an-elf",
    )
    .unwrap();
    assert!(matches!(
        validate_artifact_for_test(&missing_manifest.0),
        Err(EngineError::ManifestMissing { .. })
    ));

    let bad_manifest = TestDirectory::new("bad-manifest");
    fs::write(bad_manifest.0.join("libtransfer_engine.so"), b"not-an-elf").unwrap();
    fs::write(bad_manifest.0.join("abi-manifest.json"), b"{}").unwrap();
    assert!(matches!(
        validate_artifact_for_test(&bad_manifest.0),
        Err(EngineError::AbiMismatch { .. })
    ));

    let libc = [
        PathBuf::from("/lib/x86_64-linux-gnu/libc.so.6"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu/libc.so.6"),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .expect("test host provides libc");
    assert!(matches!(
        load_library_for_test(&libc),
        Err(EngineError::SymbolMissing { symbol }) if symbol == "createTransferEngine"
    ));
}

#[test]
#[ignore = "requires the independently built Mooncake artifact"]
fn packaged_native_artifact_manifest_hashes_abi_and_symbols_load() {
    let directory = std::env::var_os("SGLANG_PD02_ARTIFACT_DIR")
        .map(PathBuf::from)
        .expect("SGLANG_PD02_ARTIFACT_DIR must point to the packaged artifact");
    validate_and_load_artifact_for_test(&directory).unwrap();
}

#[test]
fn logical_abort_keeps_polling_and_only_frees_after_terminal() {
    let factory = MockEngineFactory::new(MockPlan::with_status_script(vec![
        vec![OperationState::Waiting],
        vec![OperationState::Pending],
        vec![OperationState::Completed],
    ]));
    let events = factory.events();
    let owner = EngineOwner::start(owner_config(), factory).unwrap();
    let (region, peer, remote) = registered_pair(&owner);
    let operation = TransferOperation::write(&region, 8, &peer, &remote, 16, 32).unwrap();
    let batch = owner.submit(vec![operation]).unwrap();

    batch.abort().unwrap();
    let snapshot = batch.wait_terminal(Duration::from_secs(1)).unwrap();
    assert_eq!(
        snapshot,
        BatchSnapshot {
            operations: vec![super::OperationProgress {
                state: OperationState::Completed,
                transferred_bytes: 32,
            }],
            logical_aborted: true,
            safe_terminal: true,
        }
    );

    let events = events.lock().unwrap();
    let submit = events
        .iter()
        .position(|event| matches!(event, MockEvent::SubmitBatch { .. }))
        .unwrap();
    let free = events
        .iter()
        .position(|event| matches!(event, MockEvent::FreeBatch { .. }))
        .unwrap();
    let terminal_poll = events
        .iter()
        .rposition(|event| {
            matches!(
                event,
                MockEvent::Poll {
                    state: OperationState::Completed,
                    ..
                }
            )
        })
        .unwrap();
    assert!(submit < terminal_poll);
    assert!(terminal_poll < free);
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, MockEvent::CancelBatch { .. }))
    );
}

#[test]
fn worker_caps_native_inflight_batches_at_four() {
    let pending = vec![vec![OperationState::Pending]; 100];
    let factory = MockEngineFactory::new(MockPlan::with_status_script(pending));
    let owner = EngineOwner::start(owner_config(), factory).unwrap();
    let (region, peer, remote) = registered_pair(&owner);

    let mut batches = Vec::new();
    for _ in 0..4 {
        let operation = TransferOperation::write(&region, 0, &peer, &remote, 0, 8).unwrap();
        batches.push(owner.submit(vec![operation]).unwrap());
    }
    let operation = TransferOperation::write(&region, 0, &peer, &remote, 0, 8).unwrap();
    assert!(matches!(
        owner.submit(vec![operation]),
        Err(EngineError::InFlightLimit { limit: 4 })
    ));

    for batch in batches {
        batch.abort().unwrap();
    }
    assert!(matches!(
        owner.shutdown().unwrap(),
        ShutdownOutcome::NotSafe { .. }
    ));
}

#[test]
fn terminal_shutdown_cleans_batch_peer_region_engine_in_reverse_order_once() {
    let factory = MockEngineFactory::new(MockPlan::default());
    let events = factory.events();
    let owner = EngineOwner::start(owner_config(), factory).unwrap();
    let (region, peer, remote) = registered_pair(&owner);
    let operation = TransferOperation::write(&region, 0, &peer, &remote, 0, 8).unwrap();
    let batch = owner.submit(vec![operation]).unwrap();
    batch.wait_terminal(Duration::from_secs(1)).unwrap();

    drop(batch);
    drop(peer);
    drop(region);
    assert_eq!(owner.shutdown().unwrap(), ShutdownOutcome::SafeTerminal);
    assert_eq!(owner.shutdown().unwrap(), ShutdownOutcome::SafeTerminal);

    let events = events.lock().unwrap();
    let free = events
        .iter()
        .position(|event| matches!(event, MockEvent::FreeBatch { .. }))
        .unwrap();
    let close = events
        .iter()
        .position(|event| matches!(event, MockEvent::ClosePeer { .. }))
        .unwrap();
    let unregister = events
        .iter()
        .position(|event| matches!(event, MockEvent::UnregisterRegion { .. }))
        .unwrap();
    let shutdown = events
        .iter()
        .position(|event| matches!(event, MockEvent::Shutdown))
        .unwrap();
    assert!(free < close && close < unregister && unregister < shutdown);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, MockEvent::FreeBatch { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, MockEvent::ClosePeer { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, MockEvent::UnregisterRegion { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, MockEvent::Shutdown))
            .count(),
        1
    );
}

#[test]
fn all_terminal_native_states_are_freed_but_unknown_state_is_not() {
    for state in [
        OperationState::Invalid,
        OperationState::Canceled,
        OperationState::Completed,
        OperationState::Timeout,
        OperationState::Failed,
    ] {
        let factory = MockEngineFactory::new(MockPlan::with_status_script(vec![vec![state]]));
        let events = factory.events();
        let owner = EngineOwner::start(owner_config(), factory).unwrap();
        let (region, peer, remote) = registered_pair(&owner);
        let operation = TransferOperation::write(&region, 0, &peer, &remote, 0, 8).unwrap();
        let batch = owner.submit(vec![operation]).unwrap();
        let snapshot = batch.wait_terminal(Duration::from_secs(1)).unwrap();
        assert_eq!(snapshot.operations[0].state, state);
        assert!(
            events
                .lock()
                .unwrap()
                .iter()
                .any(|event| matches!(event, MockEvent::FreeBatch { .. }))
        );
        drop(batch);
        drop(peer);
        drop(region);
        assert_eq!(owner.shutdown().unwrap(), ShutdownOutcome::SafeTerminal);
    }

    let factory = MockEngineFactory::new(MockPlan::with_status_script(vec![vec![
        OperationState::Unknown(77),
    ]]));
    let events = factory.events();
    let owner = EngineOwner::start(owner_config(), factory).unwrap();
    let (region, peer, remote) = registered_pair(&owner);
    let operation = TransferOperation::write(&region, 0, &peer, &remote, 0, 8).unwrap();
    let batch = owner.submit(vec![operation]).unwrap();
    assert!(matches!(
        batch.wait_terminal(Duration::from_millis(20)),
        Err(EngineError::BatchNotTerminal { .. })
    ));
    assert!(
        !events
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, MockEvent::FreeBatch { .. }))
    );
    assert!(matches!(
        owner.shutdown().unwrap(),
        ShutdownOutcome::NotSafe { .. }
    ));
}

#[test]
fn initialization_registration_connection_and_submission_failures_keep_context() {
    let create =
        MockEngineFactory::new(MockPlan::default().fail_once(MockFailurePoint::Create, -202));
    let create_error = match EngineOwner::start(owner_config(), create) {
        Ok(_) => panic!("create failure was not propagated"),
        Err(error) => error,
    };
    assert_native_error(create_error, NativeOperation::CreateEngine, -202);

    let register =
        MockEngineFactory::new(MockPlan::default().fail_once(MockFailurePoint::RegisterRegion, -7));
    let owner = EngineOwner::start(owner_config(), register).unwrap();
    let error = owner
        .register_region(
            MemoryBuffer::Host(HostMemory::new(32).unwrap()),
            MemoryLocation::Cpu0,
        )
        .err()
        .expect("register failure");
    assert_native_error(error, NativeOperation::RegisterRegion, -7);
    assert_eq!(owner.shutdown().unwrap(), ShutdownOutcome::SafeTerminal);

    let open =
        MockEngineFactory::new(MockPlan::default().fail_once(MockFailurePoint::OpenPeer, -102));
    let owner = EngineOwner::start(owner_config(), open).unwrap();
    let region = owner
        .register_region(
            MemoryBuffer::Host(HostMemory::new(32).unwrap()),
            MemoryLocation::Cpu0,
        )
        .unwrap();
    let error = owner
        .open_peer(PeerDescriptor::new("127.0.0.1:19001").unwrap())
        .err()
        .expect("open failure");
    assert_native_error(error, NativeOperation::OpenPeer, -102);
    drop(region);
    assert_eq!(owner.shutdown().unwrap(), ShutdownOutcome::SafeTerminal);

    for (point, operation, raw_code) in [
        (
            MockFailurePoint::AllocateBatch,
            NativeOperation::AllocateBatch,
            -2,
        ),
        (
            MockFailurePoint::SubmitBatch,
            NativeOperation::SubmitBatch,
            -1,
        ),
    ] {
        let factory = MockEngineFactory::new(MockPlan::default().fail_once(point, raw_code));
        let events = factory.events();
        let owner = EngineOwner::start(owner_config(), factory).unwrap();
        let (region, peer, remote) = registered_pair(&owner);
        let operation_descriptor =
            TransferOperation::write(&region, 0, &peer, &remote, 0, 8).unwrap();
        let error = owner
            .submit(vec![operation_descriptor])
            .err()
            .expect("submission phase failure");
        assert_native_error(error, operation, raw_code);
        if point == MockFailurePoint::SubmitBatch {
            assert_eq!(
                events
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|event| matches!(event, MockEvent::FreeBatch { .. }))
                    .count(),
                1
            );
        }
        drop(peer);
        drop(region);
        assert_eq!(owner.shutdown().unwrap(), ShutdownOutcome::SafeTerminal);
    }
}

#[test]
fn failed_submit_rollback_is_retained_as_not_safe() {
    let factory = MockEngineFactory::new(
        MockPlan::default()
            .fail_once(MockFailurePoint::SubmitBatch, -1)
            .fail_once(MockFailurePoint::FreeBatch, -4),
    );
    let events = factory.events();
    let owner = EngineOwner::start(owner_config(), factory).unwrap();
    let (region, peer, remote) = registered_pair(&owner);
    let operation = TransferOperation::write(&region, 0, &peer, &remote, 0, 8).unwrap();
    assert!(matches!(
        owner.submit(vec![operation]),
        Err(EngineError::Rollback {
            operation: NativeOperation::SubmitBatch,
            ..
        })
    ));
    assert!(matches!(
        owner.shutdown().unwrap(),
        ShutdownOutcome::NotSafe { batches } if batches.len() == 1
    ));
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| matches!(
                event,
                MockEvent::Failure {
                    point: MockFailurePoint::FreeBatch,
                    ..
                }
            ))
            .count(),
        1
    );
}

#[test]
fn cleanup_failure_stops_dependent_cleanup_and_is_not_retried() {
    let factory =
        MockEngineFactory::new(MockPlan::default().fail_once(MockFailurePoint::ClosePeer, -102));
    let events = factory.events();
    let owner = EngineOwner::start(owner_config(), factory).unwrap();
    let (region, peer, _) = registered_pair(&owner);
    let close_error = peer.close().expect_err("close failure");
    assert_native_error(close_error, NativeOperation::ClosePeer, -102);
    assert!(matches!(
        region.close(),
        Err(EngineError::Native {
            operation: NativeOperation::ClosePeer,
            raw_code: -102,
            ..
        })
    ));
    assert!(matches!(
        owner.shutdown(),
        Err(EngineError::Native {
            operation: NativeOperation::ClosePeer,
            raw_code: -102,
            ..
        })
    ));

    let events = events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                MockEvent::Failure {
                    point: MockFailurePoint::ClosePeer,
                    ..
                }
            ))
            .count(),
        1
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, MockEvent::UnregisterRegion { .. }))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, MockEvent::Shutdown))
    );
}

#[test]
fn bounded_queue_reports_full_and_closed_distinctly() {
    let factory = MockEngineFactory::new(
        MockPlan::default().delay(MockFailurePoint::RegisterRegion, Duration::from_millis(200)),
    );
    let events = factory.events();
    let config = OwnerConfig::new(1, Duration::from_millis(1), Duration::from_secs(2)).unwrap();
    let owner = Arc::new(EngineOwner::start(config, factory).unwrap());

    let first_owner = Arc::clone(&owner);
    let first = thread::spawn(move || {
        first_owner.register_region(
            MemoryBuffer::Host(HostMemory::new(32).unwrap()),
            MemoryLocation::Cpu0,
        )
    });
    for _ in 0..100 {
        if events
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, MockEvent::RegisterRegion { .. }))
        {
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    let second_owner = Arc::clone(&owner);
    let second = thread::spawn(move || {
        second_owner.register_region(
            MemoryBuffer::Host(HostMemory::new(32).unwrap()),
            MemoryLocation::Cpu0,
        )
    });
    thread::sleep(Duration::from_millis(20));
    assert!(matches!(
        owner.register_region(
            MemoryBuffer::Host(HostMemory::new(32).unwrap()),
            MemoryLocation::Cpu0,
        ),
        Err(EngineError::QueueFull)
    ));
    drop(first.join().unwrap().unwrap());
    drop(second.join().unwrap().unwrap());
    assert_eq!(owner.shutdown().unwrap(), ShutdownOutcome::SafeTerminal);
    assert!(matches!(
        owner.register_region(
            MemoryBuffer::Host(HostMemory::new(32).unwrap()),
            MemoryLocation::Cpu0,
        ),
        Err(EngineError::WorkerClosed)
    ));
}

#[test]
fn logical_handles_cannot_cross_engine_owners() {
    let first =
        EngineOwner::start(owner_config(), MockEngineFactory::new(MockPlan::default())).unwrap();
    let second =
        EngineOwner::start(owner_config(), MockEngineFactory::new(MockPlan::default())).unwrap();
    let (first_region, first_peer, first_remote) = registered_pair(&first);
    let (second_region, second_peer, second_remote) = registered_pair(&second);

    assert!(matches!(
        TransferOperation::write(&first_region, 0, &second_peer, &second_remote, 0, 8,),
        Err(EngineError::InvalidDescriptor {
            field: "operation.owner",
            ..
        })
    ));
    let first_operation =
        TransferOperation::write(&first_region, 0, &first_peer, &first_remote, 0, 8).unwrap();
    assert!(matches!(
        second.submit(vec![first_operation]),
        Err(EngineError::InvalidDescriptor {
            field: "operation.owner",
            ..
        })
    ));

    drop(first_peer);
    drop(first_region);
    drop(second_peer);
    drop(second_region);
    assert_eq!(first.shutdown().unwrap(), ShutdownOutcome::SafeTerminal);
    assert_eq!(second.shutdown().unwrap(), ShutdownOutcome::SafeTerminal);
}

#[test]
fn timed_out_initialization_rolls_back_the_created_engine() {
    let factory = MockEngineFactory::new(
        MockPlan::default().delay(MockFailurePoint::Create, Duration::from_millis(80)),
    );
    let events = factory.events();
    let config = OwnerConfig::new(4, Duration::from_millis(1), Duration::from_millis(10)).unwrap();
    assert!(matches!(
        EngineOwner::start(config, factory),
        Err(EngineError::ResponseTimeout {
            operation: "worker initialization"
        })
    ));
    thread::sleep(Duration::from_millis(120));
    let events = events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, MockEvent::Create))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, MockEvent::Shutdown))
            .count(),
        1
    );
}
