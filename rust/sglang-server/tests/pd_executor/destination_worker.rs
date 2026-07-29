use super::*;

#[test]
fn destination_flushes_before_reads_clears_all_regions_and_commits_once_after_ack() {
    let fixture = fixture();
    let clock = Arc::new(ManualClock::new(1_000));
    let source = SourceExecutor::new(
        Arc::clone(&fixture.ledger),
        Arc::clone(&fixture.quarantine),
        Arc::clone(&clock),
    );
    let mut fence = ComputeFence {
        ready: true,
        calls: 0,
    };
    let mut port = CpuNativePort::new(Arc::clone(&clock), [NativeMode::Success; 4]);
    let DataPlaneEffect::DataReady { identity } = source
        .execute(
            SourceExecutionRequest {
                plan: &fixture.plan,
                handle: fixture.handle,
                source_fence: &mut fence,
                aux: &fixture.aux,
                completion: &fixture.completion,
                deadline_monotonic_ms: 2_000,
            },
            &mut port,
        )
        .expect("data ready")
    else {
        panic!("expected DataReady");
    };

    let flush_calls = Arc::new(Mutex::new(0));
    let mut visibility = DestinationVisibilityFence::new(
        5,
        Flush {
            fail: false,
            calls: Arc::clone(&flush_calls),
        },
    )
    .expect("visibility");
    let mut records = CpuRecords {
        aux: fixture.aux,
        completion: fixture.completion.committed_bytes(),
        reads: 0,
        clears: Vec::new(),
    };
    let destination = DestinationExecutor::new(Arc::clone(&fixture.ledger));
    assert!(matches!(
        destination
            .validate_ready(
                &fixture.plan,
                fixture.handle,
                identity,
                &mut visibility,
                &mut records,
                &fixture.completion_input,
            )
            .expect("validate destination"),
        DataPlaneEffect::TransferComplete { .. }
    ));
    assert_eq!(*flush_calls.lock().expect("flush"), 1);
    assert_eq!(records.reads, 2);
    assert_eq!(records.clears.len(), 56);
    assert!(
        records
            .clears
            .iter()
            .all(|(_, page, tokens)| *page == 44 && *tokens == 65)
    );

    destination
        .validate_ready(
            &fixture.plan,
            fixture.handle,
            identity,
            &mut visibility,
            &mut records,
            &fixture.completion_input,
        )
        .expect("duplicate DataReady");
    assert_eq!(*flush_calls.lock().expect("flush"), 1);
    assert_eq!(records.reads, 2);
    assert_eq!(
        destination
            .commit_after_ack(fixture.handle, identity)
            .expect("handoff"),
        TransitionResult::Applied
    );
    assert_eq!(
        destination
            .commit_after_ack(fixture.handle, identity)
            .expect("duplicate handoff"),
        TransitionResult::AlreadyApplied
    );
    let completion = destination
        .consume_after_ack(identity)
        .expect("consume validated completion")
        .expect("first consume returns the result");
    assert!(completion.aux.first_token_valid);
    assert_eq!(completion.aux.first_token_id, 42);
    assert_eq!(
        destination
            .consume_after_ack(identity)
            .expect("duplicate consume is stable"),
        None
    );
    fixture
        .ledger
        .release_source_safe(fixture.handle)
        .expect("source safe");
    fixture
        .ledger
        .release_terminal(fixture.handle)
        .expect("terminal slots");
    assert_eq!(fixture.ledger.snapshot().active_rooms, 0);
}

#[test]
fn stale_data_identity_has_no_side_effect_and_failed_early_ack_can_retry_after_terminal() {
    let fixture = fixture();
    let flush_calls = Arc::new(Mutex::new(0));
    let mut visibility = DestinationVisibilityFence::new(
        5,
        Flush {
            fail: false,
            calls: Arc::clone(&flush_calls),
        },
    )
    .expect("visibility");
    let mut records = CpuRecords {
        aux: fixture.aux,
        completion: fixture.completion.committed_bytes(),
        reads: 0,
        clears: Vec::new(),
    };
    let destination = DestinationExecutor::new(Arc::clone(&fixture.ledger));
    let identity = DataPlaneIdentity::from_plan(&fixture.plan);
    let stale = DataPlaneIdentity {
        transfer_generation: identity.transfer_generation + 1,
        ..identity
    };
    assert!(matches!(
        destination.validate_ready(
            &fixture.plan,
            fixture.handle,
            stale,
            &mut visibility,
            &mut records,
            &fixture.completion_input,
        ),
        Err(BufferError::StaleHandle)
    ));
    assert_eq!(*flush_calls.lock().expect("flush"), 0);
    assert_eq!(records.reads, 0);
    assert_eq!(fixture.ledger.snapshot().active_rooms, 1);

    destination
        .validate_ready(
            &fixture.plan,
            fixture.handle,
            identity,
            &mut visibility,
            &mut records,
            &fixture.completion_input,
        )
        .expect("validated destination");
    assert!(matches!(
        destination.commit_after_ack(fixture.handle, identity),
        Err(BufferError::InvalidTransition)
    ));
    for stage in [
        sglang_server::pd::buffer::TransferStage::Kv,
        sglang_server::pd::buffer::TransferStage::Aux,
        sglang_server::pd::buffer::TransferStage::Completion,
    ] {
        fixture
            .ledger
            .begin_stage(fixture.handle, stage)
            .expect("begin remote stage");
        fixture
            .ledger
            .finish_stage(fixture.handle, stage)
            .expect("finish remote stage");
    }
    assert_eq!(
        destination
            .commit_after_ack(fixture.handle, identity)
            .expect("retry Ack after terminal"),
        TransitionResult::Applied
    );
    fixture
        .ledger
        .release_source_safe(fixture.handle)
        .expect("source release");
    fixture
        .ledger
        .release_terminal(fixture.handle)
        .expect("terminal release");
    assert_eq!(fixture.ledger.snapshot().active_rooms, 0);
}

#[test]
fn visibility_failure_performs_zero_record_reads_and_zero_destination_handoff() {
    let fixture = fixture();
    for stage in [
        sglang_server::pd::buffer::TransferStage::Kv,
        sglang_server::pd::buffer::TransferStage::Aux,
        sglang_server::pd::buffer::TransferStage::Completion,
    ] {
        fixture
            .ledger
            .begin_stage(fixture.handle, stage)
            .expect("begin");
        fixture
            .ledger
            .finish_stage(fixture.handle, stage)
            .expect("finish");
    }
    let identity = DataPlaneIdentity::from_plan(&fixture.plan);
    let mut visibility = DestinationVisibilityFence::new(
        5,
        Flush {
            fail: true,
            calls: Arc::new(Mutex::new(0)),
        },
    )
    .expect("visibility capability");
    let mut records = CpuRecords {
        aux: fixture.aux,
        completion: fixture.completion.committed_bytes(),
        reads: 0,
        clears: Vec::new(),
    };
    let destination = DestinationExecutor::new(Arc::clone(&fixture.ledger));
    assert!(matches!(
        destination.validate_ready(
            &fixture.plan,
            fixture.handle,
            identity,
            &mut visibility,
            &mut records,
            &fixture.completion_input,
        ),
        Err(BufferError::VisibilityFence)
    ));
    assert_eq!(records.reads, 0);
    assert!(records.clears.is_empty());
    assert_eq!(fixture.ledger.snapshot().active_rooms, 0);
}

#[test]
fn bounded_worker_returns_queue_full_without_blocking_the_control_caller() {
    let fixture = fixture();
    let clock = Arc::new(ManualClock::new(1_000));
    let executor = Arc::new(SourceExecutor::new(
        Arc::clone(&fixture.ledger),
        Arc::clone(&fixture.quarantine),
        Arc::clone(&clock),
    ));
    let port = CpuNativePort::new(Arc::clone(&clock), [NativeMode::Success; 4]);
    let worker = DataPlaneWorker::start(1, executor, port).expect("bounded worker");
    let started = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let first = worker
        .try_execute_source(SourceWorkRequest {
            plan: fixture.plan.clone(),
            handle: fixture.handle,
            source_fence: BlockingFence {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            },
            aux: fixture.aux,
            completion: fixture.completion.clone(),
            deadline_monotonic_ms: 2_000,
        })
        .expect("first work");
    started.wait();
    let second = worker
        .try_execute_source(SourceWorkRequest {
            plan: fixture.plan.clone(),
            handle: fixture.handle,
            source_fence: ComputeFence {
                ready: true,
                calls: 0,
            },
            aux: fixture.aux,
            completion: fixture.completion.clone(),
            deadline_monotonic_ms: 2_000,
        })
        .expect("queued work");
    assert!(matches!(
        worker.try_execute_source(SourceWorkRequest {
            plan: fixture.plan.clone(),
            handle: fixture.handle,
            source_fence: ComputeFence {
                ready: true,
                calls: 0,
            },
            aux: fixture.aux,
            completion: fixture.completion.clone(),
            deadline_monotonic_ms: 2_000,
        }),
        Err(BufferError::WorkerFull)
    ));
    release.wait();
    assert!(matches!(
        first.wait().expect("first worker result"),
        DataPlaneEffect::DataReady { .. }
    ));
    assert!(matches!(second.wait(), Err(BufferError::InvalidTransition)));
}

#[test]
fn bounded_worker_runs_visibility_crc_and_tail_clear_off_the_control_caller() {
    let fixture = fixture();
    for stage in [
        sglang_server::pd::buffer::TransferStage::Kv,
        sglang_server::pd::buffer::TransferStage::Aux,
        sglang_server::pd::buffer::TransferStage::Completion,
    ] {
        fixture
            .ledger
            .begin_stage(fixture.handle, stage)
            .expect("begin remote stage");
        fixture
            .ledger
            .finish_stage(fixture.handle, stage)
            .expect("finish remote stage");
    }
    let clock = Arc::new(ManualClock::new(1_000));
    let source = Arc::new(SourceExecutor::new(
        Arc::clone(&fixture.ledger),
        Arc::clone(&fixture.quarantine),
        Arc::clone(&clock),
    ));
    let destination = Arc::new(DestinationExecutor::new(Arc::clone(&fixture.ledger)));
    let port = CpuNativePort::new(clock, []);
    let worker = DataPlaneWorker::start_with_destination(1, source, Arc::clone(&destination), port)
        .expect("full data worker");
    let reads = Arc::new(AtomicUsize::new(0));
    let clears = Arc::new(AtomicUsize::new(0));
    let flushes = Arc::new(Mutex::new(0));
    let identity = DataPlaneIdentity::from_plan(&fixture.plan);
    let ticket = worker
        .try_validate_destination(DestinationWorkRequest {
            plan: fixture.plan.clone(),
            handle: fixture.handle,
            identity,
            device: 5,
            visibility: Flush {
                fail: false,
                calls: Arc::clone(&flushes),
            },
            records: SharedRecords {
                aux: fixture.aux,
                completion: fixture.completion.committed_bytes(),
                reads: Arc::clone(&reads),
                clears: Arc::clone(&clears),
            },
            expected: fixture.completion_input.clone(),
        })
        .expect("destination work");
    assert!(matches!(
        ticket.wait().expect("destination worker result"),
        DataPlaneEffect::TransferComplete { .. }
    ));
    assert_eq!(*flushes.lock().expect("flushes"), 1);
    assert_eq!(reads.load(Ordering::SeqCst), 2);
    assert_eq!(clears.load(Ordering::SeqCst), 56);
    destination
        .commit_after_ack(fixture.handle, identity)
        .expect("destination handoff");
    fixture
        .ledger
        .release_source_safe(fixture.handle)
        .expect("source release");
    fixture
        .ledger
        .release_terminal(fixture.handle)
        .expect("terminal release");
}

#[test]
fn typed_data_effects_drive_the_existing_prefill_and_decode_room_fsm_once() {
    let fixture = fixture();
    let profile = PdProfileV1::load_embedded().expect("profile");
    let clock = Arc::new(ManualClock::new(1_000));
    let request_digest = FixedBytes::new([0xa1; 32]);
    let prefill_spec = RoomSpec::new(
        fixture.plan.room(),
        request_digest,
        fixture.plan.source_registration_epoch(),
    )
    .expect("prefill RoomSpec");
    let mut prefill = RoomTable::new(
        RoomRole::Prefill,
        fixture.plan.room().key.decode_process_epoch,
        fixture.plan.source_registration_epoch(),
        &profile,
        Arc::clone(&clock),
    )
    .expect("prefill rooms");
    prefill.observe_local(prefill_spec.clone());
    prefill.observe_peer(prefill_spec);
    assert!(matches!(
        prefill.apply(fixture.plan.room(), RoomEvent::SourceReady),
        RoomOutcome::Applied(ref effects) if effects == &[RoomEffect::SubmitTransfer]
    ));
    let identity = DataPlaneIdentity::from_plan(&fixture.plan);
    let outcomes = apply_prefill_data_effect(&mut prefill, DataPlaneEffect::DataReady { identity })
        .expect("prefill bridge");
    assert!(matches!(
        outcomes.as_slice(),
        [
            RoomOutcome::Applied(first),
            RoomOutcome::Applied(second)
        ] if first.is_empty() && second == &[RoomEffect::SendDataReady]
    ));

    let decode_spec = RoomSpec::new(
        fixture.plan.room(),
        request_digest,
        fixture.plan.destination_registration_epoch(),
    )
    .expect("decode RoomSpec");
    let mut decode = RoomTable::new(
        RoomRole::Decode,
        fixture.plan.room().key.decode_process_epoch,
        fixture.plan.destination_registration_epoch(),
        &profile,
        Arc::clone(&clock),
    )
    .expect("decode rooms");
    decode.observe_local(decode_spec.clone());
    decode.observe_peer(decode_spec);
    assert!(matches!(
        apply_prepare_accepted(&mut decode, identity),
        RoomOutcome::Applied(ref effects) if effects.is_empty()
    ));
    assert!(matches!(
        apply_decode_data_effect(
            &mut decode,
            DataPlaneEffect::TransferComplete { identity }
        )
        .expect("decode bridge"),
        RoomOutcome::Applied(ref effects) if effects == &[RoomEffect::SendTransferComplete]
    ));
    assert!(matches!(
        apply_decode_ack(&mut decode, identity),
        RoomOutcome::Terminal {
            reason: PdReason::Success,
            duplicate: false,
            ..
        }
    ));
}
