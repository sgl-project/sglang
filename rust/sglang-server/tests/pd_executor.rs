use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

use sglang_server::mooncake::{BatchSnapshot, OperationProgress, OperationState};
use sglang_server::pd::buffer::{
    AUX_BYTES, AuxRecord, AuxRecordInput, BufferError, CapacityLedger, CompletionRecordInput,
    CompletionWrites, DataPlaneEffect, DataPlaneIdentity, DataPlaneWorker, DestinationExecutor,
    DestinationRecordPort, DestinationVisibilityFence, DestinationWorkRequest, GpuDirectFlushPort,
    NativeBatchToken, NativePhase, NativeSafety, NativeStageCommand, NativeStagePort,
    QuarantineManager, QuarantineUpdate, ReservationRequest, SourceComputeFence,
    SourceExecutionRequest, SourceExecutor, SourceWorkRequest, TableUseTracker, TransferPlan,
    TransferPlanInput, TransitionResult, apply_decode_ack, apply_decode_data_effect,
    apply_prefill_data_effect, apply_prepare_accepted,
};
use sglang_server::pd::config::PdProfileV1;
use sglang_server::pd::protocol::FixedBytes;
use sglang_server::pd::room::{
    AttemptId, Clock, ManualClock, PdReason, ProcessEpoch, RegistrationEpoch, RoomEffect,
    RoomEvent, RoomId, RoomKey, RoomOutcome, RoomRole, RoomSpec, RoomTable,
};

#[derive(Debug, Clone, Copy)]
enum NativeMode {
    Success,
    SafeFailure,
    PendingForever,
    SubmitFailure,
    PollFailure,
    FreeFailure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NativeEvent {
    Submit(NativePhase, usize, usize),
    Poll(NativeBatchToken),
    Free(NativeBatchToken),
}

struct BatchState {
    lengths: Vec<u64>,
    mode: NativeMode,
}

struct CpuNativePort {
    modes: VecDeque<NativeMode>,
    batches: HashMap<NativeBatchToken, BatchState>,
    next_batch: u64,
    in_flight: Option<NativeBatchToken>,
    events: Vec<NativeEvent>,
    clock: Arc<ManualClock>,
}

impl CpuNativePort {
    fn new(clock: Arc<ManualClock>, modes: impl IntoIterator<Item = NativeMode>) -> Self {
        Self {
            modes: modes.into_iter().collect(),
            batches: HashMap::new(),
            next_batch: 1,
            in_flight: None,
            events: Vec::new(),
            clock,
        }
    }

    fn recover(&mut self, batch: NativeBatchToken) {
        self.batches.get_mut(&batch).expect("recover batch").mode = NativeMode::Success;
    }
}

impl NativeStagePort for CpuNativePort {
    fn submit(&mut self, command: &NativeStageCommand) -> Result<NativeBatchToken, BufferError> {
        if self.in_flight.is_some() {
            return Err(BufferError::NativeTransfer);
        }
        let mode = self.modes.pop_front().unwrap_or(NativeMode::Success);
        self.events.push(NativeEvent::Submit(
            command.phase(),
            command.expected_lengths().len(),
            command.payload().len(),
        ));
        if matches!(mode, NativeMode::SubmitFailure) {
            return Err(BufferError::NativeTransfer);
        }
        let batch = NativeBatchToken::new(self.next_batch)?;
        self.next_batch += 1;
        self.batches.insert(
            batch,
            BatchState {
                lengths: command.expected_lengths().to_vec(),
                mode,
            },
        );
        self.in_flight = Some(batch);
        Ok(batch)
    }

    fn poll(&mut self, batch: NativeBatchToken) -> Result<BatchSnapshot, BufferError> {
        self.events.push(NativeEvent::Poll(batch));
        let state = self.batches.get(&batch).ok_or(BufferError::StaleHandle)?;
        let (operations, safe_terminal) = match state.mode {
            NativeMode::Success => (
                state
                    .lengths
                    .iter()
                    .map(|length| OperationProgress {
                        state: OperationState::Completed,
                        transferred_bytes: *length,
                    })
                    .collect(),
                true,
            ),
            NativeMode::SafeFailure => (
                state
                    .lengths
                    .iter()
                    .enumerate()
                    .map(|(index, length)| OperationProgress {
                        state: if index == 0 {
                            OperationState::Failed
                        } else {
                            OperationState::Completed
                        },
                        transferred_bytes: if index == 0 { 0 } else { *length },
                    })
                    .collect(),
                true,
            ),
            NativeMode::PendingForever => {
                self.clock.advance_monotonic(10);
                (
                    state
                        .lengths
                        .iter()
                        .map(|_| OperationProgress {
                            state: OperationState::Pending,
                            transferred_bytes: 0,
                        })
                        .collect(),
                    false,
                )
            }
            NativeMode::PollFailure => return Err(BufferError::NativeTransfer),
            NativeMode::FreeFailure | NativeMode::SubmitFailure => (
                state
                    .lengths
                    .iter()
                    .map(|length| OperationProgress {
                        state: OperationState::Completed,
                        transferred_bytes: *length,
                    })
                    .collect(),
                true,
            ),
        };
        Ok(BatchSnapshot {
            operations,
            logical_aborted: false,
            safe_terminal,
        })
    }

    fn free_safe(&mut self, batch: NativeBatchToken) -> Result<(), BufferError> {
        self.events.push(NativeEvent::Free(batch));
        let state = self
            .batches
            .remove(&batch)
            .ok_or(BufferError::StaleHandle)?;
        if self.in_flight == Some(batch) {
            self.in_flight = None;
        }
        if matches!(state.mode, NativeMode::FreeFailure) {
            Err(BufferError::NativeTransfer)
        } else {
            Ok(())
        }
    }
}

struct ComputeFence {
    ready: bool,
    calls: usize,
}

impl SourceComputeFence for ComputeFence {
    fn wait_ready(&mut self, _deadline_monotonic_ms: u64) -> Result<(), BufferError> {
        self.calls += 1;
        if self.ready {
            Ok(())
        } else {
            Err(BufferError::SourceFence)
        }
    }
}

struct BlockingFence {
    started: Arc<Barrier>,
    release: Arc<Barrier>,
}

struct LateFence {
    clock: Arc<ManualClock>,
}

impl SourceComputeFence for LateFence {
    fn wait_ready(&mut self, _deadline_monotonic_ms: u64) -> Result<(), BufferError> {
        self.clock.advance_monotonic(2_000);
        Ok(())
    }
}

impl SourceComputeFence for BlockingFence {
    fn wait_ready(&mut self, _deadline_monotonic_ms: u64) -> Result<(), BufferError> {
        self.started.wait();
        self.release.wait();
        Ok(())
    }
}

struct Flush {
    fail: bool,
    calls: Arc<Mutex<usize>>,
}

impl GpuDirectFlushPort for Flush {
    fn supports_flush_to_owner(&self, device: u32) -> bool {
        device == 5
    }

    fn flush_to_owner(&mut self, _device: u32) -> Result<(), BufferError> {
        *self.calls.lock().expect("flush calls") += 1;
        if self.fail {
            Err(BufferError::VisibilityFence)
        } else {
            Ok(())
        }
    }
}

struct CpuRecords {
    aux: [u8; AUX_BYTES],
    completion: [u8; 192],
    reads: usize,
    clears: Vec<(u16, u32, u32)>,
}

impl DestinationRecordPort for CpuRecords {
    fn read_completion(&mut self, _slot: u16) -> Result<[u8; 192], BufferError> {
        self.reads += 1;
        Ok(self.completion)
    }

    fn read_aux(&mut self, _slot: u16) -> Result<[u8; AUX_BYTES], BufferError> {
        self.reads += 1;
        Ok(self.aux)
    }

    fn clear_final_kv_page_tail(
        &mut self,
        region_id: u16,
        page: u32,
        valid_token_count: u32,
    ) -> Result<(), BufferError> {
        self.clears.push((region_id, page, valid_token_count));
        Ok(())
    }
}

struct SharedRecords {
    aux: [u8; AUX_BYTES],
    completion: [u8; 192],
    reads: Arc<AtomicUsize>,
    clears: Arc<AtomicUsize>,
}

impl DestinationRecordPort for SharedRecords {
    fn read_completion(&mut self, _slot: u16) -> Result<[u8; 192], BufferError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        Ok(self.completion)
    }

    fn read_aux(&mut self, _slot: u16) -> Result<[u8; AUX_BYTES], BufferError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        Ok(self.aux)
    }

    fn clear_final_kv_page_tail(
        &mut self,
        _region_id: u16,
        _page: u32,
        _valid_token_count: u32,
    ) -> Result<(), BufferError> {
        self.clears.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct Fixture {
    plan: TransferPlan,
    ledger: Arc<CapacityLedger>,
    quarantine: Arc<QuarantineManager>,
    handle: sglang_server::pd::buffer::LeaseHandle,
    aux: [u8; AUX_BYTES],
    completion_input: CompletionRecordInput,
    completion: CompletionWrites,
}

fn fixture() -> Fixture {
    let room = RoomId::new(
        RoomKey::new(ProcessEpoch::random(), 0, AttemptId::random()).expect("RoomKey"),
        1,
    )
    .expect("RoomId");
    let plan = TransferPlan::new(TransferPlanInput {
        room,
        transfer_generation: 7,
        source_registration_epoch: RegistrationEpoch::random(),
        destination_registration_epoch: RegistrationEpoch::random(),
        source_pages: vec![3, 8],
        destination_pages: vec![7, 44],
        source_aux_slot: 2,
        destination_aux_slot: 1,
        source_completion_slot: 2,
        destination_completion_slot: 1,
        valid_token_count: 65,
        chunk_sequence: 0,
        chunk_count: 1,
        is_last_chunk: true,
    })
    .expect("plan");
    let profile = PdProfileV1::load_embedded().expect("profile");
    let ledger = Arc::new(CapacityLedger::new(
        &profile,
        TableUseTracker::new(),
        TableUseTracker::new(),
    ));
    let handle = ledger
        .reserve(ReservationRequest {
            room,
            handle_generation: 1,
            source_pages: plan.source_pages().to_vec(),
            destination_pages: plan.destination_pages().to_vec(),
            aux_slot: plan.source_aux_slot(),
            completion_slot: plan.source_completion_slot(),
            request_slot: 2,
            kv_bytes: plan.expected_kv_bytes(),
            deadline_monotonic_ms: 61_000,
        })
        .expect("leases");
    let request_digest = FixedBytes::new([0xa1; 32]);
    let aux = AuxRecord::encode(AuxRecordInput {
        first_token_valid: true,
        first_token_id: 42,
        prompt_token_count: 65,
        prefill_output_count: 1,
        request_digest,
    })
    .expect("aux");
    let completion_input = CompletionRecordInput {
        decode_process_epoch: room.key.decode_process_epoch,
        attempt_id: room.key.attempt_id,
        source_registration_epoch: plan.source_registration_epoch(),
        destination_registration_epoch: plan.destination_registration_epoch(),
        bootstrap_room: room.key.bootstrap_room,
        transfer_generation: plan.transfer_generation(),
        chunk_sequence: 0,
        chunk_count: 1,
        page_count: 2,
        valid_token_count: 65,
        request_digest,
        transfer_plan_digest: plan.digest(),
    };
    let completion = CompletionWrites::encode(&completion_input, &aux).expect("completion writes");
    let quarantine = Arc::new(QuarantineManager::new(Arc::clone(&ledger)));
    Fixture {
        plan,
        ledger,
        quarantine,
        handle,
        aux,
        completion_input,
        completion,
    }
}

#[test]
fn source_executor_waits_for_compute_and_runs_four_strict_native_batches() {
    let fixture = fixture();
    let clock = Arc::new(ManualClock::new(1_000));
    let executor = SourceExecutor::new(
        Arc::clone(&fixture.ledger),
        Arc::clone(&fixture.quarantine),
        Arc::clone(&clock),
    );
    let mut fence = ComputeFence {
        ready: true,
        calls: 0,
    };
    let mut port = CpuNativePort::new(Arc::clone(&clock), [NativeMode::Success; 4]);
    let effect = executor
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
        .expect("source execution");
    assert!(matches!(effect, DataPlaneEffect::DataReady { .. }));
    assert_eq!(fence.calls, 1);
    let submits = port
        .events
        .iter()
        .filter_map(|event| match event {
            NativeEvent::Submit(phase, operations, payload) => {
                Some((*phase, *operations, *payload))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        submits,
        vec![
            (NativePhase::Kv, 112, 0),
            (NativePhase::Aux, 1, 64),
            (NativePhase::CompletionBody, 1, 188),
            (NativePhase::CompletionMarker, 1, 4),
        ]
    );
    assert_eq!(
        port.events
            .iter()
            .filter(|event| matches!(event, NativeEvent::Free(_)))
            .count(),
        4
    );
    assert_eq!(fixture.ledger.snapshot().in_flight_transfers, 0);
    assert_eq!(fixture.ledger.snapshot().active_rooms, 1);
}

#[test]
fn source_fence_and_safe_native_failure_never_submit_or_run_later_phases() {
    let first_fixture = fixture();
    let clock = Arc::new(ManualClock::new(1_000));
    let executor = SourceExecutor::new(
        Arc::clone(&first_fixture.ledger),
        Arc::clone(&first_fixture.quarantine),
        Arc::clone(&clock),
    );
    let mut fence = ComputeFence {
        ready: false,
        calls: 0,
    };
    let mut port = CpuNativePort::new(Arc::clone(&clock), []);
    assert!(matches!(
        executor
            .execute(
                SourceExecutionRequest {
                    plan: &first_fixture.plan,
                    handle: first_fixture.handle,
                    source_fence: &mut fence,
                    aux: &first_fixture.aux,
                    completion: &first_fixture.completion,
                    deadline_monotonic_ms: 2_000,
                },
                &mut port,
            )
            .expect("typed fence failure"),
        DataPlaneEffect::TransferFailed { .. }
    ));
    assert!(port.events.is_empty());
    assert_eq!(first_fixture.ledger.snapshot().active_rooms, 0);

    let second_fixture = fixture();
    let executor = SourceExecutor::new(
        Arc::clone(&second_fixture.ledger),
        Arc::clone(&second_fixture.quarantine),
        Arc::clone(&clock),
    );
    let mut fence = ComputeFence {
        ready: true,
        calls: 0,
    };
    let mut port = CpuNativePort::new(
        Arc::clone(&clock),
        [NativeMode::Success, NativeMode::SafeFailure],
    );
    assert!(matches!(
        executor
            .execute(
                SourceExecutionRequest {
                    plan: &second_fixture.plan,
                    handle: second_fixture.handle,
                    source_fence: &mut fence,
                    aux: &second_fixture.aux,
                    completion: &second_fixture.completion,
                    deadline_monotonic_ms: 2_000,
                },
                &mut port,
            )
            .expect("typed native failure"),
        DataPlaneEffect::TransferFailed { .. }
    ));
    assert_eq!(
        port.events
            .iter()
            .filter(|event| matches!(event, NativeEvent::Submit(..)))
            .count(),
        2
    );
    assert_eq!(second_fixture.ledger.snapshot().active_rooms, 0);

    let expired_fixture = fixture();
    let expired_clock = Arc::new(ManualClock::new(1_000));
    let executor = SourceExecutor::new(
        Arc::clone(&expired_fixture.ledger),
        Arc::clone(&expired_fixture.quarantine),
        Arc::clone(&expired_clock),
    );
    let mut port = CpuNativePort::new(Arc::clone(&expired_clock), []);
    assert!(matches!(
        executor
            .execute(
                SourceExecutionRequest {
                    plan: &expired_fixture.plan,
                    handle: expired_fixture.handle,
                    source_fence: &mut LateFence {
                        clock: Arc::clone(&expired_clock),
                    },
                    aux: &expired_fixture.aux,
                    completion: &expired_fixture.completion,
                    deadline_monotonic_ms: 2_000,
                },
                &mut port,
            )
            .expect("late source fence"),
        DataPlaneEffect::TransferFailed {
            reason: PdReason::TransferTimeout,
            ..
        }
    ));
    assert!(port.events.is_empty());
    assert_eq!(expired_fixture.ledger.snapshot().active_rooms, 0);
}

#[test]
fn every_native_phase_submit_poll_terminal_and_free_failure_stops_later_batches_safely() {
    let phases = [
        NativePhase::Kv,
        NativePhase::Aux,
        NativePhase::CompletionBody,
        NativePhase::CompletionMarker,
    ];
    for failure_mode in [
        NativeMode::SubmitFailure,
        NativeMode::PollFailure,
        NativeMode::SafeFailure,
        NativeMode::FreeFailure,
    ] {
        for failure_index in 0..phases.len() {
            let fixture = fixture();
            let clock = Arc::new(ManualClock::new(1_000));
            let executor = SourceExecutor::new(
                Arc::clone(&fixture.ledger),
                Arc::clone(&fixture.quarantine),
                Arc::clone(&clock),
            );
            let mut modes = vec![NativeMode::Success; failure_index];
            modes.push(failure_mode);
            let mut port = CpuNativePort::new(Arc::clone(&clock), modes);
            let mut fence = ComputeFence {
                ready: true,
                calls: 0,
            };
            let effect = executor
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
                .expect("typed native fault outcome");
            let submitted = port
                .events
                .iter()
                .filter_map(|event| match event {
                    NativeEvent::Submit(phase, _, _) => Some(*phase),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                submitted,
                phases[..=failure_index],
                "native fault submitted a later phase"
            );

            match (failure_mode, effect) {
                (NativeMode::PollFailure, DataPlaneEffect::Quarantined { batch, .. }) => {
                    assert_eq!(fixture.ledger.snapshot().quarantined_rooms, 1);
                    port.recover(batch);
                    let snapshot = port.poll(batch).expect("late native terminal");
                    let expected = snapshot
                        .operations
                        .iter()
                        .map(|operation| operation.transferred_bytes)
                        .collect::<Vec<_>>();
                    let safety =
                        sglang_server::pd::buffer::evaluate_native_fence(&snapshot, &expected);
                    assert_eq!(safety, NativeSafety::SafeSuccess);
                    port.free_safe(batch).expect("free recovered batch");
                    assert_eq!(
                        fixture
                            .quarantine
                            .observe(fixture.handle, batch, safety, clock.now_monotonic_ms(),)
                            .expect("release quarantine"),
                        QuarantineUpdate::Released
                    );
                }
                (
                    NativeMode::SubmitFailure | NativeMode::SafeFailure | NativeMode::FreeFailure,
                    DataPlaneEffect::TransferFailed { .. },
                ) => {}
                _ => panic!("native fault returned an unexpected effect"),
            }
            assert_eq!(fixture.ledger.snapshot().active_rooms, 0);
            assert!(port.batches.is_empty());
            assert!(port.in_flight.is_none());
        }
    }
}

#[test]
fn unsafe_timeout_stays_quarantined_and_emits_one_hard_deadline_fatal() {
    let fixture = fixture();
    let clock = Arc::new(ManualClock::new(1_000));
    let executor = SourceExecutor::new(
        Arc::clone(&fixture.ledger),
        Arc::clone(&fixture.quarantine),
        Arc::clone(&clock),
    );
    let mut fence = ComputeFence {
        ready: true,
        calls: 0,
    };
    let mut port = CpuNativePort::new(Arc::clone(&clock), [NativeMode::PendingForever]);
    let effect = executor
        .execute(
            SourceExecutionRequest {
                plan: &fixture.plan,
                handle: fixture.handle,
                source_fence: &mut fence,
                aux: &fixture.aux,
                completion: &fixture.completion,
                deadline_monotonic_ms: 1_020,
            },
            &mut port,
        )
        .expect("timeout outcome");
    let DataPlaneEffect::Quarantined { batch, .. } = effect else {
        panic!("expected quarantine");
    };
    assert_eq!(fixture.ledger.snapshot().quarantined_rooms, 1);
    assert!(
        !port
            .events
            .iter()
            .any(|event| matches!(event, NativeEvent::Free(_)))
    );
    let entered = clock.now_monotonic_ms();
    assert_eq!(
        fixture
            .quarantine
            .observe(
                fixture.handle,
                batch,
                NativeSafety::Pending,
                entered + 299_999,
            )
            .expect("pre-deadline"),
        QuarantineUpdate::Pending
    );
    assert_eq!(
        fixture
            .quarantine
            .observe(
                fixture.handle,
                batch,
                NativeSafety::Pending,
                entered + 300_000,
            )
            .expect("hard deadline"),
        QuarantineUpdate::LocalFatal
    );
    assert_eq!(
        fixture
            .quarantine
            .observe(
                fixture.handle,
                batch,
                NativeSafety::Pending,
                entered + 600_000,
            )
            .expect("duplicate hard deadline"),
        QuarantineUpdate::Pending
    );
    assert_eq!(fixture.ledger.snapshot().quarantined_rooms, 1);
    assert_eq!(fixture.quarantine.snapshot().fatal_effects, 1);
    assert_eq!(
        fixture
            .quarantine
            .observe(
                fixture.handle,
                batch,
                NativeSafety::SafeFailure,
                entered + 600_001,
            )
            .expect("later native terminal"),
        QuarantineUpdate::Released
    );
    port.free_safe(batch).expect("free safe terminal batch");
    assert_eq!(fixture.ledger.snapshot().active_rooms, 0);
}

#[path = "pd_executor/destination_worker.rs"]
mod destination_worker;
