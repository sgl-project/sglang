use std::collections::HashMap;
use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use sglang_server::mooncake::{BatchSnapshot, OperationProgress, OperationState};
use sglang_server::pd::buffer::{
    AUX_BYTES, AuxRecord, AuxRecordInput, BufferError, CapacityLedger, CompletionRecordInput,
    CompletionWrites, DataPlaneEffect, DestinationExecutor, DestinationRecordPort,
    DestinationVisibilityFence, GpuDirectFlushPort, NativeBatchToken, NativePhase, NativeSafety,
    NativeStageCommand, NativeStagePort, QuarantineManager, QuarantineUpdate, ReservationRequest,
    SourceComputeFence, SourceExecutionRequest, SourceExecutor, TableUseTracker, TransferPlan,
    TransferPlanInput, TransferStage, TransitionResult, clear_partial_page_tail,
    evaluate_native_fence,
};
use sglang_server::pd::config::PdProfileV1;
use sglang_server::pd::protocol::{FixedBytes, KvBlock};
use sglang_server::pd::room::{
    AttemptId, Clock, ManualClock, ProcessEpoch, RegistrationEpoch, RoomId, RoomKey,
};

const FRAME_LIMIT: usize = 32 * 1024 * 1024;
const KV_PAGE_BYTES: usize = 131_072;

const PREPARE: u8 = 1;
const ACCEPTED: u8 = 2;
const KV: u8 = 3;
const AUX: u8 = 4;
const COMPLETION_BODY: u8 = 5;
const COMPLETION_MARKER: u8 = 6;
const NATIVE_SUCCESS: u8 = 7;
const NATIVE_FAILURE: u8 = 8;
const NATIVE_PENDING: u8 = 9;
const DATA_READY: u8 = 10;
const TRANSFER_COMPLETE: u8 = 11;
const TRANSFER_COMPLETE_ACK: u8 = 12;
const TRANSFER_FAILED: u8 = 13;
const QUARANTINE_RECOVERED: u8 = 14;

type HarnessResult<T> = Result<T, String>;

#[derive(Clone)]
struct RoomShape {
    room_number: u64,
    valid_tokens: u32,
    destination_pages: Vec<u32>,
    prefill_first: bool,
}

struct Frame {
    kind: u8,
    payload: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scenario {
    Positive,
    SafeFailure,
    TimeoutRecovery,
    Disconnect,
}

impl Scenario {
    fn parse(value: &str) -> HarnessResult<Self> {
        match value {
            "positive" => Ok(Self::Positive),
            "safe_failure" => Ok(Self::SafeFailure),
            "timeout_recovery" => Ok(Self::TimeoutRecovery),
            "disconnect" => Ok(Self::Disconnect),
            _ => Err("unknown data-plane harness scenario".into()),
        }
    }
}

#[path = "pd_data_mock_peer/engine.rs"]
mod engine;
#[path = "pd_data_mock_peer/wire.rs"]
mod wire;

use engine::*;
use wire::*;

fn main() {
    if let Err(error) = run() {
        eprintln!("PD data mock peer failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> HarnessResult<()> {
    let arguments = env::args().collect::<Vec<_>>();
    if arguments.len() != 4 {
        return Err("usage: pd_data_mock_peer <prefill|decode> <address> <scenario>".into());
    }
    let scenario = Scenario::parse(&arguments[3])?;
    capacity_probe()?;
    match arguments[1].as_str() {
        "prefill" => run_prefill(&arguments[2], scenario),
        "decode" => run_decode(&arguments[2], scenario),
        _ => Err("role must be prefill or decode".into()),
    }
}

fn run_prefill(address: &str, scenario: Scenario) -> HarnessResult<()> {
    let listener = TcpListener::bind(address)
        .map_err(|error| format!("data listener bind failed: {error}"))?;
    let (mut stream, _) = listener
        .accept()
        .map_err(|error| format!("data accept failed: {error}"))?;
    stream
        .set_nodelay(true)
        .map_err(|error| format!("set TCP_NODELAY failed: {error}"))?;

    let profile = PdProfileV1::load_embedded().map_err(display)?;
    let ledger = Arc::new(CapacityLedger::new(
        &profile,
        TableUseTracker::new(),
        TableUseTracker::new(),
    ));
    let quarantine = Arc::new(QuarantineManager::new(Arc::clone(&ledger)));
    let clock = Arc::new(ManualClock::new(100));
    let executor = SourceExecutor::new(
        Arc::clone(&ledger),
        Arc::clone(&quarantine),
        Arc::clone(&clock),
    );
    let mut completed = 0;

    for expected_shape in room_shapes(scenario) {
        let prepare = receive_frame(&mut stream)?;
        if prepare.kind != PREPARE {
            return Err("prefill expected Prepare".into());
        }
        let shape = decode_prepare(&prepare.payload)?;
        if shape.room_number != expected_shape.room_number
            || shape.valid_tokens != expected_shape.valid_tokens
            || shape.destination_pages != expected_shape.destination_pages
            || shape.prefill_first != expected_shape.prefill_first
        {
            return Err("Prepare did not match the local room expectation".into());
        }

        let source_pages = source_pages(&shape);
        let plan = transfer_plan(&shape, source_pages.clone())?;
        send_frame(
            &mut stream,
            ACCEPTED,
            &encode_accepted(&source_pages, plan.digest().as_bytes()),
        )?;
        let handle = reserve_plan(&ledger, &plan, true)?;
        let (aux, completion_input, completion) = records_for(&plan)?;
        let mut fence = ReadyFence;
        let mut port = SocketNativePort::new(&mut stream, Arc::clone(&clock));
        let effect = executor
            .execute(
                SourceExecutionRequest {
                    plan: &plan,
                    handle,
                    source_fence: &mut fence,
                    aux: &aux,
                    completion: &completion,
                    deadline_monotonic_ms: 500,
                },
                &mut port,
            )
            .map_err(display)?;

        match (scenario, effect) {
            (Scenario::Positive, DataPlaneEffect::DataReady { identity })
            | (Scenario::Disconnect, DataPlaneEffect::DataReady { identity }) => {
                if identity.transfer_plan_digest != plan.digest() {
                    return Err("source executor returned a stale data identity".into());
                }
                send_frame(&mut stream, DATA_READY, plan.digest().as_bytes())?;
                if scenario == Scenario::Disconnect {
                    match receive_frame(&mut stream) {
                        Err(_) => {
                            ledger.release_failed_safe(handle).map_err(display)?;
                            assert_empty(&ledger, &quarantine)?;
                            println!(
                                "DATA_PLANE_DISCONNECT role=prefill safe_release=true quarantine=0"
                            );
                            return Ok(());
                        }
                        Ok(_) => {
                            return Err("disconnect scenario unexpectedly received Complete".into());
                        }
                    }
                }
                let complete = receive_frame(&mut stream)?;
                if complete.kind != TRANSFER_COMPLETE
                    || complete.payload.as_slice() != plan.digest().as_bytes()
                {
                    return Err("prefill expected matching TransferComplete".into());
                }
                ledger.release_source_safe(handle).map_err(display)?;
                ledger.handoff_destination(handle).map_err(display)?;
                ledger.release_terminal(handle).map_err(display)?;
                send_frame(&mut stream, TRANSFER_COMPLETE_ACK, plan.digest().as_bytes())?;
                completed += 1;
            }
            (Scenario::SafeFailure, DataPlaneEffect::TransferFailed { .. }) => {
                send_frame(&mut stream, TRANSFER_FAILED, plan.digest().as_bytes())?;
                assert_empty(&ledger, &quarantine)?;
                println!("DATA_PLANE_FAILURE role=prefill phase=aux later_submits=0 baseline=true");
                return Ok(());
            }
            (
                Scenario::TimeoutRecovery,
                DataPlaneEffect::Quarantined {
                    identity, batch, ..
                },
            ) => {
                if identity.transfer_plan_digest != plan.digest() {
                    return Err("quarantine identity did not match the plan".into());
                }
                let snapshot = port.poll(batch).map_err(display)?;
                let expected = plan
                    .kv_blocks()
                    .iter()
                    .map(|block| block.byte_length)
                    .collect::<Vec<_>>();
                let safety = evaluate_native_fence(&snapshot, &expected);
                if safety != NativeSafety::SafeSuccess {
                    return Err("mock native transfer did not recover to safe success".into());
                }
                port.free_safe(batch).map_err(display)?;
                if quarantine
                    .observe(handle, batch, safety, clock.now_monotonic_ms() + 1)
                    .map_err(display)?
                    != QuarantineUpdate::Released
                {
                    return Err("quarantine did not release after native safety".into());
                }
                send_frame(&mut stream, QUARANTINE_RECOVERED, plan.digest().as_bytes())?;
                assert_empty(&ledger, &quarantine)?;
                println!(
                    "DATA_PLANE_TIMEOUT role=prefill quarantined=true recovered=true baseline=true"
                );
                return Ok(());
            }
            _ => return Err("source executor returned an unexpected typed effect".into()),
        }

        if completion_input.transfer_plan_digest != plan.digest() {
            return Err("completion input did not retain the plan digest".into());
        }
    }

    assert_empty(&ledger, &quarantine)?;
    println!(
        "DATA_PLANE_COMPLETE role=prefill rooms={completed} room_zero=true orders=both fragmented=true capacity=true"
    );
    Ok(())
}

fn run_decode(address: &str, scenario: Scenario) -> HarnessResult<()> {
    let mut stream = connect_with_retry(address)?;
    stream
        .set_nodelay(true)
        .map_err(|error| format!("set TCP_NODELAY failed: {error}"))?;
    let profile = PdProfileV1::load_embedded().map_err(display)?;
    let ledger = Arc::new(CapacityLedger::new(
        &profile,
        TableUseTracker::new(),
        TableUseTracker::new(),
    ));
    let mut completed = 0;

    for shape in room_shapes(scenario) {
        send_frame(&mut stream, PREPARE, &encode_prepare(&shape))?;
        let accepted = receive_frame(&mut stream)?;
        if accepted.kind != ACCEPTED {
            return Err("decode expected PrepareAccepted".into());
        }
        let (source_pages, digest) = decode_accepted(&accepted.payload)?;
        let plan = transfer_plan(&shape, source_pages)?;
        if digest != *plan.digest().as_bytes() {
            return Err("decode independently recomputed a different plan digest".into());
        }
        let handle = reserve_plan(&ledger, &plan, false)?;
        let receive = receive_native_data(&mut stream, &plan, scenario)?;

        match scenario {
            Scenario::SafeFailure => {
                if receive.terminal_kind != TRANSFER_FAILED {
                    return Err("decode expected typed transfer failure".into());
                }
                ledger.release_failed_safe(handle).map_err(display)?;
                assert_ledger_empty(&ledger)?;
                println!("DATA_PLANE_FAILURE role=decode phase=aux later_submits=0 baseline=true");
                return Ok(());
            }
            Scenario::TimeoutRecovery => {
                if receive.terminal_kind != QUARANTINE_RECOVERED {
                    return Err("decode expected quarantine recovery".into());
                }
                ledger.release_failed_safe(handle).map_err(display)?;
                assert_ledger_empty(&ledger)?;
                println!(
                    "DATA_PLANE_TIMEOUT role=decode quarantined=true recovered=true baseline=true"
                );
                return Ok(());
            }
            Scenario::Disconnect => {
                if receive.terminal_kind != DATA_READY {
                    return Err("decode expected DataReady before disconnect".into());
                }
                ledger.release_failed_safe(handle).map_err(display)?;
                assert_ledger_empty(&ledger)?;
                println!("DATA_PLANE_DISCONNECT role=decode safe_release=true quarantine=0");
                return Ok(());
            }
            Scenario::Positive => {}
        }

        if receive.terminal_kind != DATA_READY {
            return Err("decode expected DataReady".into());
        }
        complete_remote_stages(&ledger, handle)?;
        let (_, completion_input, _) = records_for(&plan)?;
        let flush_calls = Arc::new(Mutex::new(0_u64));
        let mut visibility = DestinationVisibilityFence::new(
            5,
            CpuFlush {
                calls: Arc::clone(&flush_calls),
            },
        )
        .map_err(display)?;
        let destination = DestinationExecutor::new(Arc::clone(&ledger));
        let identity = sglang_server::pd::buffer::DataPlaneIdentity::from_plan(&plan);
        let mut records = receive.records;
        let effect = destination
            .validate_ready(
                &plan,
                handle,
                identity,
                &mut visibility,
                &mut records,
                &completion_input,
            )
            .map_err(display)?;
        if !matches!(effect, DataPlaneEffect::TransferComplete { .. }) {
            return Err("destination did not validate the transferred records".into());
        }
        destination
            .validate_ready(
                &plan,
                handle,
                identity,
                &mut visibility,
                &mut records,
                &completion_input,
            )
            .map_err(display)?;
        if *flush_calls.lock().map_err(|_| "flush lock poisoned")? != 1
            || records.reads != 2
            || records.clears != 56
        {
            return Err("duplicate DataReady repeated a destination side effect".into());
        }
        send_frame(&mut stream, TRANSFER_COMPLETE, plan.digest().as_bytes())?;
        let ack = receive_frame(&mut stream)?;
        if ack.kind != TRANSFER_COMPLETE_ACK || ack.payload.as_slice() != plan.digest().as_bytes() {
            return Err("decode expected matching TransferCompleteAck".into());
        }
        if destination
            .commit_after_ack(handle, identity)
            .map_err(display)?
            != TransitionResult::Applied
            || destination
                .commit_after_ack(handle, identity)
                .map_err(display)?
                != TransitionResult::AlreadyApplied
        {
            return Err("destination handoff was not exactly-once".into());
        }
        ledger.release_source_safe(handle).map_err(display)?;
        ledger.release_terminal(handle).map_err(display)?;
        completed += 1;
    }

    assert_ledger_empty(&ledger)?;
    println!(
        "DATA_PLANE_COMPLETE role=decode rooms={completed} room_zero=true orders=both fragmented=true capacity=true"
    );
    Ok(())
}

fn complete_remote_stages(
    ledger: &CapacityLedger,
    handle: sglang_server::pd::buffer::LeaseHandle,
) -> HarnessResult<()> {
    for stage in [
        TransferStage::Kv,
        TransferStage::Aux,
        TransferStage::Completion,
    ] {
        ledger.begin_stage(handle, stage).map_err(display)?;
        ledger.finish_stage(handle, stage).map_err(display)?;
    }
    Ok(())
}

fn reserve_plan(
    ledger: &CapacityLedger,
    plan: &TransferPlan,
    source_side: bool,
) -> HarnessResult<sglang_server::pd::buffer::LeaseHandle> {
    ledger
        .reserve(ReservationRequest {
            room: plan.room(),
            handle_generation: plan.transfer_generation(),
            source_pages: plan.source_pages().to_vec(),
            destination_pages: plan.destination_pages().to_vec(),
            aux_slot: if source_side {
                plan.source_aux_slot()
            } else {
                plan.destination_aux_slot()
            },
            completion_slot: if source_side {
                plan.source_completion_slot()
            } else {
                plan.destination_completion_slot()
            },
            request_slot: if source_side {
                plan.source_aux_slot()
            } else {
                plan.destination_aux_slot()
            },
            kv_bytes: plan.expected_kv_bytes(),
            deadline_monotonic_ms: 10_000,
        })
        .map_err(display)
}

fn records_for(
    plan: &TransferPlan,
) -> HarnessResult<([u8; AUX_BYTES], CompletionRecordInput, CompletionWrites)> {
    let request_digest = FixedBytes::new(
        [u8::try_from(plan.room().key.bootstrap_room)
            .unwrap_or(0)
            .saturating_add(0xa0); 32],
    );
    let aux = AuxRecord::encode(AuxRecordInput {
        first_token_valid: true,
        first_token_id: 100 + i32::try_from(plan.room().key.bootstrap_room).map_err(display)?,
        prompt_token_count: plan.valid_token_count(),
        prefill_output_count: 1,
        request_digest,
    })
    .map_err(display)?;
    let input = CompletionRecordInput {
        decode_process_epoch: plan.room().key.decode_process_epoch,
        attempt_id: plan.room().key.attempt_id,
        source_registration_epoch: plan.source_registration_epoch(),
        destination_registration_epoch: plan.destination_registration_epoch(),
        bootstrap_room: plan.room().key.bootstrap_room,
        transfer_generation: plan.transfer_generation(),
        chunk_sequence: 0,
        chunk_count: 1,
        page_count: u32::try_from(plan.destination_pages().len()).map_err(display)?,
        valid_token_count: plan.valid_token_count(),
        request_digest,
        transfer_plan_digest: plan.digest(),
    };
    let completion = CompletionWrites::encode(&input, &aux).map_err(display)?;
    Ok((aux, input, completion))
}

fn transfer_plan(shape: &RoomShape, source_pages: Vec<u32>) -> HarnessResult<TransferPlan> {
    TransferPlan::new(TransferPlanInput {
        room: room_id(shape.room_number)?,
        transfer_generation: shape.room_number + 7,
        source_registration_epoch: registration_epoch(2)?,
        destination_registration_epoch: registration_epoch(3)?,
        source_pages,
        destination_pages: shape.destination_pages.clone(),
        source_aux_slot: u16::try_from(shape.room_number + 1).map_err(display)?,
        destination_aux_slot: u16::try_from(shape.room_number + 17).map_err(display)?,
        source_completion_slot: u16::try_from(shape.room_number + 1).map_err(display)?,
        destination_completion_slot: u16::try_from(shape.room_number + 17).map_err(display)?,
        valid_token_count: shape.valid_tokens,
        chunk_sequence: 0,
        chunk_count: 1,
        is_last_chunk: true,
    })
    .map_err(display)
}

fn room_id(room_number: u64) -> HarnessResult<RoomId> {
    let process = ProcessEpoch::from_bytes(uuid_bytes(1)).map_err(display)?;
    let attempt = AttemptId::from_bytes(uuid_bytes(
        u8::try_from(room_number).unwrap_or(0).saturating_add(10),
    ))
    .map_err(display)?;
    let key = RoomKey::new(process, room_number, attempt).map_err(display)?;
    RoomId::new(key, 1).map_err(display)
}

fn registration_epoch(seed: u8) -> HarnessResult<RegistrationEpoch> {
    RegistrationEpoch::from_bytes(uuid_bytes(seed)).map_err(display)
}

fn uuid_bytes(seed: u8) -> [u8; 16] {
    let mut bytes = [seed; 16];
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    bytes
}

fn source_pages(shape: &RoomShape) -> Vec<u32> {
    (0..shape.destination_pages.len())
        .map(|logical| {
            u32::try_from(shape.room_number)
                .unwrap_or(0)
                .saturating_mul(31)
                .saturating_add(u32::try_from(logical).unwrap_or(0).saturating_mul(5))
                .saturating_add(3)
        })
        .collect()
}

fn room_shapes(scenario: Scenario) -> Vec<RoomShape> {
    let all = vec![
        RoomShape {
            room_number: 0,
            valid_tokens: 1,
            destination_pages: vec![7],
            prefill_first: true,
        },
        RoomShape {
            room_number: 1,
            valid_tokens: 63,
            destination_pages: vec![19],
            prefill_first: false,
        },
        RoomShape {
            room_number: 2,
            valid_tokens: 65,
            destination_pages: vec![44, 9],
            prefill_first: true,
        },
        RoomShape {
            room_number: 3,
            valid_tokens: 128,
            destination_pages: vec![61, 12],
            prefill_first: false,
        },
    ];
    if scenario == Scenario::Positive {
        all
    } else {
        all.into_iter().take(1).collect()
    }
}

fn capacity_probe() -> HarnessResult<()> {
    let profile = PdProfileV1::load_embedded().map_err(display)?;
    let ledger = CapacityLedger::new(&profile, TableUseTracker::new(), TableUseTracker::new());
    let mut handles = Vec::new();
    for index in 0_u16..32 {
        handles.push(
            ledger
                .reserve(ReservationRequest {
                    room: room_id(u64::from(index) + 100)?,
                    handle_generation: 1,
                    source_pages: vec![u32::from(index)],
                    destination_pages: vec![u32::from(index)],
                    aux_slot: index,
                    completion_slot: index,
                    request_slot: index,
                    kv_bytes: 1,
                    deadline_monotonic_ms: 1,
                })
                .map_err(display)?,
        );
    }
    let exhausted = ledger.reserve(ReservationRequest {
        room: room_id(999)?,
        handle_generation: 1,
        source_pages: vec![100],
        destination_pages: vec![100],
        aux_slot: 0,
        completion_slot: 0,
        request_slot: 0,
        kv_bytes: 1,
        deadline_monotonic_ms: 1,
    });
    if !matches!(
        exhausted,
        Err(BufferError::CapacityExhausted {
            resource: "active_rooms"
        })
    ) {
        return Err("32-room capacity did not fail closed".into());
    }
    for handle in handles {
        ledger.abort_pre_submit(handle).map_err(display)?;
    }
    assert_ledger_empty(&ledger)
}

fn assert_empty(ledger: &CapacityLedger, quarantine: &QuarantineManager) -> HarnessResult<()> {
    assert_ledger_empty(ledger)?;
    if quarantine.snapshot().entries != 0 {
        return Err("quarantine did not return to baseline".into());
    }
    Ok(())
}

fn assert_ledger_empty(ledger: &CapacityLedger) -> HarnessResult<()> {
    let snapshot = ledger.snapshot();
    if snapshot.active_rooms != 0
        || snapshot.source_kv_pages != 0
        || snapshot.destination_kv_pages != 0
        || snapshot.aux_slots != 0
        || snapshot.completion_slots != 0
        || snapshot.request_slots != 0
        || snapshot.in_flight_transfers != 0
        || snapshot.pending_bytes != 0
        || snapshot.quarantined_rooms != 0
    {
        return Err("capacity ledger did not return to baseline".into());
    }
    Ok(())
}
