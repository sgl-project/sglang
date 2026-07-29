use super::*;

pub(super) fn validate_wire_batch(length: usize) -> Result<(), TransportError> {
    if (1..=crate::pd::transport::MAX_TRANSPORT_BATCH).contains(&length) {
        Ok(())
    } else {
        Err(TransportError::InvalidBatch)
    }
}

pub(super) fn validate_pages(pages: &[u32], valid_token_count: u32) -> Result<(), TransportError> {
    let expected = usize::try_from(valid_token_count.div_ceil(64))
        .map_err(|_| TransportError::InvalidBatch)?;
    let unique = pages.iter().copied().collect::<BTreeSet<_>>();
    if !(1..=4096).contains(&valid_token_count)
        || pages.len() != expected
        || unique.len() != pages.len()
    {
        return Err(TransportError::InvalidBatch);
    }
    Ok(())
}

pub(super) fn room_fields(context: TransportRoomContext) -> RoomFields {
    RoomFields {
        decode_process_epoch: FixedBytes::new(context.room.key.decode_process_epoch.as_bytes()),
        bootstrap_room: context.room.key.bootstrap_room,
        attempt_id: FixedBytes::new(context.room.key.attempt_id.as_bytes()),
        generation: context.room.generation,
        request_contract_digest: context.request_digest,
    }
}

pub(super) fn room_fields_match(fields: &RoomFields, context: TransportRoomContext) -> bool {
    fields == &room_fields(context)
}

pub(super) fn validate_prepare(
    prepare: &PrepareRoom,
    context: TransportRoomContext,
) -> Result<(), TransportError> {
    if !room_fields_match(&prepare.room, context)
        || prepare
            .destination_registration_epoch
            .as_bytes()
            .iter()
            .all(|byte| *byte == 0)
        || prepare.destination_aux_slot >= 32
        || prepare.destination_completion_slot >= 32
        || prepare.chunk_sequence != 0
        || prepare.chunk_count != 1
        || !prepare.is_last_chunk
    {
        return Err(TransportError::Peer(PdReason::ProtocolMismatch));
    }
    let pages = destination_pages(prepare)?;
    validate_pages(&pages, prepare.valid_token_count)
        .map_err(|_| TransportError::Peer(PdReason::ProtocolMismatch))
}

pub(super) fn destination_blocks(pages: &[u32]) -> Vec<DestinationBlock> {
    let mut blocks = Vec::with_capacity(56 * pages.len());
    for region_id in 0_u16..56 {
        for destination_page in pages {
            blocks.push(DestinationBlock {
                region_id,
                destination_page: *destination_page,
                byte_offset: 0,
                byte_length: 131_072,
            });
        }
    }
    blocks
}

pub(super) fn destination_pages(prepare: &PrepareRoom) -> Result<Vec<u32>, TransportError> {
    if prepare.destination_blocks.is_empty() {
        return Err(TransportError::InvalidBatch);
    }
    let pages = prepare
        .destination_blocks
        .iter()
        .filter(|block| block.region_id == 0)
        .map(|block| block.destination_page)
        .collect::<Vec<_>>();
    if prepare.destination_blocks != destination_blocks(&pages) {
        return Err(TransportError::Peer(PdReason::ProtocolMismatch));
    }
    Ok(pages)
}

pub(super) fn source_pages(accepted: &PrepareAccepted) -> Result<Vec<u32>, TransportError> {
    let pages = accepted
        .kv_blocks
        .iter()
        .filter(|block| block.region_id == 0)
        .map(|block| block.source_page)
        .collect::<Vec<_>>();
    if pages.is_empty() {
        return Err(TransportError::Peer(PdReason::ProtocolMismatch));
    }
    Ok(pages)
}

pub(super) fn registration_epoch(
    bytes: FixedBytes<16>,
) -> Result<RegistrationEpoch, TransportError> {
    RegistrationEpoch::from_bytes(*bytes.as_array())
        .map_err(|_| TransportError::Peer(PdReason::ProtocolMismatch))
}

pub(super) fn planned_room(context: TransportRoomContext, plan: &TransferPlan) -> PlannedRoom {
    PlannedRoom {
        room: room_fields(context),
        transfer_plan_digest: FixedBytes::new(*plan.digest().as_bytes()),
    }
}

pub(super) fn completion_records(
    wire: &WirePlan,
) -> Result<([u8; AUX_BYTES], CompletionWrites), TransportError> {
    let aux = AuxRecord::encode(AuxRecordInput {
        first_token_valid: wire.first_token_id.is_some(),
        first_token_id: wire.first_token_id.unwrap_or(0),
        prompt_token_count: wire.plan.valid_token_count(),
        prefill_output_count: u32::from(wire.first_token_id.is_some()),
        request_digest: wire.request_digest,
    })
    .map_err(buffer_transport_error)?;
    let input = completion_input(wire);
    let completion = CompletionWrites::encode(&input, &aux).map_err(buffer_transport_error)?;
    Ok((aux, completion))
}

pub(super) fn completion_input(wire: &WirePlan) -> CompletionRecordInput {
    CompletionRecordInput {
        decode_process_epoch: wire.plan.room().key.decode_process_epoch,
        attempt_id: wire.plan.room().key.attempt_id,
        source_registration_epoch: wire.plan.source_registration_epoch(),
        destination_registration_epoch: wire.plan.destination_registration_epoch(),
        bootstrap_room: wire.plan.room().key.bootstrap_room,
        transfer_generation: wire.plan.transfer_generation(),
        chunk_sequence: 0,
        chunk_count: 1,
        page_count: wire.plan.valid_token_count().div_ceil(64),
        valid_token_count: wire.plan.valid_token_count(),
        request_digest: wire.request_digest,
        transfer_plan_digest: wire.plan.digest(),
    }
}

pub(super) fn mock_send_records(
    identity: &RuntimeIdentity,
    psk: &Psk,
    aux: &[u8; AUX_BYTES],
    completion: &[u8; COMPLETION_BYTES],
) -> Result<(), TransportError> {
    let port = identity
        .allowed_mooncake_ports
        .iter()
        .next()
        .copied()
        .ok_or(TransportError::InvalidBatch)?;
    let address = format!("{}:{port}", identity.expected_mooncake_host);
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut stream = loop {
        match StdTcpStream::connect(&address) {
            Ok(stream) => break stream,
            Err(_) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return Err(TransportError::LocalFatal(PdReason::PeerUnavailable)),
        }
    };
    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .map_err(|_| TransportError::LocalFatal(PdReason::TransferFailed))?;
    let mut payload = Vec::with_capacity(4 + AUX_BYTES + COMPLETION_BYTES);
    payload.extend_from_slice(b"SGMD");
    payload.extend_from_slice(aux);
    payload.extend_from_slice(completion);
    let tag = mock_data_tag(psk, &payload)?;
    stream
        .write_all(&payload)
        .and_then(|()| stream.write_all(&tag))
        .map_err(|_| TransportError::LocalFatal(PdReason::TransferFailed))
}

pub(super) fn mock_receive_records(
    listener: &StdTcpListener,
    psk: &Psk,
) -> Result<([u8; AUX_BYTES], [u8; COMPLETION_BYTES]), TransportError> {
    let (mut stream, _) = listener
        .accept()
        .map_err(|_| TransportError::Peer(PdReason::PeerUnavailable))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|_| TransportError::Peer(PdReason::PeerUnavailable))?;
    let mut payload = vec![0_u8; 4 + AUX_BYTES + COMPLETION_BYTES];
    let mut tag = [0_u8; 32];
    stream
        .read_exact(&mut payload)
        .and_then(|()| stream.read_exact(&mut tag))
        .map_err(|_| TransportError::Peer(PdReason::PeerUnavailable))?;
    if &payload[..4] != b"SGMD" {
        return Err(TransportError::Peer(PdReason::ProtocolMismatch));
    }
    let mut mac = Hmac::<Sha256>::new_from_slice(psk.as_bytes())
        .map_err(|_| TransportError::LocalFatal(PdReason::LocalFatal))?;
    mac.update(&payload);
    mac.verify_slice(&tag)
        .map_err(|_| TransportError::Peer(PdReason::ProtocolMismatch))?;
    let aux = payload[4..4 + AUX_BYTES]
        .try_into()
        .map_err(|_| TransportError::Peer(PdReason::ProtocolMismatch))?;
    let completion = payload[4 + AUX_BYTES..]
        .try_into()
        .map_err(|_| TransportError::Peer(PdReason::ProtocolMismatch))?;
    Ok((aux, completion))
}

pub(super) fn mock_data_tag(psk: &Psk, payload: &[u8]) -> Result<[u8; 32], TransportError> {
    let mut mac = Hmac::<Sha256>::new_from_slice(psk.as_bytes())
        .map_err(|_| TransportError::LocalFatal(PdReason::LocalFatal))?;
    mac.update(payload);
    Ok(mac.finalize().into_bytes().into())
}

pub(super) fn buffer_transport_error(error: crate::pd::buffer::BufferError) -> TransportError {
    classified_transport_error(crate::pd::runtime::FailureClass::for_buffer(&error))
}

pub(super) fn runtime_transport_error(error: RuntimeError) -> TransportError {
    classified_transport_error(crate::pd::runtime::FailureClass::for_runtime(&error))
}

fn classified_transport_error(class: crate::pd::runtime::FailureClass) -> TransportError {
    use crate::pd::runtime::FailureScope;

    match class.scope {
        FailureScope::Request => match class.reason {
            PdReason::CapacityExhausted => TransportError::CapacityExhausted,
            PdReason::StaleEpoch => TransportError::StaleHandle,
            PdReason::Unsupported => TransportError::WrongRole,
            _ => TransportError::InvalidBatch,
        },
        FailureScope::Room => TransportError::Room(class.reason),
        FailureScope::PeerSession => TransportError::Peer(class.reason),
        FailureScope::LocalFatal => TransportError::LocalFatal(class.reason),
    }
}

pub(super) fn parse_digest(value: &str) -> PyResult<FixedBytes<32>> {
    FixedBytes::from_hex(value).map_err(|_| py_transport_error(TransportError::InvalidBatch))
}

pub(super) fn opaque_handles(handles: Vec<u64>) -> Vec<OpaqueHandle> {
    handles.into_iter().map(OpaqueHandle::from_raw).collect()
}

pub(super) fn raw_handles(handles: &[OpaqueHandle]) -> Vec<u64> {
    handles.iter().map(|handle| handle.raw()).collect()
}

pub(super) fn batch_items(
    handles: Vec<u64>,
    results: Vec<Result<(), TransportError>>,
) -> Vec<PyPdBatchItem> {
    handles
        .into_iter()
        .zip(results)
        .map(|(handle, result)| batch_item(handle, result))
        .collect()
}

pub(super) fn batch_item(handle: u64, result: Result<(), TransportError>) -> PyPdBatchItem {
    match result {
        Ok(()) => PyPdBatchItem {
            handle,
            terminal_generation: OpaqueHandle::from_raw(handle).generation(),
            ok: true,
            pd_reason: PdReason::Success.code().to_string(),
            retryable: false,
        },
        Err(error) => PyPdBatchItem {
            handle,
            terminal_generation: 0,
            ok: false,
            pd_reason: error.reason().code().to_string(),
            retryable: error.reason().retryable(),
        },
    }
}

pub(super) fn created_batch_item(handle: u64, terminal_generation: u64) -> PyPdBatchItem {
    PyPdBatchItem {
        handle,
        terminal_generation,
        ok: true,
        pd_reason: PdReason::Success.code().to_string(),
        retryable: false,
    }
}

pub(super) fn poll_result(
    handle: u64,
    result: Result<TransportPollResult, TransportError>,
) -> PyPdPollResult {
    match result {
        Ok(result) => PyPdPollResult {
            handle: result.handle.raw(),
            ok: true,
            status: result.status as u8,
            pd_reason: result.reason.code().to_string(),
            retryable: result.retryable,
            transfer_bytes: result.transfer_bytes,
            transfer_latency_ms: result.transfer_latency_ms,
            terminal_generation: result.terminal_generation,
            first_token_id: result.first_token_id,
            first_token_consumed: result.first_token_consumed,
        },
        Err(error) => PyPdPollResult {
            handle,
            ok: false,
            status: 0,
            pd_reason: error.reason().code().to_string(),
            retryable: error.reason().retryable(),
            transfer_bytes: 0,
            transfer_latency_ms: 0,
            terminal_generation: 0,
            first_token_id: None,
            first_token_consumed: true,
        },
    }
}

pub(super) fn parse_reason(value: &str) -> Option<PdReason> {
    [
        PdReason::Success,
        PdReason::RequestInvalid,
        PdReason::Unsupported,
        PdReason::CapacityExhausted,
        PdReason::ProtocolMismatch,
        PdReason::PeerUnavailable,
        PdReason::RendezvousTimeout,
        PdReason::TransferTimeout,
        PdReason::TransferFailed,
        PdReason::AckTimeout,
        PdReason::Aborted,
        PdReason::StaleEpoch,
        PdReason::LocalFatal,
    ]
    .into_iter()
    .find(|reason| reason.code() == value)
}

pub(super) fn role_name(role: Role) -> &'static str {
    match role {
        Role::Prefill => "prefill",
        Role::Decode => "decode",
    }
}

pub(super) fn lifecycle_name(lifecycle: RuntimeLifecycle) -> &'static str {
    match lifecycle {
        RuntimeLifecycle::Starting => "Starting",
        RuntimeLifecycle::LocalReady => "LocalReady",
        RuntimeLifecycle::PairReady => "PairReady",
        RuntimeLifecycle::Draining => "Draining",
        RuntimeLifecycle::Fatal => "Fatal",
        RuntimeLifecycle::Stopped => "Stopped",
    }
}

pub(super) fn fatal_source_name(source: crate::pd::runtime::FatalSource) -> &'static str {
    match source {
        crate::pd::runtime::FatalSource::StartupInvariant => "StartupInvariant",
        crate::pd::runtime::FatalSource::WorkerExit => "WorkerExit",
        crate::pd::runtime::FatalSource::CommandChannelClosed => "CommandChannelClosed",
        crate::pd::runtime::FatalSource::EngineOwner => "EngineOwner",
        crate::pd::runtime::FatalSource::RegistryInvariant => "RegistryInvariant",
        crate::pd::runtime::FatalSource::QuarantineHardDeadline => "QuarantineHardDeadline",
        crate::pd::runtime::FatalSource::ShutdownUnsafe => "ShutdownUnsafe",
        crate::pd::runtime::FatalSource::ProtocolInvariant => "ProtocolInvariant",
    }
}

pub(super) fn shutdown_phase_name(phase: crate::pd::runtime::ShutdownPhase) -> &'static str {
    match phase {
        crate::pd::runtime::ShutdownPhase::Idle => "Idle",
        crate::pd::runtime::ShutdownPhase::ReadinessDown => "ReadinessDown",
        crate::pd::runtime::ShutdownPhase::GoAway => "GoAway",
        crate::pd::runtime::ShutdownPhase::StopAccepting => "StopAccepting",
        crate::pd::runtime::ShutdownPhase::DrainingRooms => "DrainingRooms",
        crate::pd::runtime::ShutdownPhase::AbortingRooms => "AbortingRooms",
        crate::pd::runtime::ShutdownPhase::NativeSafety => "NativeSafety",
        crate::pd::runtime::ShutdownPhase::SchedulerRelease => "SchedulerRelease",
        crate::pd::runtime::ShutdownPhase::WorkerJoin => "WorkerJoin",
        crate::pd::runtime::ShutdownPhase::EngineQuiesce => "EngineQuiesce",
        crate::pd::runtime::ShutdownPhase::ConnectionClose => "ConnectionClose",
        crate::pd::runtime::ShutdownPhase::RegionUnregister => "RegionUnregister",
        crate::pd::runtime::ShutdownPhase::EngineDestroy => "EngineDestroy",
        crate::pd::runtime::ShutdownPhase::Stopped => "Stopped",
    }
}

pub(super) fn shutdown_outcome_name(
    outcome: crate::pd::runtime::RuntimeShutdownOutcome,
) -> &'static str {
    match outcome {
        crate::pd::runtime::RuntimeShutdownOutcome::SafeTerminal => "SafeTerminal",
        crate::pd::runtime::RuntimeShutdownOutcome::FatalUnsafe => "FatalUnsafe",
    }
}

pub(super) fn worker_lifecycle_name(
    lifecycle: crate::pd::runtime::WorkerLifecycle,
) -> &'static str {
    match lifecycle {
        crate::pd::runtime::WorkerLifecycle::Starting => "Starting",
        crate::pd::runtime::WorkerLifecycle::Running => "Running",
        crate::pd::runtime::WorkerLifecycle::Quiescing => "Quiescing",
        crate::pd::runtime::WorkerLifecycle::Joined => "Joined",
        crate::pd::runtime::WorkerLifecycle::Failed => "Failed",
    }
}

pub(super) fn py_transport_error(error: TransportError) -> PyErr {
    let message = error.reason().code();
    match error {
        TransportError::InvalidBatch => PyErr::new::<pyo3::exceptions::PyValueError, _>(message),
        _ => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(message),
    }
}

#[cfg(test)]
mod tests {
    use crate::pd::runtime::{
        FailureClass, FailureScope, FatalSource, RuntimeShutdownOutcome, ShutdownPhase,
        WorkerLifecycle,
    };

    use super::*;

    #[test]
    fn frozen_lifecycle_names_cover_every_internal_state() {
        for (source, expected) in [
            (FatalSource::StartupInvariant, "StartupInvariant"),
            (FatalSource::WorkerExit, "WorkerExit"),
            (FatalSource::CommandChannelClosed, "CommandChannelClosed"),
            (FatalSource::EngineOwner, "EngineOwner"),
            (FatalSource::RegistryInvariant, "RegistryInvariant"),
            (
                FatalSource::QuarantineHardDeadline,
                "QuarantineHardDeadline",
            ),
            (FatalSource::ShutdownUnsafe, "ShutdownUnsafe"),
            (FatalSource::ProtocolInvariant, "ProtocolInvariant"),
        ] {
            assert_eq!(fatal_source_name(source), expected);
        }
        for (phase, expected) in [
            (ShutdownPhase::Idle, "Idle"),
            (ShutdownPhase::ReadinessDown, "ReadinessDown"),
            (ShutdownPhase::GoAway, "GoAway"),
            (ShutdownPhase::StopAccepting, "StopAccepting"),
            (ShutdownPhase::DrainingRooms, "DrainingRooms"),
            (ShutdownPhase::AbortingRooms, "AbortingRooms"),
            (ShutdownPhase::NativeSafety, "NativeSafety"),
            (ShutdownPhase::SchedulerRelease, "SchedulerRelease"),
            (ShutdownPhase::WorkerJoin, "WorkerJoin"),
            (ShutdownPhase::EngineQuiesce, "EngineQuiesce"),
            (ShutdownPhase::ConnectionClose, "ConnectionClose"),
            (ShutdownPhase::RegionUnregister, "RegionUnregister"),
            (ShutdownPhase::EngineDestroy, "EngineDestroy"),
            (ShutdownPhase::Stopped, "Stopped"),
        ] {
            assert_eq!(shutdown_phase_name(phase), expected);
        }
        assert_eq!(
            shutdown_outcome_name(RuntimeShutdownOutcome::SafeTerminal),
            "SafeTerminal"
        );
        assert_eq!(
            shutdown_outcome_name(RuntimeShutdownOutcome::FatalUnsafe),
            "FatalUnsafe"
        );
        for (lifecycle, expected) in [
            (WorkerLifecycle::Starting, "Starting"),
            (WorkerLifecycle::Running, "Running"),
            (WorkerLifecycle::Quiescing, "Quiescing"),
            (WorkerLifecycle::Joined, "Joined"),
            (WorkerLifecycle::Failed, "Failed"),
        ] {
            assert_eq!(worker_lifecycle_name(lifecycle), expected);
        }
    }

    #[test]
    fn classified_transport_errors_preserve_all_four_failure_scopes() {
        for (class, expected) in [
            (
                FailureClass::new(FailureScope::Request, PdReason::CapacityExhausted),
                TransportError::CapacityExhausted,
            ),
            (
                FailureClass::new(FailureScope::Request, PdReason::StaleEpoch),
                TransportError::StaleHandle,
            ),
            (
                FailureClass::new(FailureScope::Request, PdReason::Unsupported),
                TransportError::WrongRole,
            ),
            (
                FailureClass::new(FailureScope::Request, PdReason::RequestInvalid),
                TransportError::InvalidBatch,
            ),
            (
                FailureClass::new(FailureScope::Room, PdReason::TransferFailed),
                TransportError::Room(PdReason::TransferFailed),
            ),
            (
                FailureClass::new(FailureScope::PeerSession, PdReason::ProtocolMismatch),
                TransportError::Peer(PdReason::ProtocolMismatch),
            ),
            (
                FailureClass::new(FailureScope::LocalFatal, PdReason::LocalFatal),
                TransportError::LocalFatal(PdReason::LocalFatal),
            ),
        ] {
            assert_eq!(classified_transport_error(class), expected);
        }
    }
}
