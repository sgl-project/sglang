use super::support::*;
use super::*;

pub(super) fn transport_worker(
    mut core: PdTransportCore,
    identity: RuntimeIdentity,
    psk: Psk,
    control_host: String,
    control_port: u16,
    bootstrap_owner: BootstrapOwner,
    receiver: flume::Receiver<TransportCommand>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(_) => return,
    };
    let mut connection: Option<PairConnection> = None;
    let mut mock_endpoint: Option<MockDataEndpoint> = None;
    let mut native_sender: Option<NativeSender> = None;
    let mut native_receiver: Option<NativeReceiver> = None;
    let mut pending_prepares = HashMap::<u64, PrepareRoom>::new();
    let mut wire_plans = HashMap::<u64, WirePlan>::new();
    while let Ok(command) = receiver.recv() {
        match command {
            TransportCommand::Start { reply } => {
                let result = if connection.is_some() {
                    Err(TransportError::InvalidTransition)
                } else {
                    core.start_local(crate::pd::transport::PD_REGION_COUNT)
                        .and_then(|()| {
                            bootstrap_pair(
                                &identity,
                                &psk,
                                &control_host,
                                control_port,
                                bootstrap_owner.port(),
                                &runtime,
                            )
                        })
                        .and_then(|pair| {
                            let readiness = pair.readiness().clone();
                            core.activate_pair(
                                readiness,
                                crate::pd::transport::PD_REGION_COUNT,
                                true,
                            )?;
                            connection = Some(pair);
                            mock_endpoint = mock_endpoint_for(&identity, &bootstrap_owner)?;
                            match &bootstrap_owner {
                                BootstrapOwner::Mock(_) => {}
                                BootstrapOwner::Native(port) => match identity.role {
                                    Role::Prefill => {
                                        native_sender =
                                            Some(NativeSender::new(port, &identity.profile)?);
                                    }
                                    Role::Decode => {
                                        native_receiver =
                                            Some(NativeReceiver::new(port, &identity.profile));
                                    }
                                },
                            }
                            Ok(())
                        })
                };
                let _ = reply.send(result);
            }
            TransportCommand::SenderCreate { input, reply } => {
                let _ = reply.send(core.sender_create(input));
            }
            TransportCommand::SenderCreateMany { inputs, reply } => {
                let result = core.sender_create_many(&inputs).map(|results| {
                    results
                        .into_iter()
                        .map(|result| {
                            result.and_then(|handle| {
                                core.room_context(handle)
                                    .map(|context| (handle, context.room.generation))
                            })
                        })
                        .collect()
                });
                let _ = reply.send(result);
            }
            TransportCommand::SenderInit { handles, reply } => {
                let result = connection
                    .as_mut()
                    .ok_or(TransportError::NotReady)
                    .and_then(|pair| {
                        sender_init_wire(&mut core, pair, &runtime, &handles, &mut pending_prepares)
                    });
                let _ = reply.send(result);
            }
            TransportCommand::SenderSend {
                chunks,
                cuda_stream,
                reply,
            } => {
                let result = match connection.as_mut() {
                    Some(pair) => sender_send_wire(
                        &mut core,
                        pair,
                        &runtime,
                        &identity,
                        &psk,
                        mock_endpoint.as_mut(),
                        native_sender.as_mut(),
                        &chunks,
                        cuda_stream,
                        &mut pending_prepares,
                    ),
                    None => Err(TransportError::NotReady),
                };
                let _ = reply.send(result);
            }
            TransportCommand::ReceiverCreate { inputs, reply } => {
                let result = core.receiver_create_many(&inputs).map(|results| {
                    results
                        .into_iter()
                        .map(|result| {
                            result.and_then(|handle| {
                                core.room_context(handle)
                                    .map(|context| (handle, context.room.generation))
                            })
                        })
                        .collect()
                });
                let _ = reply.send(result);
            }
            TransportCommand::ReceiverPrepare { inputs, reply } => {
                let result = connection
                    .as_mut()
                    .ok_or(TransportError::NotReady)
                    .and_then(|pair| {
                        receiver_prepare_wire(
                            &mut core,
                            pair,
                            &runtime,
                            &identity,
                            &inputs,
                            native_receiver.as_mut(),
                            &mut wire_plans,
                        )
                    });
                let _ = reply.send(result);
            }
            TransportCommand::Poll { handles, reply } => {
                let result = match connection.as_mut() {
                    Some(pair) => receiver_poll_wire(
                        &mut core,
                        pair,
                        &runtime,
                        &psk,
                        mock_endpoint.as_mut(),
                        native_receiver.as_mut(),
                        &handles,
                        &mut wire_plans,
                    ),
                    None => Err(TransportError::NotReady),
                };
                let _ = reply.send(result);
            }
            TransportCommand::Complete { events, reply } => {
                let result = events
                    .into_iter()
                    .map(|event| core.record_terminal(event).map(|_| ()))
                    .collect();
                let _ = reply.send(Ok(result));
            }
            TransportCommand::Abort {
                handles,
                reason,
                reply,
            } => {
                let result = connection
                    .as_mut()
                    .ok_or(TransportError::NotReady)
                    .and_then(|pair| {
                        abort_wire(
                            &mut core,
                            pair,
                            &runtime,
                            &handles,
                            reason,
                            &mut pending_prepares,
                            &mut wire_plans,
                            native_sender.as_mut(),
                            native_receiver.as_mut(),
                        )
                    });
                let _ = reply.send(result);
            }
            TransportCommand::Clear { handles, reply } => {
                let _ = reply.send(core.clear_many(&handles));
            }
            TransportCommand::Snapshot { reply } => {
                let mut snapshot = native_sender
                    .as_ref()
                    .map(NativeSender::resource_snapshot)
                    .or_else(|| {
                        native_receiver
                            .as_ref()
                            .map(NativeReceiver::resource_snapshot)
                    })
                    .unwrap_or_default();
                let transport = core.readiness().snapshot();
                snapshot.active_rooms = transport.runtime.active_rooms;
                snapshot.active_handles = transport.active_handles;
                snapshot.result_slots = transport.result_slots;
                snapshot.pending_prepares = pending_prepares.len();
                snapshot.wire_plans = wire_plans.len();
                let _ = reply.send(Ok(snapshot));
            }
            TransportCommand::Shutdown { reply } => {
                core.shutdown();
                connection.take();
                mock_endpoint.take();
                let result = match &bootstrap_owner {
                    BootstrapOwner::Mock(port) => {
                        port.shutdown().map_err(TransportError::LocalFatal)
                    }
                    BootstrapOwner::Native(_) => Ok(()),
                };
                let _ = reply.send(result);
                break;
            }
        }
    }
}

fn bootstrap_pair(
    identity: &RuntimeIdentity,
    psk: &Psk,
    control_host: &str,
    control_port: u16,
    bootstrap_port: Arc<dyn BootstrapPort>,
    runtime: &tokio::runtime::Runtime,
) -> Result<PairConnection, TransportError> {
    let address = format!("{control_host}:{control_port}");
    let clock: Arc<dyn crate::pd::room::Clock> = Arc::new(SystemClock::default());
    let connection = runtime.block_on(async {
        match identity.role {
            Role::Prefill => {
                let listener = tokio::net::TcpListener::bind(&address)
                    .await
                    .map_err(|_| TransportError::LocalFatal(PdReason::LocalFatal))?;
                let (stream, _) = listener
                    .accept()
                    .await
                    .map_err(|_| TransportError::LocalFatal(PdReason::PeerUnavailable))?;
                bootstrap_prefill(stream, identity.clone(), psk, bootstrap_port, clock)
                    .await
                    .map_err(runtime_transport_error)
            }
            Role::Decode => {
                let stream = tokio::time::timeout(Duration::from_secs(30), async {
                    loop {
                        match tokio::net::TcpStream::connect(&address).await {
                            Ok(stream) => break Ok(stream),
                            Err(_) => tokio::time::sleep(Duration::from_millis(50)).await,
                        }
                    }
                })
                .await
                .map_err(|_| TransportError::NotReady)??;
                bootstrap_decode(stream, identity.clone(), psk, bootstrap_port, clock)
                    .await
                    .map_err(runtime_transport_error)
            }
        }
    })?;
    Ok(connection)
}

fn mock_endpoint_for(
    identity: &RuntimeIdentity,
    owner: &BootstrapOwner,
) -> Result<Option<MockDataEndpoint>, TransportError> {
    if !matches!(owner, BootstrapOwner::Mock(_)) {
        return Ok(None);
    }
    match identity.role {
        Role::Prefill => Ok(Some(MockDataEndpoint::Prefill)),
        Role::Decode => {
            let data_port = identity
                .allowed_mooncake_ports
                .iter()
                .next()
                .copied()
                .ok_or(TransportError::InvalidBatch)?;
            let listener =
                StdTcpListener::bind((identity.expected_mooncake_host.as_str(), data_port))
                    .map_err(|_| TransportError::LocalFatal(PdReason::LocalFatal))?;
            Ok(Some(MockDataEndpoint::Decode { listener }))
        }
    }
}

fn sender_init_wire(
    core: &mut PdTransportCore,
    connection: &mut PairConnection,
    runtime: &tokio::runtime::Runtime,
    handles: &[OpaqueHandle],
    pending: &mut HashMap<u64, PrepareRoom>,
) -> Result<Vec<Result<(), TransportError>>, TransportError> {
    validate_wire_batch(handles.len())?;
    let mut unmatched = handles.to_vec();
    for _ in handles {
        let frame = runtime
            .block_on(connection.receive_expected(MessageKind::PrepareRoom))
            .map_err(runtime_transport_error)?;
        let ControlPayload::PrepareRoom(prepare) = frame.payload else {
            return Err(TransportError::LocalFatal(PdReason::ProtocolMismatch));
        };
        let position = unmatched
            .iter()
            .position(|handle| {
                core.room_context(*handle)
                    .is_ok_and(|context| room_fields_match(&prepare.room, context))
            })
            .ok_or(TransportError::LocalFatal(PdReason::ProtocolMismatch))?;
        let handle = unmatched.remove(position);
        validate_prepare(&prepare, core.room_context(handle)?)?;
        if pending.insert(handle.raw(), prepare).is_some() {
            return Err(TransportError::StaleHandle);
        }
    }
    core.sender_init_many(handles)
}

fn receiver_prepare_wire(
    core: &mut PdTransportCore,
    connection: &mut PairConnection,
    runtime: &tokio::runtime::Runtime,
    identity: &RuntimeIdentity,
    inputs: &[ReceiverWirePrepare],
    mut native: Option<&mut NativeReceiver>,
    plans: &mut HashMap<u64, WirePlan>,
) -> Result<Vec<Result<(), TransportError>>, TransportError> {
    validate_wire_batch(inputs.len())?;
    for input in inputs {
        validate_pages(&input.destination_pages, input.valid_token_count)?;
        let context = core.room_context(input.handle)?;
        let prepare = PrepareRoom {
            room: room_fields(context),
            destination_registration_epoch: FixedBytes::new(identity.registration_epoch.as_bytes()),
            destination_blocks: destination_blocks(&input.destination_pages),
            destination_aux_slot: input.handle.slot() as u16,
            destination_completion_slot: input.handle.slot() as u16,
            valid_token_count: input.valid_token_count,
            chunk_sequence: 0,
            chunk_count: 1,
            is_last_chunk: true,
        };
        runtime
            .block_on(connection.send(&ControlPayload::PrepareRoom(prepare)))
            .map_err(runtime_transport_error)?;
    }

    for input in inputs {
        let frame = runtime
            .block_on(connection.receive_expected(MessageKind::PrepareAccepted))
            .map_err(runtime_transport_error)?;
        let ControlPayload::PrepareAccepted(accepted) = frame.payload else {
            return Err(TransportError::LocalFatal(PdReason::ProtocolMismatch));
        };
        let context = core.room_context(input.handle)?;
        if !room_fields_match(&accepted.room, context) {
            return Err(TransportError::StaleHandle);
        }
        let source_pages = source_pages(&accepted)?;
        let plan = TransferPlan::new(TransferPlanInput {
            room: context.room,
            transfer_generation: context.room.generation,
            source_registration_epoch: registration_epoch(accepted.source_registration_epoch)?,
            destination_registration_epoch: identity.registration_epoch,
            source_pages,
            destination_pages: input.destination_pages.clone(),
            source_aux_slot: input.handle.slot() as u16,
            destination_aux_slot: input.handle.slot() as u16,
            source_completion_slot: input.handle.slot() as u16,
            destination_completion_slot: input.handle.slot() as u16,
            valid_token_count: input.valid_token_count,
            chunk_sequence: 0,
            chunk_count: 1,
            is_last_chunk: true,
        })
        .map_err(buffer_transport_error)?;
        plan.verify_prepare_accepted(&accepted)
            .map_err(buffer_transport_error)?;
        if let Some(receiver) = native.as_deref_mut() {
            let deadline = SystemClock::default()
                .now_monotonic_ms()
                .checked_add(identity.profile.deadline_ms.native_transfer)
                .ok_or(TransportError::LocalFatal(PdReason::LocalFatal))?;
            receiver.reserve(input.handle.raw(), &plan, deadline)?;
        }
        if plans
            .insert(
                input.handle.raw(),
                WirePlan {
                    plan,
                    request_digest: context.request_digest,
                    first_token_id: None,
                },
            )
            .is_some()
        {
            return Err(TransportError::StaleHandle);
        }
    }
    let handles = inputs.iter().map(|input| input.handle).collect::<Vec<_>>();
    core.receiver_prepare_many(&handles)
}

#[allow(clippy::too_many_arguments)]
fn sender_send_wire(
    core: &mut PdTransportCore,
    connection: &mut PairConnection,
    runtime: &tokio::runtime::Runtime,
    identity: &RuntimeIdentity,
    psk: &Psk,
    mut endpoint: Option<&mut MockDataEndpoint>,
    mut native: Option<&mut NativeSender>,
    chunks: &[SenderWireChunk],
    cuda_stream: u64,
    pending: &mut HashMap<u64, PrepareRoom>,
) -> Result<Vec<Result<(), TransportError>>, TransportError> {
    validate_wire_batch(chunks.len())?;
    if endpoint
        .as_ref()
        .is_some_and(|endpoint| !matches!(endpoint, MockDataEndpoint::Prefill))
        || (endpoint.is_none() && native.is_none())
    {
        return Err(TransportError::WrongRole);
    }
    let mut wire = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        validate_pages(&chunk.source_pages, chunk.valid_token_count)?;
        let context = core.room_context(chunk.chunk.handle)?;
        let prepare = pending
            .remove(&chunk.chunk.handle.raw())
            .ok_or(TransportError::InvalidTransition)?;
        validate_prepare(&prepare, context)?;
        let destination_pages = destination_pages(&prepare)?;
        let destination_registration_epoch =
            registration_epoch(prepare.destination_registration_epoch)?;
        let plan = TransferPlan::new(TransferPlanInput {
            room: context.room,
            transfer_generation: context.room.generation,
            source_registration_epoch: identity.registration_epoch,
            destination_registration_epoch,
            source_pages: chunk.source_pages.clone(),
            destination_pages,
            source_aux_slot: chunk.chunk.handle.slot() as u16,
            destination_aux_slot: prepare.destination_aux_slot,
            source_completion_slot: chunk.chunk.handle.slot() as u16,
            destination_completion_slot: prepare.destination_completion_slot,
            valid_token_count: chunk.valid_token_count,
            chunk_sequence: 0,
            chunk_count: 1,
            is_last_chunk: true,
        })
        .map_err(buffer_transport_error)?;
        if plan.expected_kv_bytes() != chunk.chunk.transfer_bytes {
            return Err(TransportError::InvalidBatch);
        }
        let accepted = PrepareAccepted {
            room: room_fields(context),
            source_registration_epoch: FixedBytes::new(identity.registration_epoch.as_bytes()),
            destination_registration_epoch: prepare.destination_registration_epoch,
            kv_blocks: plan.kv_blocks().to_vec(),
            source_aux_slot: plan.source_aux_slot(),
            destination_aux_slot: plan.destination_aux_slot(),
            source_completion_slot: plan.source_completion_slot(),
            destination_completion_slot: plan.destination_completion_slot(),
            valid_token_count: plan.valid_token_count(),
            chunk_sequence: 0,
            chunk_count: 1,
            is_last_chunk: true,
            transfer_plan_digest: FixedBytes::new(*plan.digest().as_bytes()),
        };
        runtime
            .block_on(connection.send(&ControlPayload::PrepareAccepted(accepted)))
            .map_err(runtime_transport_error)?;
        wire.push(WirePlan {
            plan,
            request_digest: context.request_digest,
            first_token_id: chunk.first_token_id,
        });
    }

    let core_chunks = chunks.iter().map(|chunk| chunk.chunk).collect::<Vec<_>>();
    let mut results = core.sender_send_chunks(&core_chunks)?;
    for (index, (chunk, wire_plan)) in chunks.iter().zip(wire).enumerate() {
        if results[index].is_err() {
            continue;
        }
        let (aux, completion) = completion_records(&wire_plan)?;
        if let Some(endpoint) = endpoint.as_deref_mut() {
            if !matches!(endpoint, MockDataEndpoint::Prefill) {
                return Err(TransportError::WrongRole);
            }
            mock_send_records(identity, psk, &aux, &completion.committed_bytes())?;
        } else if let Some(sender) = native.as_deref_mut() {
            sender.execute(
                chunk.chunk.handle.raw(),
                wire_plan.plan.clone(),
                aux,
                completion,
                cuda_stream,
            )?;
        } else {
            return Err(TransportError::NotReady);
        }
        let planned = planned_room(core.room_context(chunk.chunk.handle)?, &wire_plan.plan);
        runtime
            .block_on(connection.send(&ControlPayload::DataReady(planned.clone())))
            .map_err(runtime_transport_error)?;
        let complete = runtime
            .block_on(connection.receive_expected(MessageKind::TransferComplete))
            .map_err(runtime_transport_error)?;
        let ControlPayload::TransferComplete(complete) = complete.payload else {
            return Err(TransportError::LocalFatal(PdReason::ProtocolMismatch));
        };
        if complete != planned {
            return Err(TransportError::StaleHandle);
        }
        runtime
            .block_on(connection.send(&ControlPayload::TransferCompleteAck(planned)))
            .map_err(runtime_transport_error)?;
        if let Some(sender) = native.as_deref_mut() {
            sender.finish_after_ack(chunk.chunk.handle.raw())?;
        }
        results[index] = core
            .record_terminal(TerminalEvent {
                handle: chunk.chunk.handle,
                reason: PdReason::Success,
                first_token_id: None,
                transfer_bytes: chunk.chunk.transfer_bytes,
            })
            .map(|_| ());
    }
    Ok(results)
}

// The wire poll joins the core, authenticated pair, mock/native ports, and plan table.
#[allow(clippy::too_many_arguments)]
fn receiver_poll_wire(
    core: &mut PdTransportCore,
    connection: &mut PairConnection,
    runtime: &tokio::runtime::Runtime,
    psk: &Psk,
    mut endpoint: Option<&mut MockDataEndpoint>,
    mut native: Option<&mut NativeReceiver>,
    handles: &[OpaqueHandle],
    plans: &mut HashMap<u64, WirePlan>,
) -> Result<Vec<Result<TransportPollResult, TransportError>>, TransportError> {
    validate_wire_batch(handles.len())?;
    if endpoint
        .as_ref()
        .is_some_and(|endpoint| !matches!(endpoint, MockDataEndpoint::Decode { .. }))
        || (endpoint.is_none() && native.is_none())
    {
        return core.poll_many(handles);
    }
    for handle in handles {
        let Some(mut wire_plan) = plans.remove(&handle.raw()) else {
            continue;
        };
        let ready = runtime
            .block_on(connection.receive_expected(MessageKind::DataReady))
            .map_err(runtime_transport_error)?;
        let ControlPayload::DataReady(ready) = ready.payload else {
            return Err(TransportError::LocalFatal(PdReason::ProtocolMismatch));
        };
        let expected = planned_room(core.room_context(*handle)?, &wire_plan.plan);
        if ready != expected {
            return Err(TransportError::StaleHandle);
        }
        let expected_record = completion_input(&wire_plan);
        if let Some(MockDataEndpoint::Decode { listener }) = endpoint.as_deref_mut() {
            let (aux, completion) = mock_receive_records(listener, psk)?;
            let validated = validate_completion(&completion, &aux, &expected_record)
                .map_err(buffer_transport_error)?;
            wire_plan.first_token_id = validated
                .aux
                .first_token_valid
                .then_some(validated.aux.first_token_id);
        } else if let Some(receiver) = native.as_deref_mut() {
            receiver.validate(handle.raw(), &wire_plan.plan, &expected_record)?;
        } else {
            return Err(TransportError::NotReady);
        }
        runtime
            .block_on(connection.send(&ControlPayload::TransferComplete(expected.clone())))
            .map_err(runtime_transport_error)?;
        let ack = runtime
            .block_on(connection.receive_expected(MessageKind::TransferCompleteAck))
            .map_err(runtime_transport_error)?;
        let ControlPayload::TransferCompleteAck(ack) = ack.payload else {
            return Err(TransportError::LocalFatal(PdReason::ProtocolMismatch));
        };
        if ack != expected {
            return Err(TransportError::StaleHandle);
        }
        if let Some(receiver) = native.as_deref_mut() {
            let validated = receiver.finish_after_ack(handle.raw(), &wire_plan.plan)?;
            wire_plan.first_token_id = validated
                .aux
                .first_token_valid
                .then_some(validated.aux.first_token_id);
        }
        core.record_terminal(TerminalEvent {
            handle: *handle,
            reason: PdReason::Success,
            first_token_id: wire_plan.first_token_id,
            transfer_bytes: wire_plan.plan.expected_kv_bytes(),
        })?;
    }
    core.poll_many(handles)
}

#[allow(clippy::too_many_arguments)]
fn abort_wire(
    core: &mut PdTransportCore,
    connection: &mut PairConnection,
    runtime: &tokio::runtime::Runtime,
    handles: &[OpaqueHandle],
    reason: PdReason,
    pending: &mut HashMap<u64, PrepareRoom>,
    plans: &mut HashMap<u64, WirePlan>,
    mut native_sender: Option<&mut NativeSender>,
    mut native_receiver: Option<&mut NativeReceiver>,
) -> Result<Vec<Result<(), TransportError>>, TransportError> {
    validate_wire_batch(handles.len())?;
    if reason == PdReason::Success {
        return Err(TransportError::InvalidBatch);
    }
    let mut expected = Vec::with_capacity(handles.len());
    for handle in handles {
        let context = core.room_context(*handle)?;
        let transfer_plan_digest = plans
            .get(&handle.raw())
            .map_or_else(PlanDigest::empty, |wire| {
                PlanDigest::from_digest(FixedBytes::new(*wire.plan.digest().as_bytes()))
            });
        let terminal = TerminalRoom {
            room: room_fields(context),
            transfer_plan_digest,
            reason: reason.code().to_string(),
        };
        runtime
            .block_on(connection.send(&ControlPayload::Abort(terminal.clone())))
            .map_err(runtime_transport_error)?;
        expected.push(terminal);
    }

    while !expected.is_empty() {
        let frame = runtime
            .block_on(connection.receive_next(MessageKind::AbortAck))
            .map_err(runtime_transport_error)?;
        match frame.payload {
            ControlPayload::Abort(peer_abort) => {
                let position = handles
                    .iter()
                    .position(|handle| {
                        core.room_context(*handle)
                            .is_ok_and(|context| room_fields_match(&peer_abort.room, context))
                    })
                    .ok_or(TransportError::StaleHandle)?;
                let peer_reason = parse_reason(&peer_abort.reason)
                    .filter(|peer_reason| *peer_reason != PdReason::Success)
                    .ok_or(TransportError::InvalidBatch)?;
                let handle = handles[position];
                cleanup_native_abort(
                    handle,
                    native_sender.as_deref_mut(),
                    native_receiver.as_deref_mut(),
                )?;
                core.record_terminal(TerminalEvent {
                    handle,
                    reason: peer_reason,
                    first_token_id: None,
                    transfer_bytes: 0,
                })?;
                runtime
                    .block_on(connection.send(&ControlPayload::AbortAck(peer_abort)))
                    .map_err(runtime_transport_error)?;
            }
            ControlPayload::AbortAck(ack) => {
                let position = expected
                    .iter()
                    .position(|terminal| terminal == &ack)
                    .ok_or(TransportError::StaleHandle)?;
                expected.swap_remove(position);
            }
            _ => {
                return Err(TransportError::LocalFatal(PdReason::ProtocolMismatch));
            }
        }
    }

    for handle in handles {
        cleanup_native_abort(
            *handle,
            native_sender.as_deref_mut(),
            native_receiver.as_deref_mut(),
        )?;
        pending.remove(&handle.raw());
        plans.remove(&handle.raw());
    }
    core.abort_many(handles, reason)
}

fn cleanup_native_abort(
    handle: OpaqueHandle,
    native_sender: Option<&mut NativeSender>,
    native_receiver: Option<&mut NativeReceiver>,
) -> Result<(), TransportError> {
    if let Some(sender) = native_sender {
        sender.abort(handle.raw())?;
    }
    if let Some(receiver) = native_receiver {
        receiver.abort(handle.raw())?;
    }
    Ok(())
}
