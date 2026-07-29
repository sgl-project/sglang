use super::support::*;
use super::*;

const WORKER_TICK: Duration = Duration::from_millis(50);
const INITIAL_BOOTSTRAP_TIMEOUT: Duration = Duration::from_secs(30);
const RECONNECT_BOOTSTRAP_TIMEOUT: Duration = Duration::from_secs(1);
const NATIVE_JOIN_TIMEOUT: Duration = Duration::from_secs(30);

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
        Err(_) => {
            core.publish_fatal(
                crate::pd::runtime::FatalSource::StartupInvariant,
                PdReason::LocalFatal,
            );
            return;
        }
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_transport_worker(
            &mut core,
            &identity,
            &psk,
            &control_host,
            control_port,
            &bootstrap_owner,
            &receiver,
            &runtime,
        );
    }));
    if result.is_err() {
        core.publish_fatal(
            crate::pd::runtime::FatalSource::WorkerExit,
            PdReason::LocalFatal,
        );
        let _ = core.shutdown();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_transport_worker(
    core: &mut PdTransportCore,
    identity: &RuntimeIdentity,
    psk: &Psk,
    control_host: &str,
    control_port: u16,
    bootstrap_owner: &BootstrapOwner,
    receiver: &flume::Receiver<TransportCommand>,
    runtime: &tokio::runtime::Runtime,
) {
    let mut connection: Option<PairConnection> = None;
    let mut control_endpoint: Option<ControlEndpoint> = None;
    let mut mock_endpoint: Option<MockDataEndpoint> = None;
    let mut native_sender: Option<NativeSender> = None;
    let mut retired_native_senders = Vec::<NativeSender>::new();
    let mut native_receiver: Option<NativeReceiver> = None;
    let mut pending_prepares = HashMap::<u64, PrepareRoom>::new();
    let mut wire_plans = HashMap::<u64, WirePlan>::new();
    let mut next_reconnect_attempt = Instant::now();
    loop {
        drive_lifecycle(
            core,
            identity,
            psk,
            bootstrap_owner,
            runtime,
            control_endpoint.as_ref(),
            &mut connection,
            &mut mock_endpoint,
            &mut native_sender,
            &mut retired_native_senders,
            &mut native_receiver,
            &mut pending_prepares,
            &mut wire_plans,
            &mut next_reconnect_attempt,
        );
        if core.readiness().snapshot().runtime.lifecycle == RuntimeLifecycle::Fatal {
            let _ = shutdown_worker_resources(
                core,
                identity,
                bootstrap_owner,
                runtime,
                &mut control_endpoint,
                &mut connection,
                &mut mock_endpoint,
                &mut native_sender,
                &mut retired_native_senders,
                &mut native_receiver,
                &mut pending_prepares,
                &mut wire_plans,
                crate::pd::runtime::ShutdownMode::Fatal,
            );
            break;
        }
        let command = match receiver.recv_timeout(WORKER_TICK) {
            Ok(command) => command,
            Err(flume::RecvTimeoutError::Timeout) => continue,
            Err(flume::RecvTimeoutError::Disconnected) => {
                core.publish_fatal(
                    crate::pd::runtime::FatalSource::CommandChannelClosed,
                    PdReason::LocalFatal,
                );
                let _ = shutdown_worker_resources(
                    core,
                    identity,
                    bootstrap_owner,
                    runtime,
                    &mut control_endpoint,
                    &mut connection,
                    &mut mock_endpoint,
                    &mut native_sender,
                    &mut retired_native_senders,
                    &mut native_receiver,
                    &mut pending_prepares,
                    &mut wire_plans,
                    crate::pd::runtime::ShutdownMode::Fatal,
                );
                break;
            }
        };
        match command {
            TransportCommand::Start { reply } => {
                let result = if control_endpoint.is_some() {
                    Err(TransportError::InvalidTransition)
                } else {
                    core.start_local(crate::pd::transport::PD_REGION_COUNT)
                        .and_then(|()| {
                            control_endpoint = Some(open_control_endpoint(
                                identity,
                                control_host,
                                control_port,
                                runtime,
                            )?);
                            mock_endpoint = mock_endpoint_for(identity, bootstrap_owner)?;
                            bootstrap_pair(
                                identity,
                                psk,
                                control_endpoint
                                    .as_ref()
                                    .ok_or(TransportError::InvalidTransition)?,
                                bootstrap_owner.port(),
                                runtime,
                                INITIAL_BOOTSTRAP_TIMEOUT,
                            )
                        })
                        .and_then(|pair| {
                            install_pair(
                                core,
                                identity,
                                bootstrap_owner,
                                pair,
                                runtime,
                                &mut connection,
                                &mut mock_endpoint,
                                &mut native_sender,
                                &mut native_receiver,
                            )
                        })
                };
                observe_command_error(
                    core,
                    identity,
                    bootstrap_owner,
                    &result,
                    &mut connection,
                    &mut native_sender,
                    &mut retired_native_senders,
                    &mut native_receiver,
                    &mut pending_prepares,
                    &mut wire_plans,
                );
                let _ = reply.send(result);
            }
            TransportCommand::SenderCreate { input, reply } => {
                let result = core.sender_create(input);
                observe_command_error(
                    core,
                    identity,
                    bootstrap_owner,
                    &result,
                    &mut connection,
                    &mut native_sender,
                    &mut retired_native_senders,
                    &mut native_receiver,
                    &mut pending_prepares,
                    &mut wire_plans,
                );
                let _ = reply.send(result);
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
                observe_command_error(
                    core,
                    identity,
                    bootstrap_owner,
                    &result,
                    &mut connection,
                    &mut native_sender,
                    &mut retired_native_senders,
                    &mut native_receiver,
                    &mut pending_prepares,
                    &mut wire_plans,
                );
                let _ = reply.send(result);
            }
            TransportCommand::SenderInit { handles, reply } => {
                let result = connection
                    .as_mut()
                    .ok_or(TransportError::NotReady)
                    .and_then(|pair| {
                        sender_init_wire(core, pair, runtime, &handles, &mut pending_prepares)
                    });
                observe_command_error(
                    core,
                    identity,
                    bootstrap_owner,
                    &result,
                    &mut connection,
                    &mut native_sender,
                    &mut retired_native_senders,
                    &mut native_receiver,
                    &mut pending_prepares,
                    &mut wire_plans,
                );
                let _ = reply.send(result);
            }
            TransportCommand::SenderSend {
                chunks,
                cuda_stream,
                reply,
            } => {
                let result = match connection.as_mut() {
                    Some(pair) => sender_send_wire(
                        core,
                        pair,
                        runtime,
                        identity,
                        psk,
                        mock_endpoint.as_mut(),
                        native_sender.as_mut(),
                        &chunks,
                        cuda_stream,
                        &mut pending_prepares,
                    ),
                    None => Err(TransportError::NotReady),
                };
                observe_command_error(
                    core,
                    identity,
                    bootstrap_owner,
                    &result,
                    &mut connection,
                    &mut native_sender,
                    &mut retired_native_senders,
                    &mut native_receiver,
                    &mut pending_prepares,
                    &mut wire_plans,
                );
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
                observe_command_error(
                    core,
                    identity,
                    bootstrap_owner,
                    &result,
                    &mut connection,
                    &mut native_sender,
                    &mut retired_native_senders,
                    &mut native_receiver,
                    &mut pending_prepares,
                    &mut wire_plans,
                );
                let _ = reply.send(result);
            }
            TransportCommand::ReceiverPrepare { inputs, reply } => {
                let result = connection
                    .as_mut()
                    .ok_or(TransportError::NotReady)
                    .and_then(|pair| {
                        receiver_prepare_wire(
                            core,
                            pair,
                            runtime,
                            identity,
                            &inputs,
                            native_receiver.as_mut(),
                            &mut wire_plans,
                        )
                    });
                observe_command_error(
                    core,
                    identity,
                    bootstrap_owner,
                    &result,
                    &mut connection,
                    &mut native_sender,
                    &mut retired_native_senders,
                    &mut native_receiver,
                    &mut pending_prepares,
                    &mut wire_plans,
                );
                let _ = reply.send(result);
            }
            TransportCommand::Poll { handles, reply } => {
                let result = match connection.as_mut() {
                    Some(pair) => receiver_poll_wire(
                        core,
                        pair,
                        runtime,
                        psk,
                        mock_endpoint.as_mut(),
                        native_receiver.as_mut(),
                        &handles,
                        &mut wire_plans,
                    ),
                    // Peer isolation clears every wire plan after first
                    // terminalizing active handles. Their tombstones must
                    // remain observable while the endpoint reconnects.
                    None => core.poll_many(&handles),
                };
                observe_command_error(
                    core,
                    identity,
                    bootstrap_owner,
                    &result,
                    &mut connection,
                    &mut native_sender,
                    &mut retired_native_senders,
                    &mut native_receiver,
                    &mut pending_prepares,
                    &mut wire_plans,
                );
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
                            core,
                            pair,
                            runtime,
                            &handles,
                            reason,
                            &mut pending_prepares,
                            &mut wire_plans,
                            native_sender.as_mut(),
                            native_receiver.as_mut(),
                        )
                    });
                observe_command_error(
                    core,
                    identity,
                    bootstrap_owner,
                    &result,
                    &mut connection,
                    &mut native_sender,
                    &mut retired_native_senders,
                    &mut native_receiver,
                    &mut pending_prepares,
                    &mut wire_plans,
                );
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
                for retired in &retired_native_senders {
                    merge_resource_snapshot(&mut snapshot, retired.resource_snapshot());
                }
                let transport = core.readiness().snapshot();
                snapshot.active_rooms = transport.runtime.active_rooms;
                snapshot.active_handles = transport.active_handles;
                snapshot.result_slots = transport.result_slots;
                snapshot.pending_prepares = pending_prepares.len();
                snapshot.wire_plans = wire_plans.len();
                let _ = reply.send(Ok(snapshot));
            }
            TransportCommand::Shutdown { reply } => {
                let mode =
                    if core.readiness().snapshot().runtime.lifecycle == RuntimeLifecycle::Fatal {
                        crate::pd::runtime::ShutdownMode::Fatal
                    } else {
                        crate::pd::runtime::ShutdownMode::Graceful
                    };
                let result = shutdown_worker_resources(
                    core,
                    identity,
                    bootstrap_owner,
                    runtime,
                    &mut control_endpoint,
                    &mut connection,
                    &mut mock_endpoint,
                    &mut native_sender,
                    &mut retired_native_senders,
                    &mut native_receiver,
                    &mut pending_prepares,
                    &mut wire_plans,
                    mode,
                );
                let _ = reply.send(result);
                break;
            }
        }
    }
}

fn open_control_endpoint(
    identity: &RuntimeIdentity,
    control_host: &str,
    control_port: u16,
    runtime: &tokio::runtime::Runtime,
) -> Result<ControlEndpoint, TransportError> {
    let address = format!("{control_host}:{control_port}");
    match identity.role {
        Role::Prefill => runtime
            .block_on(tokio::net::TcpListener::bind(&address))
            .map(|listener| ControlEndpoint::Prefill { listener })
            .map_err(|_| TransportError::LocalFatal(PdReason::LocalFatal)),
        Role::Decode => Ok(ControlEndpoint::Decode { address }),
    }
}

fn bootstrap_pair(
    identity: &RuntimeIdentity,
    psk: &Psk,
    endpoint: &ControlEndpoint,
    bootstrap_port: Arc<dyn BootstrapPort>,
    runtime: &tokio::runtime::Runtime,
    wait: Duration,
) -> Result<PairConnection, TransportError> {
    let clock: Arc<dyn crate::pd::room::Clock> = Arc::new(SystemClock::default());
    let connection = runtime.block_on(async {
        match endpoint {
            ControlEndpoint::Prefill { listener } => {
                let (stream, _) = tokio::time::timeout(wait, listener.accept())
                    .await
                    .map_err(|_| TransportError::NotReady)?
                    .map_err(|_| TransportError::Peer(PdReason::PeerUnavailable))?;
                bootstrap_prefill(stream, identity.clone(), psk, bootstrap_port, clock)
                    .await
                    .map_err(runtime_transport_error)
            }
            ControlEndpoint::Decode { address } => {
                let stream = tokio::time::timeout(wait, async {
                    loop {
                        match tokio::net::TcpStream::connect(address.as_str()).await {
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

#[allow(clippy::too_many_arguments)]
fn install_pair(
    core: &mut PdTransportCore,
    identity: &RuntimeIdentity,
    bootstrap_owner: &BootstrapOwner,
    mut pair: PairConnection,
    runtime: &tokio::runtime::Runtime,
    connection: &mut Option<PairConnection>,
    mock_endpoint: &mut Option<MockDataEndpoint>,
    native_sender: &mut Option<NativeSender>,
    native_receiver: &mut Option<NativeReceiver>,
) -> Result<(), TransportError> {
    if let Err(error) = core.validate_pair_candidate(pair.readiness()) {
        let generation = core
            .readiness()
            .snapshot()
            .runtime
            .reconnect_generation
            .saturating_add(1);
        let _ = runtime.block_on(pair.send_goaway(generation));
        return Err(error);
    }
    match bootstrap_owner {
        BootstrapOwner::Mock(_) => {
            // A control-session replacement must not inherit a queued mock
            // data connection from the retired peer.
            *mock_endpoint = None;
            *mock_endpoint = mock_endpoint_for(identity, bootstrap_owner)?;
        }
        BootstrapOwner::Native(port) => match identity.role {
            Role::Prefill => {
                *native_sender = Some(NativeSender::new(port, &identity.profile)?);
            }
            Role::Decode => {
                if native_receiver.is_none() {
                    *native_receiver = Some(NativeReceiver::new(port, &identity.profile)?);
                }
            }
        },
    }
    core.activate_pair(
        pair.readiness().clone(),
        crate::pd::transport::PD_REGION_COUNT,
        true,
    )?;
    *connection = Some(pair);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn drive_lifecycle(
    core: &mut PdTransportCore,
    identity: &RuntimeIdentity,
    psk: &Psk,
    bootstrap_owner: &BootstrapOwner,
    runtime: &tokio::runtime::Runtime,
    control_endpoint: Option<&ControlEndpoint>,
    connection: &mut Option<PairConnection>,
    mock_endpoint: &mut Option<MockDataEndpoint>,
    native_sender: &mut Option<NativeSender>,
    retired_native_senders: &mut Vec<NativeSender>,
    native_receiver: &mut Option<NativeReceiver>,
    pending_prepares: &mut HashMap<u64, PrepareRoom>,
    wire_plans: &mut HashMap<u64, WirePlan>,
    next_reconnect_attempt: &mut Instant,
) {
    let connection_event = connection
        .as_mut()
        .map(|pair| runtime.block_on(pair.lifecycle_tick()));
    match connection_event {
        Some(Ok(
            crate::pd::runtime::ConnectionLifecycle::PeerLost
            | crate::pd::runtime::ConnectionLifecycle::PeerDraining(_),
        )) => isolate_peer(
            core,
            bootstrap_owner,
            connection,
            native_sender,
            retired_native_senders,
            native_receiver,
            pending_prepares,
            wire_plans,
        ),
        Some(Err(error)) => {
            let class = crate::pd::runtime::FailureClass::for_runtime(&error);
            match class.scope {
                crate::pd::runtime::FailureScope::PeerSession => isolate_peer(
                    core,
                    bootstrap_owner,
                    connection,
                    native_sender,
                    retired_native_senders,
                    native_receiver,
                    pending_prepares,
                    wire_plans,
                ),
                crate::pd::runtime::FailureScope::LocalFatal => {
                    core.publish_fatal(
                        crate::pd::runtime::FatalSource::ProtocolInvariant,
                        class.reason,
                    );
                }
                crate::pd::runtime::FailureScope::Request
                | crate::pd::runtime::FailureScope::Room => {}
            }
        }
        Some(Ok(_)) | None => {}
    }

    if let Some(sender) = native_sender.as_mut() {
        observe_native_lifecycle(core, sender);
    }
    if let Some(receiver) = native_receiver.as_mut() {
        observe_native_receiver_lifecycle(core, receiver);
    }
    let mut index = 0;
    while index < retired_native_senders.len() {
        observe_native_lifecycle(core, &mut retired_native_senders[index]);
        if !retired_native_senders[index].has_unsafe_leases() {
            let mut sender = retired_native_senders.swap_remove(index);
            if sender.shutdown_worker(NATIVE_JOIN_TIMEOUT).is_err() {
                core.publish_fatal(
                    crate::pd::runtime::FatalSource::WorkerExit,
                    PdReason::LocalFatal,
                );
            }
        } else {
            index += 1;
        }
    }

    let snapshot = core.readiness().snapshot();
    if snapshot.runtime.lifecycle != RuntimeLifecycle::LocalReady
        || connection.is_some()
        || control_endpoint.is_none()
        || Instant::now() < *next_reconnect_attempt
    {
        return;
    }
    *next_reconnect_attempt = Instant::now() + Duration::from_millis(200);
    match bootstrap_pair(
        identity,
        psk,
        control_endpoint.expect("checked control endpoint"),
        bootstrap_owner.port(),
        runtime,
        RECONNECT_BOOTSTRAP_TIMEOUT,
    ) {
        Ok(pair) => {
            if let Err(error) = install_pair(
                core,
                identity,
                bootstrap_owner,
                pair,
                runtime,
                connection,
                mock_endpoint,
                native_sender,
                native_receiver,
            ) {
                if bootstrap_owner.reset_peer().is_err() {
                    core.publish_fatal(
                        crate::pd::runtime::FatalSource::EngineOwner,
                        PdReason::LocalFatal,
                    );
                }
                publish_classified_error(core, &error);
            }
        }
        Err(error) => {
            if crate::pd::runtime::FailureClass::for_transport(&error).scope
                == crate::pd::runtime::FailureScope::LocalFatal
            {
                publish_classified_error(core, &error);
            }
        }
    }
}

fn observe_native_lifecycle(core: &mut PdTransportCore, sender: &mut NativeSender) {
    match sender.lifecycle_tick() {
        Ok(native::NativeLifecycleEffect::Idle)
        | Ok(native::NativeLifecycleEffect::Released { .. }) => {}
        Ok(native::NativeLifecycleEffect::HardDeadline) => {
            core.publish_fatal(
                crate::pd::runtime::FatalSource::QuarantineHardDeadline,
                PdReason::LocalFatal,
            );
        }
        Err(_) => {
            core.publish_fatal(
                crate::pd::runtime::FatalSource::WorkerExit,
                PdReason::LocalFatal,
            );
        }
    }
}

fn observe_native_receiver_lifecycle(core: &mut PdTransportCore, receiver: &mut NativeReceiver) {
    match receiver.lifecycle_tick() {
        Ok(native::NativeLifecycleEffect::Idle)
        | Ok(native::NativeLifecycleEffect::Released { .. }) => {}
        Ok(native::NativeLifecycleEffect::HardDeadline) | Err(_) => {
            core.publish_fatal(
                crate::pd::runtime::FatalSource::QuarantineHardDeadline,
                PdReason::LocalFatal,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn observe_command_error<T>(
    core: &mut PdTransportCore,
    _identity: &RuntimeIdentity,
    bootstrap_owner: &BootstrapOwner,
    result: &Result<T, TransportError>,
    connection: &mut Option<PairConnection>,
    native_sender: &mut Option<NativeSender>,
    retired_native_senders: &mut Vec<NativeSender>,
    native_receiver: &mut Option<NativeReceiver>,
    pending_prepares: &mut HashMap<u64, PrepareRoom>,
    wire_plans: &mut HashMap<u64, WirePlan>,
) {
    let Err(error) = result else {
        return;
    };
    match crate::pd::runtime::FailureClass::for_transport(error).scope {
        crate::pd::runtime::FailureScope::PeerSession => isolate_peer(
            core,
            bootstrap_owner,
            connection,
            native_sender,
            retired_native_senders,
            native_receiver,
            pending_prepares,
            wire_plans,
        ),
        crate::pd::runtime::FailureScope::LocalFatal => publish_classified_error(core, error),
        crate::pd::runtime::FailureScope::Request | crate::pd::runtime::FailureScope::Room => {}
    }
}

fn publish_classified_error(core: &mut PdTransportCore, error: &TransportError) {
    let class = crate::pd::runtime::FailureClass::for_transport(error);
    if class.scope == crate::pd::runtime::FailureScope::LocalFatal {
        core.publish_fatal(
            crate::pd::runtime::FatalSource::ProtocolInvariant,
            class.reason,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn isolate_peer(
    core: &mut PdTransportCore,
    bootstrap_owner: &BootstrapOwner,
    connection: &mut Option<PairConnection>,
    native_sender: &mut Option<NativeSender>,
    retired_native_senders: &mut Vec<NativeSender>,
    native_receiver: &mut Option<NativeReceiver>,
    pending_prepares: &mut HashMap<u64, PrepareRoom>,
    wire_plans: &mut HashMap<u64, WirePlan>,
) {
    core.peer_lost();
    connection.take();
    pending_prepares.clear();
    if let Some(receiver) = native_receiver.as_mut()
        && receiver.isolate_peer().is_err()
    {
        core.publish_fatal(
            crate::pd::runtime::FatalSource::RegistryInvariant,
            PdReason::LocalFatal,
        );
    }
    wire_plans.clear();
    if let Some(mut sender) = native_sender.take() {
        match sender.isolate_peer() {
            Ok(true) => retired_native_senders.push(sender),
            Ok(false) => {
                if sender.shutdown_worker(NATIVE_JOIN_TIMEOUT).is_err() {
                    core.publish_fatal(
                        crate::pd::runtime::FatalSource::WorkerExit,
                        PdReason::LocalFatal,
                    );
                }
            }
            Err(_) => {
                retired_native_senders.push(sender);
                core.publish_fatal(
                    crate::pd::runtime::FatalSource::RegistryInvariant,
                    PdReason::LocalFatal,
                );
            }
        }
    }
    if bootstrap_owner.reset_peer().is_err() {
        core.publish_fatal(
            crate::pd::runtime::FatalSource::EngineOwner,
            PdReason::LocalFatal,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn shutdown_worker_resources(
    core: &mut PdTransportCore,
    identity: &RuntimeIdentity,
    bootstrap_owner: &BootstrapOwner,
    runtime: &tokio::runtime::Runtime,
    control_endpoint: &mut Option<ControlEndpoint>,
    connection: &mut Option<PairConnection>,
    mock_endpoint: &mut Option<MockDataEndpoint>,
    native_sender: &mut Option<NativeSender>,
    retired_native_senders: &mut Vec<NativeSender>,
    native_receiver: &mut Option<NativeReceiver>,
    pending_prepares: &mut HashMap<u64, PrepareRoom>,
    wire_plans: &mut HashMap<u64, WirePlan>,
    mode: crate::pd::runtime::ShutdownMode,
) -> Result<crate::pd::runtime::RuntimeShutdownOutcome, TransportError> {
    if let Some(outcome) = core.readiness().snapshot().runtime.shutdown_outcome {
        return Ok(outcome);
    }
    let generation = core.begin_shutdown(mode)?;
    if let Some(pair) = connection.as_mut()
        && runtime.block_on(pair.send_goaway(generation)).is_err()
    {
        core.peer_lost();
    }
    core.advance_shutdown(crate::pd::runtime::ShutdownPhase::GoAway)?;
    core.advance_shutdown(crate::pd::runtime::ShutdownPhase::StopAccepting)?;
    core.advance_shutdown(crate::pd::runtime::ShutdownPhase::DrainingRooms)?;
    core.advance_shutdown(crate::pd::runtime::ShutdownPhase::AbortingRooms)?;

    pending_prepares.clear();
    if let Some(receiver) = native_receiver.as_mut() {
        receiver.isolate_peer()?;
    }
    wire_plans.clear();
    if let Some(sender) = native_sender.as_mut() {
        let _ = sender.isolate_peer();
    }
    core.advance_shutdown(crate::pd::runtime::ShutdownPhase::NativeSafety)?;

    let native_deadline = Instant::now() + Duration::from_secs(300);
    let unsafe_native = loop {
        let unsafe_now = native_sender
            .as_ref()
            .is_some_and(NativeSender::has_unsafe_leases)
            || retired_native_senders
                .iter()
                .any(NativeSender::has_unsafe_leases)
            || native_receiver
                .as_ref()
                .is_some_and(NativeReceiver::has_unsafe_leases);
        if !unsafe_now {
            break false;
        }
        if Instant::now() >= native_deadline {
            core.publish_fatal(
                crate::pd::runtime::FatalSource::QuarantineHardDeadline,
                PdReason::LocalFatal,
            );
            break true;
        }
        if let Some(sender) = native_sender.as_mut() {
            observe_native_lifecycle(core, sender);
        }
        for sender in retired_native_senders.iter_mut() {
            observe_native_lifecycle(core, sender);
        }
        if let Some(receiver) = native_receiver.as_mut() {
            observe_native_receiver_lifecycle(core, receiver);
        }
        if core.readiness().snapshot().runtime.lifecycle == RuntimeLifecycle::Fatal {
            break true;
        }
        std::thread::sleep(WORKER_TICK);
    };

    core.advance_shutdown(crate::pd::runtime::ShutdownPhase::SchedulerRelease)?;
    let mut worker_safe = true;
    if let Some(sender) = native_sender.as_mut() {
        let result = if sender.has_unsafe_leases() {
            sender.shutdown_worker_unsafe(NATIVE_JOIN_TIMEOUT)
        } else {
            sender.shutdown_worker(NATIVE_JOIN_TIMEOUT)
        };
        worker_safe &= result.is_ok();
    }
    for sender in retired_native_senders.iter_mut() {
        let result = if sender.has_unsafe_leases() {
            sender.shutdown_worker_unsafe(NATIVE_JOIN_TIMEOUT)
        } else {
            sender.shutdown_worker(NATIVE_JOIN_TIMEOUT)
        };
        worker_safe &= result.is_ok();
    }
    core.advance_shutdown(crate::pd::runtime::ShutdownPhase::WorkerJoin)?;
    native_sender.take();
    retired_native_senders.clear();
    native_receiver.take();
    core.advance_shutdown(crate::pd::runtime::ShutdownPhase::EngineQuiesce)?;

    connection.take();
    mock_endpoint.take();
    control_endpoint.take();
    let peer_safe = bootstrap_owner.reset_peer().is_ok();
    core.advance_shutdown(crate::pd::runtime::ShutdownPhase::ConnectionClose)?;
    core.advance_shutdown(crate::pd::runtime::ShutdownPhase::RegionUnregister)?;

    let owner_safe = match bootstrap_owner {
        BootstrapOwner::Mock(port) => port.shutdown().is_ok(),
        BootstrapOwner::Native(port) => matches!(
            port.shutdown(),
            Ok(crate::pd::runtime::RuntimeShutdownOutcome::SafeTerminal)
        ),
    };
    core.advance_shutdown(crate::pd::runtime::ShutdownPhase::EngineDestroy)?;
    let safe = mode == crate::pd::runtime::ShutdownMode::Graceful
        && !unsafe_native
        && worker_safe
        && peer_safe
        && owner_safe
        && identity.role == core.readiness().snapshot().runtime.role;
    let outcome = if safe {
        crate::pd::runtime::RuntimeShutdownOutcome::SafeTerminal
    } else {
        crate::pd::runtime::RuntimeShutdownOutcome::FatalUnsafe
    };
    core.complete_shutdown(outcome)
}

fn merge_resource_snapshot(target: &mut PyPdResourceSnapshot, source: PyPdResourceSnapshot) {
    target.active_rooms += source.active_rooms;
    target.active_handles += source.active_handles;
    target.result_slots += source.result_slots;
    target.pending_prepares += source.pending_prepares;
    target.wire_plans += source.wire_plans;
    target.native_leases += source.native_leases;
    target.source_kv_pages += source.source_kv_pages;
    target.destination_kv_pages += source.destination_kv_pages;
    target.aux_slots += source.aux_slots;
    target.completion_slots += source.completion_slots;
    target.request_slots += source.request_slots;
    target.in_flight_transfers += source.in_flight_transfers;
    target.native_batches += source.native_batches;
    target.pending_bytes = target.pending_bytes.saturating_add(source.pending_bytes);
    target.quarantined_rooms += source.quarantined_rooms;
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
            return Err(TransportError::Peer(PdReason::ProtocolMismatch));
        };
        let position = unmatched
            .iter()
            .position(|handle| {
                core.room_context(*handle)
                    .is_ok_and(|context| room_fields_match(&prepare.room, context))
            })
            .ok_or(TransportError::Peer(PdReason::ProtocolMismatch))?;
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
            return Err(TransportError::Peer(PdReason::ProtocolMismatch));
        };
        let context = core.room_context(input.handle)?;
        if !room_fields_match(&accepted.room, context) {
            return Err(TransportError::Peer(PdReason::ProtocolMismatch));
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
        .map_err(|_| TransportError::Peer(PdReason::ProtocolMismatch))?;
        plan.verify_prepare_accepted(&accepted)
            .map_err(|_| TransportError::Peer(PdReason::ProtocolMismatch))?;
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
            return Err(TransportError::Peer(PdReason::ProtocolMismatch));
        };
        if complete != planned {
            return Err(TransportError::Peer(PdReason::ProtocolMismatch));
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
            return Err(TransportError::Peer(PdReason::ProtocolMismatch));
        };
        let expected = planned_room(core.room_context(*handle)?, &wire_plan.plan);
        if ready != expected {
            return Err(TransportError::Peer(PdReason::ProtocolMismatch));
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
            return Err(TransportError::Peer(PdReason::ProtocolMismatch));
        };
        if ack != expected {
            return Err(TransportError::Peer(PdReason::ProtocolMismatch));
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
                    .ok_or(TransportError::Peer(PdReason::ProtocolMismatch))?;
                let peer_reason = parse_reason(&peer_abort.reason)
                    .filter(|peer_reason| *peer_reason != PdReason::Success)
                    .ok_or(TransportError::Peer(PdReason::ProtocolMismatch))?;
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
                    .ok_or(TransportError::Peer(PdReason::ProtocolMismatch))?;
                expected.swap_remove(position);
            }
            _ => {
                return Err(TransportError::Peer(PdReason::ProtocolMismatch));
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
        receiver.abort_after_peer_ack(handle.raw())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_snapshots_merge_every_owned_counter_without_overflow() {
        let mut target = PyPdResourceSnapshot {
            active_rooms: 1,
            active_handles: 2,
            result_slots: 3,
            pending_prepares: 4,
            wire_plans: 5,
            native_leases: 6,
            source_kv_pages: 7,
            destination_kv_pages: 8,
            aux_slots: 9,
            completion_slots: 10,
            request_slots: 11,
            in_flight_transfers: 12,
            native_batches: 13,
            pending_bytes: u64::MAX,
            quarantined_rooms: 14,
        };
        merge_resource_snapshot(
            &mut target,
            PyPdResourceSnapshot {
                active_rooms: 10,
                active_handles: 20,
                result_slots: 30,
                pending_prepares: 40,
                wire_plans: 50,
                native_leases: 60,
                source_kv_pages: 70,
                destination_kv_pages: 80,
                aux_slots: 90,
                completion_slots: 100,
                request_slots: 110,
                in_flight_transfers: 120,
                native_batches: 130,
                pending_bytes: 1,
                quarantined_rooms: 140,
            },
        );
        assert_eq!(target.active_rooms, 11);
        assert_eq!(target.active_handles, 22);
        assert_eq!(target.result_slots, 33);
        assert_eq!(target.pending_prepares, 44);
        assert_eq!(target.wire_plans, 55);
        assert_eq!(target.native_leases, 66);
        assert_eq!(target.source_kv_pages, 77);
        assert_eq!(target.destination_kv_pages, 88);
        assert_eq!(target.aux_slots, 99);
        assert_eq!(target.completion_slots, 110);
        assert_eq!(target.request_slots, 121);
        assert_eq!(target.in_flight_transfers, 132);
        assert_eq!(target.native_batches, 143);
        assert_eq!(target.pending_bytes, u64::MAX);
        assert_eq!(target.quarantined_rooms, 154);
    }
}
