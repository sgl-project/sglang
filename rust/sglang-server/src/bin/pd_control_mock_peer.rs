use std::collections::{HashMap, HashSet};
use std::env;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use sglang_server::pd::protocol::{
    ControlPayload, MessageKind, PlanDigest, PlannedRoom, PrepareRejected, Psk, Role, TerminalRoom,
};
use sglang_server::pd::room::{
    ManualClock, PdReason, ProcessEpoch, RegistrationEpoch, RoomEvent, RoomRole, RoomTable,
};
use sglang_server::pd::runtime::{
    CpuMockBootstrapPort, PairConnection, RuntimeIdentity, bootstrap_decode, bootstrap_prefill,
};
use tokio::net::{TcpListener, TcpStream};

#[path = "pd_control_mock_peer/support.rs"]
mod support;

use support::{
    RoomContext, accepted_payload, context, context_from_prepare, identity, parse_reason,
    prepare_payload, reason, room_error,
};

const NORMAL_ROOMS: u64 = 8;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("PD mock peer failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() != 5 {
        return Err(format!(
            "Usage: {} <prefill|decode> <control_addr> <psk_file> <positive|auth_failure|disconnect|reconnect>",
            arguments
                .first()
                .map(String::as_str)
                .unwrap_or("pd_control_mock_peer")
        ));
    }
    let role = match arguments[1].as_str() {
        "prefill" => Role::Prefill,
        "decode" => Role::Decode,
        _ => return Err("role must be prefill or decode".into()),
    };
    let control_address = arguments[2].clone();
    let psk = Psk::load(Path::new(&arguments[3])).map_err(|error| error.to_string())?;
    let scenario = arguments[4].as_str();
    match role {
        Role::Prefill => run_prefill(&control_address, &psk, scenario).await,
        Role::Decode => run_decode(&control_address, &psk, scenario).await,
    }
}

async fn run_prefill(address: &str, psk: &Psk, scenario: &str) -> Result<(), String> {
    let listener = TcpListener::bind(address)
        .await
        .map_err(|error| format!("control bind failed: {error}"))?;
    let identity = identity(Role::Prefill)?;
    match scenario {
        "positive" => {
            let (stream, _) = listener
                .accept()
                .await
                .map_err(|error| format!("accept failed: {error}"))?;
            run_prefill_connection(stream, identity, psk, ConnectionMode::Positive).await
        }
        "auth_failure" => {
            let (stream, _) = listener
                .accept()
                .await
                .map_err(|error| format!("accept failed: {error}"))?;
            let port = Arc::new(CpuMockBootstrapPort::new(&identity).map_err(reason)?);
            let result = bootstrap_prefill(
                stream,
                identity,
                psk,
                port.clone(),
                Arc::new(sglang_server::pd::room::SystemClock::default()),
            )
            .await;
            port.shutdown().map_err(reason)?;
            if result.is_ok() {
                return Err("wrong PSK unexpectedly reached PairReady".into());
            }
            println!("AUTH_REJECTED role=prefill");
            Ok(())
        }
        "reconnect" => {
            let (first, _) = listener
                .accept()
                .await
                .map_err(|error| format!("first accept failed: {error}"))?;
            run_prefill_connection(first, identity.clone(), psk, ConnectionMode::Disconnect)
                .await?;
            let (second, _) = listener
                .accept()
                .await
                .map_err(|error| format!("second accept failed: {error}"))?;
            run_prefill_connection(second, identity, psk, ConnectionMode::Positive).await?;
            println!("RECONNECTED role=prefill sessions=2");
            Ok(())
        }
        _ => Err("prefill scenario must be positive, auth_failure, or reconnect".into()),
    }
}

async fn run_prefill_connection(
    stream: TcpStream,
    identity: RuntimeIdentity,
    psk: &Psk,
    mode: ConnectionMode,
) -> Result<(), String> {
    let port = Arc::new(CpuMockBootstrapPort::new(&identity).map_err(reason)?);
    let connection = bootstrap_prefill(
        stream,
        identity.clone(),
        psk,
        port.clone(),
        Arc::new(sglang_server::pd::room::SystemClock::default()),
    )
    .await
    .map_err(|error| error.to_string())?;
    if !connection.readiness().ready {
        return Err("prefill reported a false PairReady".into());
    }
    let result = match mode {
        ConnectionMode::Positive => run_prefill_rooms(connection, &identity).await,
        ConnectionMode::Disconnect => expect_disconnect(connection).await,
    };
    port.shutdown().map_err(reason)?;
    result
}

async fn run_decode(address: &str, psk: &Psk, scenario: &str) -> Result<(), String> {
    let identity = identity(Role::Decode)?;
    let stream = connect_with_retry(address).await?;
    let port = Arc::new(CpuMockBootstrapPort::new(&identity).map_err(reason)?);
    if scenario == "auth_failure" {
        let result = bootstrap_decode(
            stream,
            identity,
            psk,
            port.clone(),
            Arc::new(sglang_server::pd::room::SystemClock::default()),
        )
        .await;
        port.shutdown().map_err(reason)?;
        if result.is_ok() {
            return Err("wrong PSK unexpectedly reached PairReady".into());
        }
        println!("AUTH_REJECTED role=decode");
        return Ok(());
    }
    let connection = bootstrap_decode(
        stream,
        identity.clone(),
        psk,
        port.clone(),
        Arc::new(sglang_server::pd::room::SystemClock::default()),
    )
    .await
    .map_err(|error| error.to_string())?;
    if !connection.readiness().ready {
        return Err("decode reported a false PairReady".into());
    }
    let result = match scenario {
        "positive" => run_decode_rooms(connection, &identity).await,
        "disconnect" => {
            drop(connection);
            println!("DISCONNECTED role=decode ready=true");
            Ok(())
        }
        _ => Err("decode scenario must be positive, auth_failure, or disconnect".into()),
    };
    port.shutdown().map_err(reason)?;
    result
}

async fn run_prefill_rooms(
    mut connection: PairConnection,
    identity: &RuntimeIdentity,
) -> Result<(), String> {
    let readiness = connection.readiness().clone();
    let decode_process =
        ProcessEpoch::from_bytes(readiness.peer_process_epoch.into_array()).map_err(room_error)?;
    let destination_registration = RegistrationEpoch::from_bytes(
        readiness
            .peer_registration_epoch
            .ok_or_else(|| "prefill is missing destination registration epoch".to_string())?
            .into_array(),
    )
    .map_err(room_error)?;
    let clock = Arc::new(ManualClock::new(0));
    let mut rooms = RoomTable::new(
        RoomRole::Prefill,
        decode_process,
        destination_registration,
        &identity.profile,
        clock.clone(),
    )
    .map_err(room_error)?;
    let mut contexts = HashMap::new();
    let mut prepared = HashSet::new();

    for _ in 0..=NORMAL_ROOMS {
        let frame = connection
            .receive_expected(MessageKind::PrepareRoom)
            .await
            .map_err(|error| error.to_string())?;
        let ControlPayload::PrepareRoom(prepare) = frame.payload else {
            return Err("expected PrepareRoom payload".into());
        };
        if prepare.room.decode_process_epoch != readiness.peer_process_epoch
            || prepare.destination_registration_epoch
                != readiness
                    .peer_registration_epoch
                    .expect("checked destination registration")
        {
            return Err("authenticated PrepareRoom used a stale epoch".into());
        }
        let context = context_from_prepare(&prepare, destination_registration)?;
        let room_number = context.spec.id.key.bootstrap_room;
        if prepared.insert(room_number) {
            if room_number % 2 == 0 {
                rooms.observe_local(context.spec.clone());
                rooms.observe_peer(context.spec.clone());
            } else {
                rooms.observe_peer(context.spec.clone());
                rooms.observe_local(context.spec.clone());
            }
            rooms.apply(context.spec.id, RoomEvent::SourceReady);
            rooms.apply(
                context.spec.id,
                RoomEvent::TransferSubmitted {
                    plan_digest: context.plan_digest,
                },
            );
            rooms.apply(context.spec.id, RoomEvent::TransferTerminal);
            contexts.insert(room_number, context.clone());
        } else {
            rooms.observe_peer(context.spec.clone());
        }
        connection
            .send(&ControlPayload::PrepareAccepted(accepted_payload(
                &context,
                identity,
                prepare.destination_registration_epoch,
            )))
            .await
            .map_err(|error| error.to_string())?;
    }

    for room_number in 0..NORMAL_ROOMS {
        let context = contexts
            .get(&room_number)
            .ok_or_else(|| "normal Room context missing".to_string())?;
        connection
            .send(&ControlPayload::DataReady(PlannedRoom {
                room: context.wire.clone(),
                transfer_plan_digest: context.plan_digest,
            }))
            .await
            .map_err(|error| error.to_string())?;
    }
    for _ in 0..NORMAL_ROOMS {
        let frame = connection
            .receive_expected(MessageKind::TransferComplete)
            .await
            .map_err(|error| error.to_string())?;
        let ControlPayload::TransferComplete(complete) = frame.payload else {
            return Err("expected TransferComplete payload".into());
        };
        let context = contexts
            .get(&complete.room.bootstrap_room)
            .ok_or_else(|| "completed Room context missing".to_string())?;
        rooms.apply(
            context.spec.id,
            RoomEvent::TransferComplete {
                plan_digest: complete.transfer_plan_digest,
            },
        );
        connection
            .send(&ControlPayload::TransferCompleteAck(complete))
            .await
            .map_err(|error| error.to_string())?;
    }

    handle_prefill_stale(&mut connection, &readiness).await?;
    handle_prefill_out_of_order(
        &mut connection,
        &mut rooms,
        identity,
        destination_registration,
    )
    .await?;
    handle_prefill_timeout(
        &mut connection,
        &mut rooms,
        &clock,
        destination_registration,
    )
    .await?;

    let snapshot = rooms.snapshot();
    if snapshot.active_rooms != 0 || snapshot.terminal_notifications < NORMAL_ROOMS + 2 {
        return Err("prefill Room table did not converge".into());
    }
    println!(
        "PAIR_READY role=prefill rooms={} tombstones={} notifications={}",
        NORMAL_ROOMS, snapshot.tombstones, snapshot.terminal_notifications
    );
    Ok(())
}

async fn run_decode_rooms(
    mut connection: PairConnection,
    identity: &RuntimeIdentity,
) -> Result<(), String> {
    let readiness = connection.readiness().clone();
    let decode_process =
        ProcessEpoch::from_bytes(readiness.local_process_epoch.into_array()).map_err(room_error)?;
    let destination_registration =
        RegistrationEpoch::from_bytes(readiness.local_registration_epoch.into_array())
            .map_err(room_error)?;
    let clock = Arc::new(ManualClock::new(0));
    let mut rooms = RoomTable::new(
        RoomRole::Decode,
        decode_process,
        destination_registration,
        &identity.profile,
        clock,
    )
    .map_err(room_error)?;
    let contexts: Vec<RoomContext> = (0..NORMAL_ROOMS)
        .map(|room_number| {
            context(
                decode_process,
                destination_registration,
                room_number,
                room_number + 1,
            )
        })
        .collect::<Result<_, _>>()?;

    for context in &contexts {
        rooms.observe_local(context.spec.clone());
        rooms.observe_peer(context.spec.clone());
        connection
            .send(&ControlPayload::PrepareRoom(prepare_payload(
                context,
                readiness.local_registration_epoch,
            )))
            .await
            .map_err(|error| error.to_string())?;
        if context.spec.id.key.bootstrap_room == 0 {
            connection
                .send(&ControlPayload::PrepareRoom(prepare_payload(
                    context,
                    readiness.local_registration_epoch,
                )))
                .await
                .map_err(|error| error.to_string())?;
        }
    }

    for _ in 0..=NORMAL_ROOMS {
        let frame = connection
            .receive_expected(MessageKind::PrepareAccepted)
            .await
            .map_err(|error| error.to_string())?;
        let ControlPayload::PrepareAccepted(accepted) = frame.payload else {
            return Err("expected PrepareAccepted payload".into());
        };
        let context = contexts
            .get(accepted.room.bootstrap_room as usize)
            .ok_or_else(|| "accepted Room context missing".to_string())?;
        rooms.apply(
            context.spec.id,
            RoomEvent::PrepareAccepted {
                plan_digest: accepted.transfer_plan_digest,
            },
        );
    }

    for _ in 0..NORMAL_ROOMS {
        let frame = connection
            .receive_expected(MessageKind::DataReady)
            .await
            .map_err(|error| error.to_string())?;
        let ControlPayload::DataReady(data_ready) = frame.payload else {
            return Err("expected DataReady payload".into());
        };
        let context = contexts
            .get(data_ready.room.bootstrap_room as usize)
            .ok_or_else(|| "DataReady Room context missing".to_string())?;
        rooms.apply(
            context.spec.id,
            RoomEvent::DataReady {
                plan_digest: data_ready.transfer_plan_digest,
            },
        );
        connection
            .send(&ControlPayload::TransferComplete(data_ready))
            .await
            .map_err(|error| error.to_string())?;
    }
    for _ in 0..NORMAL_ROOMS {
        let frame = connection
            .receive_expected(MessageKind::TransferCompleteAck)
            .await
            .map_err(|error| error.to_string())?;
        let ControlPayload::TransferCompleteAck(ack) = frame.payload else {
            return Err("expected TransferCompleteAck payload".into());
        };
        let context = contexts
            .get(ack.room.bootstrap_room as usize)
            .ok_or_else(|| "Ack Room context missing".to_string())?;
        rooms.apply(
            context.spec.id,
            RoomEvent::TransferCompleteAck {
                plan_digest: ack.transfer_plan_digest,
            },
        );
    }

    send_decode_stale(&mut connection, &readiness).await?;
    send_decode_out_of_order(
        &mut connection,
        &mut rooms,
        decode_process,
        destination_registration,
        &readiness,
    )
    .await?;
    send_decode_timeout(
        &mut connection,
        &mut rooms,
        decode_process,
        destination_registration,
        &readiness,
    )
    .await?;

    let snapshot = rooms.snapshot();
    if snapshot.active_rooms != 0 || snapshot.terminal_notifications < NORMAL_ROOMS + 2 {
        return Err("decode Room table did not converge".into());
    }
    println!(
        "PAIR_READY role=decode rooms={} tombstones={} notifications={}",
        NORMAL_ROOMS, snapshot.tombstones, snapshot.terminal_notifications
    );
    Ok(())
}

async fn handle_prefill_stale(
    connection: &mut PairConnection,
    readiness: &sglang_server::pd::runtime::PairReadiness,
) -> Result<(), String> {
    let frame = connection
        .receive_expected(MessageKind::PrepareRoom)
        .await
        .map_err(|error| error.to_string())?;
    let ControlPayload::PrepareRoom(stale) = frame.payload else {
        return Err("expected stale PrepareRoom".into());
    };
    if stale.room.decode_process_epoch == readiness.peer_process_epoch {
        return Err("stale process epoch mutation was not stale".into());
    }
    connection
        .send(&ControlPayload::PrepareRejected(PrepareRejected {
            room: stale.room,
            reason: PdReason::StaleEpoch.code().into(),
        }))
        .await
        .map_err(|error| error.to_string())
}

async fn send_decode_stale(
    connection: &mut PairConnection,
    readiness: &sglang_server::pd::runtime::PairReadiness,
) -> Result<(), String> {
    let stale_process = ProcessEpoch::random();
    let registration =
        RegistrationEpoch::from_bytes(readiness.local_registration_epoch.into_array())
            .map_err(room_error)?;
    let context = context(stale_process, registration, 90, 90)?;
    connection
        .send(&ControlPayload::PrepareRoom(prepare_payload(
            &context,
            readiness.local_registration_epoch,
        )))
        .await
        .map_err(|error| error.to_string())?;
    let frame = connection
        .receive_expected(MessageKind::PrepareRejected)
        .await
        .map_err(|error| error.to_string())?;
    let ControlPayload::PrepareRejected(rejected) = frame.payload else {
        return Err("expected stale PrepareRejected".into());
    };
    if rejected.reason != PdReason::StaleEpoch.code() {
        return Err("stale Room did not receive PD_STALE_EPOCH".into());
    }
    Ok(())
}

async fn handle_prefill_out_of_order(
    connection: &mut PairConnection,
    rooms: &mut RoomTable,
    identity: &RuntimeIdentity,
    registration: RegistrationEpoch,
) -> Result<(), String> {
    let frame = connection
        .receive_expected(MessageKind::PrepareRoom)
        .await
        .map_err(|error| error.to_string())?;
    let ControlPayload::PrepareRoom(prepare) = frame.payload else {
        return Err("expected out-of-order PrepareRoom".into());
    };
    let context = context_from_prepare(&prepare, registration)?;
    rooms.observe_local(context.spec.clone());
    rooms.observe_peer(context.spec.clone());
    connection
        .send(&ControlPayload::PrepareAccepted(accepted_payload(
            &context,
            identity,
            prepare.destination_registration_epoch,
        )))
        .await
        .map_err(|error| error.to_string())?;
    let frame = connection
        .receive_expected(MessageKind::TransferComplete)
        .await
        .map_err(|error| error.to_string())?;
    let ControlPayload::TransferComplete(complete) = frame.payload else {
        return Err("expected early TransferComplete".into());
    };
    rooms.apply(
        context.spec.id,
        RoomEvent::TransferComplete {
            plan_digest: complete.transfer_plan_digest,
        },
    );
    connection
        .send(&ControlPayload::TransferFailed(TerminalRoom {
            room: context.wire,
            transfer_plan_digest: PlanDigest::from_digest(context.plan_digest),
            reason: PdReason::ProtocolMismatch.code().into(),
        }))
        .await
        .map_err(|error| error.to_string())
}

async fn send_decode_out_of_order(
    connection: &mut PairConnection,
    rooms: &mut RoomTable,
    process: ProcessEpoch,
    registration: RegistrationEpoch,
    readiness: &sglang_server::pd::runtime::PairReadiness,
) -> Result<(), String> {
    let context = context(process, registration, 91, 91)?;
    rooms.observe_local(context.spec.clone());
    rooms.observe_peer(context.spec.clone());
    connection
        .send(&ControlPayload::PrepareRoom(prepare_payload(
            &context,
            readiness.local_registration_epoch,
        )))
        .await
        .map_err(|error| error.to_string())?;
    let frame = connection
        .receive_expected(MessageKind::PrepareAccepted)
        .await
        .map_err(|error| error.to_string())?;
    let ControlPayload::PrepareAccepted(accepted) = frame.payload else {
        return Err("expected out-of-order PrepareAccepted".into());
    };
    rooms.apply(
        context.spec.id,
        RoomEvent::PrepareAccepted {
            plan_digest: accepted.transfer_plan_digest,
        },
    );
    connection
        .send(&ControlPayload::TransferComplete(PlannedRoom {
            room: context.wire.clone(),
            transfer_plan_digest: context.plan_digest,
        }))
        .await
        .map_err(|error| error.to_string())?;
    let frame = connection
        .receive_expected(MessageKind::TransferFailed)
        .await
        .map_err(|error| error.to_string())?;
    let ControlPayload::TransferFailed(failed) = frame.payload else {
        return Err("expected TransferFailed".into());
    };
    rooms.apply(
        context.spec.id,
        RoomEvent::TransferFailed(parse_reason(&failed.reason)?),
    );
    Ok(())
}

async fn handle_prefill_timeout(
    connection: &mut PairConnection,
    rooms: &mut RoomTable,
    clock: &ManualClock,
    registration: RegistrationEpoch,
) -> Result<(), String> {
    let frame = connection
        .receive_expected(MessageKind::PrepareRoom)
        .await
        .map_err(|error| error.to_string())?;
    let ControlPayload::PrepareRoom(prepare) = frame.payload else {
        return Err("expected timeout PrepareRoom".into());
    };
    let context = context_from_prepare(&prepare, registration)?;
    rooms.observe_peer(context.spec.clone());
    clock.advance(300_000);
    let expired = rooms.expire_due();
    if !expired
        .iter()
        .any(|(id, reason)| *id == context.spec.id && *reason == PdReason::RendezvousTimeout)
    {
        return Err("rendezvous timeout did not fire at 300 seconds".into());
    }
    connection
        .send(&ControlPayload::PrepareRejected(PrepareRejected {
            room: context.wire,
            reason: PdReason::RendezvousTimeout.code().into(),
        }))
        .await
        .map_err(|error| error.to_string())
}

async fn send_decode_timeout(
    connection: &mut PairConnection,
    rooms: &mut RoomTable,
    process: ProcessEpoch,
    registration: RegistrationEpoch,
    readiness: &sglang_server::pd::runtime::PairReadiness,
) -> Result<(), String> {
    let context = context(process, registration, 92, 92)?;
    rooms.observe_local(context.spec.clone());
    rooms.observe_peer(context.spec.clone());
    connection
        .send(&ControlPayload::PrepareRoom(prepare_payload(
            &context,
            readiness.local_registration_epoch,
        )))
        .await
        .map_err(|error| error.to_string())?;
    let frame = connection
        .receive_expected(MessageKind::PrepareRejected)
        .await
        .map_err(|error| error.to_string())?;
    let ControlPayload::PrepareRejected(rejected) = frame.payload else {
        return Err("expected timeout PrepareRejected".into());
    };
    let reason = parse_reason(&rejected.reason)?;
    if reason != PdReason::RendezvousTimeout {
        return Err("timeout Room did not receive PD_RENDEZVOUS_TIMEOUT".into());
    }
    rooms.apply(context.spec.id, RoomEvent::PrepareRejected(reason));
    Ok(())
}

async fn expect_disconnect(mut connection: PairConnection) -> Result<(), String> {
    if connection.receive_expected(MessageKind::Ping).await.is_ok() {
        return Err("disconnected peer unexpectedly sent a Ping".into());
    }
    println!("PEER_LOST role=prefill ready=false reason=PD_PEER_UNAVAILABLE");
    Ok(())
}

async fn connect_with_retry(address: &str) -> Result<TcpStream, String> {
    for _ in 0..150 {
        match TcpStream::connect(address).await {
            Ok(stream) => return Ok(stream),
            Err(_) => tokio::time::sleep(Duration::from_millis(20)).await,
        }
    }
    Err("control connect did not succeed within the harness startup window".into())
}

#[derive(Clone, Copy)]
enum ConnectionMode {
    Positive,
    Disconnect,
}
