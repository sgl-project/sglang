use std::collections::{BTreeSet, HashMap};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener as StdTcpListener, TcpStream as StdTcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use hmac::{Hmac, Mac};
use pyo3::prelude::*;
use serde::Deserialize;
use sha2::Sha256;

#[path = "python/native.rs"]
mod native;
#[path = "python/support.rs"]
mod support;
#[path = "python/worker.rs"]
mod worker;

use native::{NativeReceiver, NativeSender};
use support::*;
use worker::transport_worker;

use crate::pd::buffer::{
    AUX_BYTES, AuxRecord, AuxRecordInput, COMPLETION_BYTES, CompletionRecordInput,
    CompletionWrites, TransferPlan, TransferPlanInput, validate_completion,
};
use crate::pd::config::PdProfileV1;
use crate::pd::protocol::{
    ControlPayload, DestinationBlock, FixedBytes, MessageKind, PlanDigest, PlannedRoom,
    PrepareAccepted, PrepareRoom, Psk, Role, RoomFields, TerminalRoom,
};
use crate::pd::room::{AttemptId, Clock, PdReason, ProcessEpoch, RegistrationEpoch, SystemClock};
use crate::pd::runtime::{
    BootstrapPort, CpuMockBootstrapPort, NativeBootstrapPort, NativeRegionDescriptor,
    PairConnection, RuntimeError, RuntimeIdentity, RuntimeLifecycle, bootstrap_decode,
    bootstrap_prefill,
};
use crate::pd::transport::{
    OpaqueHandle, PdReadinessHandle as CoreReadinessHandle, PdTransportCore, ReceiverCreateInput,
    SenderChunk, SenderCreateInput, TerminalEvent, TransportError, TransportPollResult,
    TransportRoomContext,
};

type Reply<T> = std::sync::mpsc::SyncSender<Result<T, TransportError>>;

enum TransportCommand {
    Start {
        reply: Reply<()>,
    },
    SenderCreate {
        input: SenderCreateInput,
        reply: Reply<OpaqueHandle>,
    },
    SenderCreateMany {
        inputs: Vec<SenderCreateInput>,
        reply: Reply<Vec<Result<(OpaqueHandle, u64), TransportError>>>,
    },
    SenderInit {
        handles: Vec<OpaqueHandle>,
        reply: Reply<Vec<Result<(), TransportError>>>,
    },
    SenderSend {
        chunks: Vec<SenderWireChunk>,
        cuda_stream: u64,
        reply: Reply<Vec<Result<(), TransportError>>>,
    },
    ReceiverCreate {
        inputs: Vec<ReceiverCreateInput>,
        reply: Reply<Vec<Result<(OpaqueHandle, u64), TransportError>>>,
    },
    ReceiverPrepare {
        inputs: Vec<ReceiverWirePrepare>,
        reply: Reply<Vec<Result<(), TransportError>>>,
    },
    Poll {
        handles: Vec<OpaqueHandle>,
        reply: Reply<Vec<Result<TransportPollResult, TransportError>>>,
    },
    Complete {
        events: Vec<TerminalEvent>,
        reply: Reply<Vec<Result<(), TransportError>>>,
    },
    Abort {
        handles: Vec<OpaqueHandle>,
        reason: PdReason,
        reply: Reply<Vec<Result<(), TransportError>>>,
    },
    Clear {
        handles: Vec<OpaqueHandle>,
        reply: Reply<Vec<Result<(), TransportError>>>,
    },
    Snapshot {
        reply: Reply<PyPdResourceSnapshot>,
    },
    Shutdown {
        reply: Reply<crate::pd::runtime::RuntimeShutdownOutcome>,
    },
}

#[derive(Debug, Clone)]
struct SenderWireChunk {
    chunk: SenderChunk,
    source_pages: Vec<u32>,
    first_token_id: Option<i32>,
    valid_token_count: u32,
}

#[derive(Debug, Clone)]
struct ReceiverWirePrepare {
    handle: OpaqueHandle,
    destination_pages: Vec<u32>,
    valid_token_count: u32,
}

#[derive(Debug, Clone)]
struct WirePlan {
    plan: TransferPlan,
    request_digest: FixedBytes<32>,
    first_token_id: Option<i32>,
}

enum MockDataEndpoint {
    Prefill,
    Decode { listener: StdTcpListener },
}

enum ControlEndpoint {
    Prefill { listener: tokio::net::TcpListener },
    Decode { address: String },
}

enum BootstrapOwner {
    Mock(Arc<CpuMockBootstrapPort>),
    Native(Arc<NativeBootstrapPort>),
}

impl BootstrapOwner {
    fn port(&self) -> Arc<dyn BootstrapPort> {
        match self {
            Self::Mock(port) => port.clone(),
            Self::Native(port) => port.clone(),
        }
    }

    fn reset_peer(&self) -> Result<(), TransportError> {
        match self {
            Self::Mock(port) => port.reset_peer().map_err(TransportError::LocalFatal),
            Self::Native(port) => port.reset_peer().map_err(TransportError::LocalFatal),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TransportConfig {
    role: Role,
    #[serde(default)]
    process_epoch: Option<String>,
    #[serde(default)]
    registration_epoch: Option<String>,
    model_manifest_digest: String,
    tokenizer_manifest_digest: String,
    layout_fingerprint: String,
    native_abi_digest: String,
    mooncake_host: String,
    mooncake_ports: Vec<u16>,
    control_host: String,
    control_port: u16,
    pd_control_psk_file: String,
    #[serde(default)]
    mock_data_plane: bool,
    #[serde(default)]
    regions: Vec<NativeRegionDescriptor>,
}

#[pyclass(name = "PdBatchItem", frozen, get_all)]
pub struct PyPdBatchItem {
    handle: u64,
    terminal_generation: u64,
    ok: bool,
    pd_reason: String,
    retryable: bool,
}

#[pyclass(name = "PdPollResult", frozen, get_all)]
pub struct PyPdPollResult {
    handle: u64,
    ok: bool,
    status: u8,
    pd_reason: String,
    retryable: bool,
    transfer_bytes: u64,
    transfer_latency_ms: u64,
    terminal_generation: u64,
    first_token_id: Option<i32>,
    first_token_consumed: bool,
}

#[pyclass(name = "PdReadinessSnapshot", frozen, get_all)]
pub struct PyPdReadinessSnapshot {
    role: String,
    lifecycle: String,
    local_ready: bool,
    pair_ready: bool,
    accepting_rooms: bool,
    session_count: u64,
    reconnect_generation: u64,
    process_epoch: String,
    registration_epoch: String,
    peer_process_epoch: Option<String>,
    peer_registration_epoch: Option<String>,
    active_handles: usize,
    result_slots: usize,
    abort_generation: u64,
    last_pd_reason: Option<String>,
    fatal_generation: Option<u64>,
    fatal_source: Option<String>,
    fatal_duplicate_sources: u64,
    drain_generation: u64,
    shutdown_phase: String,
    shutdown_outcome: Option<String>,
    worker_lifecycle: String,
}

#[pyclass(name = "PdResourceSnapshot", frozen, get_all)]
#[derive(Default)]
pub struct PyPdResourceSnapshot {
    active_rooms: usize,
    active_handles: usize,
    result_slots: usize,
    pending_prepares: usize,
    wire_plans: usize,
    native_leases: usize,
    source_kv_pages: usize,
    destination_kv_pages: usize,
    aux_slots: usize,
    completion_slots: usize,
    request_slots: usize,
    in_flight_transfers: usize,
    native_batches: usize,
    pending_bytes: u64,
    quarantined_rooms: usize,
}

/// Read-only Arc snapshot. Holding or dropping this object cannot start,
/// stop, or otherwise own the transport worker.
#[pyclass(name = "PdReadinessHandle", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyPdReadinessHandle {
    pub(crate) inner: CoreReadinessHandle,
}

#[pymethods]
impl PyPdReadinessHandle {
    fn snapshot(&self) -> PyPdReadinessSnapshot {
        let snapshot = self.inner.snapshot();
        PyPdReadinessSnapshot {
            role: role_name(snapshot.runtime.role).to_string(),
            lifecycle: lifecycle_name(snapshot.runtime.lifecycle).to_string(),
            local_ready: snapshot.runtime.local_ready,
            pair_ready: snapshot.runtime.pair_ready,
            accepting_rooms: snapshot.accepting_rooms,
            session_count: snapshot.runtime.session_count,
            reconnect_generation: snapshot.runtime.reconnect_generation,
            process_epoch: uuid::Uuid::from_bytes(snapshot.runtime.process_epoch.as_bytes())
                .to_string(),
            registration_epoch: uuid::Uuid::from_bytes(
                snapshot.runtime.registration_epoch.as_bytes(),
            )
            .to_string(),
            peer_process_epoch: snapshot
                .runtime
                .peer_process_epoch
                .map(|epoch| uuid::Uuid::from_bytes(epoch.into_array()).to_string()),
            peer_registration_epoch: snapshot
                .runtime
                .peer_registration_epoch
                .map(|epoch| uuid::Uuid::from_bytes(epoch.into_array()).to_string()),
            active_handles: snapshot.active_handles,
            result_slots: snapshot.result_slots,
            abort_generation: snapshot.abort_generation,
            last_pd_reason: snapshot
                .runtime
                .last_reason
                .or(snapshot.last_abort_reason)
                .map(|reason| reason.code().to_string()),
            fatal_generation: snapshot.runtime.fatal.map(|fatal| fatal.generation),
            fatal_source: snapshot
                .runtime
                .fatal
                .map(|fatal| fatal_source_name(fatal.source).to_string()),
            fatal_duplicate_sources: snapshot.runtime.fatal_duplicate_sources,
            drain_generation: snapshot.runtime.drain_generation,
            shutdown_phase: shutdown_phase_name(snapshot.runtime.shutdown_phase).to_string(),
            shutdown_outcome: snapshot
                .runtime
                .shutdown_outcome
                .map(|outcome| shutdown_outcome_name(outcome).to_string()),
            worker_lifecycle: worker_lifecycle_name(snapshot.runtime.worker).to_string(),
        }
    }
}

/// Independent Python handle for the bounded Rust PD command worker.
#[pyclass(name = "PdTransport")]
pub struct PyPdTransport {
    commands: flume::Sender<TransportCommand>,
    readiness: CoreReadinessHandle,
    worker: Mutex<Option<JoinHandle<()>>>,
}

#[pymethods]
impl PyPdTransport {
    #[new]
    #[pyo3(signature = (config_json, command_capacity = 64))]
    fn new(py: Python<'_>, config_json: &str, command_capacity: usize) -> PyResult<Self> {
        if !(1..=1024).contains(&command_capacity) {
            return Err(py_transport_error(TransportError::InvalidBatch));
        }
        let config: TransportConfig = serde_json::from_str(config_json)
            .map_err(|_| py_transport_error(TransportError::InvalidBatch))?;
        let process_epoch = config
            .process_epoch
            .as_deref()
            .map(ProcessEpoch::parse)
            .transpose()
            .map_err(|_| py_transport_error(TransportError::InvalidBatch))?
            .unwrap_or_else(ProcessEpoch::random);
        let configured_registration_epoch = config
            .registration_epoch
            .as_deref()
            .map(RegistrationEpoch::parse)
            .transpose()
            .map_err(|_| py_transport_error(TransportError::InvalidBatch))?;
        if config.control_host.is_empty()
            || config.control_port == 0
            || config.pd_control_psk_file.is_empty()
        {
            return Err(py_transport_error(TransportError::InvalidBatch));
        }
        let psk = Psk::load(Path::new(&config.pd_control_psk_file))
            .map_err(|_| py_transport_error(TransportError::InvalidBatch))?;
        let layout_fingerprint = parse_digest(&config.layout_fingerprint)?;
        let (registration_epoch, bootstrap_owner) = if config.mock_data_plane {
            let registration_epoch =
                configured_registration_epoch.unwrap_or_else(RegistrationEpoch::random);
            let profile = Arc::new(
                PdProfileV1::load_embedded()
                    .map_err(|_| py_transport_error(TransportError::InvalidBatch))?,
            );
            let provisional_identity = RuntimeIdentity::new(
                config.role,
                process_epoch,
                registration_epoch,
                parse_digest(&config.model_manifest_digest)?,
                parse_digest(&config.tokenizer_manifest_digest)?,
                layout_fingerprint,
                parse_digest(&config.native_abi_digest)?,
                config.mooncake_host.clone(),
                config
                    .mooncake_ports
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>(),
                profile,
            )
            .map_err(|_| py_transport_error(TransportError::InvalidBatch))?;
            let port = CpuMockBootstrapPort::new(&provisional_identity)
                .map_err(|reason| py_transport_error(TransportError::LocalFatal(reason)))?;
            (registration_epoch, BootstrapOwner::Mock(Arc::new(port)))
        } else {
            let mooncake_port = config
                .mooncake_ports
                .first()
                .copied()
                .ok_or_else(|| py_transport_error(TransportError::InvalidBatch))?;
            let endpoint: SocketAddr = format!("{}:{mooncake_port}", config.mooncake_host)
                .parse()
                .map_err(|_| py_transport_error(TransportError::InvalidBatch))?;
            let role = config.role;
            let regions = config.regions.clone();
            let port = py
                .detach(move || {
                    NativeBootstrapPort::new(role, endpoint, layout_fingerprint, regions)
                })
                .map_err(|reason| py_transport_error(TransportError::LocalFatal(reason)))?;
            let registration_epoch = port
                .registration_epoch()
                .map_err(|reason| py_transport_error(TransportError::LocalFatal(reason)))?;
            if configured_registration_epoch
                .is_some_and(|configured| configured != registration_epoch)
            {
                return Err(py_transport_error(TransportError::StaleHandle));
            }
            (registration_epoch, BootstrapOwner::Native(Arc::new(port)))
        };
        let identity = RuntimeIdentity::new(
            config.role,
            process_epoch,
            registration_epoch,
            parse_digest(&config.model_manifest_digest)?,
            parse_digest(&config.tokenizer_manifest_digest)?,
            layout_fingerprint,
            parse_digest(&config.native_abi_digest)?,
            config.mooncake_host,
            config.mooncake_ports.into_iter().collect::<BTreeSet<_>>(),
            Arc::new(
                PdProfileV1::load_embedded()
                    .map_err(|_| py_transport_error(TransportError::InvalidBatch))?,
            ),
        )
        .map_err(|_| py_transport_error(TransportError::InvalidBatch))?;
        let mut core = PdTransportCore::new(identity.clone(), Arc::new(SystemClock::default()))
            .map_err(py_transport_error)?;
        core.configure_gateway_bootstrap(
            config.control_host.clone(),
            BTreeSet::from([config.control_port]),
        )
        .map_err(py_transport_error)?;
        let readiness = core.readiness();
        let (commands, receiver) = flume::bounded(command_capacity);
        let worker = std::thread::Builder::new()
            .name("sglang-pd-transport".to_string())
            .spawn(move || {
                transport_worker(
                    core,
                    identity,
                    psk,
                    config.control_host,
                    config.control_port,
                    bootstrap_owner,
                    receiver,
                );
            })
            .map_err(|_| py_transport_error(TransportError::LocalFatal(PdReason::LocalFatal)))?;
        Ok(Self {
            commands,
            readiness,
            worker: Mutex::new(Some(worker)),
        })
    }

    fn start(&self, py: Python<'_>) -> PyResult<()> {
        self.dispatch(py, |reply| TransportCommand::Start { reply })
    }

    fn sender_create(
        &self,
        py: Python<'_>,
        decode_process_epoch: &str,
        bootstrap_room: u64,
        attempt_id: &str,
        request_digest: &str,
    ) -> PyResult<u64> {
        let input = SenderCreateInput {
            decode_process_epoch: ProcessEpoch::parse(decode_process_epoch)
                .map_err(|_| py_transport_error(TransportError::InvalidBatch))?,
            bootstrap_room,
            attempt_id: AttemptId::parse(attempt_id)
                .map_err(|_| py_transport_error(TransportError::InvalidBatch))?,
            request_digest: parse_digest(request_digest)?,
        };
        self.dispatch(py, |reply| TransportCommand::SenderCreate { input, reply })
            .map(OpaqueHandle::raw)
    }

    fn sender_create_many(
        &self,
        py: Python<'_>,
        decode_process_epochs: Vec<String>,
        bootstrap_rooms: Vec<u64>,
        attempt_ids: Vec<String>,
        request_digests: Vec<String>,
    ) -> PyResult<Vec<PyPdBatchItem>> {
        let length = decode_process_epochs.len();
        if bootstrap_rooms.len() != length
            || attempt_ids.len() != length
            || request_digests.len() != length
        {
            return Err(py_transport_error(TransportError::InvalidBatch));
        }
        let mut inputs = Vec::with_capacity(length);
        for (((decode_process_epoch, bootstrap_room), attempt_id), request_digest) in
            decode_process_epochs
                .into_iter()
                .zip(bootstrap_rooms)
                .zip(attempt_ids)
                .zip(request_digests)
        {
            inputs.push(SenderCreateInput {
                decode_process_epoch: ProcessEpoch::parse(&decode_process_epoch)
                    .map_err(|_| py_transport_error(TransportError::InvalidBatch))?,
                bootstrap_room,
                attempt_id: AttemptId::parse(&attempt_id)
                    .map_err(|_| py_transport_error(TransportError::InvalidBatch))?,
                request_digest: parse_digest(&request_digest)?,
            });
        }
        let results = self.dispatch(py, |reply| TransportCommand::SenderCreateMany {
            inputs,
            reply,
        })?;
        Ok(results
            .into_iter()
            .map(|result| match result {
                Ok((handle, terminal_generation)) => {
                    created_batch_item(handle.raw(), terminal_generation)
                }
                Err(error) => batch_item(0, Err(error)),
            })
            .collect())
    }

    fn sender_init_many(&self, py: Python<'_>, handles: Vec<u64>) -> PyResult<Vec<PyPdBatchItem>> {
        let handles = opaque_handles(handles);
        let raw = raw_handles(&handles);
        let results = self.dispatch(py, |reply| TransportCommand::SenderInit { handles, reply })?;
        Ok(batch_items(raw, results))
    }

    #[pyo3(signature = (
        handles,
        transfer_bytes,
        source_pages,
        first_token_ids,
        valid_token_counts,
        cuda_stream = 0
    ))]
    // Frozen batch API copies seven parallel arrays plus the current CUDA stream.
    #[allow(clippy::too_many_arguments)]
    fn sender_send_chunks(
        &self,
        py: Python<'_>,
        handles: Vec<u64>,
        transfer_bytes: Vec<u64>,
        source_pages: Vec<Vec<u32>>,
        first_token_ids: Vec<Option<i32>>,
        valid_token_counts: Vec<u32>,
        cuda_stream: u64,
    ) -> PyResult<Vec<PyPdBatchItem>> {
        let length = handles.len();
        if transfer_bytes.len() != length
            || source_pages.len() != length
            || first_token_ids.len() != length
            || valid_token_counts.len() != length
        {
            return Err(py_transport_error(TransportError::InvalidBatch));
        }
        let raw = handles.clone();
        let chunks = handles
            .into_iter()
            .zip(transfer_bytes)
            .zip(source_pages)
            .zip(first_token_ids)
            .zip(valid_token_counts)
            .map(
                |(
                    (((handle, transfer_bytes), source_pages), first_token_id),
                    valid_token_count,
                )| {
                    SenderWireChunk {
                        chunk: SenderChunk {
                            handle: OpaqueHandle::from_raw(handle),
                            transfer_bytes,
                        },
                        source_pages,
                        first_token_id,
                        valid_token_count,
                    }
                },
            )
            .collect();
        let results = self.dispatch(py, |reply| TransportCommand::SenderSend {
            chunks,
            cuda_stream,
            reply,
        })?;
        Ok(batch_items(raw, results))
    }

    fn receiver_prepare_many(
        &self,
        py: Python<'_>,
        handles: Vec<u64>,
        destination_pages: Vec<Vec<u32>>,
        valid_token_counts: Vec<u32>,
    ) -> PyResult<Vec<PyPdBatchItem>> {
        let length = handles.len();
        if destination_pages.len() != length || valid_token_counts.len() != length {
            return Err(py_transport_error(TransportError::InvalidBatch));
        }
        let raw = handles.clone();
        let inputs = handles
            .into_iter()
            .zip(destination_pages)
            .zip(valid_token_counts)
            .map(
                |((handle, destination_pages), valid_token_count)| ReceiverWirePrepare {
                    handle: OpaqueHandle::from_raw(handle),
                    destination_pages,
                    valid_token_count,
                },
            )
            .collect();
        let results = self.dispatch(py, |reply| TransportCommand::ReceiverPrepare {
            inputs,
            reply,
        })?;
        Ok(batch_items(raw, results))
    }

    fn receiver_create_many(
        &self,
        py: Python<'_>,
        bootstrap_rooms: Vec<u64>,
        attempt_ids: Vec<String>,
        request_digests: Vec<String>,
    ) -> PyResult<Vec<PyPdBatchItem>> {
        let length = bootstrap_rooms.len();
        if attempt_ids.len() != length || request_digests.len() != length {
            return Err(py_transport_error(TransportError::InvalidBatch));
        }
        let mut inputs = Vec::with_capacity(length);
        for ((bootstrap_room, attempt_id), request_digest) in bootstrap_rooms
            .into_iter()
            .zip(attempt_ids)
            .zip(request_digests)
        {
            inputs.push(ReceiverCreateInput {
                bootstrap_room,
                attempt_id: AttemptId::parse(&attempt_id)
                    .map_err(|_| py_transport_error(TransportError::InvalidBatch))?,
                request_digest: parse_digest(&request_digest)?,
            });
        }
        let results = self.dispatch(py, |reply| TransportCommand::ReceiverCreate {
            inputs,
            reply,
        })?;
        Ok(results
            .into_iter()
            .map(|result| match result {
                Ok((handle, terminal_generation)) => {
                    created_batch_item(handle.raw(), terminal_generation)
                }
                Err(error) => batch_item(0, Err(error)),
            })
            .collect())
    }

    fn poll_many(&self, py: Python<'_>, handles: Vec<u64>) -> PyResult<Vec<PyPdPollResult>> {
        let handles = opaque_handles(handles);
        let raw = raw_handles(&handles);
        let results = self.dispatch(py, |reply| TransportCommand::Poll { handles, reply })?;
        Ok(raw
            .into_iter()
            .zip(results)
            .map(|(handle, result)| poll_result(handle, result))
            .collect())
    }

    fn complete_many(
        &self,
        py: Python<'_>,
        handles: Vec<u64>,
        pd_reasons: Vec<String>,
        first_token_ids: Vec<Option<i32>>,
        transfer_bytes: Vec<u64>,
    ) -> PyResult<Vec<PyPdBatchItem>> {
        let length = handles.len();
        if pd_reasons.len() != length
            || first_token_ids.len() != length
            || transfer_bytes.len() != length
        {
            return Err(py_transport_error(TransportError::InvalidBatch));
        }
        let raw = handles.clone();
        let mut events = Vec::with_capacity(length);
        for (((handle, reason), first_token_id), transfer_bytes) in handles
            .into_iter()
            .zip(pd_reasons)
            .zip(first_token_ids)
            .zip(transfer_bytes)
        {
            events.push(TerminalEvent {
                handle: OpaqueHandle::from_raw(handle),
                reason: parse_reason(&reason)
                    .ok_or_else(|| py_transport_error(TransportError::InvalidBatch))?,
                first_token_id,
                transfer_bytes,
            });
        }
        let results = self.dispatch(py, |reply| TransportCommand::Complete { events, reply })?;
        Ok(batch_items(raw, results))
    }

    fn abort_many(
        &self,
        py: Python<'_>,
        handles: Vec<u64>,
        pd_reason: &str,
    ) -> PyResult<Vec<PyPdBatchItem>> {
        let reason = parse_reason(pd_reason)
            .filter(|reason| *reason != PdReason::Success)
            .ok_or_else(|| py_transport_error(TransportError::InvalidBatch))?;
        let handles = opaque_handles(handles);
        let raw = raw_handles(&handles);
        let results = self.dispatch(py, |reply| TransportCommand::Abort {
            handles,
            reason,
            reply,
        })?;
        Ok(batch_items(raw, results))
    }

    fn clear_many(&self, py: Python<'_>, handles: Vec<u64>) -> PyResult<Vec<PyPdBatchItem>> {
        let handles = opaque_handles(handles);
        let raw = raw_handles(&handles);
        let results = self.dispatch(py, |reply| TransportCommand::Clear { handles, reply })?;
        Ok(batch_items(raw, results))
    }

    fn readiness(&self) -> PyPdReadinessHandle {
        PyPdReadinessHandle {
            inner: self.readiness.clone(),
        }
    }

    fn resource_snapshot(&self, py: Python<'_>) -> PyResult<PyPdResourceSnapshot> {
        self.dispatch(py, |reply| TransportCommand::Snapshot { reply })
    }

    fn shutdown(&self, py: Python<'_>) -> PyResult<String> {
        if self
            .worker
            .lock()
            .map_err(|_| py_transport_error(TransportError::LocalFatal(PdReason::LocalFatal)))?
            .is_none()
        {
            let outcome = self
                .readiness
                .snapshot()
                .runtime
                .shutdown_outcome
                .unwrap_or(crate::pd::runtime::RuntimeShutdownOutcome::FatalUnsafe);
            return Ok(shutdown_outcome_name(outcome).to_string());
        }
        let outcome = self.dispatch(py, |reply| TransportCommand::Shutdown { reply })?;
        if let Some(worker) = self
            .worker
            .lock()
            .map_err(|_| py_transport_error(TransportError::LocalFatal(PdReason::LocalFatal)))?
            .take()
        {
            py.detach(move || worker.join()).map_err(|_| {
                py_transport_error(TransportError::LocalFatal(PdReason::LocalFatal))
            })?;
        }
        Ok(shutdown_outcome_name(outcome).to_string())
    }
}

impl PyPdTransport {
    fn dispatch<T, F>(&self, py: Python<'_>, command: F) -> PyResult<T>
    where
        T: Send + 'static,
        F: FnOnce(Reply<T>) -> TransportCommand,
    {
        let (reply, response) = std::sync::mpsc::sync_channel(1);
        let command = command(reply);
        py.detach(move || {
            self.commands
                .try_send(command)
                .map_err(|error| match error {
                    flume::TrySendError::Full(_) => {
                        py_transport_error(TransportError::CapacityExhausted)
                    }
                    flume::TrySendError::Disconnected(_) => {
                        py_transport_error(TransportError::LocalFatal(PdReason::LocalFatal))
                    }
                })?;
            response
                .recv()
                .map_err(|_| py_transport_error(TransportError::LocalFatal(PdReason::LocalFatal)))?
                .map_err(py_transport_error)
        })
    }
}

impl Drop for PyPdTransport {
    fn drop(&mut self) {
        let Some(worker) = self
            .worker
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        else {
            return;
        };
        let (reply, response) = std::sync::mpsc::sync_channel(1);
        if self
            .commands
            .try_send(TransportCommand::Shutdown { reply })
            .is_ok()
        {
            let _ = response.recv_timeout(Duration::from_millis(100));
        }
        let deadline = Instant::now() + Duration::from_millis(100);
        while !worker.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        if worker.is_finished() {
            let _ = worker.join();
        }
    }
}
