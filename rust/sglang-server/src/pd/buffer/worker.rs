use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use flume::{Receiver, Sender, TryRecvError, TrySendError};

use crate::pd::buffer::{
    AUX_BYTES, BufferError, CompletionRecordInput, CompletionWrites, DataPlaneEffect,
    DataPlaneIdentity, DestinationExecutor, DestinationRecordPort, DestinationVisibilityFence,
    GpuDirectFlushPort, LeaseHandle, NativeBatchToken, NativeSafety, NativeStagePort,
    SourceComputeFence, SourceExecutionRequest, SourceExecutor, TransferPlan,
    evaluate_native_fence,
};

struct SourceWork {
    plan: TransferPlan,
    handle: LeaseHandle,
    source_fence: Box<dyn SourceComputeFence>,
    aux: [u8; AUX_BYTES],
    completion: CompletionWrites,
    deadline_monotonic_ms: u64,
    reply: Sender<Result<DataPlaneEffect, BufferError>>,
}

enum WorkerCommand {
    Source(SourceWork),
    Destination(DestinationWork),
    ObserveNative {
        batch: NativeBatchToken,
        expected_lengths: Vec<u64>,
        reply: Sender<Result<NativeSafety, BufferError>>,
    },
    Shutdown {
        reply: Sender<()>,
    },
}

struct DestinationWork {
    plan: TransferPlan,
    handle: LeaseHandle,
    identity: DataPlaneIdentity,
    device: u32,
    visibility: Box<dyn GpuDirectFlushPort>,
    records: Box<dyn DestinationRecordPort>,
    expected: CompletionRecordInput,
    reply: Sender<Result<DataPlaneEffect, BufferError>>,
}

pub struct SourceWorkRequest<F> {
    pub plan: TransferPlan,
    pub handle: LeaseHandle,
    pub source_fence: F,
    pub aux: [u8; AUX_BYTES],
    pub completion: CompletionWrites,
    pub deadline_monotonic_ms: u64,
}

pub struct DestinationWorkRequest<V, R> {
    pub plan: TransferPlan,
    pub handle: LeaseHandle,
    pub identity: DataPlaneIdentity,
    pub device: u32,
    pub visibility: V,
    pub records: R,
    pub expected: CompletionRecordInput,
}

pub struct DataPlaneWorker {
    commands: Option<Sender<WorkerCommand>>,
    worker: Option<JoinHandle<()>>,
    lifecycle: Arc<AtomicU8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataPlaneWorkerState {
    Starting = 0,
    Running = 1,
    Quiescing = 2,
    Joined = 3,
    Failed = 4,
}

impl DataPlaneWorkerState {
    fn from_raw(value: u8) -> Self {
        match value {
            0 => Self::Starting,
            1 => Self::Running,
            2 => Self::Quiescing,
            3 => Self::Joined,
            _ => Self::Failed,
        }
    }
}

impl DataPlaneWorker {
    pub fn start<P>(
        capacity: usize,
        executor: Arc<SourceExecutor>,
        port: P,
    ) -> Result<Self, BufferError>
    where
        P: NativeStagePort + 'static,
    {
        Self::start_inner(capacity, executor, None, port)
    }

    pub fn start_with_destination<P>(
        capacity: usize,
        source: Arc<SourceExecutor>,
        destination: Arc<DestinationExecutor>,
        port: P,
    ) -> Result<Self, BufferError>
    where
        P: NativeStagePort + 'static,
    {
        Self::start_inner(capacity, source, Some(destination), port)
    }

    fn start_inner<P>(
        capacity: usize,
        executor: Arc<SourceExecutor>,
        destination: Option<Arc<DestinationExecutor>>,
        mut port: P,
    ) -> Result<Self, BufferError>
    where
        P: NativeStagePort + 'static,
    {
        if capacity == 0 {
            return Err(BufferError::InvalidDescriptor {
                field: "worker_capacity",
                detail: "must be non-zero",
            });
        }
        let (commands, receiver) = flume::bounded(capacity);
        let lifecycle = Arc::new(AtomicU8::new(DataPlaneWorkerState::Starting as u8));
        let worker_lifecycle = Arc::clone(&lifecycle);
        let worker = thread::Builder::new()
            .name("sglang-pd-data-plane".into())
            .spawn(move || {
                worker_lifecycle.store(DataPlaneWorkerState::Running as u8, Ordering::Release);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    run_worker(receiver, executor, destination, &mut port)
                }));
                let terminal = match result {
                    Ok(true) => DataPlaneWorkerState::Joined,
                    Ok(false)
                        if DataPlaneWorkerState::from_raw(
                            worker_lifecycle.load(Ordering::Acquire),
                        ) == DataPlaneWorkerState::Quiescing =>
                    {
                        DataPlaneWorkerState::Joined
                    }
                    Ok(false) | Err(_) => DataPlaneWorkerState::Failed,
                };
                worker_lifecycle.store(terminal as u8, Ordering::Release);
            })
            .map_err(|_| BufferError::NativeTransfer)?;
        Ok(Self {
            commands: Some(commands),
            worker: Some(worker),
            lifecycle,
        })
    }

    pub fn try_execute_source<F>(
        &self,
        request: SourceWorkRequest<F>,
    ) -> Result<DataPlaneWorkerTicket, BufferError>
    where
        F: SourceComputeFence + 'static,
    {
        let (reply, receiver) = flume::bounded(1);
        let command = WorkerCommand::Source(SourceWork {
            plan: request.plan,
            handle: request.handle,
            source_fence: Box::new(request.source_fence),
            aux: request.aux,
            completion: request.completion,
            deadline_monotonic_ms: request.deadline_monotonic_ms,
            reply,
        });
        self.try_send(command, receiver)
    }

    pub fn try_validate_destination<V, R>(
        &self,
        request: DestinationWorkRequest<V, R>,
    ) -> Result<DataPlaneWorkerTicket, BufferError>
    where
        V: GpuDirectFlushPort + 'static,
        R: DestinationRecordPort + 'static,
    {
        let (reply, receiver) = flume::bounded(1);
        let command = WorkerCommand::Destination(DestinationWork {
            plan: request.plan,
            handle: request.handle,
            identity: request.identity,
            device: request.device,
            visibility: Box::new(request.visibility),
            records: Box::new(request.records),
            expected: request.expected,
            reply,
        });
        self.try_send(command, receiver)
    }

    fn try_send(
        &self,
        command: WorkerCommand,
        receiver: Receiver<Result<DataPlaneEffect, BufferError>>,
    ) -> Result<DataPlaneWorkerTicket, BufferError> {
        let commands = self
            .commands
            .as_ref()
            .ok_or(BufferError::InvalidTransition)?;
        match commands.try_send(command) {
            Ok(()) => Ok(DataPlaneWorkerTicket { receiver }),
            Err(TrySendError::Full(_)) => Err(BufferError::WorkerFull),
            Err(TrySendError::Disconnected(_)) => Err(BufferError::NativeTransfer),
        }
    }

    pub fn try_observe_native(
        &self,
        batch: NativeBatchToken,
        expected_lengths: &[u64],
    ) -> Result<NativeObservationTicket, BufferError> {
        if expected_lengths.is_empty() {
            return Err(BufferError::InvalidTransition);
        }
        let commands = self
            .commands
            .as_ref()
            .ok_or(BufferError::InvalidTransition)?;
        let (reply, receiver) = flume::bounded(1);
        match commands.try_send(WorkerCommand::ObserveNative {
            batch,
            expected_lengths: expected_lengths.to_vec(),
            reply,
        }) {
            Ok(()) => Ok(NativeObservationTicket { receiver }),
            Err(TrySendError::Full(_)) => Err(BufferError::WorkerFull),
            Err(TrySendError::Disconnected(_)) => Err(BufferError::NativeTransfer),
        }
    }

    pub fn observe_native(
        &self,
        batch: NativeBatchToken,
        expected_lengths: &[u64],
    ) -> Result<NativeSafety, BufferError> {
        self.try_observe_native(batch, expected_lengths)?.wait()
    }

    pub fn pending_count(&self) -> usize {
        self.commands.as_ref().map_or(0, Sender::len)
    }

    pub fn lifecycle(&self) -> DataPlaneWorkerState {
        DataPlaneWorkerState::from_raw(self.lifecycle.load(Ordering::Acquire))
    }

    pub fn begin_shutdown(&mut self) -> Result<(), BufferError> {
        match self.lifecycle() {
            DataPlaneWorkerState::Joined => return Ok(()),
            DataPlaneWorkerState::Failed => return Err(BufferError::NativeTransfer),
            DataPlaneWorkerState::Quiescing => return Ok(()),
            DataPlaneWorkerState::Starting | DataPlaneWorkerState::Running => {}
        }
        self.lifecycle
            .store(DataPlaneWorkerState::Quiescing as u8, Ordering::Release);
        let commands = self
            .commands
            .as_ref()
            .ok_or(BufferError::InvalidTransition)?;
        let (reply, response) = flume::bounded(1);
        match commands.try_send(WorkerCommand::Shutdown { reply }) {
            Ok(()) => {
                self.commands.take();
                response
                    .recv_timeout(Duration::from_secs(30))
                    .map_err(|_| BufferError::Deadline)?;
                Ok(())
            }
            Err(TrySendError::Full(_)) => Err(BufferError::WorkerFull),
            Err(TrySendError::Disconnected(_)) => Err(BufferError::NativeTransfer),
        }
    }

    pub fn shutdown(&mut self, timeout: Duration) -> Result<DataPlaneWorkerState, BufferError> {
        if timeout.is_zero() {
            return Err(BufferError::Deadline);
        }
        match self.lifecycle() {
            DataPlaneWorkerState::Joined => {
                self.join_finished();
                return Ok(DataPlaneWorkerState::Joined);
            }
            DataPlaneWorkerState::Failed => {
                self.join_finished();
                return Err(BufferError::NativeTransfer);
            }
            DataPlaneWorkerState::Starting
            | DataPlaneWorkerState::Running
            | DataPlaneWorkerState::Quiescing => {}
        }
        if self.lifecycle() != DataPlaneWorkerState::Quiescing {
            self.begin_shutdown()?;
        }
        let deadline = Instant::now() + timeout;
        while self
            .worker
            .as_ref()
            .is_some_and(|worker| !worker.is_finished())
        {
            if Instant::now() >= deadline {
                return Err(BufferError::Deadline);
            }
            thread::sleep(Duration::from_millis(1));
        }
        self.join_finished();
        match self.lifecycle() {
            DataPlaneWorkerState::Joined => Ok(DataPlaneWorkerState::Joined),
            _ => Err(BufferError::NativeTransfer),
        }
    }

    fn join_finished(&mut self) {
        if self.worker.as_ref().is_some_and(JoinHandle::is_finished)
            && let Some(worker) = self.worker.take()
        {
            let _ = worker.join();
        }
    }
}

impl Drop for DataPlaneWorker {
    fn drop(&mut self) {
        if matches!(
            self.lifecycle(),
            DataPlaneWorkerState::Starting | DataPlaneWorkerState::Running
        ) {
            self.lifecycle
                .store(DataPlaneWorkerState::Quiescing as u8, Ordering::Release);
            if let Some(commands) = self.commands.take() {
                let (reply, _response) = flume::bounded(1);
                let _ = commands.try_send(WorkerCommand::Shutdown { reply });
            }
        }
    }
}

pub struct DataPlaneWorkerTicket {
    receiver: Receiver<Result<DataPlaneEffect, BufferError>>,
}

impl DataPlaneWorkerTicket {
    pub fn wait(self) -> Result<DataPlaneEffect, BufferError> {
        self.receiver
            .recv()
            .map_err(|_| BufferError::NativeTransfer)?
    }
}

pub struct NativeObservationTicket {
    receiver: Receiver<Result<NativeSafety, BufferError>>,
}

impl NativeObservationTicket {
    pub fn try_wait(&self) -> Result<Option<NativeSafety>, BufferError> {
        match self.receiver.try_recv() {
            Ok(result) => result.map(Some),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(BufferError::NativeTransfer),
        }
    }

    pub fn wait(self) -> Result<NativeSafety, BufferError> {
        self.receiver
            .recv()
            .map_err(|_| BufferError::NativeTransfer)?
    }
}

fn run_worker<P>(
    receiver: Receiver<WorkerCommand>,
    executor: Arc<SourceExecutor>,
    destination: Option<Arc<DestinationExecutor>>,
    port: &mut P,
) -> bool
where
    P: NativeStagePort,
{
    while let Ok(command) = receiver.recv() {
        match command {
            WorkerCommand::Source(mut work) => {
                let result = executor.execute(
                    SourceExecutionRequest {
                        plan: &work.plan,
                        handle: work.handle,
                        source_fence: work.source_fence.as_mut(),
                        aux: &work.aux,
                        completion: &work.completion,
                        deadline_monotonic_ms: work.deadline_monotonic_ms,
                    },
                    port,
                );
                let _ = work.reply.send(result);
            }
            WorkerCommand::Destination(mut work) => {
                let result = destination
                    .as_ref()
                    .ok_or(BufferError::InvalidTransition)
                    .and_then(|destination| {
                        let mut visibility =
                            DestinationVisibilityFence::new(work.device, work.visibility)?;
                        destination.validate_ready(
                            &work.plan,
                            work.handle,
                            work.identity,
                            &mut visibility,
                            work.records.as_mut(),
                            &work.expected,
                        )
                    });
                let _ = work.reply.send(result);
            }
            WorkerCommand::ObserveNative {
                batch,
                expected_lengths,
                reply,
            } => {
                let result = match port.poll(batch) {
                    Ok(snapshot) => {
                        let safety = evaluate_native_fence(&snapshot, &expected_lengths);
                        if safety.is_safe() {
                            port.free_safe(batch).map(|()| safety)
                        } else {
                            Ok(safety)
                        }
                    }
                    Err(_) => Ok(NativeSafety::Pending),
                };
                let _ = reply.send(result);
            }
            WorkerCommand::Shutdown { reply } => {
                let _ = reply.send(());
                return true;
            }
        }
    }
    false
}
