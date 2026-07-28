use std::sync::Arc;
use std::thread;

use flume::{Receiver, Sender, TrySendError};

use crate::pd::buffer::{
    AUX_BYTES, BufferError, CompletionRecordInput, CompletionWrites, DataPlaneEffect,
    DataPlaneIdentity, DestinationExecutor, DestinationRecordPort, DestinationVisibilityFence,
    GpuDirectFlushPort, LeaseHandle, NativeStagePort, SourceComputeFence, SourceExecutionRequest,
    SourceExecutor, TransferPlan,
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
    commands: Sender<WorkerCommand>,
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
        thread::Builder::new()
            .name("sglang-pd-data-plane".into())
            .spawn(move || run_worker(receiver, executor, destination, &mut port))
            .map_err(|_| BufferError::NativeTransfer)?;
        Ok(Self { commands })
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
        match self.commands.try_send(command) {
            Ok(()) => Ok(DataPlaneWorkerTicket { receiver }),
            Err(TrySendError::Full(_)) => Err(BufferError::WorkerFull),
            Err(TrySendError::Disconnected(_)) => Err(BufferError::NativeTransfer),
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

fn run_worker<P>(
    receiver: Receiver<WorkerCommand>,
    executor: Arc<SourceExecutor>,
    destination: Option<Arc<DestinationExecutor>>,
    port: &mut P,
) where
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
        }
    }
}
