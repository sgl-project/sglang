use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use flume::{Sender, TrySendError};

use crate::mooncake::worker::Worker;
use crate::mooncake::{
    BatchId, BatchSnapshot, EngineError, EngineFactory, MemoryBuffer, MemoryLocation,
    PeerDescriptor, PeerId, RegionDescriptor, RegionId, RemoteRegionDescriptor, TransferOperation,
};

static NEXT_OWNER_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct OwnerConfig {
    command_queue_capacity: usize,
    poll_interval: Duration,
    response_timeout: Duration,
}

impl OwnerConfig {
    pub fn new(
        command_queue_capacity: usize,
        poll_interval: Duration,
        response_timeout: Duration,
    ) -> Result<Self, EngineError> {
        if command_queue_capacity == 0 {
            return Err(EngineError::InvalidDescriptor {
                field: "command_queue_capacity",
                detail: "must be non-zero".into(),
            });
        }
        if poll_interval.is_zero() {
            return Err(EngineError::InvalidDescriptor {
                field: "poll_interval",
                detail: "must be non-zero".into(),
            });
        }
        if response_timeout.is_zero() {
            return Err(EngineError::InvalidDescriptor {
                field: "response_timeout",
                detail: "must be non-zero".into(),
            });
        }
        Ok(Self {
            command_queue_capacity,
            poll_interval,
            response_timeout,
        })
    }
}

impl Default for OwnerConfig {
    fn default() -> Self {
        Self {
            command_queue_capacity: 256,
            poll_interval: Duration::from_millis(1),
            response_timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShutdownOutcome {
    SafeTerminal,
    NotSafe { batches: Vec<BatchId> },
}

pub struct EngineOwner {
    client: Arc<Client>,
}

impl EngineOwner {
    pub fn start<F>(config: OwnerConfig, factory: F) -> Result<Self, EngineError>
    where
        F: EngineFactory,
    {
        let (command_tx, command_rx) = flume::bounded(config.command_queue_capacity);
        let (init_tx, init_rx) = flume::bounded(1);
        let safe_shutdown = Arc::new(Mutex::new(None));
        let worker_safe_shutdown = Arc::clone(&safe_shutdown);
        let poll_interval = config.poll_interval;

        thread::Builder::new()
            .name("sglang-mooncake-owner".into())
            .spawn(move || {
                let mut engine = match factory.create() {
                    Ok(engine) => engine,
                    Err(error) => {
                        let _ = init_tx.send(Err(error));
                        return;
                    }
                };
                if init_tx.send(Ok(())).is_err() {
                    let _ = engine.shutdown();
                    return;
                }
                Worker::new(engine, command_rx, poll_interval, worker_safe_shutdown).run();
            })
            .map_err(|error| EngineError::WorkerStart {
                detail: error.to_string(),
            })?;

        match init_rx.recv_timeout(config.response_timeout) {
            Ok(Ok(())) => Ok(Self {
                client: Arc::new(Client {
                    owner_id: NEXT_OWNER_ID.fetch_add(1, Ordering::Relaxed),
                    commands: command_tx,
                    response_timeout: config.response_timeout,
                    poll_interval: config.poll_interval,
                    safe_shutdown,
                }),
            }),
            Ok(Err(error)) => Err(error),
            Err(flume::RecvTimeoutError::Timeout) => Err(EngineError::ResponseTimeout {
                operation: "worker initialization",
            }),
            Err(flume::RecvTimeoutError::Disconnected) => Err(EngineError::WorkerClosed),
        }
    }

    pub fn register_region(
        &self,
        buffer: MemoryBuffer,
        location: MemoryLocation,
    ) -> Result<Region, EngineError> {
        let descriptor = RegionDescriptor::new(buffer, location)?;
        let remote = RemoteRegionDescriptor::from_local(&descriptor);
        let id = self
            .client
            .request("register_region", |reply| Command::RegisterRegion {
                descriptor,
                reply,
            })?;
        Ok(Region {
            id,
            remote,
            owner_id: self.client.owner_id,
            client: Arc::clone(&self.client),
            released: false,
        })
    }

    pub fn local_peer_descriptor(&self) -> Result<PeerDescriptor, EngineError> {
        self.client.request("local_peer_descriptor", |reply| {
            Command::LocalPeerDescriptor { reply }
        })
    }

    pub fn open_peer(&self, descriptor: PeerDescriptor) -> Result<Peer, EngineError> {
        let id = self
            .client
            .request("open_peer", |reply| Command::OpenPeer { descriptor, reply })?;
        Ok(Peer {
            id,
            owner_id: self.client.owner_id,
            client: Arc::clone(&self.client),
            released: false,
        })
    }

    pub fn submit(&self, operations: Vec<TransferOperation>) -> Result<Batch, EngineError> {
        if operations
            .iter()
            .any(|operation| operation.owner_id() != self.client.owner_id)
        {
            return Err(EngineError::InvalidDescriptor {
                field: "operation.owner",
                detail: "region and peer handles belong to another engine owner".into(),
            });
        }
        let id = self
            .client
            .request("submit", |reply| Command::Submit { operations, reply })?;
        Ok(Batch {
            id,
            client: Arc::clone(&self.client),
            forgotten: false,
        })
    }

    pub fn shutdown(&self) -> Result<ShutdownOutcome, EngineError> {
        if self.client.safe_shutdown()? {
            return Ok(ShutdownOutcome::SafeTerminal);
        }
        self.client
            .request("shutdown", |reply| Command::Shutdown { reply })
            .or_else(|error| {
                if matches!(error, EngineError::WorkerClosed) && self.client.safe_shutdown()? {
                    Ok(ShutdownOutcome::SafeTerminal)
                } else {
                    Err(error)
                }
            })
    }
}

impl Drop for EngineOwner {
    fn drop(&mut self) {
        self.client.best_effort(Command::BeginShutdown);
    }
}

pub struct Region {
    id: RegionId,
    remote: RemoteRegionDescriptor,
    owner_id: u64,
    client: Arc<Client>,
    released: bool,
}

impl Region {
    pub fn id(&self) -> RegionId {
        self.id
    }

    pub fn length(&self) -> u64 {
        self.remote.length()
    }

    pub(crate) fn owner_id(&self) -> u64 {
        self.owner_id
    }

    pub fn remote_descriptor(&self) -> RemoteRegionDescriptor {
        self.remote.clone()
    }

    pub fn close(mut self) -> Result<(), EngineError> {
        self.client
            .request("unregister_region", |reply| Command::ReleaseRegion {
                id: self.id,
                reply: Some(reply),
            })?;
        self.released = true;
        Ok(())
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        if !self.released {
            self.client.best_effort(Command::ReleaseRegion {
                id: self.id,
                reply: None,
            });
        }
    }
}

pub struct Peer {
    id: PeerId,
    owner_id: u64,
    client: Arc<Client>,
    released: bool,
}

impl Peer {
    pub fn id(&self) -> PeerId {
        self.id
    }

    pub(crate) fn owner_id(&self) -> u64 {
        self.owner_id
    }

    pub fn close(mut self) -> Result<(), EngineError> {
        self.client
            .request("close_peer", |reply| Command::ReleasePeer {
                id: self.id,
                reply: Some(reply),
            })?;
        self.released = true;
        Ok(())
    }
}

impl Drop for Peer {
    fn drop(&mut self) {
        if !self.released {
            self.client.best_effort(Command::ReleasePeer {
                id: self.id,
                reply: None,
            });
        }
    }
}

pub struct Batch {
    id: BatchId,
    client: Arc<Client>,
    forgotten: bool,
}

impl Batch {
    pub fn id(&self) -> BatchId {
        self.id
    }

    pub fn abort(&self) -> Result<(), EngineError> {
        self.client
            .request("logical_abort", |reply| Command::Abort {
                id: self.id,
                reply,
            })
    }

    pub fn status(&self) -> Result<BatchSnapshot, EngineError> {
        self.client
            .request("batch_status", |reply| Command::Status {
                id: self.id,
                reply,
            })
    }

    pub fn wait_terminal(&self, timeout: Duration) -> Result<BatchSnapshot, EngineError> {
        let deadline = Instant::now() + timeout;
        loop {
            let snapshot = self.status()?;
            if snapshot.safe_terminal {
                return Ok(snapshot);
            }
            if Instant::now() >= deadline {
                return Err(EngineError::BatchNotTerminal { id: self.id.get() });
            }
            thread::sleep(self.client.poll_interval.min(Duration::from_millis(10)));
        }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        if !self.forgotten {
            self.client
                .best_effort(Command::ForgetBatch { id: self.id });
            self.forgotten = true;
        }
    }
}

struct Client {
    owner_id: u64,
    commands: Sender<Command>,
    response_timeout: Duration,
    poll_interval: Duration,
    safe_shutdown: Arc<Mutex<Option<ShutdownOutcome>>>,
}

impl Client {
    fn request<T>(
        &self,
        operation: &'static str,
        build: impl FnOnce(Sender<Result<T, EngineError>>) -> Command,
    ) -> Result<T, EngineError> {
        let (reply_tx, reply_rx) = flume::bounded(1);
        match self.commands.try_send(build(reply_tx)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return Err(EngineError::QueueFull),
            Err(TrySendError::Disconnected(_)) => return Err(EngineError::WorkerClosed),
        }
        match reply_rx.recv_timeout(self.response_timeout) {
            Ok(result) => result,
            Err(flume::RecvTimeoutError::Timeout) => {
                Err(EngineError::ResponseTimeout { operation })
            }
            Err(flume::RecvTimeoutError::Disconnected) => Err(EngineError::WorkerClosed),
        }
    }

    fn best_effort(&self, command: Command) {
        let _ = self.commands.try_send(command);
    }

    fn safe_shutdown(&self) -> Result<bool, EngineError> {
        Ok(matches!(
            *self
                .safe_shutdown
                .lock()
                .map_err(|_| EngineError::LockPoisoned)?,
            Some(ShutdownOutcome::SafeTerminal)
        ))
    }
}

pub(crate) enum Command {
    LocalPeerDescriptor {
        reply: Sender<Result<PeerDescriptor, EngineError>>,
    },
    RegisterRegion {
        descriptor: RegionDescriptor,
        reply: Sender<Result<RegionId, EngineError>>,
    },
    ReleaseRegion {
        id: RegionId,
        reply: Option<Sender<Result<(), EngineError>>>,
    },
    OpenPeer {
        descriptor: PeerDescriptor,
        reply: Sender<Result<PeerId, EngineError>>,
    },
    ReleasePeer {
        id: PeerId,
        reply: Option<Sender<Result<(), EngineError>>>,
    },
    Submit {
        operations: Vec<TransferOperation>,
        reply: Sender<Result<BatchId, EngineError>>,
    },
    Status {
        id: BatchId,
        reply: Sender<Result<BatchSnapshot, EngineError>>,
    },
    Abort {
        id: BatchId,
        reply: Sender<Result<(), EngineError>>,
    },
    ForgetBatch {
        id: BatchId,
    },
    Shutdown {
        reply: Sender<Result<ShutdownOutcome, EngineError>>,
    },
    BeginShutdown,
}
