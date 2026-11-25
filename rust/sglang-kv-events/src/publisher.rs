//! PUB + ROUTER publisher for encoded KV event batches: the Rust counterpart of
//! `ZmqEventPublisher` in `python/sglang/srt/disaggregation/kv_events.py`.
//!
//! Live batches go out as `[topic, seq (u64 BE), payload]`. A replay request
//! `[identity, "", start_seq (BE)]` is answered with `[identity, "", topic, seq,
//! payload]` for every retained batch from `start_seq` on, then
//! `[identity, "", "", END_SEQ, ""]`.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use bytes::Bytes;
use tokio::sync::{Semaphore, mpsc, oneshot};
use zeromq::{PubSocket, RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

/// The sequence frame of the message that ends a replay: -1 as i64 BE.
pub const END_SEQ: [u8; 8] = [0xff; 8];

// Same as ZmqEventPublisher.SHUTDOWN_TIMEOUT.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(1);
// Same as ZmqEventPublisher.MAX_PENDING_REPLAYS and REPLAY_SEND_BUDGET.
const MAX_PENDING_REPLAYS: usize = 16;
const REPLAY_SEND_BUDGET: usize = 256;
// Arbitrary. zeromq has no per-peer high-water mark or nonblocking send, so a
// peer that holds up one send this long is dropped instead.
const SEND_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Debug, Clone)]
pub struct PublisherConfig {
    /// `tcp://host:port` or `ipc://path`; always bound, never connected.
    pub endpoint: String,
    pub replay_endpoint: Option<String>,
    pub topic: String,
    /// Batches retained for replay.
    pub buffer_steps: usize,
    /// Batches queued between `publish` and the publisher thread; 0 for no limit.
    pub max_queue_size: usize,
    /// Batches queued for the PUB socket, shared by all subscribers; 0 for no limit.
    pub hwm: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum PublisherError {
    #[error("unsupported KV event endpoint {0:?}: expected tcp://host:port or ipc://path")]
    UnsupportedEndpoint(String),
    #[error("cannot bind KV event socket {endpoint}: {reason}")]
    Bind { endpoint: String, reason: String },
    #[error("cannot start the KV event publisher thread: {0}")]
    Thread(#[from] std::io::Error),
    #[error("Publisher is closed")]
    Closed,
}

/// 0 means no limit, as for `queue.Queue(maxsize=0)` and ZMQ_SNDHWM.
fn queue_capacity(limit: usize) -> usize {
    match limit {
        0 => Semaphore::MAX_PERMITS,
        limit => limit.min(Semaphore::MAX_PERMITS),
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

pub struct Publisher {
    queue: Mutex<Option<mpsc::Sender<Bytes>>>,
    worker: Mutex<Option<Worker>>,
    local_pub_endpoint: String,
    local_replay_endpoint: Option<String>,
}

struct Worker {
    thread: JoinHandle<()>,
    /// Fires when the thread has drained its queue and closed its sockets.
    done: std::sync::mpsc::Receiver<()>,
    abort: oneshot::Sender<()>,
}

struct Bound {
    pub_endpoint: String,
    replay_endpoint: Option<String>,
}

impl Publisher {
    /// Bind the sockets and start the publisher thread.
    pub fn start(config: PublisherConfig) -> Result<Self, PublisherError> {
        let (queue, batches) = mpsc::channel(queue_capacity(config.max_queue_size));
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let (abort_tx, abort_rx) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("kv-events-pub".to_string())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build();
                match runtime {
                    Ok(runtime) => runtime.block_on(async {
                        tokio::select! {
                            () = run(config, batches, ready_tx) => {}
                            _ = abort_rx => {}
                        }
                    }),
                    Err(error) => {
                        let _ = ready_tx.send(Err(PublisherError::Thread(error)));
                    }
                }
                let _ = done_tx.send(());
            })?;
        let bound = match ready_rx.recv() {
            Ok(Ok(bound)) => bound,
            Ok(Err(error)) => {
                let _ = thread.join();
                return Err(error);
            }
            Err(_) => {
                let _ = thread.join();
                return Err(PublisherError::Closed);
            }
        };
        Ok(Self {
            queue: Mutex::new(Some(queue)),
            worker: Mutex::new(Some(Worker {
                thread,
                done: done_rx,
                abort: abort_tx,
            })),
            local_pub_endpoint: bound.pub_endpoint,
            local_replay_endpoint: bound.replay_endpoint,
        })
    }

    /// Queue one encoded batch. Blocks while the queue is full, as
    /// `queue.Queue.put` does; call it outside an async runtime.
    pub fn publish(&self, payload: Bytes) -> Result<(), PublisherError> {
        let queue = lock(&self.queue).clone().ok_or(PublisherError::Closed)?;
        queue
            .blocking_send(payload)
            .map_err(|_| PublisherError::Closed)
    }

    /// The bound PUB endpoint, with the real port for a `:0` bind.
    pub fn local_pub_endpoint(&self) -> &str {
        &self.local_pub_endpoint
    }

    pub fn local_replay_endpoint(&self) -> Option<&str> {
        self.local_replay_endpoint.as_deref()
    }

    /// Publish what is queued, then close the sockets. Idempotent.
    pub fn shutdown(&self) {
        lock(&self.queue).take();
        let Some(worker) = lock(&self.worker).take() else {
            return;
        };
        if worker.done.recv_timeout(SHUTDOWN_TIMEOUT).is_err() {
            tracing::warn!(
                "KV event publisher did not drain its queue within {SHUTDOWN_TIMEOUT:?}"
            );
            let _ = worker.abort.send(());
            // A thread that never reaches an await point cannot be stopped;
            // leave it behind rather than hang the caller.
            if worker.done.recv_timeout(SHUTDOWN_TIMEOUT).is_err() {
                return;
            }
        }
        let _ = worker.thread.join();
    }
}

impl Drop for Publisher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

type SequencedBatch = (u64, Bytes);

/// The retained batches, oldest first, with strictly increasing sequence numbers.
struct Ring {
    entries: VecDeque<SequencedBatch>,
    capacity: usize,
}

impl Ring {
    fn push(&mut self, seq: u64, payload: Bytes) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back((seq, payload));
    }

    fn since(&self, start_seq: u64) -> VecDeque<SequencedBatch> {
        let first = self.entries.partition_point(|(seq, _)| *seq < start_seq);
        self.entries.range(first..).cloned().collect()
    }
}

/// Removes the bound ipc socket files however `run` ends. zeromq only does so
/// from its accept task, which an early return or an abort never polls again.
#[derive(Default)]
struct IpcFiles(Vec<String>);

impl Drop for IpcFiles {
    fn drop(&mut self) {
        for path in &self.0 {
            let _ = std::fs::remove_file(path);
        }
    }
}

async fn run(
    config: PublisherConfig,
    mut batches: mpsc::Receiver<Bytes>,
    ready: std::sync::mpsc::Sender<Result<Bound, PublisherError>>,
) {
    let mut ipc_files = IpcFiles::default();
    let mut pub_socket = PubSocket::new();
    let pub_endpoint = match bind(&mut pub_socket, &config.endpoint, &mut ipc_files).await {
        Ok(endpoint) => endpoint,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let mut router = None;
    let mut replay_endpoint = None;
    if let Some(endpoint) = &config.replay_endpoint {
        let mut socket = RouterSocket::new();
        match bind(&mut socket, endpoint, &mut ipc_files).await {
            Ok(endpoint) => replay_endpoint = Some(endpoint),
            Err(error) => {
                let _ = ready.send(Err(error));
                return;
            }
        }
        router = Some(socket);
    }
    let _ = ready.send(Ok(Bound {
        pub_endpoint,
        replay_endpoint,
    }));

    let topic = Bytes::from(config.topic.into_bytes());
    let ring = Arc::new(Mutex::new(Ring {
        entries: VecDeque::new(),
        capacity: config.buffer_steps,
    }));
    let (live_tx, live_rx) = mpsc::channel(queue_capacity(config.hwm));
    let live = tokio::spawn(send_live(pub_socket, live_rx));
    let replay =
        router.map(|router| tokio::spawn(serve_replays(router, ring.clone(), topic.clone())));

    let mut next_seq = 0u64;
    while let Some(payload) = batches.recv().await {
        let seq = next_seq;
        next_seq += 1;
        let message = frames([
            topic.clone(),
            Bytes::copy_from_slice(&seq.to_be_bytes()),
            payload.clone(),
        ]);
        // A full PUB queue drops the live copy, as libzmq does at the high-water
        // mark; the batch is still retained for replay.
        let _ = live_tx.try_send(message);
        lock(&ring).push(seq, payload);
    }

    drop(live_tx);
    if let Ok(mut pub_socket) = live.await {
        pub_socket.unbind_all().await;
    }
    if let Some(replay) = replay {
        replay.abort();
    }
}

fn frames<const N: usize>(frames: [Bytes; N]) -> ZmqMessage {
    ZmqMessage::try_from(Vec::from(frames)).expect("a message with at least one frame")
}

async fn send_live(mut socket: PubSocket, mut live: mpsc::Receiver<ZmqMessage>) -> PubSocket {
    while let Some(message) = live.recv().await {
        match tokio::time::timeout(SEND_TIMEOUT, socket.send(message)).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => tracing::warn!(%error, "KV event publish failed"),
            // zeromq writes to the subscribers in turn and cannot say which one
            // stalled, so drop them all: they reconnect, see a sequence gap and
            // repair it through replay.
            Err(_stalled) => {
                tracing::warn!("a KV event subscriber stopped reading; dropping the subscribers");
                socket.backend().shutdown();
            }
        }
    }
    socket
}

/// The batches one requester is owed, cut from the ring when it asked.
struct Replay {
    identity: Bytes,
    batches: VecDeque<SequencedBatch>,
}

enum ReplayProgress {
    /// The end marker went out.
    Done,
    /// The send budget ran out first.
    Pending,
    /// The requester is gone or stopped reading; it gets no end marker.
    Abandoned,
}

/// One task owns the whole ROUTER socket. Its receive side removes a peer that
/// disconnected, and blocks the thread to do so if a send to that peer is
/// suspended, so a send and a receive must never be polled concurrently.
async fn serve_replays(mut socket: RouterSocket, ring: Arc<Mutex<Ring>>, topic: Bytes) {
    let mut pending: VecDeque<Replay> = VecDeque::new();
    loop {
        if pending.len() < MAX_PENDING_REPLAYS {
            let message = if pending.is_empty() {
                Some(socket.recv().await)
            } else {
                // Poll once without waiting, so pending replays keep moving.
                tokio::time::timeout(Duration::ZERO, socket.recv())
                    .await
                    .ok()
            };
            match message {
                Some(Ok(message)) => match parse_request(message) {
                    Some((identity, start_seq)) => pending.push_back(Replay {
                        identity,
                        batches: lock(&ring).since(start_seq),
                    }),
                    None => tracing::warn!("invalid KV event replay request"),
                },
                Some(Err(error)) => {
                    tracing::warn!(%error, "KV event replay receive failed");
                    if pending.is_empty() {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
                None => {}
            }
        }
        if let Some(mut replay) = pending.pop_front() {
            match send_replay(&mut socket, &mut replay, &topic).await {
                ReplayProgress::Done => {}
                ReplayProgress::Pending => pending.push_back(replay),
                ReplayProgress::Abandoned => {
                    tracing::warn!(identity = ?replay.identity, "abandoning KV event replay");
                }
            }
        }
    }
}

/// `(identity, start_seq)` of `[identity, "", start_seq]`. Like the Python
/// publisher, any length of big-endian start sequence is accepted.
fn parse_request(message: ZmqMessage) -> Option<(Bytes, u64)> {
    let frames = message.into_vec();
    let [identity, _delimiter, start_seq] = frames.as_slice() else {
        return None;
    };
    let start_seq = start_seq.iter().try_fold(0u64, |value, byte| {
        value.checked_mul(256).map(|value| value | u64::from(*byte))
    });
    // A start past every u64 sequence number replays nothing.
    Some((identity.clone(), start_seq.unwrap_or(u64::MAX)))
}

/// Send up to `REPLAY_SEND_BUDGET` messages of `replay`, then its end marker
/// once every batch is out.
async fn send_replay(
    socket: &mut RouterSocket,
    replay: &mut Replay,
    topic: &Bytes,
) -> ReplayProgress {
    for _ in 0..REPLAY_SEND_BUDGET {
        let (message, last) = match replay.batches.front() {
            Some((seq, payload)) => {
                let seq = Bytes::copy_from_slice(&seq.to_be_bytes());
                let message = [
                    replay.identity.clone(),
                    Bytes::new(),
                    topic.clone(),
                    seq,
                    payload.clone(),
                ];
                (frames(message), false)
            }
            None => {
                let end = Bytes::from_static(&END_SEQ);
                let message = [
                    replay.identity.clone(),
                    Bytes::new(),
                    Bytes::new(),
                    end,
                    Bytes::new(),
                ];
                (frames(message), true)
            }
        };
        let sent = tokio::time::timeout(SEND_TIMEOUT, socket.send(message)).await;
        if !matches!(sent, Ok(Ok(()))) {
            return ReplayProgress::Abandoned;
        }
        if last {
            return ReplayProgress::Done;
        }
        replay.batches.pop_front();
    }
    ReplayProgress::Pending
}

async fn bind(
    socket: &mut impl Socket,
    endpoint: &str,
    ipc_files: &mut IpcFiles,
) -> Result<String, PublisherError> {
    let address = bind_address(endpoint)?;
    if let Some(path) = address.strip_prefix("ipc://") {
        remove_stale_socket(Path::new(path));
    }
    match socket.bind(&address).await {
        Ok(bound) => {
            let bound = bound.to_string();
            if let Some(path) = bound.strip_prefix("ipc://") {
                ipc_files.0.push(path.to_string());
            }
            Ok(bound)
        }
        Err(error) => Err(PublisherError::Bind {
            endpoint: endpoint.to_string(),
            reason: error.to_string(),
        }),
    }
}

/// zeromq takes the host literally, so spell out the wildcards libzmq accepts.
fn bind_address(endpoint: &str) -> Result<String, PublisherError> {
    let unsupported = || PublisherError::UnsupportedEndpoint(endpoint.to_string());
    if let Some(address) = endpoint.strip_prefix("tcp://") {
        let (host, port) = address.rsplit_once(':').ok_or_else(unsupported)?;
        let host = match host {
            "*" => "0.0.0.0",
            "::" => "[::]",
            host => host,
        };
        return Ok(format!("tcp://{host}:{port}"));
    }
    match endpoint.strip_prefix("ipc://") {
        // An abstract-namespace name has no filesystem path to bind.
        Some(path) if !path.is_empty() && !path.starts_with('@') => Ok(endpoint.to_string()),
        _ => Err(unsupported()),
    }
}

/// A socket file left by a previous process would fail the bind, as it does
/// not for libzmq, which unlinks it first.
fn remove_stale_socket(path: &Path) {
    use std::os::unix::fs::FileTypeExt;
    if std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_socket()) {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_queue_limit_means_unlimited() {
        // queue.Queue(maxsize=0) and ZMQ_SNDHWM=0 are unbounded; a one-slot
        // queue here would block or drop almost every batch of such a config.
        assert_eq!(queue_capacity(0), Semaphore::MAX_PERMITS);
        assert_eq!(queue_capacity(7), 7);
    }

    #[test]
    fn ring_replays_from_the_requested_sequence_and_evicts_the_oldest() {
        let mut ring = Ring {
            entries: VecDeque::new(),
            capacity: 3,
        };
        for seq in 0..5 {
            ring.push(seq, Bytes::new());
        }
        let seqs = |start| {
            ring.since(start)
                .into_iter()
                .map(|(seq, _)| seq)
                .collect::<Vec<_>>()
        };
        assert_eq!(seqs(0), [2, 3, 4]);
        assert_eq!(seqs(3), [3, 4]);
        assert_eq!(seqs(5), [0u64; 0]);
    }

    #[test]
    fn replay_start_is_any_length_big_endian() {
        let request = |start: &[u8]| {
            let message = frames([
                Bytes::from_static(b"id"),
                Bytes::new(),
                Bytes::copy_from_slice(start),
            ]);
            parse_request(message).map(|(_, start)| start)
        };
        assert_eq!(request(&7u64.to_be_bytes()), Some(7));
        assert_eq!(request(&[1, 0]), Some(256));
        assert_eq!(request(&[1; 9]), Some(u64::MAX));
    }

    #[test]
    fn wildcard_hosts_are_spelled_out() {
        assert_eq!(bind_address("tcp://*:5557").unwrap(), "tcp://0.0.0.0:5557");
        assert_eq!(bind_address("tcp://:::5557").unwrap(), "tcp://[::]:5557");
        assert_eq!(
            bind_address("tcp://127.0.0.1:5557").unwrap(),
            "tcp://127.0.0.1:5557"
        );
        assert!(bind_address("inproc://kv").is_err());
        assert!(bind_address("ipc://@abstract").is_err());
    }
}
