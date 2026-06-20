//! The two Rust↔Python boundary queues.
//!
//! In embedded mode the Rust frontend threads and the Python scheduler loop
//! share one process, so these are in-process `flume` channels — literal
//! `mpsc`/`mpmc`, no shared memory, no serialization beyond the msgpack bytes
//! the payload already is.
//!
//! GIL note: the Python side only ever calls the *non-blocking* `drain` /
//! `try_push` methods while holding the GIL, and the Rust worker threads only
//! ever push/drain raw `Bytes` — neither side touches a `PyObject` off-thread,
//! so the producer threads never need the GIL.

use bytes::Bytes;

/// Ingress: TokenizerManager → scheduler `recv_requests`.
/// Producers are Rust TM workers; the single consumer is the Python thread.
#[derive(Clone)]
pub struct IngressProducer {
    tx: flume::Sender<Bytes>,
}

pub struct IngressConsumer {
    rx: flume::Receiver<Bytes>,
}

impl IngressProducer {
    /// Non-blocking push. Returns `false` on a full ring (backpressure) so the
    /// caller can fail the request rather than block a worker thread.
    #[inline]
    pub fn try_push(&self, msg: Bytes) -> bool {
        self.tx.try_send(msg).is_ok()
    }
}

impl IngressConsumer {
    /// Drain up to `max` messages, returning immediately when the ring runs
    /// dry — mirrors the scheduler's existing `zmq.NOBLOCK` loop in
    /// `request_receiver._pull_raw_reqs`.
    ///
    /// Non-blocking by construction: try_recv` returns `Err(TryRecvError::Empty)`
    /// instantly when the ring is empty, and `Err(_) => break` exits the loop
    /// right away.
    pub fn drain(&self, max: usize) -> Vec<Bytes> {
        let mut out = Vec::new();
        while out.len() < max {
            match self.rx.try_recv() {
                Ok(b) => out.push(b),
                Err(_) => break, // Empty or Disconnected -> stop now
            }
        }
        out
    }
}

/// Egress: scheduler output (`push_chunk`) → Rust egress dispatcher.
/// The single producer is the Python thread; the consumer is the dispatcher.
#[derive(Clone)]
pub struct EgressProducer {
    tx: flume::Sender<Bytes>,
}

pub struct EgressConsumer {
    rx: flume::Receiver<Bytes>,
}

impl EgressProducer {
    /// Non-blocking push from the Python scheduler. `false` = ring full.
    #[inline]
    pub fn try_push(&self, msg: Bytes) -> bool {
        self.tx.try_send(msg).is_ok()
    }
}

impl EgressConsumer {
    /// Blocking receive used by the dedicated dispatcher thread.
    pub fn recv(&self) -> Option<Bytes> {
        self.rx.recv().ok()
    }
}

/// Build both halves of a bounded ring.
pub fn ingress_ring(cap: usize) -> (IngressProducer, IngressConsumer) {
    let (tx, rx) = flume::bounded(cap);
    (IngressProducer { tx }, IngressConsumer { rx })
}

pub fn egress_ring(cap: usize) -> (EgressProducer, EgressConsumer) {
    let (tx, rx) = flume::bounded(cap);
    (EgressProducer { tx }, EgressConsumer { rx })
}
