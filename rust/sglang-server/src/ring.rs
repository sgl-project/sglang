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

use std::sync::Mutex;
use std::time::Duration;

use bytes::Bytes;

/// Ingress: TokenizerManager → scheduler `recv_requests`.
/// Producers are Rust TM workers; the single consumer is the Python thread.
/// Carries each request's msgpack header only — `input_ids` cross via the
/// [`InputIdsStore`](crate::input_ids_store::InputIdsStore), not the ring.
#[derive(Clone)]
pub struct IngressProducer {
    tx: flume::Sender<Bytes>,
}

pub struct IngressConsumer {
    rx: flume::Receiver<Bytes>,
    /// One-slot buffer holding a message consumed by a blocking [`wait`] so the
    /// scheduler can park on idle without losing it — the next [`drain`] returns
    /// it first. Only ever touched by the single consumer (the Python thread),
    /// so contention is nil; the `Mutex` is just for interior mutability across
    /// the `&self` methods.
    ///
    /// [`wait`]: IngressConsumer::wait
    /// [`drain`]: IngressConsumer::drain
    stash: Mutex<Option<Bytes>>,
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
    /// Drain up to `max` headers, returning immediately when the ring runs dry
    /// — mirrors the scheduler's existing `zmq.NOBLOCK` loop in
    /// `request_receiver._pull_raw_reqs`.
    pub fn drain(&self, max: usize) -> Vec<Bytes> {
        let mut batch = Vec::new();
        // A message parked by a prior blocking `wait` is delivered first.
        if let Some(m) = self.stash.lock().unwrap().take() {
            batch.push(m);
        }
        while batch.len() < max {
            match self.rx.try_recv() {
                Ok(m) => batch.push(m),
                Err(_) => break, // Empty or Disconnected -> stop now
            }
        }
        batch
    }

    /// Park up to `timeout` for at least one incoming message, so the idle
    /// scheduler loop sleeps instead of spinning at 100% CPU. The message is
    /// **stashed, not returned** — the next [`drain`](Self::drain) yields it —
    /// so this composes with the existing non-blocking drain flow. Returns
    /// whether a message is now available. `flume` wakes the parked thread the
    /// instant a producer pushes, so this adds no latency to real requests.
    pub fn wait(&self, timeout: Duration) -> bool {
        if self.stash.lock().unwrap().is_some() {
            return true;
        }
        match self.rx.recv_timeout(timeout) {
            Ok(m) => {
                *self.stash.lock().unwrap() = Some(m);
                true
            }
            Err(_) => false, // Timeout or Disconnected
        }
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
    /// Blocking push: parks until the ring has space, so a full ring applies
    /// backpressure to the scheduler instead of dropping output the scheduler has
    /// already committed (advanced `send_token_offset` for). The GIL is released
    /// around the call, so parking here doesn't stall other Python threads.
    /// `false` only when the consumer is gone (runtime shutdown), where the frame
    /// is unavoidably lost.
    pub fn push(&self, msg: Bytes) -> bool {
        self.tx.send(msg).is_ok()
    }

    /// Non-blocking push, so the pyo3 boundary can try to hand the frame over
    /// while still holding the GIL and detach only when it would actually park.
    /// Releasing the GIL is not free: reacquiring it waits out the interpreter's
    /// switch interval (5 ms by default), which dwarfs the sub-microsecond push
    /// it was protecting.
    ///
    /// Hands the frame BACK on a full ring (`Err(Some(msg))`) so the caller can
    /// retry it under [`push`](Self::push) without rebuilding it. `Err(None)` is
    /// the consumer being gone (shutdown), where the frame is unavoidably lost.
    #[inline]
    pub fn try_push(&self, msg: Bytes) -> Result<(), Option<Bytes>> {
        match self.tx.try_send(msg) {
            Ok(()) => Ok(()),
            Err(flume::TrySendError::Full(msg)) => Err(Some(msg)),
            Err(flume::TrySendError::Disconnected(_)) => Err(None),
        }
    }
}

impl EgressConsumer {
    /// The underlying receiver, so the dispatcher can drain it via
    /// [`tokenizer_manager::recv`](crate::tokenizer_manager::recv) (data + shutdown select).
    pub fn receiver(&self) -> &flume::Receiver<Bytes> {
        &self.rx
    }
}

/// Build both halves of a bounded ring.
pub fn ingress_ring(cap: usize) -> (IngressProducer, IngressConsumer) {
    let (tx, rx) = flume::bounded(cap);
    (
        IngressProducer { tx },
        IngressConsumer {
            rx,
            stash: Mutex::new(None),
        },
    )
}

pub fn egress_ring(cap: usize) -> (EgressProducer, EgressConsumer) {
    let (tx, rx) = flume::bounded(cap);
    (EgressProducer { tx }, EgressConsumer { rx })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `wait` parks when empty (times out), stashes a pushed message
    /// non-destructively, and the next `drain` returns it.
    #[test]
    fn wait_stashes_then_drain_returns_it() {
        let (tx, rx) = ingress_ring(8);
        // Empty ring → times out, nothing stashed.
        assert!(!rx.wait(Duration::from_millis(1)));
        // Push one, then wait stashes it (returns true).
        assert!(tx.try_push(Bytes::from_static(b"a")));
        assert!(rx.wait(Duration::from_millis(200)));
        // Idempotent: already stashed, returns true without touching the ring.
        assert!(rx.wait(Duration::from_millis(1)));
        // Drain yields the stashed message, then the ring is empty.
        assert_eq!(rx.drain(16).len(), 1);
        assert!(rx.drain(16).is_empty());
    }

    /// A blocked `wait` is woken the instant a producer pushes (no polling).
    #[test]
    fn wait_wakes_on_push() {
        let (tx, rx) = ingress_ring(8);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            let _ = tx.try_push(Bytes::from_static(b"a"));
        });
        // Generous timeout, but it should return well before it as soon as the
        // push lands.
        assert!(rx.wait(Duration::from_secs(5)));
        assert_eq!(rx.drain(16).len(), 1);
    }

    /// A full egress ring parks the producer until the consumer drains — the
    /// committed frame is delivered in order, never dropped.
    #[test]
    fn egress_push_blocks_until_drained() {
        let (tx, rx) = egress_ring(1);
        assert!(tx.push(Bytes::from_static(b"a"))); // fits; ring now full
        let t = std::thread::spawn(move || tx.push(Bytes::from_static(b"b")));
        // The parked push can't have completed while the ring is full.
        std::thread::sleep(Duration::from_millis(20));
        // Drain one → frees a slot → the parked push lands.
        assert_eq!(rx.receiver().recv().unwrap(), Bytes::from_static(b"a"));
        assert!(t.join().unwrap(), "push should succeed once space frees");
        assert_eq!(rx.receiver().recv().unwrap(), Bytes::from_static(b"b"));
    }

    /// A closed ring (consumer gone → shutdown) returns `false` instead of
    /// parking forever, so a scheduler blocked in `push` unblocks on teardown.
    #[test]
    fn egress_push_returns_false_when_closed() {
        let (tx, rx) = egress_ring(1);
        drop(rx);
        assert!(!tx.push(Bytes::from_static(b"x")));
    }
}
