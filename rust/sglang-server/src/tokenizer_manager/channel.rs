//! The two Rust↔Python boundary queues.
//!
//! In embedded mode the Rust frontend threads and the Python scheduler loop
//! share one process, so these are in-process `flume` channels — literal
//! `mpsc`/`mpmc`, no shared memory, no serialization beyond the msgpack bytes
//! the payload already is.

use std::sync::Mutex;
use std::time::Duration;

use bytes::Bytes;

use crate::message::request::SchedulerRequest;

/// ToSchedulerTx: TokenizerManager → scheduler `recv_requests`.
/// Producers are Rust TM workers; the single consumer is the Python thread.
/// Carries [`SchedulerRequest`] (columnar: scalar header + raw int64 ids cell), not a
/// single msgpack blob, so the large `input_ids` tensor bypasses msgpack.
#[derive(Clone)]
pub struct ToSchedulerTx {
    tx: flume::Sender<SchedulerRequest>,
}

pub struct ToSchedulerRx {
    rx: flume::Receiver<SchedulerRequest>,
    /// One-slot buffer holding a message consumed by a blocking [`wait`] so the
    /// scheduler can park on idle without losing it — the next [`drain`] returns
    /// it first. Only ever touched by the single consumer (the Python thread),
    /// so contention is nil; the `Mutex` is just for interior mutability across
    /// the `&self` methods.
    ///
    /// [`wait`]: ToSchedulerRx::wait
    /// [`drain`]: ToSchedulerRx::drain
    stash: Mutex<Option<SchedulerRequest>>,
}

/// A drained request batch in **columnar** (struct-of-arrays) form. The `ids`
/// cells are kept *un-concatenated* so the pyo3 boundary can copy them straight
/// into one `PyBytes` (no intermediate buffer); `ids_total` is their summed
/// length, precomputed for that single allocation.
#[derive(Default)]
pub struct RequestColumns {
    /// Per-request scalar msgpack header (`input_ids` omitted).
    pub headers: Vec<Bytes>,
    /// Per-request raw little-endian int64 ids cell (empty for control reqs).
    pub ids: Vec<Bytes>,
    /// Per-request token count (`ids` cell length / 8).
    pub lengths: Vec<u32>,
    /// Sum of all `ids` cell byte lengths.
    pub ids_total: usize,
}

impl RequestColumns {
    /// Concatenate the `ids` cells into `buf`, which must be exactly
    /// `ids_total` bytes — the pyo3 boundary hands in the freshly allocated
    /// `PyBytes` so the ids are copied once, straight to their destination.
    pub fn copy_ids_into(&self, mut buf: &mut [u8]) {
        debug_assert_eq!(buf.len(), self.ids_total);
        for cell in &self.ids {
            let (dst, rest) = buf.split_at_mut(cell.len());
            dst.copy_from_slice(cell);
            buf = rest;
        }
    }
}

impl ToSchedulerTx {
    /// Non-blocking push. Returns `false` on a full ring (backpressure) so the
    /// caller can fail the request rather than block a worker thread.
    #[inline]
    pub fn try_push(&self, msg: SchedulerRequest) -> bool {
        self.tx.try_send(msg).is_ok()
    }
}

impl ToSchedulerRx {
    /// Drain up to `max` messages into a columnar [`RequestColumns`], returning
    /// immediately when the ring runs dry — mirrors the scheduler's existing
    /// `zmq.NOBLOCK` loop in `request_receiver._pull_raw_reqs`.
    ///
    /// Non-blocking by construction: `try_recv` returns `Err(TryRecvError::Empty)`
    /// instantly when the ring is empty, and `Err(_) => break` exits the loop
    /// right away.
    pub fn drain(&self, max: usize) -> RequestColumns {
        let mut batch = RequestColumns::default();
        // A message parked by a prior blocking `wait` is delivered first.
        if let Some(m) = self.stash.lock().unwrap().take() {
            push_msg(&mut batch, m);
        }
        while batch.headers.len() < max {
            match self.rx.try_recv() {
                Ok(m) => push_msg(&mut batch, m),
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

/// Append one drained message's columnar cells to the batch.
#[inline]
fn push_msg(batch: &mut RequestColumns, m: SchedulerRequest) {
    batch.ids_total += m.ids.len();
    batch.lengths.push((m.ids.len() / 8) as u32); // int64 cell → tokens
    batch.headers.push(m.header);
    batch.ids.push(m.ids);
}

/// Scheduler output (`Server.push_decode_result_batch` / `push_control_result`
/// / `push_error`) → Rust response dispatcher. The single producer is the
/// Python thread; the consumer is the dispatcher.
#[derive(Clone)]
pub struct FromSchedulerTx {
    tx: flume::Sender<Bytes>,
}

pub struct FromSchedulerRx {
    rx: flume::Receiver<Bytes>,
}

impl FromSchedulerTx {
    /// Blocking push.
    pub fn push(&self, msg: Bytes) -> bool {
        self.tx.send(msg).is_ok()
    }

    /// Non-blocking push.
    #[inline]
    pub fn try_push(&self, msg: Bytes) -> Result<(), Option<Bytes>> {
        match self.tx.try_send(msg) {
            Ok(()) => Ok(()),
            Err(flume::TrySendError::Full(msg)) => Err(Some(msg)),
            Err(flume::TrySendError::Disconnected(_)) => Err(None),
        }
    }
}

impl FromSchedulerRx {
    /// The underlying receiver, so the dispatcher can drain it via
    /// [`wiring::recv`](crate::tokenizer_manager::wiring::recv) (data + shutdown select).
    pub fn receiver(&self) -> &flume::Receiver<Bytes> {
        &self.rx
    }
}

/// Build both halves of a bounded ring.
pub fn to_scheduler(cap: usize) -> (ToSchedulerTx, ToSchedulerRx) {
    let (tx, rx) = flume::bounded(cap);
    (
        ToSchedulerTx { tx },
        ToSchedulerRx {
            rx,
            stash: Mutex::new(None),
        },
    )
}

pub fn from_scheduler(cap: usize) -> (FromSchedulerTx, FromSchedulerRx) {
    let (tx, rx) = flume::bounded(cap);
    (FromSchedulerTx { tx }, FromSchedulerRx { rx })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(h: &'static [u8]) -> SchedulerRequest {
        SchedulerRequest {
            header: Bytes::from_static(h),
            ids: Bytes::new(),
        }
    }

    /// `wait` parks when empty (times out), stashes a pushed message
    /// non-destructively, and the next `drain` returns it.
    #[test]
    fn wait_stashes_then_drain_returns_it() {
        let (tx, rx) = to_scheduler(8);
        // Empty ring → times out, nothing stashed.
        assert!(!rx.wait(Duration::from_millis(1)));
        // Push one, then wait stashes it (returns true).
        assert!(tx.try_push(msg(b"a")));
        assert!(rx.wait(Duration::from_millis(200)));
        // Idempotent: already stashed, returns true without touching the ring.
        assert!(rx.wait(Duration::from_millis(1)));
        // Drain yields the stashed message, then the ring is empty.
        assert_eq!(rx.drain(16).headers.len(), 1);
        assert!(rx.drain(16).headers.is_empty());
    }

    /// A blocked `wait` is woken the instant a producer pushes (no polling).
    #[test]
    fn wait_wakes_on_push() {
        let (tx, rx) = to_scheduler(8);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            let _ = tx.try_push(msg(b"a"));
        });
        // Generous timeout, but it should return well before it as soon as the
        // push lands.
        assert!(rx.wait(Duration::from_secs(5)));
        assert_eq!(rx.drain(16).headers.len(), 1);
    }

    /// A full from_scheduler channel parks the producer until the consumer drains — the
    /// committed frame is delivered in order, never dropped.
    #[test]
    fn response_push_blocks_until_drained() {
        let (tx, rx) = from_scheduler(1);
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
    fn response_push_returns_false_when_closed() {
        let (tx, rx) = from_scheduler(1);
        drop(rx);
        assert!(!tx.push(Bytes::from_static(b"x")));
    }
}
