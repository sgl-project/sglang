//! Multimodal worker pool (step 1 of the all-Rust MM roadmap).
//!
//! Rust-owned threads drain requests parked in the `Encoding` stage and drive
//! the *Python* `mm_processor` path (`rust_server.MmProcessorHost.handle_sync`:
//! media download/load, resize, HF AutoProcessor) through a handler callable
//! registered via `Server.start_mm_workers`. This is the one deliberate
//! exception to the "pipeline stages never touch a `PyObject`" rule: a worker
//! holds the GIL only for the handler call (which itself releases it while
//! waiting on the processor's event loop) and parks GIL-free on the channel.
//! Later roadmap steps replace the Python call with native Rust processing.

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::message::MmRequest;
use crate::runtime::channels::TmEvent;

/// Spawn `workers` `mm-worker-{i}` threads (unpinned — the heavy work happens
/// Python-side) and return their join handles for the runtime's shutdown join.
pub fn spawn_workers(
    py: Python<'_>,
    rx: flume::Receiver<MmRequest>,
    tm: flume::Sender<TmEvent>,
    handler: Py<PyAny>,
    workers: usize,
) -> Vec<std::thread::JoinHandle<()>> {
    (0..workers.max(1))
        .map(|i| {
            let worker = MmWorker {
                rx: rx.clone(),
                tm: tm.clone(),
                handler: handler.clone_ref(py),
            };
            std::thread::Builder::new()
                .name(format!("mm-worker-{i}"))
                .spawn(move || worker.run())
                .expect("spawn mm worker")
        })
        .collect()
}

struct MmWorker {
    rx: flume::Receiver<MmRequest>,
    tm: flume::Sender<TmEvent>,
    handler: Py<PyAny>,
}

impl MmWorker {
    /// Drain until the mm channel closes (tm-ingress drops its sender on
    /// shutdown). Pool size bounds MM concurrency: each worker processes one
    /// request at a time.
    fn run(self) {
        while let Ok(req) = self.rx.recv() {
            let rid = req.rid.clone();
            let event = match self.process(req) {
                Ok(input_ids) => TmEvent::MmEncoded { rid, input_ids },
                Err(message) => {
                    tracing::warn!(%rid, %message, "mm processing failed");
                    TmEvent::MmFailed { rid, message }
                }
            };
            if self.tm.send(event).is_err() {
                return; // tm-ingress gone: shutdown
            }
        }
    }

    /// One handler call: `handler(rid, payload) -> bytes` — the final
    /// placeholder-expanded prompt ids as raw little-endian int64 bytes
    /// (`array("q").tobytes()`). The handler stores the processed `mm_inputs`
    /// in its rid-keyed table *before* returning, so `MmEncoded` (and the
    /// scheduler drain that follows) always finds the entry. An exception
    /// rejects the request back to the client as a 400.
    fn process(&self, req: MmRequest) -> Result<Vec<i32>, String> {
        Python::attach(|py| {
            let out = self
                .handler
                .call1(py, (req.rid.as_str(), PyBytes::new(py, &req.payload)))
                .map_err(|e| e.to_string())?;
            let ids = out
                .extract::<PyBackedBytes>(py)
                .map_err(|e| format!("mm handler must return bytes: {e}"))?;
            decode_ids(&ids)
        })
    }
}

/// Raw little-endian int64 bytes → `Vec<i32>`.
fn decode_ids(bytes: &[u8]) -> Result<Vec<i32>, String> {
    if !bytes.len().is_multiple_of(8) {
        return Err("mm handler returned malformed input_ids bytes".into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as i32)
        .collect())
}
