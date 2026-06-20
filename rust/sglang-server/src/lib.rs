//! sglang-server: a multi-threaded Rust frontend (API server → TokenizerManager
//! → Tokenizer/Detokenizer) embedded in the Python scheduler process.
//!
//! Pipeline stages 1–5 are pure Rust and never touch a `PyObject`, so they run
//! concurrently with the Python scheduler without contending for the GIL. The
//! only GIL crossings are the two boundary methods on [`Server`]:
//!   * `recv_requests` — Python scheduler thread drains the ingress ring.
//!   * `push_chunk`    — Python scheduler thread pushes one output chunk.
//! Both are non-blocking, so the GIL is never held across a wait.

mod api_server;
mod detokenizer;
mod error;
mod fsm;
mod ids;
mod message;
mod runtime;
mod tokenizer;
mod tokenizer_manager;

use std::net::SocketAddr;

use bytes::Bytes;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::runtime::{Runtime, RuntimeConfig};

/// Handle owned by the Python scheduler process. Construct once via
/// [`Server::start`], then poll it from the scheduler event loop.
#[pyclass]
struct Server {
    rt: Runtime,
}

#[pymethods]
impl Server {
    /// Boot the frontend (spawns all threads) and return immediately.
    #[new]
    #[pyo3(signature = (
        bind = "127.0.0.1:30000",
        api_threads = 2,
        tokenizer_threads = 2,
        detok_shards = 2,
        ingress_ring_cap = 8192,
        egress_ring_cap = 8192,
        channel_cap = 8192,
        pin_cores = true,
        cores = None,
        tokenizer_path = None,
        tokenizer_revision = None,
    ))]
    fn start(
        bind: &str,
        api_threads: usize,
        tokenizer_threads: usize,
        detok_shards: usize,
        ingress_ring_cap: usize,
        egress_ring_cap: usize,
        channel_cap: usize,
        pin_cores: bool,
        cores: Option<Vec<usize>>,
        tokenizer_path: Option<String>,
        tokenizer_revision: Option<String>,
    ) -> PyResult<Self> {
        let bind: SocketAddr = bind
            .parse()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("bad bind: {e}")))?;
        let cfg = RuntimeConfig {
            bind,
            api_threads,
            tokenizer_threads,
            detok_shards,
            ingress_ring_cap,
            egress_ring_cap,
            channel_cap,
            pin_cores,
            cores,
            tokenizer_path,
            tokenizer_revision,
        };
        Ok(Server {
            rt: runtime::start(cfg),
        })
    }

    /// Non-blocking drain of the ingress ring. Returns up to `max` msgpack
    /// payloads (`bytes`) for the scheduler's `recv_requests` to decode into
    /// `TokenizedGenerateReqInput`. Releases the GIL while popping from the
    /// Rust ring so producer threads never contend with us.
    #[pyo3(signature = (max = 256))]
    fn recv_requests<'py>(&self, py: Python<'py>, max: usize) -> Vec<Bound<'py, PyBytes>> {
        let msgs = py.detach(|| self.rt.ingress.drain(max));
        msgs.into_iter()
            .map(|b| PyBytes::new(py, b.as_ref()))
            .collect()
    }

    /// Push one scheduler-output chunk (already msgpack-encoded `ChunkEvent`)
    /// into the egress ring. Returns `False` on backpressure (ring full).
    fn push_chunk(&self, py: Python<'_>, chunk: &[u8]) -> bool {
        let bytes = Bytes::copy_from_slice(chunk);
        py.detach(|| self.rt.egress.try_push(bytes))
    }

    /// Signal all threads to stop (best effort).
    fn shutdown(&self) {
        self.rt.request_shutdown();
    }
}

/// The Python module: `import sglang_server`.
#[pymodule]
fn sglang_server(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize tracing once; ignore if already set by the host process.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    m.add_class::<Server>()?;
    Ok(())
}
