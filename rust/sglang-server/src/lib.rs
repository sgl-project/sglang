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

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::cmp::max;

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
        bind = None,
        api_worker_num = None,
        tokenizer_worker_num = None,
        detokenizer_worker_num = None,
        ingress_ring_cap = 8192,
        egress_ring_cap = 8192,
        channel_cap = 8192,
        pin_cores = true,
        cores = None,
        tokenizer_path = None,
        revision = None,
        server_args_json = "{}",
    ))]
    fn start(
        bind: Option<String>,
        api_worker_num: Option<usize>,
        tokenizer_worker_num: Option<usize>,
        detokenizer_worker_num: Option<usize>,
        ingress_ring_cap: usize,
        egress_ring_cap: usize,
        channel_cap: usize,
        pin_cores: bool,
        cores: Option<Vec<usize>>,
        tokenizer_path: Option<String>,
        revision: Option<String>,
        server_args_json: &str,
    ) -> PyResult<Self> {
        // Static server metadata (server_args + model_config) dumped by the
        // scheduler; parse and validate mandatory fields now so a bad/missing
        // field is a boot error, not a request-time 500.
        let server_args: runtime::ServerArgs = runtime::ServerArgs::from_json(server_args_json)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "bad server_args_json: {e}"
                ))
            })?;
        server_args.validate_mandatory().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("server_args: {e}"))
        })?;
        // The bind address, tokenizer source/threads/shards all live in the
        // `server_args` blob; resolve them from there so the scheduler doesn't
        // re-pass them. The explicit params stay as optional overrides for
        // standalone callers (tests) that construct a `Server` without a full
        // `server_args`.
        let bind: SocketAddr = bind
            .or_else(|| server_args.bind())
            .unwrap_or_else(|| "127.0.0.1:30000".to_string())
            .parse()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("bad bind: {e}"))
            })?;

        let tokenizer_worker_num =
            tokenizer_worker_num.unwrap_or_else(|| server_args.tokenizer_worker_num());
        let detokenizer_worker_num =
            detokenizer_worker_num.unwrap_or_else(|| server_args.detokenizer_worker_num());
        let api_worker_num = api_worker_num.unwrap_or_else(|| {
            max(
                4,
                max(
                    (tokenizer_worker_num / 2) as usize,
                    (detokenizer_worker_num / 2) as usize,
                ),
            )
        });

        let tokenizer_path = tokenizer_path.or_else(|| server_args.tokenizer_path());
        let revision = revision.or_else(|| server_args.revision());
        let server_args = std::sync::Arc::new(server_args);

        let cfg = RuntimeConfig {
            bind,
            api_worker_num,
            tokenizer_worker_num,
            detokenizer_worker_num,
            ingress_ring_cap,
            egress_ring_cap,
            channel_cap,
            pin_cores,
            cores,
            tokenizer_path,
            revision,
            server_args,
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
    /// into the egress ring → detok shard. Returns `False` on backpressure.
    fn push_chunk(&self, py: Python<'_>, chunk: &[u8]) -> bool {
        let bytes = crate::message::frame_egress_chunk(chunk);
        py.detach(|| self.rt.egress.try_push(bytes))
    }

    /// Push a control-request result (e.g. the `/server_info` JSON) into the
    /// egress ring, routed by `rid` to the waiting request's sink as a single
    /// non-streamed response. Returns `False` on backpressure.
    fn push_result(&self, py: Python<'_>, rid: &str, payload: &[u8]) -> bool {
        let bytes = crate::message::frame_egress_result(rid, payload);
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
