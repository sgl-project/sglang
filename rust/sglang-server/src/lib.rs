//! sglang-server: a multi-threaded Rust frontend (API server → TokenizerManager
//! → Tokenizer/Detokenizer) embedded in the Python scheduler process.
//!
//! Pipeline stages 1–5 are pure Rust and never touch a `PyObject`, so they run
//! concurrently with the Python scheduler without contending for the GIL. The
//! only GIL crossings are the two boundary methods on [`Server`]:
//!   * `recv_requests` — Python scheduler thread drains the ingress ring.
//!   * `push_chunk`    — Python scheduler thread pushes one output chunk.
//!
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
mod transport;

use std::net::SocketAddr;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::runtime::{Runtime, RuntimeConfig};

/// Columnar ingress batch handed to Python by [`Server::recv_requests`]:
/// `(headers, ids_buf, lengths)` — per-request scalar msgpack headers, all
/// requests' raw int64 ids concatenated, and per-request token counts.
type IngressBatch<'py> = (Vec<Bound<'py, PyBytes>>, Bound<'py, PyBytes>, Vec<u32>);

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
        headless_server_bind = None,
        ingress_ring_cap = 8192,
        egress_ring_cap = 8192,
        channel_cap = 8192,
        pin_cores = true,
        cores = None,

        server_args_json = "{}",
    ))]
    // pyo3 `#[new]` constructor: the wide arg list is the Python-facing boot
    // surface (all optional overrides), not a call-site ergonomics problem.
    #[allow(clippy::too_many_arguments)]
    fn start(
        bind: Option<String>,
        headless_server_bind: Option<String>,
        ingress_ring_cap: usize,
        egress_ring_cap: usize,
        channel_cap: usize,
        pin_cores: bool,
        cores: Option<Vec<usize>>,
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
            .unwrap_or_else(|| server_args.bind())
            .parse()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("bad bind: {e}"))
            })?;

        let tokenizer_worker_num = server_args.tokenizer_worker_num();
        let detokenizer_worker_num = server_args.detokenizer_worker_num();
        let api_worker_num = server_args.api_worker_num();

        let tokenizer_path = server_args.tokenizer_path();
        let revision = server_args.revision();
        // Headless TCP transport (`dp_size > 1`): when set, replaces the embedded
        // HTTP api-server with a TCP listener a standalone api-server drives.
        let headless: Option<SocketAddr> = match headless_server_bind {
            Some(addr) => Some(addr.parse().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "bad headless_server_bind: {e}"
                ))
            })?),
            None => None,
        };

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
            headless,
            server_args,
        };
        let rt = runtime::start(cfg).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("runtime start failed: {e}"))
        })?;
        Ok(Server { rt })
    }

    /// Non-blocking drain of the ingress ring, returned **columnar** so the large
    /// `input_ids` tensor never goes through msgpack. Yields a 3-tuple:
    ///   * `headers`: `list[bytes]` — one msgpack scalar header per request
    ///     (`input_ids` omitted), decoded individually by the scheduler;
    ///   * `ids_buf`: `bytes` — all requests' raw little-endian int64 ids,
    ///     concatenated; sliced per request and wrapped as `array("q")`;
    ///   * `lengths`: `list[int]` — per-request token count (0 for control reqs),
    ///     so the scheduler can slice `ids_buf`.
    /// The GIL is released for the drain + columnar split; only the `PyBytes`
    /// marshaling needs it. The `ids` cells are copied **directly into the result
    /// `bytes`** (one copy, no intermediate buffer).
    #[pyo3(signature = (max = 256))]
    fn recv_requests<'py>(&self, py: Python<'py>, max: usize) -> PyResult<IngressBatch<'py>> {
        let cols = py.detach(|| self.rt.ingress.drain(max));
        let headers = cols.headers.iter().map(|h| PyBytes::new(py, h)).collect();
        // Single pass: copy each raw ids cell straight into the output `bytes`.
        let ids_buf = PyBytes::new_with(py, cols.ids_total, |buf| {
            let mut pos = 0;
            for cell in &cols.ids {
                let end = pos + cell.len();
                buf[pos..end].copy_from_slice(cell);
                pos = end;
            }
            Ok(())
        })?;
        Ok((headers, ids_buf, cols.lengths))
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

/// Run the standalone api-server process (`dp_size > 1`): serve the OpenAI /
/// `/generate` HTTP API on `bind` and the internal `/internal/register` endpoint.
/// The TCP pool to the DP ranks is **deferred** — each headless rank reports its
/// endpoint via registration, and the pool connects once all `dp_size` have
/// (handlers answer 503 until then). Blocks for the process lifetime; the GIL is
/// released while the Rust HTTP server runs, so the host process can still handle
/// signals.
#[pyfunction]
#[pyo3(signature = (
    bind,
    egress_buf = 8192,
    server_args_json = "{}",
))]
fn run_api_server(
    py: Python<'_>,
    bind: String,
    egress_buf: usize,
    server_args_json: &str,
) -> PyResult<()> {
    let to_val = |e: String| PyErr::new::<pyo3::exceptions::PyValueError, _>(e);

    let server_args = runtime::ServerArgs::from_json(server_args_json)
        .map_err(|e| to_val(format!("bad server_args_json: {e}")))?;
    server_args
        .validate_mandatory()
        .map_err(|e| to_val(format!("server_args: {e}")))?;
    let bind: SocketAddr = bind.parse().map_err(|e| to_val(format!("bad bind: {e}")))?;
    let standalone_api_worker_num = server_args.standalone_api_server_num();
    // Readiness state: ranks register their endpoints, then the pool connects.
    let ready = transport::NetReady::new(
        server_args.dp_size(),
        server_args.ingress_pool_size(),
        server_args.egress_pool_size(),
    );
    let server_args = std::sync::Arc::new(server_args);
    let id_gen = std::sync::Arc::new(crate::ids::RequestIdGen::default());

    // Release the GIL: pure-Rust HTTP server that runs until the process is
    // signalled. The pool connect is driven by registrations inside `block_on`.
    py.detach(move || -> Result<(), String> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(standalone_api_worker_num)
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        rt.block_on(async move {
            api_server::serve(
                bind,
                api_server::Transport::Net(ready),
                id_gen,
                egress_buf,
                server_args,
            )
            .await;
            Ok(())
        })
    })
    .map_err(to_val)
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
    m.add_function(wrap_pyfunction!(run_api_server, m)?)?;
    Ok(())
}
