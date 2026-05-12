use pyo3::prelude::*;

pub mod detokenizer;
pub mod engine;
pub mod http_server;
pub mod ipc;
pub mod tokenizer_manager;

// ──────────────────────── DetokenizerConfig (existing) ──────────────────────

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct DetokenizerConfig {
    /// ZMQ endpoint to PULL batches from the scheduler (bind).
    #[pyo3(get, set)]
    pub detokenizer_ipc_name: String,

    /// ZMQ endpoint to PUSH results to the tokenizer worker (connect).
    #[pyo3(get, set)]
    pub tokenizer_ipc_name: String,

    /// HuggingFace model id or local path that contains tokenizer.json.
    #[pyo3(get, set)]
    pub tokenizer_path: String,

    /// Do not load a tokenizer; output empty strings (used with skip_tokenizer_init).
    #[pyo3(get, set)]
    pub skip_tokenizer_init: bool,

    /// Maximum number of in-flight decode states before oldest are evicted.
    #[pyo3(get, set)]
    pub max_states: usize,

    /// Decode each sequence individually instead of using batch_decode.
    #[pyo3(get, set)]
    pub disable_tokenizer_batch_decode: bool,

    /// Tool-call parser mode. Set to "gpt-oss" to enable special stop-token handling.
    #[pyo3(get, set)]
    pub tool_call_parser: String,
}

#[pymethods]
impl DetokenizerConfig {
    #[new]
    fn new(
        detokenizer_ipc_name: String,
        tokenizer_ipc_name: String,
        tokenizer_path: String,
        skip_tokenizer_init: bool,
        max_states: usize,
        disable_tokenizer_batch_decode: bool,
        tool_call_parser: String,
    ) -> Self {
        DetokenizerConfig {
            detokenizer_ipc_name,
            tokenizer_ipc_name,
            tokenizer_path,
            skip_tokenizer_init,
            max_states,
            disable_tokenizer_batch_decode,
            tool_call_parser,
        }
    }
}

// ──────────────────────────── HttpServerConfig ───────────────────────────────

/// Configuration for the Rust HTTP server + TokenizerManager process.
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct HttpServerConfig {
    /// ZMQ PULL endpoint where this process receives BatchStrOutput from detokenizer.
    #[pyo3(get, set)]
    pub tokenizer_ipc_name: String,

    /// ZMQ PUSH endpoint of the scheduler's input socket (we connect and push requests).
    #[pyo3(get, set)]
    pub scheduler_ipc_name: String,

    /// HuggingFace model id or local path that contains tokenizer.json.
    #[pyo3(get, set)]
    pub tokenizer_path: String,

    /// Skip tokenizer loading (useful when skip_tokenizer_init=True on the engine).
    #[pyo3(get, set)]
    pub skip_tokenizer_init: bool,

    /// Model name reported in /v1/models responses.
    #[pyo3(get, set)]
    pub model_name: String,

    /// HTTP bind host, e.g. "0.0.0.0".
    #[pyo3(get, set)]
    pub host: String,

    /// HTTP bind port.
    #[pyo3(get, set)]
    pub port: u16,

    /// Number of tokio worker threads (default 8).
    #[pyo3(get, set)]
    pub worker_threads: Option<u32>,
}

#[pymethods]
impl HttpServerConfig {
    #[new]
    #[pyo3(signature = (
        tokenizer_ipc_name,
        scheduler_ipc_name,
        tokenizer_path,
        model_name,
        host = "0.0.0.0".to_string(),
        port = 30000,
        skip_tokenizer_init = false,
        worker_threads = None,
    ))]
    fn new(
        tokenizer_ipc_name: String,
        scheduler_ipc_name: String,
        tokenizer_path: String,
        model_name: String,
        host: String,
        port: u16,
        skip_tokenizer_init: bool,
        worker_threads: Option<u32>,
    ) -> Self {
        HttpServerConfig {
            tokenizer_ipc_name,
            scheduler_ipc_name,
            tokenizer_path,
            skip_tokenizer_init,
            model_name,
            host,
            port,
            worker_threads,
        }
    }
}

// ───────────────────────────── EngineConfig ──────────────────────────────────

/// All-in-one configuration for the Rust engine (detokenizer + HTTP server + TM).
///
/// All three components run as tokio tasks on a single runtime in one process.
/// ZMQ sockets are used at all three process boundaries with the Python scheduler:
///   • TokenizerManager → Scheduler  (PUSH at scheduler_ipc_name)
///   • Scheduler → Detokenizer       (PULL at detokenizer_ipc_name)
///   • Scheduler → TokenizerManager  (PULL at tokenizer_ipc_name, for direct messages)
///
/// The Rust detokenizer → TokenizerManager path uses an `mpsc` channel instead
/// of a ZMQ socket, eliminating the intra-process ZMQ hop.
///
/// IPC addresses are allocated by the Python `PortArgs` helper and passed in.
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct EngineConfig {
    // ── IPC addresses ──────────────────────────────────────────────────────
    /// ZMQ address the detokenizer binds as PULL (scheduler connects and pushes here).
    #[pyo3(get, set)]
    pub detokenizer_ipc_name: String,

    /// ZMQ address the TokenizerManager binds as PULL for direct scheduler messages
    /// (also used as `http_worker_ipc` in outgoing requests so the scheduler knows
    /// where to route responses).
    #[pyo3(get, set)]
    pub tokenizer_ipc_name: String,

    /// ZMQ address the scheduler binds as PULL (TM connects and pushes requests here).
    #[pyo3(get, set)]
    pub scheduler_ipc_name: String,

    // ── Tokenizer settings ─────────────────────────────────────────────────
    /// HuggingFace model id or local path that contains tokenizer.json.
    #[pyo3(get, set)]
    pub tokenizer_path: String,

    /// Do not load a tokenizer; output empty strings.
    #[pyo3(get, set)]
    pub skip_tokenizer_init: bool,

    /// Decode each sequence individually instead of using batch_decode.
    #[pyo3(get, set)]
    pub disable_tokenizer_batch_decode: bool,

    /// Tool-call parser mode. Set to "gpt-oss" to enable special stop-token handling.
    #[pyo3(get, set)]
    pub tool_call_parser: String,

    /// Maximum number of in-flight decode states before oldest are evicted.
    #[pyo3(get, set)]
    pub max_states: usize,

    // ── HTTP server settings ───────────────────────────────────────────────
    /// HTTP bind host, e.g. "0.0.0.0".
    #[pyo3(get, set)]
    pub host: String,

    /// HTTP bind port.
    #[pyo3(get, set)]
    pub port: u16,

    /// Model name reported in /v1/models responses.
    #[pyo3(get, set)]
    pub model_name: String,

    /// Number of tokio worker threads for the HTTP server (default 8).
    #[pyo3(get, set)]
    pub worker_threads: Option<u32>,
}

#[pymethods]
impl EngineConfig {
    #[new]
    #[pyo3(signature = (
        detokenizer_ipc_name,
        tokenizer_ipc_name,
        scheduler_ipc_name,
        tokenizer_path,
        model_name,
        host = "0.0.0.0".to_string(),
        port = 30000,
        skip_tokenizer_init = false,
        disable_tokenizer_batch_decode = false,
        tool_call_parser = "".to_string(),
        max_states = 4096,
        worker_threads = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        detokenizer_ipc_name: String,
        tokenizer_ipc_name: String,
        scheduler_ipc_name: String,
        tokenizer_path: String,
        model_name: String,
        host: String,
        port: u16,
        skip_tokenizer_init: bool,
        disable_tokenizer_batch_decode: bool,
        tool_call_parser: String,
        max_states: usize,
        worker_threads: Option<u32>,
    ) -> Self {
        EngineConfig {
            detokenizer_ipc_name,
            tokenizer_ipc_name,
            scheduler_ipc_name,
            tokenizer_path,
            skip_tokenizer_init,
            disable_tokenizer_batch_decode,
            tool_call_parser,
            max_states,
            model_name,
            host,
            port,
            worker_threads,
        }
    }
}

// ──────────────────────────── pymodule ──────────────────────────────────────

#[pymodule]
mod sglang_detokenizer {
    use pyo3::prelude::*;
    use crate::{engine, http_server};

    #[pymodule_export]
    use super::DetokenizerConfig;

    #[pymodule_export]
    use super::HttpServerConfig;

    #[pymodule_export]
    use super::EngineConfig;

    /// Start the Rust detokenizer process (ZMQ PULL → decode → ZMQ PUSH).
    #[pyfunction]
    fn start_detokenizer(config: DetokenizerConfig) -> PyResult<()> {
        crate::detokenizer::runner::start(config.into());
        Ok(())
    }

    /// Start the Rust HTTP server + TokenizerManager process.
    ///
    /// Blocks until the server shuts down.
    ///
    /// IPC topology:
    ///   • Binds a ZMQ PULL socket at `tokenizer_ipc_name` — detokenizer connects here.
    ///   • Connects a ZMQ PUSH socket to `scheduler_ipc_name` — scheduler binds there.
    #[pyfunction]
    fn start_http_server(config: HttpServerConfig) -> PyResult<()> {
        http_server::start(config);
        Ok(())
    }

    /// Start the full Rust engine: detokenizer + HTTP server + TokenizerManager.
    ///
    /// Releases the Python GIL so the calling thread can run other Python code
    /// concurrently (e.g. launching the scheduler subprocess).
    #[pyfunction]
    fn start_engine(py: Python<'_>, config: EngineConfig) -> PyResult<()> {
        py.detach(|| engine::start(config));
        Ok(())
    }
}
