use pyo3::prelude::*;

mod cli;
mod ipc;
mod detokenizer;

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct DetokenizerConfig {
    /// ZMQ endpoint to PULL batches from the scheduler (bind).
    #[pyo3(get, set)]
    detokenizer_ipc_name: String,

    /// ZMQ endpoint to PUSH results to the tokenizer worker (connect).
    #[pyo3(get, set)]
    tokenizer_ipc_name: String,

    /// HuggingFace model id or local path that contains tokenizer.json.
    #[pyo3(get, set)]
    tokenizer_path: String,

    /// Do not load a tokenizer; output empty strings (used with skip_tokenizer_init).
    #[pyo3(get, set)]
    skip_tokenizer_init: bool,

    /// Maximum number of in-flight decode states before oldest are evicted.
    #[pyo3(get, set)]
    max_states: usize,

    /// Decode each sequence individually instead of using batch_decode.
    #[pyo3(get, set)]
    disable_tokenizer_batch_decode: bool,

    /// Tool-call parser mode. Set to "gpt-oss" to enable special stop-token handling.
    #[pyo3(get, set)]
    tool_call_parser: String,
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

/// A Python module implemented in Rust.
#[pymodule]
mod sglang_detokenizer {
    use pyo3::prelude::*;
    use crate::cli;

    #[pymodule_export]
    use super::DetokenizerConfig;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn start_detokenizer(config: DetokenizerConfig) -> PyResult<()> {
        cli::start(config.into());
        Ok(())
    }
}
