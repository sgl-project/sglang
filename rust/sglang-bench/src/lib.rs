//! The serving benchmark client, ported from
//! `python/sglang/benchmark/serving.py`.
//!
//! Why a port: the Python client drives every request and parses every
//! response stream on one asyncio thread. Past a few thousand concurrent
//! streams that thread, not the server, sets the measured throughput, and the
//! benchmark reports the client's limit. Here each request is a Tokio task, so
//! stream parsing spreads across `--worker-threads` cores, and the one
//! remaining serial step (re-tokenizing the outputs) is batched.
//!
//! The report, the result JSON keys, and the flag names are kept identical, so
//! an existing command line and anything reading the `.jsonl` keep working.
//! See README.md for what this port does not cover.
//!
//! Python loads this as `sglang.srt.rust_extensions._bench`; see
//! `python/sglang/benchmark/rust_client.py`. The core is pyo3-free, so
//! `cargo test` builds it without Python.

mod args;
mod dataset;
mod hf;
mod metrics;
mod request;
mod run;

pub use args::Args;
pub use run::run_blocking;

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    use crate::Args;

    /// Run a benchmark described by a command line, as the CLI would.
    ///
    /// `argv` starts with the program name, so clap's errors and `--help` name
    /// the caller correctly. A usage error raises `SystemExit` with clap's own
    /// exit code after printing clap's message, which keeps
    /// `python -m sglang.benchmark.rust_client --help` behaving like any other
    /// command line tool.
    #[pyfunction]
    fn run_argv(py: Python<'_>, argv: Vec<String>) -> PyResult<String> {
        use clap::Parser;

        let args = match Args::try_parse_from(&argv) {
            Ok(args) => args,
            Err(error) => {
                // clap renders both help and usage errors; print it the way it
                // asks to (help on stdout, errors on stderr).
                let _ = error.print();
                return Err(PyErr::new::<pyo3::exceptions::PySystemExit, _>(
                    error.exit_code(),
                ));
            }
        };
        run(py, args)
    }

    /// Run a benchmark described by a JSON object whose keys are the long flag
    /// names without their dashes. Missing keys take the flag's default, so a
    /// caller sends only what it means to set.
    ///
    /// This is the entry point for a Python caller that already parsed its own
    /// arguments; `run_argv` is the one for a bare command line.
    #[pyfunction]
    fn run_config(py: Python<'_>, config: &str) -> PyResult<String> {
        let args: Args = serde_json::from_str(config).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "invalid benchmark config: {error}"
            ))
        })?;
        run(py, args)
    }

    /// The GIL is released for the whole run: it owns a Tokio runtime with its
    /// own threads and never touches Python, so holding the GIL would only
    /// block the caller's other threads for the length of the benchmark.
    fn run(py: Python<'_>, args: Args) -> PyResult<String> {
        let result = py.detach(|| crate::run_blocking(args)).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{error:#}"))
        })?;
        serde_json::to_string(&result).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "cannot serialize the benchmark result: {error}"
            ))
        })
    }

    #[pymodule]
    fn _bench(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(run_argv, module)?)?;
        module.add_function(wrap_pyfunction!(run_config, module)?)?;
        Ok(())
    }
}
