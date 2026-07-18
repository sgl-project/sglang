//! sglang-bench: Rust load-generation client for `sglang.benchmark.serving`.
//!
//! The Python benchmark script prepares the workload (datasets, tokenizer,
//! payload JSON, arrival offsets) and computes the metrics; this crate owns
//! the hot loop — pacing, concurrency limiting, HTTP streaming, SSE/JSON
//! parsing, and per-chunk timing — on tokio threads, off the GIL. The GIL is
//! crossed twice per run: [`start_run`] and [`BenchRun::results`].

mod client;
mod output;
mod runner;
mod sse;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::client::{RequestSpec, RunConfig};
use crate::output::RequestOutput;

/// Per-request record from Python:
/// `(payload_json, prompt_len, output_len, arrival_offset_s, routing_key)`.
type RequestRecord = (Vec<u8>, u64, i64, f64, Option<String>);

/// Handle for an in-flight benchmark run. Poll `completed()` to drive the
/// progress bar, then collect with `results()` once `is_done()`.
#[pyclass]
struct BenchRun {
    handle: runner::RunHandle,
}

#[pymethods]
impl BenchRun {
    fn total(&self) -> usize {
        self.handle.state.total
    }

    fn completed(&self) -> usize {
        self.handle
            .state
            .completed
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn is_done(&self) -> bool {
        self.handle
            .join
            .as_ref()
            .is_none_or(|join| join.is_finished())
    }

    /// Return the per-request outputs in submission order. Joins the runner
    /// thread (GIL released while waiting); consumes the run — a second call
    /// raises.
    fn results(&mut self, py: Python<'_>) -> PyResult<Vec<RequestOutput>> {
        let join = self
            .handle
            .join
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("results() already consumed"))?;
        py.detach(|| join.join())
            .map_err(|_| PyRuntimeError::new_err("benchmark runner thread panicked"))
    }
}

/// Start a benchmark run in the background and return a [`BenchRun`] handle.
#[pyfunction]
#[pyo3(signature = (
    requests,
    *,
    api_url,
    headers,
    routing_key_header,
    max_concurrency = None,
    cache_report = false,
))]
fn start_run(
    requests: Vec<RequestRecord>,
    api_url: String,
    headers: Vec<(String, String)>,
    routing_key_header: String,
    max_concurrency: Option<usize>,
    cache_report: bool,
) -> PyResult<BenchRun> {
    let specs = requests
        .into_iter()
        .map(
            |(payload, prompt_len, output_len, arrival_offset_s, routing_key)| RequestSpec {
                payload,
                prompt_len,
                output_len,
                arrival_offset_s,
                routing_key,
            },
        )
        .collect();
    let cfg = RunConfig {
        api_url,
        headers,
        routing_key_header,
        max_concurrency,
        cache_report,
    };
    let handle = runner::start(cfg, specs).map_err(PyRuntimeError::new_err)?;
    Ok(BenchRun { handle })
}

/// The Python module: `import sglang_bench`.
#[pymodule]
fn sglang_bench(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_run, m)?)?;
    m.add_class::<BenchRun>()?;
    m.add_class::<RequestOutput>()?;
    Ok(())
}
