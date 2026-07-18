//! Python-visible result type.

use pyo3::prelude::*;

/// Mirror of the `RequestFuncOutput` fields the sglang-native path fills.
/// `ttft` / `latency` / `itl` are durations from the request send instant
/// (like Python). `start_time` is seconds relative to the run anchor (the
/// instant `start_run` was called); the Python wrapper re-bases it onto
/// `time.perf_counter()`.
#[pyclass(get_all, frozen)]
pub struct RequestOutput {
    pub generated_text: String,
    pub success: bool,
    pub latency: f64,
    pub ttft: f64,
    pub itl: Vec<f64>,
    pub prompt_len: u64,
    pub error: String,
    pub output_len: i64,
    pub start_time: f64,
    pub cached_tokens: u64,
    /// `meta_info.cached_tokens_details` as raw JSON (parsed lazily in Python).
    pub cached_tokens_details_json: Option<String>,
    pub spec_accept_length: f64,
}
