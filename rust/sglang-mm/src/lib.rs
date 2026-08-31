//! sglang-mm: SGLang's adapter over `dynamo-mm-preprocessor`.
//!
//! The preprocessing pipeline itself (families, driver, resize kernels, fetch)
//! lives in the `dynamo-mm-preprocessor` crate; this crate keeps what is
//! SGLang-shaped: the PyO3 bindings, the Inkling batch path, the
//! scheduler-drain packing (`qwen_vl::pack_output`), and the env-var shims
//! (the dynamo crate reads no environment variables).
//!
//! Built two ways:
//! * PyO3 extension `sglang.srt.rust_extensions._multimodal` (feature `python`),
//!   used by Python processors (e.g. Inkling) and by parity tests.
//! * Pure-Rust `rlib` (`default-features = false`), linked by `sglang-server`'s
//!   MM worker path — no pyo3 in that dependency graph.

pub mod common;
pub mod inkling;
pub mod qwen_vl;
pub mod registry;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _multimodal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Pool size from the SGLang env (zero/unset falls through to the dynamo
    // crate's own default of min(cores, 8)).
    #[cfg(feature = "parallel")]
    dynamo_mm_preprocessor::par::init_pool(
        std::env::var("SGL_MM_RS_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
    );
    common::register(m)?;
    inkling::register(m)?;
    qwen_vl::register(m)?;
    Ok(())
}
