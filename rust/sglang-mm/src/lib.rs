//! sglang-mm: Rust-accelerated multimodal preprocessing for SGLang.
//!
//! Built two ways:
//! * PyO3 extension `sglang.srt.rust_extensions._multimodal` (feature `python`),
//!   used by Python processors (e.g. Inkling) and by parity tests.
//! * Pure-Rust `rlib` (`default-features = false`), linked by `sglang-server`'s
//!   MM worker path — no pyo3 in that dependency graph.

pub mod common;
pub mod driver;
pub mod inkling;
pub mod pipeline;
pub mod qwen_vl;
pub mod registry;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _multimodal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    common::register(m)?;
    inkling::register(m)?;
    qwen_vl::register(m)?;
    Ok(())
}
