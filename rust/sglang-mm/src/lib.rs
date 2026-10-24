//! sglang-mm: Rust-accelerated multimodal preprocessing for SGLang.
//!
//! Built two ways:
//! * PyO3 extension `sglang.srt.rust_extensions._multimodal` (feature `python`),
//!   used by Python processors (e.g. Inkling) and by parity tests.
//! * Pure-Rust `rlib` (`default-features = false`) for the crate's own tests.
//!
//! The Rust server's multimodal pipeline lives in `sglang-processor` (`mm`).

pub mod common;
pub mod dsv41;
pub mod inkling;
pub mod registry;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _multimodal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    common::register(m)?;
    inkling::register(m)?;
    dsv41::register(m)?;
    Ok(())
}
