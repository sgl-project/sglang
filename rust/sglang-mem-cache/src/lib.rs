//! SGLang unified radix cache — Rust core logic + PyO3 wrapper.
//!
//! This crate is the Rust backing for SGLang's unified radix cache (the
//! component design in `python/sglang/srt/mem_cache/unified_radix_cache.py`).
//! It shares:
//!   * the core data-structure / algorithm modules, and
//!   * the PyO3 wrapper layer (`radix_cache_wrapper.rs`) plus the torch-tensor
//!     bridge (`py_interop::PyTensor`, vendored from `pyo3-tch`) that the Python
//!     `RustUnifiedRadixCache` orchestrator calls into.
//!
//! The Python build wiring lives in `python/pyproject.toml`; test and bench
//! wrappers are staged separately from the core wrapper.

// PyO3 proc macro generates Into<PyErr> conversions that clippy flags.
#![allow(clippy::useless_conversion)]

pub mod component_type;
pub mod components;
pub mod deferred_action;
pub mod error;
pub mod py_interop;
pub mod radix_cache;
pub mod radix_cache_wrapper;
pub mod tree_node_lru;
pub mod tree_node_pool;
pub mod utils;

use pyo3::prelude::*;

/// Native module imported by the Python orchestrator as
/// `sglang.srt.mem_cache._mem_cache_core`. The `#[pymodule]` function name, the
/// `[lib].name` in Cargo.toml, and the `setuptools-rust` target's last
/// component must all stay equal to `_mem_cache_core`.
#[pymodule]
fn _mem_cache_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "RadixCacheInitPyError",
        m.py().get_type_bound::<error::RadixCacheInitPyError>(),
    )?;
    m.add(
        "RadixCacheRuntimePyError",
        m.py().get_type_bound::<error::RadixCacheRuntimePyError>(),
    )?;
    m.add(
        "RadixCacheInfraPyError",
        m.py().get_type_bound::<error::RadixCacheInfraPyError>(),
    )?;

    m.add_class::<component_type::ComponentType>()?;
    m.add_class::<radix_cache_wrapper::RustPageRadixCacheWrapper>()?;
    m.add_class::<radix_cache_wrapper::RustBigramRadixCacheWrapper>()?;
    m.add_class::<radix_cache_wrapper::RustEvictResult>()?;
    m.add_class::<radix_cache_wrapper::RustInsertResult>()?;
    m.add_class::<radix_cache_wrapper::RustMatchResult>()?;
    m.add_class::<radix_cache_wrapper::RustPrepareLoadBackResult>()?;
    Ok(())
}
