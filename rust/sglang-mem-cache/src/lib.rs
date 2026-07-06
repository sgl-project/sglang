//! Rust unified radix cache with PyO3 bindings.

#![allow(clippy::useless_conversion)]

pub mod component_type;
pub mod components;
pub mod deferred_action;
pub mod error;
pub mod py_interop;
pub mod radix_cache;
pub mod radix_cache_wrapper;
pub mod tree_node_pool;
pub mod utils;

use pyo3::prelude::*;

/// Native module imported by Python as `sglang.srt.mem_cache._mem_cache_core`.
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
    Ok(())
}
