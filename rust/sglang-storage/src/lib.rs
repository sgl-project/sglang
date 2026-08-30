//! Native storage helpers used by the Python serving runtime.

#[cfg(target_os = "linux")]
mod io_uring_reader;

use pyo3::prelude::*;

#[pymodule]
fn _storage(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(target_os = "linux")]
    m.add_class::<io_uring_reader::IoUringReader>()?;
    Ok(())
}
