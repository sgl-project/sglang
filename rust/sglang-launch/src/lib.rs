//! Host-side launcher for chains of small CUDA kernels.
//!
//! A diffusion step at low resolution is not one big kernel; it is on the order
//! of a thousand launches whose kernels run for tens of microseconds each. The
//! device finishes each one long before Python and the PyTorch dispatcher can
//! describe the next, so the GPU idles waiting on the host.
//!
//! This crate takes the issuing loop out of Python. A chain is recorded once as
//! (function, grid, block, arguments), then replayed through `cuLaunchKernel`
//! with one FFI crossing for the whole chain instead of one dispatcher round
//! trip per op. The kernels themselves are unchanged and run in the recorded
//! order on the caller's stream, so results are bit-identical to the eager path
//! -- this removes host latency, it does not alter arithmetic.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::os::raw::{c_int, c_uint, c_void};

type CUresult = c_int;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;

#[link(name = "cuda")]
extern "C" {
    fn cuLaunchKernel(
        f: CUfunction,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    fn cuGetErrorName(error: CUresult, str_: *mut *const std::os::raw::c_char) -> CUresult;
}

fn cuda_err(name: &str, code: CUresult) -> PyErr {
    let mut text: *const std::os::raw::c_char = std::ptr::null();
    let label = unsafe {
        if cuGetErrorName(code, &mut text) == 0 && !text.is_null() {
            std::ffi::CStr::from_ptr(text).to_string_lossy().into_owned()
        } else {
            format!("code {code}")
        }
    };
    PyRuntimeError::new_err(format!("{name} failed: {label}"))
}

/// One recorded launch. Arguments are stored by value in the order the kernel
/// declares them; `cuLaunchKernel` wants an array of pointers *to* those values,
/// which is rebuilt at replay so the boxed storage can move freely until then.
struct Launch {
    /// Held as an integer rather than `CUfunction` so the chain stays `Send`
    /// and the replay loop can run with the GIL released.
    func: usize,
    grid: (c_uint, c_uint, c_uint),
    block: (c_uint, c_uint, c_uint),
    shared: c_uint,
    /// Each entry is an 8-byte argument slot: a device pointer or a scalar
    /// already widened by the caller. Kernels compiled for these ops take
    /// pointers and 32/64-bit scalars, so one slot width covers both.
    args: Vec<u64>,
}

/// A replayable chain of launches.
///
/// Recording is a Python-side cost paid once. Replay is the steady-state path
/// and touches no Python object, no allocator and no dispatcher.
#[pyclass]
struct LaunchChain {
    launches: Vec<Launch>,
}

#[pymethods]
impl LaunchChain {
    #[new]
    fn new() -> Self {
        LaunchChain {
            launches: Vec::new(),
        }
    }

    /// Record one launch.
    ///
    /// `func` is a `CUfunction` as an integer, as returned by the runtime that
    /// loaded the module. `args` are the kernel's parameters in declaration
    /// order: device pointers as integers, scalars already widened to 64 bits.
    fn record(
        &mut self,
        func: usize,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        args: Vec<u64>,
    ) -> PyResult<()> {
        if func == 0 {
            return Err(PyRuntimeError::new_err("null CUfunction"));
        }
        self.launches.push(Launch {
            func,
            grid,
            block,
            shared,
            args,
        });
        Ok(())
    }

    /// Replay every recorded launch on `stream`.
    ///
    /// The GIL is released for the whole chain: nothing here calls back into
    /// Python, and holding it would serialize this against any other thread
    /// trying to issue work.
    fn replay(&mut self, py: Python<'_>, stream: usize) -> PyResult<()> {
        let stream_addr = stream;
        let failure = py.allow_threads(|| {
            let mut slots: Vec<*mut c_void> = Vec::with_capacity(16);
            for (index, launch) in self.launches.iter_mut().enumerate() {
                slots.clear();
                for arg in launch.args.iter_mut() {
                    slots.push(arg as *mut u64 as *mut c_void);
                }
                let code = unsafe {
                    cuLaunchKernel(
                        launch.func as CUfunction,
                        launch.grid.0,
                        launch.grid.1,
                        launch.grid.2,
                        launch.block.0,
                        launch.block.1,
                        launch.block.2,
                        launch.shared,
                        stream_addr as CUstream,
                        slots.as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                };
                if code != 0 {
                    return Some((index, code));
                }
            }
            None
        });
        match failure {
            Some((index, code)) => Err(cuda_err(
                &format!("cuLaunchKernel (chain index {index})"),
                code,
            )),
            None => Ok(()),
        }
    }

    /// Rebind one argument slot without re-recording the chain.
    ///
    /// Shapes and buffers change between steps while the op sequence does not.
    /// Rebinding is what lets one recording serve every step, and is the reason
    /// this is not just a CUDA graph: a graph would have to be recaptured.
    fn rebind(&mut self, launch_index: usize, arg_index: usize, value: u64) -> PyResult<()> {
        let launch = self
            .launches
            .get_mut(launch_index)
            .ok_or_else(|| PyRuntimeError::new_err("launch index out of range"))?;
        let slot = launch
            .args
            .get_mut(arg_index)
            .ok_or_else(|| PyRuntimeError::new_err("argument index out of range"))?;
        *slot = value;
        Ok(())
    }

    /// Set a launch's grid, for chains whose shape changes between steps.
    fn set_grid(&mut self, launch_index: usize, grid: (u32, u32, u32)) -> PyResult<()> {
        let launch = self
            .launches
            .get_mut(launch_index)
            .ok_or_else(|| PyRuntimeError::new_err("launch index out of range"))?;
        launch.grid = grid;
        Ok(())
    }

    fn clear(&mut self) {
        self.launches.clear();
    }

    #[getter]
    fn len(&self) -> usize {
        self.launches.len()
    }
}

#[pymodule]
fn sgl_launch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LaunchChain>()?;
    Ok(())
}
