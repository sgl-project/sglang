pub mod io_struct;
pub mod lb_state;
pub mod server;
pub mod strategy_lb;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use lb_state::{LBConfig, LBState};
use server::{periodic_logging, startup};
use tokio::signal;

#[pyclass]
pub struct LoadBalancer {
    lb_config: LBConfig,
}

#[pymethods]
impl LoadBalancer {
    #[new]
    pub fn new(
        host: String,
        port: u16,
        policy: String,
        prefill_infos: Vec<(String, Option<u16>)>,
        decode_infos: Vec<String>,
        log_interval: u64,
        timeout: u64,
    ) -> PyResult<Self> {
        let lb_config = LBConfig {
            host,
            port,
            policy,
            prefill_infos,
            decode_infos,
            log_interval,
            timeout,
        };
        Ok(LoadBalancer { lb_config })
    }

    pub fn start(&self) -> PyResult<()> {
        let lb_state = LBState::new(self.lb_config.clone()).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to build load balancer: {}", e))
        })?;

        let ret: PyResult<()> = actix_web::rt::System::new().block_on(async move {
            tokio::select! {
                _ = periodic_logging(lb_state.clone()) => {
                    unreachable!()
                }
                res = startup(self.lb_config.clone(), lb_state) => {
                    res.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    unreachable!()
                }
                _ = signal::ctrl_c() => {
                    println!("Received Ctrl+C, shutting down");
                    std::process::exit(0);
                }
            }
        });
        ret
    }
}

#[pymodule]
fn _rust(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<LoadBalancer>()?;
    Ok(())
}
