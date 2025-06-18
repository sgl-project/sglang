mod io_struct;
mod lb_state;
mod server;
mod strategy_lb;

use lb_state::{LBConfig, LBState};
use server::{periodic_logging, startup};
use tokio::signal;

fn main() -> anyhow::Result<()> {
    // FIXME: test code, move to test folder
    let prefill_infos = (0..8)
        .map(|i| (format!("123.123.123.123:{}", i), None))
        .collect::<Vec<(String, Option<u16>)>>();

    let decode_infos = (0..32)
        .map(|i| format!("233.233.233.233:{}", i))
        .collect::<Vec<String>>();

    let lb_config = LBConfig {
        host: "localhost".to_string(),
        port: 8080,
        policy: "random".to_string(),
        prefill_infos,
        decode_infos,
        log_interval: 5,
        timeout: 600,
    };
    let lb_state = LBState::new(lb_config.clone()).map_err(|e| anyhow::anyhow!(e))?;
    let ret: anyhow::Result<()> = actix_web::rt::System::new().block_on(async move {
        tokio::select! {
            _ = periodic_logging(lb_state.clone()) => {
                unreachable!()
            }
            res = startup(lb_config.clone(), lb_state) => {
                res.map_err(|e| anyhow::anyhow!(e))?;
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
