mod server;
mod strategy_lb;

fn main() {
    // test code
    let prefill_infos = (0..8)
        .map(|i| (format!("123.123.123.123:{}", i), None))
        .collect::<Vec<(String, Option<u16>)>>();

    let decode_infos = (0..32)
        .map(|i| format!("233.233.233.233:{}", i))
        .collect::<Vec<String>>();

    let lb_config = server::LBConfig {
        host: "localhost".to_string(),
        port: 8080,
        policy: "random".to_string(),
        prefill_infos,
        decode_infos,
        log_interval: 5,
        timeout: 600,
    };
    let lb_state = server::LBState::new(lb_config.clone());
    actix_web::rt::System::new().block_on(async move {
        tokio::spawn(server::periodic_logging(lb_state.clone()));
        server::startup(lb_config.clone(), lb_state).await.unwrap();
    });
}
