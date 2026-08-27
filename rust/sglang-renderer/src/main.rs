use std::path::PathBuf;

use sglang_renderer::{RendererRuntimeConfig, serve};

fn main() {
    let config_path = parse_config_path().unwrap_or_else(|error| exit(error));
    let config = std::fs::read_to_string(&config_path)
        .map_err(|error| format!("reading {} failed: {error}", config_path.display()))
        .and_then(|contents| {
            serde_json::from_str::<RendererRuntimeConfig>(&contents)
                .map_err(|error| format!("parsing {} failed: {error}", config_path.display()))
        })
        .unwrap_or_else(|error| exit(error));

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(config.http_workers.max(1))
        .enable_all()
        .build()
        .unwrap_or_else(|error| exit(format!("building renderer runtime failed: {error}")));
    runtime
        .block_on(serve(config))
        .unwrap_or_else(|error| exit(error));
}

fn parse_config_path() -> Result<PathBuf, String> {
    let mut args = std::env::args_os().skip(1);
    match (args.next().as_deref(), args.next(), args.next()) {
        (Some(flag), Some(path), None) if flag == "--config" => Ok(path.into()),
        _ => Err("usage: sglang-renderer --config PATH".into()),
    }
}

fn exit(message: impl std::fmt::Display) -> ! {
    eprintln!("sglang-renderer: {message}");
    std::process::exit(2)
}
