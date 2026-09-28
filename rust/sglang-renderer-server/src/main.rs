//! Standalone renderer process. The shared libraries own no process runtime.
mod engine;
mod launcher;
mod proxy;
mod routes;
mod runtime;
#[cfg(test)]
mod test_utils;
#[cfg(test)]
mod tests;

fn main() {
    launcher::run_cli().unwrap_or_else(|error| exit(error));
}

fn exit(message: impl std::fmt::Display) -> ! {
    eprintln!("sglang-renderer: {message}");
    std::process::exit(2)
}
