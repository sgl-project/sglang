fn main() {
    crate::launcher::run_cli().unwrap_or_else(|error| exit(error));
}

fn exit(message: impl std::fmt::Display) -> ! {
    eprintln!("sglang-renderer: {message}");
    std::process::exit(2)
}
