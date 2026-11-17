// Libpython symbols in a PyO3 cdylib resolve at import time; on macOS that needs
// `-undefined dynamic_lookup`, which plain `cargo build` (the runtime loader) never passes.
fn main() {
    #[cfg(feature = "python")]
    pyo3_build_config::add_extension_module_link_args();
}
