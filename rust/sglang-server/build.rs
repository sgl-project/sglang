fn main() {
    // `sglang-server` is loaded by an existing Python interpreter through
    // PyO3, so it intentionally does not link a separate libpython. Linux
    // permits the Python C API symbols to remain unresolved until load time,
    // but Apple's linker requires `-undefined dynamic_lookup` for the same
    // extension-module layout. This helper adds that flag on macOS and is a
    // no-op on other platforms.
    pyo3_build_config::add_extension_module_link_args();
}
