fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Only the `python` feature links pyo3 with `extension-module`, which
    // leaves `_Py*` unresolved for the importing Python to supply.
    // setuptools-rust passes macOS's lazy-resolution flag for the wheel's
    // cdylib; nothing passes it for the test executable, and ld rejects it
    // without one.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && std::env::var_os("CARGO_FEATURE_PYTHON").is_some()
    {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
