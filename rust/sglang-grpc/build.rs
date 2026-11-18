fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // `extension-module` leaves `_Py*` unresolved for the importing Python to
    // supply; pyo3 passes macOS's lazy-resolution flag for the cdylib only,
    // and ld rejects the test executable without it.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
