fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // sglang-server links pyo3 with `extension-module`, which leaves `_Py*`
    // unresolved for the importing Python to supply; macOS's ld rejects a
    // test executable with them undefined unless told to resolve lazily.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
