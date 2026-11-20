fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // `extension-module` leaves `_Py*` unresolved for the importing Python to
    // supply. setuptools-rust passes macOS's lazy-resolution flag for the
    // wheel's cdylib; nothing passes it for the test executable, and ld
    // rejects it without one.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
