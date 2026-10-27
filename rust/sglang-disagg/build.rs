//! Builds the NIXL C-API shim (csrc/nixl_shim.cc) only when the `nixl`
//! feature is enabled. NIXL is Linux/CUDA-only, so feature-off builds —
//! including macOS dev machines — never touch the C++ sources.

fn main() {
    println!("cargo:rerun-if-changed=csrc/nixl_shim.h");
    if std::env::var("CARGO_FEATURE_NIXL").is_err() {
        return;
    }

    let include =
        std::env::var("NIXL_INCLUDE_DIR").unwrap_or_else(|_| "/usr/local/include".to_string());
    let header = std::path::Path::new(&include).join("nixl.h");
    if !header.exists() {
        panic!(
            "the `nixl` feature requires the NIXL C++ headers; \
             {} not found — set NIXL_INCLUDE_DIR to the directory containing nixl.h",
            header.display()
        );
    }

    // nixl.h >= 1.5 declares std::span overloads, so C++20 is required.
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .include(&include)
        .file("csrc/nixl_shim.cc")
        .compile("sgd_nixl_shim");

    // NIXL's shared library; NIXL_LIB_DIR overrides the linker search path
    // on non-standard installs.
    if let Ok(lib_dir) = std::env::var("NIXL_LIB_DIR") {
        println!("cargo:rustc-link-search=native={lib_dir}");
    }
    println!("cargo:rustc-link-lib=dylib=nixl");
    println!("cargo:rerun-if-changed=csrc/nixl_shim.cc");
    println!("cargo:rerun-if-env-changed=NIXL_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=NIXL_LIB_DIR");
}
