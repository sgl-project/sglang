use std::process::Command;

const DEFAULT_VERSION: &str = "0.0.0";
const DEFAULT_PROJECT_NAME: &str = "sgl-model-gateway";

/// Set a compile-time environment variable with the SGL_MODEL_GATEWAY_ prefix
macro_rules! set_env {
    ($name:expr, $value:expr) => {
        println!("cargo:rustc-env=SGL_MODEL_GATEWAY_{}={}", $name, $value);
    };
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rebuild triggers
    println!("cargo:rerun-if-changed=src/proto/sglang_scheduler.proto");
    println!("cargo:rerun-if-changed=src/proto/vllm_engine.proto");
    println!("cargo:rerun-if-changed=Cargo.toml");

    // Compile protobuf files
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .type_attribute("GetModelInfoResponse", "#[derive(serde::Serialize)]")
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &[
                "src/proto/sglang_scheduler.proto",
                "src/proto/vllm_engine.proto",
            ],
            &["src/proto"],
        )?;

    // Set version info environment variables
    let version = read_cargo_version().unwrap_or_else(|_| DEFAULT_VERSION.to_string());
    let target = std::env::var("TARGET").unwrap_or_else(|_| get_rustc_host().unwrap_or_default());
    let profile = std::env::var("PROFILE").unwrap_or_default();

    set_env!("PROJECT_NAME", DEFAULT_PROJECT_NAME);
    set_env!("VERSION", version);
    set_env!(
        "BUILD_TIME",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    );
    set_env!(
        "BUILD_MODE",
        if profile == "release" {
            "release"
        } else {
            "debug"
        }
    );
    set_env!("TARGET_TRIPLE", target);
    set_env!(
        "GIT_BRANCH",
        git_branch().unwrap_or_else(|| "unknown".into())
    );
    set_env!(
        "GIT_COMMIT",
        git_commit().unwrap_or_else(|| "unknown".into())
    );
    set_env!(
        "GIT_STATUS",
        git_status().unwrap_or_else(|| "unknown".into())
    );
    set_env!(
        "RUSTC_VERSION",
        rustc_version().unwrap_or_else(|| "unknown".into())
    );
    set_env!(
        "CARGO_VERSION",
        cargo_version().unwrap_or_else(|| "unknown".into())
    );

    Ok(())
}

fn read_cargo_version() -> Result<String, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string("Cargo.toml")?;
    let toml: toml::Value = toml::from_str(&content)?;
    toml.get("package")
        .and_then(|p| p.get("version"))
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| "Missing version in Cargo.toml".into())
}

fn run_cmd(cmd: &str, args: &[&str]) -> Option<String> {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

fn git_branch() -> Option<String> {
    run_cmd("git", &["rev-parse", "--abbrev-ref", "HEAD"])
}

fn git_commit() -> Option<String> {
    run_cmd("git", &["rev-parse", "--short", "HEAD"])
}

fn git_status() -> Option<String> {
    run_cmd("git", &["status", "--porcelain"])
        .map(|s| if s.is_empty() { "clean" } else { "dirty" }.into())
}

fn rustc_version() -> Option<String> {
    run_cmd("rustc", &["--version"])
}

fn cargo_version() -> Option<String> {
    run_cmd("cargo", &["--version"])
}

fn get_rustc_host() -> Option<String> {
    run_cmd("rustc", &["-vV"])?
        .lines()
        .find(|l| l.starts_with("host: "))
        .and_then(|l| l.strip_prefix("host: "))
        .map(|s| s.trim().to_string())
}
