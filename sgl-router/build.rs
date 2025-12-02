use std::process::Command;

// Default values for version and project name when pyproject.toml is unavailable
const DEFAULT_VERSION: &str = "0.0.0";
const DEFAULT_PROJECT_NAME: &str = "sgl-router";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only regenerate if proto files change
    println!("cargo:rerun-if-changed=src/proto/sglang_scheduler.proto");
    println!("cargo:rerun-if-changed=src/proto/vllm_engine.proto");
    println!("cargo:rerun-if-changed=pyproject.toml");

    // Configure tonic-prost-build for gRPC code generation
    tonic_prost_build::configure()
        // Generate both client and server code
        .build_server(true)
        .build_client(true)
        // Add serde Serialize for model info messages (we only need to serialize to labels)
        .type_attribute("GetModelInfoResponse", "#[derive(serde::Serialize)]")
        // Allow proto3 optional fields
        .protoc_arg("--experimental_allow_proto3_optional")
        // Compile both proto files
        .compile_protos(
            &[
                "src/proto/sglang_scheduler.proto",
                "src/proto/vllm_engine.proto",
            ],
            &["src/proto"],
        )?;

    println!("cargo:info=Protobuf compilation completed successfully");

    // Only regenerate if proto files change
    println!("cargo:rerun-if-changed=src/mesh/proto/gossip.proto");

    tonic_prost_build::configure()
        // Generate both client and server code
        .build_server(true)
        .build_client(true)
        .compile_protos(&["src/mesh/proto/gossip.proto"], &["src/mesh/proto"])?;

    println!("cargo:info=Protobuf compilation completed successfully");

    // Read version and project name from pyproject.toml with fallback
    let version =
        read_field_from_pyproject("version").unwrap_or_else(|_| DEFAULT_VERSION.to_string());
    let project_name =
        read_field_from_pyproject("name").unwrap_or_else(|_| DEFAULT_PROJECT_NAME.to_string());
    println!("cargo:rustc-env=SGL_ROUTER_VERSION={}", version);
    println!("cargo:rustc-env=SGL_ROUTER_PROJECT_NAME={}", project_name);

    // Generate build time (UTC)
    let build_time = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S UTC")
        .to_string();
    println!("cargo:rustc-env=SGL_ROUTER_BUILD_TIME={}", build_time);

    // Try to get Git branch
    let git_branch = get_git_branch().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=SGL_ROUTER_GIT_BRANCH={}", git_branch);

    // Try to get Git commit hash
    let git_commit = get_git_commit().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=SGL_ROUTER_GIT_COMMIT={}", git_commit);

    // Try to get Git status (clean/dirty)
    let git_status = get_git_status().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=SGL_ROUTER_GIT_STATUS={}", git_status);

    // Get Rustc version
    let rustc_version = get_rustc_version().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=SGL_ROUTER_RUSTC_VERSION={}", rustc_version);

    // Get Cargo version
    let cargo_version = get_cargo_version().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=SGL_ROUTER_CARGO_VERSION={}", cargo_version);

    // Get target triple (platform)
    let target_triple = std::env::var("TARGET").unwrap_or_else(|_| {
        // Try to get from rustc if not set
        get_target_from_rustc().unwrap_or_else(|| "unknown".to_string())
    });
    println!("cargo:rustc-env=SGL_ROUTER_TARGET_TRIPLE={}", target_triple);

    // Get build mode (debug/release)
    let build_mode = if std::env::var("PROFILE").unwrap_or_default() == "release" {
        "release"
    } else {
        "debug"
    };
    println!("cargo:rustc-env=SGL_ROUTER_BUILD_MODE={}", build_mode);

    Ok(())
}

fn read_field_from_pyproject(field: &str) -> Result<String, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string("pyproject.toml")?;
    let toml: toml::Value = toml::from_str(&content)?;

    // Navigate to [project] section
    let project = toml
        .get("project")
        .ok_or("Missing [project] section in pyproject.toml")?;

    // Get the field value
    let value = project
        .get(field)
        .ok_or_else(|| format!("Missing '{}' field in [project] section", field))?;

    // Convert to string
    match value {
        toml::Value::String(s) => Ok(s.clone()),
        toml::Value::Integer(i) => Ok(i.to_string()),
        toml::Value::Float(f) => Ok(f.to_string()),
        toml::Value::Boolean(b) => Ok(b.to_string()),
        _ => Err(format!("Field '{}' is not a string value", field).into()),
    }
}

/// Execute a command and return its output as a trimmed string
fn run_command(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;

    if output.status.success() {
        String::from_utf8(output.stdout)
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    }
}

fn get_git_branch() -> Option<String> {
    run_command("git", &["rev-parse", "--abbrev-ref", "HEAD"])
}

fn get_git_commit() -> Option<String> {
    run_command("git", &["rev-parse", "--short", "HEAD"])
}

fn get_git_status() -> Option<String> {
    // Check if there are uncommitted changes
    let output = run_command("git", &["status", "--porcelain"])?;
    if output.is_empty() {
        Some("clean".to_string())
    } else {
        Some("dirty".to_string())
    }
}

fn get_rustc_version() -> Option<String> {
    run_command("rustc", &["--version"])
}

fn get_cargo_version() -> Option<String> {
    run_command("cargo", &["--version"])
}

fn get_target_from_rustc() -> Option<String> {
    let output_str = run_command("rustc", &["-vV"])?;
    for line in output_str.lines() {
        if line.starts_with("host: ") {
            if let Some(host) = line.strip_prefix("host: ") {
                return Some(host.trim().to_string());
            }
        }
    }
    None
}
