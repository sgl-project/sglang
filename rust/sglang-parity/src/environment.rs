//! Prepare one versioned SGLang environment for both serving implementations.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::time::{Instant, sleep};

use crate::process::{ServerConfig, isolated_command, run_command};
use crate::runner::RunConfig;

pub(crate) mod lock;
pub use lock::{Backend, Profile};

/// Regenerate one committed dependency lock from the repository declarations.
pub async fn update_lock(
    repo: &Path,
    backend: Backend,
    log: &Path,
    timeout: Duration,
) -> Result<PathBuf, String> {
    lock::update_lock(repo, backend, log, timeout).await
}

/// Source and installation policy shared by every API suite.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct EnvironmentConfig {
    pub source_root: Option<PathBuf>,
    pub backend: Backend,
    pub cache_dir: Option<PathBuf>,
    pub setup_timeout_secs: u64,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            source_root: None,
            backend: Backend::Auto,
            cache_dir: None,
            setup_timeout_secs: 1800,
        }
    }
}

/// Read-only description of the source and committed installation contract.
#[derive(Clone, Debug, Serialize)]
pub struct EnvironmentPlan {
    pub source_root: PathBuf,
    pub commit: String,
    pub profile: Profile,
    pub lock_file: PathBuf,
    pub lock_sha256: String,
    pub cache_dir: PathBuf,
    pub source_snapshot: PathBuf,
    pub environment_dir: PathBuf,
    pub python: PathBuf,
    pub managed: bool,
    pub build_environment: BTreeMap<String, String>,
}

fn git(repo: &Path, args: &[&str]) -> Result<String, String> {
    let output = isolated_command("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .map_err(|e| format!("cannot execute Git: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map(|s| s.trim_end().to_owned())
        .map_err(|e| e.to_string())
}

/// Find the checkout without changing it or creating an environment.
pub fn source_root(path: Option<&Path>) -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let root = git(path.unwrap_or(&cwd), &["rev-parse", "--show-toplevel"])?;
    fs::canonicalize(root).map_err(|e| e.to_string())
}

fn clean_source(repo: &Path) -> Result<(), String> {
    if !git(repo, &["diff", "HEAD", "--name-only"])?.is_empty() {
        return Err(
            "source checkout has uncommitted changes; commit them before testing HEAD".into(),
        );
    }
    let untracked = git(
        repo,
        &[
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "python",
            "rust",
        ],
    )?;
    if !untracked.is_empty() {
        return Err(format!(
            "source checkout has untracked source files; commit or remove them:\n{untracked}"
        ));
    }
    Ok(())
}

fn absolute(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path))
    }
}

fn python_path(path: &Path) -> Result<PathBuf, String> {
    if path.components().count() > 1 || path.is_absolute() {
        return absolute(path);
    }
    std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|directory| directory.join(path))
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| format!("Python executable {} was not found in PATH", path.display()))
        .and_then(|path| absolute(&path))
}

fn backend(requested: Backend) -> Result<Backend, String> {
    let available = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Backend::Mlx
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Backend::Cuda
    } else {
        return Err(
            "managed parity environments support Apple Silicon macOS and Linux x86_64 CUDA".into(),
        );
    };
    if requested != Backend::Auto && requested != available {
        return Err("selected environment backend does not match this host".into());
    }
    Ok(available)
}

fn build_environment(
    server: &ServerConfig,
    backend: Backend,
) -> Result<BTreeMap<String, String>, String> {
    for key in server.env.keys() {
        if key.starts_with("GIT_")
            || key.starts_with("PYTHON")
            || key.starts_with("UV_")
            || key.starts_with("PIP_")
            || matches!(
                key.as_str(),
                "VIRTUAL_ENV"
                    | "CONDA_PREFIX"
                    | "SGLANG_RUST_BUILD_MODE"
                    | "PYO3_PYTHON"
                    | "CARGO_TARGET_DIR"
            )
        {
            return Err(format!(
                "server.env contains source/environment control {key}"
            ));
        }
    }
    let mut environment = server.env.clone();
    for name in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET"] {
        if !environment.contains_key(name)
            && let Ok(value) = std::env::var(name)
        {
            environment.insert(name.into(), value);
        }
    }
    if backend == Backend::Mlx {
        if environment.get("SGLANG_USE_MLX").is_some_and(|v| v != "1") {
            return Err("MLX profile requires SGLANG_USE_MLX=1".into());
        }
        environment.insert("SGLANG_USE_MLX".into(), "1".into());
        if !environment.contains_key("CARGO_ENCODED_RUSTFLAGS") {
            environment
                .entry("RUSTFLAGS".into())
                .or_insert_with(|| "-C link-arg=-undefined -C link-arg=dynamic_lookup".into());
        }
    } else {
        if environment.get("SGLANG_USE_MLX").is_some_and(|v| v != "0") {
            return Err("CUDA profile cannot enable SGLANG_USE_MLX".into());
        }
        environment.insert("SGLANG_USE_MLX".into(), "0".into());
    }
    Ok(environment)
}

/// Resolve the committed contract without downloads, installations or builds.
pub fn describe(config: &RunConfig) -> Result<EnvironmentPlan, String> {
    if config.environment.setup_timeout_secs == 0 {
        return Err("environment setup timeout must be positive".into());
    }
    let repo = source_root(config.environment.source_root.as_deref())?;
    clean_source(&repo)?;
    let commit = git(&repo, &["rev-parse", "HEAD"])?;
    let profile = lock::load_profile(&repo, backend(config.environment.backend)?)?;
    let spec = lock::inspect_lock(&repo, &profile)?;
    let cache = absolute(
        config
            .environment
            .cache_dir
            .as_deref()
            .unwrap_or(&repo.join("rust/target/parity-environments")),
    )?;
    let build_environment = build_environment(&config.server, profile.backend)?;
    let mut identity = json!({
        "commit": commit, "profile": profile, "lock": spec.sha256, "build": build_environment,
    });
    identity.sort_all_objects();
    let identity = serde_json::to_vec(&identity).map_err(|e| e.to_string())?;
    let key = format!("{:x}", Sha256::digest(identity));
    let source_snapshot = cache.join("sources").join(&commit);
    let environment_dir = cache.join("environments").join(&key);
    let python = match &config.server.python {
        Some(path) => python_path(path)?,
        None => environment_dir.join("bin/python"),
    };
    Ok(EnvironmentPlan {
        source_root: repo,
        commit,
        profile,
        lock_file: spec.path,
        lock_sha256: spec.sha256,
        cache_dir: cache,
        source_snapshot,
        environment_dir,
        python,
        managed: config.server.python.is_none(),
        build_environment,
    })
}

struct Lease(File);

impl Lease {
    async fn acquire(path: &Path, deadline: Instant) -> Result<Self, String> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        loop {
            // SAFETY: the owned descriptor stays open for the lock's lifetime.
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
                return Ok(Self(file));
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::WouldBlock {
                return Err(error.to_string());
            }
            remaining(deadline)?;
            sleep(Duration::from_millis(50)).await;
        }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        // SAFETY: this descriptor is owned by the lease and is still open.
        unsafe {
            libc::flock(self.0.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

fn remaining(deadline: Instant) -> Result<Duration, String> {
    let duration = deadline.saturating_duration_since(Instant::now());
    if duration.is_zero() {
        Err("environment preparation timed out".into())
    } else {
        Ok(duration)
    }
}

/// A prepared interpreter and its source/installation evidence.
pub(crate) struct PreparedEnvironment {
    pub server: ServerConfig,
    pub record: Value,
    source_snapshot: PathBuf,
    commit: String,
    _lease: Lease,
    _source_lease: Lease,
}

impl PreparedEnvironment {
    pub fn verify_source(&self) -> Result<(), String> {
        verify_snapshot(&self.source_snapshot, &self.commit)
    }
}

fn verify_snapshot(snapshot: &Path, commit: &str) -> Result<(), String> {
    if git(snapshot, &["rev-parse", "--abbrev-ref", "HEAD"])? != "HEAD" {
        return Err("prepared source must be a detached Git worktree".into());
    }
    if git(snapshot, &["rev-parse", "HEAD"])? != commit {
        return Err("prepared source commit changed".into());
    }
    clean_source(snapshot)
}

fn command(
    program: impl AsRef<std::ffi::OsStr>,
    environment: &BTreeMap<String, String>,
) -> Command {
    let mut command = isolated_command(program);
    command.envs(environment).env("UV_NO_CONFIG", "1");
    command
}

async fn preflight(plan: &EnvironmentPlan, log: &Path, deadline: Instant) -> Result<(), String> {
    for program in ["cc", "cargo", "rustc"] {
        let mut check = command(program, &plan.build_environment);
        check
            .current_dir(plan.source_root.join("rust"))
            .arg("--version");
        run_command(&mut check, log, remaining(deadline)?).await?;
    }
    let mut platform = command(
        if plan.profile.backend == Backend::Mlx {
            "sw_vers"
        } else {
            "getconf"
        },
        &plan.build_environment,
    );
    platform.arg(if plan.profile.backend == Backend::Mlx {
        "-productVersion"
    } else {
        "GNU_LIBC_VERSION"
    });
    let platform_log = log.with_file_name("platform.log");
    run_command(&mut platform, &platform_log, remaining(deadline)?).await?;
    let version = fs::read_to_string(platform_log).map_err(|e| e.to_string())?;
    let numbers: Vec<u32> = version
        .split_whitespace()
        .last()
        .unwrap_or("")
        .split('.')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|_| format!("cannot parse host platform version: {version}"))?;
    let minimum = if plan.profile.backend == Backend::Mlx {
        [14, 0]
    } else {
        [2, 31]
    };
    if numbers.as_slice() < minimum.as_slice() {
        return Err(format!(
            "{} requires platform version {}.{} or newer, got {}",
            plan.profile.platform,
            minimum[0],
            minimum[1],
            version.trim()
        ));
    }
    Ok(())
}

/// Prepare once, retaining a lease until both implementations finish.
pub(crate) async fn prepare(
    config: &RunConfig,
    plan: &EnvironmentPlan,
    output: &Path,
) -> Result<PreparedEnvironment, String> {
    let deadline = Instant::now()
        .checked_add(Duration::from_secs(config.environment.setup_timeout_secs))
        .ok_or("environment timeout is too large")?;
    let log = output.join("setup.log");
    fs::write(
        &log,
        format!(
            "Preparing {} at {}\n",
            plan.commit,
            plan.source_snapshot.display()
        ),
    )
    .map_err(|e| e.to_string())?;
    fs::copy(&plan.lock_file, output.join("environment.lock")).map_err(|e| e.to_string())?;
    let copied_lock = fs::read(output.join("environment.lock")).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&copied_lock)) != plan.lock_sha256 {
        return Err("dependency lock changed after configuration validation".into());
    }
    if plan.managed {
        preflight(plan, &log, deadline).await?;
        let version_log = output.join("uv-version.log");
        run_command(
            command("uv", &plan.build_environment).arg("--version"),
            &version_log,
            remaining(deadline)?,
        )
        .await?;
        let version = fs::read_to_string(version_log).map_err(|e| e.to_string())?;
        if version.split_whitespace().take(2).collect::<Vec<_>>()
            != ["uv", plan.profile.uv.as_str()]
        {
            return Err(format!(
                "uv {} is required, got {}",
                plan.profile.uv,
                version.trim()
            ));
        }
    }
    fs::create_dir_all(plan.cache_dir.join("sources")).map_err(|e| e.to_string())?;
    fs::create_dir_all(plan.cache_dir.join("environments")).map_err(|e| e.to_string())?;
    let _source_lease = Lease::acquire(
        &plan
            .cache_dir
            .join("sources")
            .join(format!("{}.lock", plan.commit)),
        deadline,
    )
    .await?;
    if !plan.source_snapshot.exists() {
        let mut checkout = command("git", &BTreeMap::new());
        checkout
            .arg("-C")
            .arg(&plan.source_root)
            .args(["worktree", "add", "--detach"])
            .arg(&plan.source_snapshot)
            .arg(&plan.commit);
        run_command(&mut checkout, &log, remaining(deadline)?).await?;
    }
    verify_snapshot(&plan.source_snapshot, &plan.commit).map_err(|error| {
        format!(
            "invalid cached source {}: {error}; remove this cached Git worktree before retrying",
            plan.source_snapshot.display()
        )
    })?;
    let snapshot_lock = lock::inspect_lock(&plan.source_snapshot, &plan.profile)?;
    if snapshot_lock.sha256 != plan.lock_sha256 {
        return Err("source snapshot dependency lock differs from the described commit".into());
    }
    let lease = Lease::acquire(&plan.environment_dir.with_extension("lock"), deadline).await?;
    let mut environment = plan.build_environment.clone();
    environment.insert(
        "PYTHONPATH".into(),
        plan.source_snapshot
            .join("python")
            .to_string_lossy()
            .into_owned(),
    );
    environment.insert("PYTHONSAFEPATH".into(), "1".into());
    environment.insert("PYTHONNOUSERSITE".into(), "1".into());
    environment.insert("PYTHONDONTWRITEBYTECODE".into(), "1".into());
    environment.insert("SGLANG_RUST_BUILD_MODE".into(), "auto".into());
    environment
        .entry("SGLANG_CACHE_DIR".into())
        .or_insert_with(|| {
            plan.cache_dir
                .join("build-cache")
                .to_string_lossy()
                .into_owned()
        });
    if let Some(target) = &plan.profile.macos_deployment_target {
        environment.insert("MACOSX_DEPLOYMENT_TARGET".into(), target.clone());
    }
    let ready = plan.environment_dir.join("parity-ready.json");
    let reused = plan.managed && ready.is_file();
    if reused {
        let previous: Value = serde_json::from_slice(&fs::read(&ready).map_err(|e| e.to_string())?)
            .map_err(|e| format!("invalid environment completion marker: {e}"))?;
        if previous["plan"]["commit"] != plan.commit
            || previous["plan"]["lock_sha256"] != plan.lock_sha256
            || previous["plan"]["python"] != plan.python.to_string_lossy().as_ref()
            || previous["probe"]["status"] != "passed"
        {
            return Err("environment completion marker does not match this run".into());
        }
    }
    if plan.managed && !reused {
        if plan.environment_dir.exists() {
            fs::remove_dir_all(&plan.environment_dir).map_err(|e| e.to_string())?;
        }
        let mut create = command("uv", &environment);
        create
            .args([
                "venv",
                "--python-preference",
                "only-managed",
                "--python",
                &plan.profile.python,
            ])
            .arg(&plan.environment_dir);
        run_command(&mut create, &log, remaining(deadline)?).await?;
        let mut install = command("uv", &environment);
        install
            .args([
                "pip",
                "sync",
                "--require-hashes",
                "--only-binary",
                ":all:",
                "--default-index",
                "https://pypi.org/simple",
                "--python",
            ])
            .arg(&plan.python)
            .arg(output.join("environment.lock"));
        if let Some(backend) = &plan.profile.torch_backend {
            install.args(["--torch-backend", backend]);
        }
        for index in &plan.profile.indexes {
            install.args(["--index", index]);
        }
        install.args(["--index-strategy", "first-index"]);
        run_command(&mut install, &log, remaining(deadline)?).await?;
        let mut install_source = command("uv", &environment);
        install_source
            .env("SGLANG_BUILD_RUST_EXTS", "none")
            .args(["pip", "install", "--python"])
            .arg(&plan.python)
            .args(["--no-deps", "--no-build-isolation", "-e"])
            .arg(plan.source_snapshot.join("python"));
        run_command(&mut install_source, &log, remaining(deadline)?).await?;
    }
    let probe_path = plan
        .source_snapshot
        .join("rust/sglang-parity/environments/probe.py");
    if plan.profile.backend == Backend::Cuda {
        let paths_file = output.join("runtime-library-paths.json");
        let mut discover = command(&plan.python, &environment);
        discover
            .arg(&probe_path)
            .arg("--library-paths")
            .arg(&paths_file);
        run_command(&mut discover, &log, remaining(deadline)?).await?;
        let mut paths: Vec<PathBuf> =
            serde_json::from_slice(&fs::read(paths_file).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if let Some(previous) = environment
            .get("LD_LIBRARY_PATH")
            .cloned()
            .or_else(|| std::env::var("LD_LIBRARY_PATH").ok())
        {
            paths.extend(std::env::split_paths(&previous));
        }
        environment.insert(
            "LD_LIBRARY_PATH".into(),
            std::env::join_paths(paths)
                .map_err(|e| e.to_string())?
                .to_string_lossy()
                .into_owned(),
        );
    }
    let verification = output.join("environment-probe.json");
    let mut probe = command(&plan.python, &environment);
    probe
        .arg(&probe_path)
        .arg("--python")
        .arg(&plan.python)
        .arg("--python-version")
        .arg(&plan.profile.python)
        .arg("--source")
        .arg(&plan.source_snapshot)
        .arg("--lock")
        .arg(output.join("environment.lock"))
        .arg("--backend")
        .arg(if plan.profile.backend == Backend::Mlx {
            "mlx"
        } else {
            "cuda"
        })
        .arg("--output")
        .arg(&verification);
    run_command(&mut probe, &log, remaining(deadline)?).await?;
    verify_snapshot(&plan.source_snapshot, &plan.commit)?;
    let probe: Value = serde_json::from_slice(&fs::read(&verification).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    for (pointer, expected) in [
        ("/status", json!("passed")),
        ("/dependencies_only", json!(false)),
        (
            "/source",
            json!(fs::canonicalize(&plan.source_snapshot).map_err(|e| e.to_string())?),
        ),
        (
            "/lock",
            json!(fs::canonicalize(output.join("environment.lock")).map_err(|e| e.to_string())?),
        ),
        ("/python_executable", json!(plan.python)),
        ("/python_version", json!(plan.profile.python)),
        ("/backend/name", json!(plan.profile.backend.name())),
    ] {
        if probe.pointer(pointer) != Some(&expected) {
            return Err(format!(
                "environment probe {pointer} does not match this run"
            ));
        }
    }
    let record = json!({"plan": plan, "uv_version": plan.managed.then_some(&plan.profile.uv), "reused": reused, "probe": probe});
    if plan.managed && !reused {
        let pending = ready.with_extension("pending");
        fs::write(
            &pending,
            serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        fs::rename(pending, ready).map_err(|e| e.to_string())?;
    }
    fs::write(
        output.join("environment.json"),
        serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut server = config.server.clone();
    server.python = Some(plan.python.clone());
    server.env = environment;
    Ok(PreparedEnvironment {
        server,
        record,
        source_snapshot: plan.source_snapshot.clone(),
        commit: plan.commit.clone(),
        _lease: lease,
        _source_lease,
    })
}

#[cfg(test)]
mod tests;
