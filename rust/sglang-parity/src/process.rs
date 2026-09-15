//! Own process groups for environment preparation and local SGLang servers.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io;
use std::net::TcpListener;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::time::{Instant, sleep};

/// Shared configuration for both implementations; lifecycle controls are reserved.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default)]
    pub python: Option<PathBuf>,
    pub model: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub working_dir: Option<PathBuf>,
}

fn default_port() -> u16 {
    30000
}

fn default_seed() -> u64 {
    42
}

// argparse accepts abbreviated long options. Reserve their prefixes too, so
// e.g. --random-s=7 cannot replace the suite's controlled seed.
const CONTROLLED_OPTIONS: &[&str] = &[
    "--model-path",
    "--model",
    "--host",
    "--port",
    "--random-seed",
    "--enable-deterministic-inference",
    "--disable-radix-cache",
    "--config",
    "--enable-metrics",
    "--enable-metrics-for-all-schedulers",
    "--enable-mfu-metrics",
    "--enable-forward-pass-metrics",
    "--export-metrics-to-file",
    "--speculative-algorithm",
    "--uno-lora-path",
    "--grpc-mode",
    "--grpc-port",
    "--smg-grpc-mode",
    "--smg-http-sidecar-port",
    "--enable-grpc",
    "--encoder-only",
    "--use-ray",
    "--sidecar",
];

fn controlled_option(option: &str) -> bool {
    CONTROLLED_OPTIONS
        .iter()
        .any(|reserved| reserved.starts_with(option))
        || option.starts_with("--speculative-")
        || option.starts_with("--metrics-")
        || option.starts_with("--enable-metrics-")
        || option.starts_with("--config-")
        || option.starts_with("--grpc-")
        || option.starts_with("--smg-grpc-")
}

fn controlled_env(key: &str) -> bool {
    matches!(
        key,
        "SGLANG_RUST_SERVER"
            | "SGLANG_ENABLE_DETERMINISTIC_INFERENCE"
            | "SGLANG_PORT"
            | "SGLANG_GRPC_PORT"
    ) || (key.starts_with("SGLANG_")
        && key.split('_').any(|part| {
            matches!(
                part,
                "METRICS"
                    | "GRPC"
                    | "SPEC"
                    | "SPECULATIVE"
                    | "DRAFT"
                    | "DSPARK"
                    | "DFLASH"
                    | "EAGLE"
                    | "NGRAM"
            )
        }))
}

fn configure_environment(
    command: &mut Command,
    config: &ServerConfig,
    implementation: Implementation,
    inherited_keys: impl Iterator<Item = OsString>,
) {
    // Remove controls inherited from the calling shell as well as rejecting
    // explicit overrides. Hardware/library variables remain shared unchanged.
    for key in inherited_keys {
        if key.to_str().is_some_and(controlled_env) {
            command.env_remove(key);
        }
    }
    command
        .envs(&config.env)
        .env(
            "SGLANG_RUST_SERVER",
            if implementation == Implementation::Rust {
                "1"
            } else {
                "0"
            },
        )
        .env("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "1");
}

impl ServerConfig {
    /// Reject ambiguous configurations before starting or downloading a model.
    pub fn validate(&self) -> Result<(), String> {
        if self
            .python
            .as_ref()
            .is_some_and(|python| python.as_os_str().is_empty())
        {
            return Err("server.python must name a Python executable".into());
        }
        if self.model.trim().is_empty() || self.model.contains('\0') {
            return Err("server.model must be nonempty and contain no NUL".into());
        }
        if self.port == 0 {
            return Err("server.port must be between 1 and 65535".into());
        }
        if self.seed > u32::MAX as u64 {
            return Err("server.seed must fit uint32, as required by SGLang's NumPy seed".into());
        }
        if let Some(directory) = &self.working_dir
            && !directory.is_dir()
        {
            return Err(format!(
                "server.working_dir is not a directory: {}",
                directory.display()
            ));
        }
        for arg in &self.args {
            if arg.contains('\0') {
                return Err("server.args may not contain NUL".into());
            }
            if arg == "--" {
                return Err("server.args may not contain the option terminator --".into());
            }
            if arg.starts_with('-')
                && !arg.starts_with("--")
                && arg.chars().nth(1).is_some_and(|c| c.is_ascii_alphabetic())
            {
                return Err(format!(
                    "server.args must use full --option spellings, not {arg}"
                ));
            }
            if arg.starts_with("--") {
                let option = arg.split('=').next().unwrap();
                if controlled_option(option) {
                    return Err(format!(
                        "server.args contains reserved or unsupported option {option}"
                    ));
                }
                // This is an argparse store_true flag, not a bool-valued option.
                if incremental_option(option) && option != arg {
                    return Err(format!("{option} takes no value"));
                }
            }
        }
        for (key, value) in &self.env {
            if key.is_empty() || key.contains(['=', '\0']) || value.contains('\0') {
                return Err("server.env contains an invalid name or NUL value".into());
            }
            if controlled_env(key) {
                return Err(format!(
                    "server.env contains reserved or unsupported variable {key}"
                ));
            }
        }
        Ok(())
    }

    /// Whether the shared SGLang streaming output mode is incremental.
    pub fn incremental_output(&self) -> bool {
        self.args.iter().any(|arg| incremental_option(arg))
    }
}

fn incremental_option(option: &str) -> bool {
    // --inc is unambiguous in SGLang; do not mistake arbitrary
    // values containing the flag, or =false, for a enabled store_true flag.
    option.starts_with("--inc") && "--incremental-streaming-output".starts_with(option)
}

/// The two actual SGLang serving implementations.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Implementation {
    Python,
    Rust,
}

impl Implementation {
    pub const ALL: [Self; 2] = [Self::Python, Self::Rust];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::Rust => "rust",
        }
    }
}

/// Own a process group across completion, errors, and cancelled futures.
///
/// Keep the direct child until cleanup even after it exits: descendants may
/// still be running. Drop kills the group and reaps the child synchronously.
#[derive(Debug)]
pub(crate) struct ManagedChild {
    child: Option<Child>,
    process_group: i32,
}

impl ManagedChild {
    pub(crate) fn spawn(command: &mut Command) -> io::Result<Self> {
        let child = command.process_group(0).spawn()?;
        Ok(Self {
            process_group: child.id() as i32,
            child: Some(child),
        })
    }

    pub(crate) fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.child
            .as_mut()
            .ok_or_else(|| io::Error::other("process has already been shut down"))?
            .try_wait()
    }

    pub(crate) async fn shutdown(&mut self, grace: Duration) -> Result<(), String> {
        if self.child.is_none() {
            return Ok(());
        }
        let deadline = Instant::now()
            .checked_add(grace)
            .ok_or("shutdown timeout is too large")?;
        signal_group(self.process_group, libc::SIGTERM)
            .map_err(|e| format!("cannot terminate process group: {e}"))?;
        loop {
            let exited = self
                .try_wait()
                .map_err(|e| format!("cannot inspect shutting down process: {e}"))?
                .is_some();
            if exited && !group_exists(self.process_group) {
                self.child.take();
                return Ok(());
            }
            if Instant::now() >= deadline {
                return self.kill_and_reap();
            }
            sleep(
                Duration::from_millis(25).min(deadline.saturating_duration_since(Instant::now())),
            )
            .await;
        }
    }

    fn kill_and_reap(&mut self) -> Result<(), String> {
        let Some(mut child) = self.child.take() else {
            return Ok(());
        };
        let signal = signal_group(self.process_group, libc::SIGKILL);
        // Also target the direct child if group signaling failed unexpectedly.
        if signal.is_err() {
            let _ = child.kill();
        }
        let wait = child.wait();
        signal.map_err(|e| format!("cannot kill process group: {e}"))?;
        wait.map_err(|e| format!("cannot reap process: {e}"))?;
        Ok(())
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        let _ = self.kill_and_reap();
    }
}

/// Remove inherited Python and installer overrides before applying run settings.
pub(crate) fn isolated_command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    let mut command = Command::new(program);
    for (key, _) in std::env::vars_os() {
        if key.to_str().is_some_and(|key| {
            key.starts_with("GIT_")
                || key.starts_with("UV_")
                || key.starts_with("PIP_")
                || key.starts_with("PYTHON")
                || matches!(
                    key,
                    "VIRTUAL_ENV"
                        | "CONDA_PREFIX"
                        | "SGLANG_RUST_BUILD_MODE"
                        | "PYO3_PYTHON"
                        | "CARGO_TARGET_DIR"
                )
        }) {
            command.env_remove(key);
        }
    }
    command
}

/// Run a setup command with retained output and cancellation-safe cleanup.
pub(crate) async fn run_command(
    command: &mut Command,
    log: &Path,
    timeout: Duration,
) -> Result<(), String> {
    if timeout.is_zero() {
        return Err("command timeout must be positive".into());
    }
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or("command timeout is too large")?;
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .map_err(|e| format!("cannot open command log {}: {e}", log.display()))?;
    let stderr = stdout
        .try_clone()
        .map_err(|e| format!("cannot duplicate command log: {e}"))?;
    command.stdin(Stdio::null()).stdout(stdout).stderr(stderr);
    let program = command.get_program().to_string_lossy().into_owned();
    let mut child = ManagedChild::spawn(command)
        .map_err(|e| format!("cannot start {program}: {e}; see {}", log.display()))?;
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|e| format!("cannot inspect {program}: {e}; see {}", log.display()))?
        {
            child.shutdown(Duration::ZERO).await?;
            return if status.success() {
                Ok(())
            } else {
                Err(format!(
                    "{program} exited with {status}; see {}",
                    log.display()
                ))
            };
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(format!("{program} timed out; see {}", log.display()));
        }
        sleep(Duration::from_millis(25).min(remaining)).await;
    }
}

/// Owns the server group and waits for its port to be released at shutdown.
#[derive(Debug)]
pub struct SglangProcess {
    child: ManagedChild,
    port: u16,
    shutdown_timeout: Duration,
}

impl SglangProcess {
    /// Start a real SGLang server and await successful generation readiness.
    pub async fn start(
        config: &ServerConfig,
        implementation: Implementation,
        log: &Path,
        startup_timeout: Duration,
        shutdown_timeout: Duration,
    ) -> Result<Self, String> {
        config.validate()?;
        let python = config
            .python
            .as_ref()
            .ok_or("server.python must be prepared before starting SGLang")?;
        if startup_timeout.is_zero() {
            return Err("startup timeout must be positive".into());
        }
        if shutdown_timeout > Duration::from_secs(60) {
            return Err("shutdown timeout must not exceed 60 seconds".into());
        }
        let deadline = Instant::now()
            .checked_add(startup_timeout)
            .ok_or("startup timeout is too large")?;
        // Refuse an already occupied port instead of accepting another server's
        // /health_generate response as readiness for the child we just launched.
        let reservation = reserve_port(config.port)
            .map_err(|e| format!("cannot use parity server port {}: {e}", config.port))?;
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| format!("cannot create readiness client: {e}"))?;
        let stdout = File::create(log)
            .map_err(|e| format!("cannot create server log {}: {e}", log.display()))?;
        let stderr = stdout
            .try_clone()
            .map_err(|e| format!("cannot duplicate server log: {e}"))?;
        // Resolve explicit relative paths before changing the child's directory;
        // a bare executable name must still use PATH lookup.
        let python = if python.is_relative() && python.components().count() > 1 {
            std::env::current_dir()
                .map_err(|e| format!("cannot resolve Python executable: {e}"))?
                .join(python)
        } else {
            python.clone()
        };
        let mut command = isolated_command(python);
        command
            .args(["-m", "sglang.launch_server", "--model-path", &config.model])
            .args(&config.args)
            .args([
                "--enable-deterministic-inference",
                "--random-seed",
                &config.seed.to_string(),
                "--disable-radix-cache",
                "--host",
                "127.0.0.1",
                "--port",
                &config.port.to_string(),
            ])
            .stdin(Stdio::null())
            .stdout(stdout)
            .stderr(stderr);
        configure_environment(
            &mut command,
            config,
            implementation,
            std::env::vars_os().map(|(key, _)| key),
        );
        if let Some(directory) = &config.working_dir {
            command.current_dir(directory);
        }
        drop(reservation);
        let child = ManagedChild::spawn(&mut command).map_err(|e| {
            format!(
                "cannot start {} SGLang server: {e}",
                implementation.as_str()
            )
        })?;
        let mut process = Self {
            child,
            port: config.port,
            shutdown_timeout,
        };
        let ready_url = format!("{}/health_generate", process.base_url());
        loop {
            if let Some(status) = process
                .child
                .try_wait()
                .map_err(|e| format!("cannot inspect server process: {e}"))?
            {
                let error = format!(
                    "{} SGLang server exited before readiness ({status}); see {}",
                    implementation.as_str(),
                    log.display()
                );
                return Err(process.startup_error(error).await);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let error = format!(
                    "{} SGLang server startup timed out; see {}",
                    implementation.as_str(),
                    log.display()
                );
                return Err(process.startup_error(error).await);
            }
            if let Ok(Ok(response)) =
                tokio::time::timeout(remaining, client.get(&ready_url).send()).await
                && response.status().is_success()
            {
                // A child can exit while a readiness response is in flight.
                if process
                    .child
                    .try_wait()
                    .map_err(|e| format!("cannot inspect ready server: {e}"))?
                    .is_none()
                {
                    return Ok(process);
                }
                let error = format!("server exited during readiness; see {}", log.display());
                return Err(process.startup_error(error).await);
            }
            sleep(
                Duration::from_millis(50).min(deadline.saturating_duration_since(Instant::now())),
            )
            .await;
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    async fn startup_error(mut self, error: String) -> String {
        match self.shutdown().await {
            Ok(()) => error,
            Err(cleanup) => format!("{error}; cleanup failed: {cleanup}"),
        }
    }

    /// Terminate the entire group, escalating after the bounded grace period.
    pub async fn shutdown(&mut self) -> Result<(), String> {
        if self.child.child.is_none() {
            return Ok(());
        }
        self.child.shutdown(self.shutdown_timeout).await?;
        self.wait_port_release().await
    }

    async fn wait_port_release(&self) -> Result<(), String> {
        // SIGKILL delivery to workers is asynchronous. Wait briefly before the
        // next implementation starts, without mistaking TCP TIME_WAIT for a
        // surviving listener (the serving stack also uses SO_REUSEADDR).
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match reserve_port(self.port) {
                Ok(_) => return Ok(()),
                Err(error) if Instant::now() >= deadline => {
                    return Err(format!(
                        "server port {} was not released after shutdown: {error}",
                        self.port
                    ));
                }
                Err(_) => sleep(Duration::from_millis(20)).await,
            }
        }
    }
}

fn reserve_port(port: u16) -> io::Result<TcpListener> {
    // socket2 sets close-on-exec: concurrent server launches must not inherit
    // another run's reservation and keep its port occupied after this drop.
    let socket = socket2::Socket::new(socket2::Domain::IPV4, socket2::Type::STREAM, None)?;
    socket.set_reuse_address(true)?;
    let address = std::net::SocketAddrV4::new(std::net::Ipv4Addr::LOCALHOST, port);
    socket.bind(&std::net::SocketAddr::V4(address).into())?;
    Ok(socket.into())
}

fn signal_group(group: i32, signal: i32) -> io::Result<()> {
    // SAFETY: the child was spawned with process_group(0), so its PID is a
    // positive group ID owned by this object; negative IDs signal that group.
    if unsafe { libc::kill(-group, signal) } == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(error)
    }
}

fn group_exists(group: i32) -> bool {
    // SAFETY: signal 0 only checks existence/permission and changes no state.
    unsafe {
        libc::kill(-group, 0) == 0 || io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::PermissionsExt;

    fn config() -> ServerConfig {
        serde_json::from_value(serde_json::json!({"python": "/usr/bin/python3", "model": "model"}))
            .unwrap()
    }

    fn available_port() -> u16 {
        // Allocate in a child so concurrent parent forks cannot inherit the socket.
        let output = Command::new("python3")
            .args([
                "-c",
                "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])",
            ])
            .output()
            .unwrap();
        assert!(output.status.success(), "{:?}", output);
        String::from_utf8(output.stdout)
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    }

    #[test]
    fn reservation_is_closed_in_an_executed_child() {
        let reservation = reserve_port(0).unwrap();
        let port = reservation.local_addr().unwrap().port();
        // A different concurrent child can hold even a CLOEXEC descriptor until
        // it execs. Inspect this executed child's descriptor instead of treating
        // any immediate rebind failure as a leak from this particular child.
        let child = Command::new("python3")
            .args([
                "-c",
                r#"import socket, sys
try:
    inherited = socket.socket(fileno=int(sys.argv[1]))
except OSError:
    sys.exit(0)
assert inherited.getsockname() != ('127.0.0.1', int(sys.argv[2])), 'port reservation survived exec'
"#,
            ])
            .arg(reservation.as_raw_fd().to_string())
            .arg(port.to_string())
            .output()
            .unwrap();
        assert!(
            child.status.success(),
            "reservation descriptor check failed: {}",
            String::from_utf8_lossy(&child.stderr)
        );
    }

    #[test]
    fn rejects_control_overrides_and_unsupported_modes() {
        for argument in [
            "--model=other",
            "--model-p",
            "--host",
            "--por=9000",
            "--random-s=1",
            "--enable-deterministic",
            "--disable-radix",
            "--config",
            "--config-file=x",
            "--enable-metrics",
            "--enable-mfu-metrics",
            "--enable-forward-pass-metrics",
            "--export-metrics-to-file=x",
            "--enable-met",
            "--metrics-port=3",
            "--spec",
            "--speculative-draft-model=x",
            "--uno-lora-path",
            "--grpc-mode",
            "--smg-grpc-mode",
            "--grpc-port=8000",
            "--enable-grpc",
            "--encoder-only",
            "--use-ray",
            "-m",
            "-tp=2",
            "--",
        ] {
            let mut c = config();
            c.args.push(argument.into());
            assert!(c.validate().is_err(), "accepted {argument}");
        }
        for key in [
            "SGLANG_RUST_SERVER",
            "SGLANG_PORT",
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE",
            "SGLANG_ENABLE_METRICS_DEVICE_TIMER",
        ] {
            let mut c = config();
            c.env.insert(key.into(), "0".into());
            assert!(c.validate().is_err(), "accepted {key}");
        }
        let mut c = config();
        c.args = vec![
            "--attention-backend".into(),
            "triton".into(),
            "--model-loader-extra-config={}".into(),
            "--mlx-enable-sampling".into(),
        ];
        c.env.insert("SGLANG_USE_MLX".into(), "1".into());
        c.validate().unwrap();
    }

    #[test]
    fn inherited_controls_are_removed_before_setting_the_implementation() {
        let inherited = [
            "SGLANG_RUST_SERVER",
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE",
            "SGLANG_PORT",
            "SGLANG_GRPC_PORT",
            "SGLANG_ENABLE_GRPC",
            "SGLANG_ENABLE_METRICS_DEVICE_TIMER",
            "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH",
            "SGLANG_USE_MLX",
            "HF_HOME",
        ];
        for implementation in Implementation::ALL {
            let mut command = Command::new("python3");
            configure_environment(
                &mut command,
                &config(),
                implementation,
                inherited.into_iter().map(OsString::from),
            );
            let overrides: BTreeMap<_, _> = command
                .get_envs()
                .map(|(key, value)| {
                    (
                        key.to_string_lossy().into_owned(),
                        value.map(|v| v.to_string_lossy().into_owned()),
                    )
                })
                .collect();
            assert_eq!(
                overrides["SGLANG_RUST_SERVER"].as_deref(),
                Some(if implementation == Implementation::Rust {
                    "1"
                } else {
                    "0"
                })
            );
            assert_eq!(
                overrides["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"].as_deref(),
                Some("1")
            );
            for key in &inherited[2..7] {
                assert_eq!(overrides[*key], None);
            }
            assert!(!overrides.contains_key("SGLANG_USE_MLX"));
            assert!(!overrides.contains_key("HF_HOME"));
        }
    }

    #[test]
    fn port_reservation_rejects_a_live_listener() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(reserve_port(port).is_err());
    }

    #[test]
    fn incremental_mode_matches_the_actual_store_true_option() {
        for flag in ["--inc", "--incremental", "--incremental-streaming-output"] {
            let mut c = config();
            c.args = vec![flag.into()];
            c.validate().unwrap();
            assert!(c.incremental_output());
        }
        let mut c = config();
        c.args = vec!["--tokenizer-path=--incremental-streaming-output".into()];
        assert!(!c.incremental_output());
        c.args = vec!["--incremental-streaming-output=false".into()];
        assert!(!c.incremental_output());
        assert!(c.validate().is_err());
        assert!(
            serde_json::from_value::<ServerConfig>(serde_json::json!({
                "python": "python3", "model": "x", "unknown": true
            }))
            .is_err()
        );
    }

    #[tokio::test]
    async fn omitted_python_requires_environment_preparation() {
        let config: ServerConfig =
            serde_json::from_value(serde_json::json!({"model": "model"})).unwrap();
        config.validate().unwrap();
        let error = SglangProcess::start(
            &config,
            Implementation::Python,
            Path::new("unused.log"),
            Duration::from_secs(1),
            Duration::ZERO,
        )
        .await
        .unwrap_err();
        assert!(error.contains("must be prepared"), "{error}");
    }

    #[tokio::test]
    async fn python_paths_and_path_lookup_survive_a_different_server_working_dir() {
        let invocation_dir = std::env::current_dir().unwrap();
        let directory = tempfile::tempdir_in(&invocation_dir).unwrap();
        let relative_dir = directory.path().strip_prefix(&invocation_dir).unwrap();
        let executable = directory.path().join("fake-python");
        std::fs::write(&executable, "#!/bin/sh\npwd\nexit 17\n").unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
        let working_dir = directory.path().join("server-dir");
        std::fs::create_dir(&working_dir).unwrap();
        let mut config = config();
        config.working_dir = Some(relative_dir.join("server-dir"));
        config.env.insert(
            "PATH".into(),
            directory.path().to_string_lossy().into_owned(),
        );
        config.port = available_port();
        for python in [
            PathBuf::from(".").join(relative_dir).join("fake-python"),
            PathBuf::from("fake-python"),
        ] {
            config.python = Some(python);
            let log = directory.path().join("server.log");
            let error = SglangProcess::start(
                &config,
                Implementation::Python,
                &log,
                Duration::from_secs(5),
                Duration::from_millis(50),
            )
            .await
            .unwrap_err();
            assert!(
                error.contains("exited before readiness"),
                "did not execute {:?}: {error}",
                config.python
            );
            assert_eq!(
                Path::new(std::fs::read_to_string(log).unwrap().trim()),
                working_dir
            );
        }
    }

    struct Fixture {
        directory: tempfile::TempDir,
        config: ServerConfig,
    }

    impl Fixture {
        fn new(mode: &str) -> Self {
            let directory = tempfile::tempdir().unwrap();
            let executable = directory.path().join("fake-python");
            // A real process tree, without model, GPU, or Python dependencies.
            std::fs::write(
                &executable,
                r#"#!/bin/sh
printf '%s\n' "$$" > "$PARITY_PIDS/root"
printf '%s\n' "$@"
printf 'implementation=%s\n' "$SGLANG_RUST_SERVER"
if [ "$PARITY_MODE" = ignore ]; then trap '' TERM; fi
sh -c '
  sleep 120 &
  descendant=$!
  printf "%s\n" "$descendant" > "$PARITY_PIDS/grandchild"
  trap '\''kill "$descendant" 2>/dev/null; wait "$descendant" 2>/dev/null; exit 0'\'' TERM
  wait "$descendant"
' &
worker=$!
printf '%s\n' "$worker" > "$PARITY_PIDS/child"
trap 'kill "$worker" 2>/dev/null; wait "$worker" 2>/dev/null; exit 0' TERM
if [ "$PARITY_MODE" = early ]; then
  while [ ! -f "$PARITY_PIDS/grandchild" ]; do sleep 0.01; done
  exit 17
fi
if [ "$PARITY_MODE" = ignore ]; then trap '' TERM; fi
wait "$worker"
"#,
            )
            .unwrap();
            std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
            let mut config = config();
            config.python = Some(executable);
            config.port = available_port();
            config.env.insert(
                "PARITY_PIDS".into(),
                directory.path().to_string_lossy().into_owned(),
            );
            config.env.insert("PARITY_MODE".into(), mode.into());
            Self { directory, config }
        }

        fn log(&self) -> PathBuf {
            self.directory.path().join("server.log")
        }

        async fn await_tree(&self) {
            let deadline = Instant::now() + Duration::from_secs(5);
            while std::fs::read_to_string(self.directory.path().join("grandchild"))
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok())
                .is_none()
            {
                assert!(Instant::now() < deadline, "fake process tree did not start");
                sleep(Duration::from_millis(10)).await;
            }
        }

        async fn assert_dead(&self) {
            for name in ["root", "child", "grandchild"] {
                let pid: i32 = std::fs::read_to_string(self.directory.path().join(name))
                    .unwrap()
                    .trim()
                    .parse()
                    .unwrap();
                let deadline = Instant::now() + Duration::from_secs(3);
                while pid_running(pid) && Instant::now() < deadline {
                    sleep(Duration::from_millis(10)).await;
                }
                assert!(!pid_running(pid), "{name} process {pid} survived cleanup");
            }
        }

        fn command(&self) -> Command {
            let mut command = Command::new(self.config.python.as_ref().unwrap());
            command
                .envs(&self.config.env)
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            command
        }

        fn owner(&self, grace: Duration) -> SglangProcess {
            SglangProcess {
                child: ManagedChild::spawn(&mut self.command()).unwrap(),
                port: self.config.port,
                shutdown_timeout: grace,
            }
        }
    }

    fn pid_running(pid: i32) -> bool {
        // Orphan grandchildren are reaped by the OS, not by our direct-child
        // wait. A zombie is terminated and cannot hold files, ports, or GPU state.
        let state = Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output()
            .unwrap();
        let state = String::from_utf8_lossy(&state.stdout);
        !state.trim().is_empty() && !state.trim().starts_with('Z')
    }

    #[tokio::test]
    async fn setup_commands_append_logs_and_report_exit_status() {
        let directory = tempfile::tempdir().unwrap();
        let log = directory.path().join("setup.log");
        for status in [0, 17] {
            let mut command = Command::new("sh");
            command.args([
                "-c",
                &format!("echo output-{status}; echo error-{status} >&2; exit {status}"),
            ]);
            let result = run_command(&mut command, &log, Duration::from_secs(5)).await;
            if status == 0 {
                result.unwrap();
            } else {
                let error = result.unwrap_err();
                assert!(
                    error.contains("17") && error.contains("setup.log"),
                    "{error}"
                );
            }
        }
        let output = std::fs::read_to_string(log).unwrap();
        assert_eq!(output, "output-0\nerror-0\noutput-17\nerror-17\n");
    }

    #[tokio::test]
    async fn setup_exit_timeout_and_cancellation_clean_descendants() {
        for (mode, timeout, expected) in [
            ("early", Duration::from_secs(5), "exited with"),
            ("wait", Duration::from_secs(2), "timed out"),
        ] {
            let fixture = Fixture::new(mode);
            let error = run_command(&mut fixture.command(), &fixture.log(), timeout)
                .await
                .unwrap_err();
            assert!(error.contains(expected), "{error}");
            fixture.assert_dead().await;
        }
        let fixture = Fixture::new("wait");
        let mut command = fixture.command();
        let log = fixture.log();
        let task =
            tokio::spawn(
                async move { run_command(&mut command, &log, Duration::from_secs(60)).await },
            );
        fixture.await_tree().await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        fixture.assert_dead().await;
    }

    #[tokio::test]
    async fn startup_timeout_kills_the_whole_group() {
        let fixture = Fixture::new("wait");
        let error = SglangProcess::start(
            &fixture.config,
            Implementation::Python,
            &fixture.log(),
            Duration::from_secs(2),
            Duration::from_millis(50),
        )
        .await
        .unwrap_err();
        assert!(error.contains("timed out"), "{error}");
        fixture.assert_dead().await;
        let log = std::fs::read_to_string(fixture.log()).unwrap();
        assert!(log.contains("sglang.launch_server\n--model-path\nmodel\n"));
        assert!(log.contains(
            "--enable-deterministic-inference\n--random-seed\n42\n--disable-radix-cache\n"
        ));
        assert!(log.contains("implementation=0"));
    }

    #[tokio::test]
    async fn cancelled_start_reaps_the_server_and_kills_descendants() {
        let fixture = Fixture::new("wait");
        let config = fixture.config.clone();
        let log = fixture.log();
        let task = tokio::spawn(async move {
            SglangProcess::start(
                &config,
                Implementation::Rust,
                &log,
                Duration::from_secs(60),
                Duration::from_secs(1),
            )
            .await
        });
        fixture.await_tree().await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        fixture.assert_dead().await;
        assert!(
            std::fs::read_to_string(fixture.log())
                .unwrap()
                .contains("implementation=1")
        );
    }

    #[tokio::test]
    async fn early_exit_also_cleans_descendants() {
        let fixture = Fixture::new("early");
        let error = SglangProcess::start(
            &fixture.config,
            Implementation::Python,
            &fixture.log(),
            Duration::from_secs(5),
            Duration::from_millis(50),
        )
        .await
        .unwrap_err();
        assert!(error.contains("exited before readiness"), "{error}");
        fixture.assert_dead().await;
    }

    #[tokio::test]
    async fn cancelled_shutdown_still_cleans_the_group() {
        let fixture = Fixture::new("ignore");
        let mut owner = fixture.owner(Duration::from_secs(60));
        fixture.await_tree().await;
        let task = tokio::spawn(async move { owner.shutdown().await });
        // Abort before the normal graceful path can finish. Its moved owner
        // must clean up even though the shutdown future is never polled again.
        sleep(Duration::from_millis(10)).await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        fixture.assert_dead().await;
    }

    #[tokio::test]
    async fn shutdown_is_bounded_and_idempotent() {
        let fixture = Fixture::new("ignore");
        let mut owner = fixture.owner(Duration::from_millis(50));
        fixture.await_tree().await;
        tokio::time::timeout(Duration::from_secs(3), owner.shutdown())
            .await
            .unwrap()
            .unwrap();
        owner.shutdown().await.unwrap();
        fixture.assert_dead().await;
    }
}
