//! Rust-owned offline Engine API foundations.
//!
//! The crate separates validated launch configuration, engine lifecycle, and
//! runtime ownership from the runtime implementation. Runtime backends must
//! implement [`EngineRuntime`]; the production scheduler/model backend is a
//! later migration phase and is deliberately not emulated here.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use thiserror::Error;

pub mod admission;
pub mod checkpoint;
pub mod model;
pub mod model_config;
pub mod runtime;
pub mod sampling;
pub mod scheduler;

/// Engine configuration that can be resolved without initializing a model.
///
/// This is the initial typed core, not the complete Python `ServerArgs`
/// surface. New fields are added as their launch and runtime semantics are
/// migrated and covered by parity tests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineConfig {
    /// Model repository identifier or local checkpoint path.
    pub model: String,
    /// Optional tokenizer override; defaults to the model's tokenizer.
    pub tokenizer: Option<String>,
    /// Optional local cache directory for model and tokenizer artifacts.
    pub cache_dir: Option<PathBuf>,
}

impl EngineConfig {
    /// Validate fields whose contract is independent of model metadata.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model.trim().is_empty() {
            return Err(ConfigError::EmptyModel);
        }
        if self.tokenizer.as_deref().is_some_and(str::is_empty) {
            return Err(ConfigError::EmptyTokenizer);
        }
        Ok(())
    }
}

/// Configuration validation errors.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ConfigError {
    #[error("model must be a non-empty repository identifier or path")]
    EmptyModel,
    #[error("tokenizer override must not be empty")]
    EmptyTokenizer,
}

/// Observable lifecycle of an offline Engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineState {
    /// No runtime has been started.
    Created,
    /// Runtime startup is in progress.
    Starting,
    /// Runtime is ready to accept work.
    Ready,
    /// Runtime is shutting down.
    Stopping,
    /// Runtime has shut down successfully.
    Stopped,
    /// Startup or runtime operation failed.
    Failed,
}

/// Runtime boundary owned by an [`Engine`].
///
/// Implementations own all runtime resources and must make `shutdown`
/// idempotent. The initial trait permits lifecycle behavior to be tested
/// independently while scheduler and model execution are migrated.
pub trait EngineRuntime: Send {
    /// Start runtime resources and return only after they are usable.
    fn start(&mut self, config: &EngineConfig) -> Result<(), RuntimeError>;

    /// Stop runtime resources and release owned processes/devices.
    fn shutdown(&mut self) -> Result<(), RuntimeError>;
}

/// Configuration for a supervised runtime worker process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessRuntimeConfig {
    /// Executable to launch.
    pub program: PathBuf,
    /// Arguments passed to the executable.
    pub args: Vec<String>,
    /// Readiness line emitted on standard output after the worker is usable.
    pub ready_message: String,
    /// Maximum time to wait for readiness.
    pub startup_timeout: Duration,
}

/// Runtime backend that supervises one worker process.
///
/// The child must print [`ProcessRuntimeConfig::ready_message`] as a complete
/// line on stdout after initialization. Stdout is reserved for this readiness
/// protocol; diagnostics should go to stderr. This is the process substrate
/// for the later scheduler/model runtime, not an implementation of that
/// runtime itself.
pub struct ProcessRuntime {
    config: ProcessRuntimeConfig,
    child: Option<Child>,
}

impl ProcessRuntime {
    /// Create a process runtime. The process is started by `Engine::start`.
    pub fn new(config: ProcessRuntimeConfig) -> Self {
        Self {
            config,
            child: None,
        }
    }

    fn stop_child(&mut self) -> Result<(), RuntimeError> {
        let Some(mut child) = self.child.take() else {
            return Ok(());
        };
        if child
            .try_wait()
            .map_err(|e| RuntimeError::new("shutdown", e.to_string()))?
            .is_none()
        {
            child
                .kill()
                .map_err(|e| RuntimeError::new("shutdown", e.to_string()))?;
        }
        child
            .wait()
            .map_err(|e| RuntimeError::new("shutdown", e.to_string()))?;
        Ok(())
    }
}

impl EngineRuntime for ProcessRuntime {
    fn start(&mut self, _: &EngineConfig) -> Result<(), RuntimeError> {
        if self.child.is_some() {
            return Err(RuntimeError::new("startup", "worker is already running"));
        }
        let mut child = Command::new(&self.config.program)
            .args(&self.config.args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| RuntimeError::new("startup", e.to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| RuntimeError::new("startup", "worker stdout was not piped"))?;
        let (line_tx, line_rx) = mpsc::channel();
        thread::Builder::new()
            .name("sglang-engine-readiness".into())
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    match line {
                        Ok(line) => {
                            if line_tx.send(Ok(line)).is_err() {
                                break;
                            }
                        }
                        Err(error) => {
                            let _ = line_tx.send(Err(error.to_string()));
                            break;
                        }
                    }
                }
            })
            .map_err(|e| RuntimeError::new("startup", e.to_string()))?;

        let deadline = Instant::now() + self.config.startup_timeout;
        let mut ready = false;
        while !ready {
            if let Some(status) = child
                .try_wait()
                .map_err(|e| RuntimeError::new("startup", e.to_string()))?
            {
                return Err(RuntimeError::new(
                    "startup",
                    format!("worker exited before readiness with status {status}"),
                ));
            }
            let now = Instant::now();
            if now >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                return Err(RuntimeError::new(
                    "startup",
                    format!(
                        "worker did not become ready within {:?}",
                        self.config.startup_timeout
                    ),
                ));
            }
            let wait = (deadline - now).min(Duration::from_millis(20));
            match line_rx.recv_timeout(wait) {
                Ok(Ok(line)) if line == self.config.ready_message => ready = true,
                Ok(Ok(_)) => {}
                Ok(Err(error)) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(RuntimeError::new("startup", error));
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    let _ = child.kill();
                    let status = child.wait().ok();
                    return Err(RuntimeError::new(
                        "startup",
                        format!("worker stdout closed before readiness (status {status:?})"),
                    ));
                }
            }
        }
        self.child = Some(child);
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), RuntimeError> {
        self.stop_child()
    }
}

impl Drop for ProcessRuntime {
    fn drop(&mut self) {
        let _ = self.stop_child();
    }
}

/// Runtime startup or shutdown error.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("engine runtime {operation} failed: {message}")]
pub struct RuntimeError {
    operation: &'static str,
    message: String,
}

impl RuntimeError {
    /// Create a runtime error at a backend boundary.
    pub fn new(operation: &'static str, message: impl Into<String>) -> Self {
        Self {
            operation,
            message: message.into(),
        }
    }
}

/// Offline Engine lifecycle handle.
pub struct Engine {
    config: EngineConfig,
    runtime: Box<dyn EngineRuntime>,
    state: EngineState,
}

impl Engine {
    /// Validate configuration, start the runtime, and return a ready engine.
    pub fn start(
        config: EngineConfig,
        mut runtime: Box<dyn EngineRuntime>,
    ) -> Result<Self, EngineError> {
        config.validate().map_err(EngineError::InvalidConfig)?;
        runtime.start(&config).map_err(EngineError::Runtime)?;
        Ok(Self {
            config,
            runtime,
            state: EngineState::Ready,
        })
    }

    /// Engine configuration used to start this instance.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Current lifecycle state.
    pub fn state(&self) -> EngineState {
        self.state
    }

    /// Shut down the runtime. Repeated calls after success are harmless.
    pub fn shutdown(&mut self) -> Result<(), EngineError> {
        if self.state == EngineState::Stopped {
            return Ok(());
        }
        self.state = EngineState::Stopping;
        match self.runtime.shutdown() {
            Ok(()) => {
                self.state = EngineState::Stopped;
                Ok(())
            }
            Err(error) => {
                self.state = EngineState::Failed;
                Err(EngineError::Runtime(error))
            }
        }
    }
}

/// Engine construction or lifecycle failure.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum EngineError {
    #[error(transparent)]
    InvalidConfig(ConfigError),
    #[error(transparent)]
    Runtime(RuntimeError),
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Default)]
    struct RuntimeState {
        starts: usize,
        stops: usize,
        fail_start: bool,
    }

    struct TestRuntime(Arc<Mutex<RuntimeState>>);

    impl EngineRuntime for TestRuntime {
        fn start(&mut self, _: &EngineConfig) -> Result<(), RuntimeError> {
            let mut state = self.0.lock().unwrap();
            state.starts += 1;
            if state.fail_start {
                return Err(RuntimeError::new("startup", "injected failure"));
            }
            Ok(())
        }

        fn shutdown(&mut self) -> Result<(), RuntimeError> {
            self.0.lock().unwrap().stops += 1;
            Ok(())
        }
    }

    fn config(model: &str) -> EngineConfig {
        EngineConfig {
            model: model.into(),
            tokenizer: None,
            cache_dir: None,
        }
    }

    #[test]
    fn validates_required_model_and_tokenizer() {
        assert_eq!(config(" ").validate(), Err(ConfigError::EmptyModel));
        let mut cfg = config("model");
        cfg.tokenizer = Some(String::new());
        assert_eq!(cfg.validate(), Err(ConfigError::EmptyTokenizer));
    }

    #[test]
    fn starts_ready_and_shutdown_is_idempotent() {
        let shared = Arc::new(Mutex::new(RuntimeState::default()));
        let mut engine = Engine::start(config("model"), Box::new(TestRuntime(shared.clone())))
            .expect("runtime starts");
        assert_eq!(engine.state(), EngineState::Ready);
        assert_eq!(engine.config().model, "model");
        engine.shutdown().unwrap();
        engine.shutdown().unwrap();
        assert_eq!(engine.state(), EngineState::Stopped);
        let state = shared.lock().unwrap();
        assert_eq!(state.starts, 1);
        assert_eq!(state.stops, 1);
    }

    #[test]
    fn invalid_config_does_not_start_runtime() {
        let shared = Arc::new(Mutex::new(RuntimeState::default()));
        let result = Engine::start(config(""), Box::new(TestRuntime(shared.clone())));
        assert!(matches!(result, Err(EngineError::InvalidConfig(_))));
        assert_eq!(shared.lock().unwrap().starts, 0);
    }

    #[test]
    fn startup_failure_is_reported() {
        let shared = Arc::new(Mutex::new(RuntimeState {
            fail_start: true,
            ..RuntimeState::default()
        }));
        let result = Engine::start(config("model"), Box::new(TestRuntime(shared)));
        assert!(matches!(result, Err(EngineError::Runtime(_))));
    }

    #[cfg(unix)]
    fn process_config(script: &str, timeout: Duration) -> ProcessRuntimeConfig {
        ProcessRuntimeConfig {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), script.into()],
            ready_message: "ENGINE_READY".into(),
            startup_timeout: timeout,
        }
    }

    #[cfg(unix)]
    #[test]
    fn process_runtime_waits_for_readiness_and_reaps_worker() {
        let runtime = ProcessRuntime::new(process_config(
            "printf 'ENGINE_READY\\n'; exec sleep 30",
            Duration::from_secs(2),
        ));
        let mut engine = Engine::start(config("model"), Box::new(runtime)).unwrap();
        assert_eq!(engine.state(), EngineState::Ready);
        engine.shutdown().unwrap();
        assert_eq!(engine.state(), EngineState::Stopped);
    }

    #[cfg(unix)]
    #[test]
    fn process_runtime_times_out_and_stops_worker() {
        let runtime =
            ProcessRuntime::new(process_config("exec sleep 30", Duration::from_millis(50)));
        let result = Engine::start(config("model"), Box::new(runtime));
        assert!(matches!(result, Err(EngineError::Runtime(_))));
    }

    #[cfg(unix)]
    #[test]
    fn process_runtime_reports_early_exit() {
        let runtime = ProcessRuntime::new(process_config("exit 7", Duration::from_secs(2)));
        let result = Engine::start(config("model"), Box::new(runtime));
        assert!(matches!(result, Err(EngineError::Runtime(_))));
    }
}
