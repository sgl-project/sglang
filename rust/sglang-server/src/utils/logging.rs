//! Process-wide `tracing` setup for the embedded server.

use std::sync::OnceLock;

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;

/// Keeps the non-blocking log writer's background thread alive for the process
/// lifetime (dropping the guard would stop log delivery).
static LOG_GUARD: OnceLock<WorkerGuard> = OnceLock::new();

/// Install the global `tracing` subscriber once; a no-op if the host process
/// (or an earlier call) already set one.
pub fn init_tracing() {
    let (writer, guard) = tracing_appender::non_blocking(std::io::stdout());
    let _ = LOG_GUARD.set(guard);
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(writer)
        .try_init();
}

/// The log lines this thread emits while the returned value is alive, formatted
/// as [`init_tracing`]'s are; for tests that check what the server logs.
///
/// One process-wide subscriber writes each thread's lines to that thread's
/// capture, if any. A per-test scoped subscriber would race: a log site first
/// reached on another thread while no scoped subscriber exists can cache
/// "never log" after this thread's subscriber registered, dropping its lines.
#[cfg(any(test, feature = "test-support"))]
pub fn capture_logs() -> CapturedLogs {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(|| ThreadCapture)
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("tests install no other global subscriber");
    });
    let lines = CapturedLines::default();
    CAPTURE.with(|capture| *capture.borrow_mut() = Some(lines.clone()));
    CapturedLogs { lines }
}

#[cfg(any(test, feature = "test-support"))]
thread_local! {
    static CAPTURE: std::cell::RefCell<Option<CapturedLines>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(any(test, feature = "test-support"))]
pub struct CapturedLogs {
    lines: CapturedLines,
}

#[cfg(any(test, feature = "test-support"))]
impl CapturedLogs {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.lines.0.lock().unwrap()).into_owned()
    }

    /// The `rejected request` lines logged so far.
    pub fn rejections(&self) -> Vec<String> {
        self.text()
            .lines()
            .filter(|line| line.contains("rejected request"))
            .map(str::to_owned)
            .collect()
    }
}

#[cfg(any(test, feature = "test-support"))]
impl Drop for CapturedLogs {
    fn drop(&mut self) {
        CAPTURE.with(|capture| {
            let mut capture = capture.borrow_mut();
            if capture
                .as_ref()
                .is_some_and(|lines| std::sync::Arc::ptr_eq(&lines.0, &self.lines.0))
            {
                *capture = None;
            }
        });
    }
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Default)]
struct CapturedLines(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

/// Writes to the current thread's capture; drops the line if there is none.
#[cfg(any(test, feature = "test-support"))]
struct ThreadCapture;

#[cfg(any(test, feature = "test-support"))]
impl std::io::Write for ThreadCapture {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        CAPTURE.with(|capture| {
            if let Some(lines) = &*capture.borrow() {
                lines.0.lock().unwrap().extend_from_slice(bytes);
            }
        });
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
