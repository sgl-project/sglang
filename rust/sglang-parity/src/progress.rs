//! Report long-running stages without spawning tasks or changing cancellation.

use std::future::Future;
use std::path::Path;
use std::time::Duration;

use tokio::time::{Instant, sleep};

/// Emit a stage immediately and every ten seconds until its operation finishes.
pub(crate) async fn track<T>(stage: &str, log: &Path, operation: impl Future<Output = T>) -> T {
    tracing::info!(stage, log = %log.display(), "Starting");
    let started = Instant::now();
    tokio::pin!(operation);
    loop {
        tokio::select! {
            result = &mut operation => return result,
            _ = sleep(Duration::from_secs(10)) => {
                tracing::info!(stage, elapsed_secs = started.elapsed().as_secs(), log = %log.display(), "Still running");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};

    use tracing::instrument::WithSubscriber;

    #[derive(Clone, Default)]
    struct Output(Arc<Mutex<Vec<u8>>>);

    impl Write for Output {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[tokio::test(start_paused = true)]
    async fn pending_operations_emit_periodic_progress_and_preserve_errors() {
        let output = Output::default();
        let writer = output.clone();
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let result = track("Installing dependencies", Path::new("setup.log"), async {
            sleep(Duration::from_secs(21)).await;
            Err::<(), _>("installation failed")
        })
        .with_subscriber(subscriber)
        .await;
        assert_eq!(result, Err("installation failed"));
        let log = String::from_utf8(output.0.lock().unwrap().clone()).unwrap();
        assert_eq!(log.matches("Starting").count(), 1);
        assert_eq!(log.matches("Still running").count(), 2);
        for expected in [
            "Installing dependencies",
            "setup.log",
            "elapsed_secs=10",
            "elapsed_secs=20",
        ] {
            assert!(log.contains(expected), "{log}");
        }
    }
}
