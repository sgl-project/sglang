use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::debug;

pub struct PeriodicTask {
    name: &'static str,
    shutdown_flag: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for PeriodicTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeriodicTask")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl PeriodicTask {
    /// Spawn a background thread that periodically executes a task.
    pub fn spawn<F>(interval_secs: u64, name: &'static str, task: F) -> Self
    where
        F: Fn() + Send + 'static,
    {
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown_flag);

        let handle = thread::spawn(move || {
            let check_interval_ms = 100u64;
            let total_sleep_ms = interval_secs * 1000;

            loop {
                // Sleep in small increments, checking shutdown flag periodically
                let mut slept_ms = 0u64;
                while slept_ms < total_sleep_ms {
                    if shutdown_clone.load(Ordering::Relaxed) {
                        debug!("{} thread received shutdown signal", name);
                        return;
                    }
                    thread::sleep(Duration::from_millis(check_interval_ms));
                    slept_ms += check_interval_ms;
                }

                // Check shutdown before starting task
                if shutdown_clone.load(Ordering::Relaxed) {
                    debug!("{} thread received shutdown signal", name);
                    return;
                }

                task();
            }
        });

        Self {
            name,
            shutdown_flag,
            handle: Some(handle),
        }
    }
}

impl Drop for PeriodicTask {
    fn drop(&mut self) {
        self.shutdown_flag.store(true, Ordering::Relaxed);

        if let Some(handle) = self.handle.take() {
            match handle.join() {
                Ok(()) => debug!("{} thread shut down cleanly", self.name),
                Err(_) => debug!("{} thread panicked during shutdown", self.name),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Instant;

    #[test]
    fn test_periodic_task_executes() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let _task = PeriodicTask::spawn(1, "test", move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Wait for at least one execution
        thread::sleep(Duration::from_millis(1200));
        assert!(counter.load(Ordering::SeqCst) >= 1);

        // Task will be stopped on drop
    }

    #[test]
    fn test_periodic_task_responds_to_shutdown() {
        let task = PeriodicTask::spawn(60, "test", || {
            // Long interval task
        });

        let start = Instant::now();
        drop(task);
        let elapsed = start.elapsed();

        // Should shutdown within ~200ms (2 check intervals), not 60 seconds
        assert!(elapsed < Duration::from_millis(500));
    }
}

