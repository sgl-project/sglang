use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::debug;

/// Spawn a background thread that periodically executes a task.
///
/// The thread sleeps in small increments (100ms) to check the shutdown flag frequently,
/// ensuring responsive shutdown behavior.
///
/// # Arguments
/// * `interval_secs` - How often to run the task (in seconds)
/// * `shutdown_flag` - Atomic flag to signal thread termination
/// * `task_name` - Name for logging purposes
/// * `task` - The closure to execute periodically
///
/// # Returns
/// A JoinHandle for the spawned thread
pub fn spawn_periodic_task<F>(
    interval_secs: u64,
    shutdown_flag: Arc<AtomicBool>,
    task_name: &'static str,
    task: F,
) -> JoinHandle<()>
where
    F: Fn() + Send + 'static,
{
    thread::spawn(move || {
        let check_interval_ms = 100u64;
        let total_sleep_ms = interval_secs * 1000;

        loop {
            // Sleep in small increments, checking shutdown flag periodically
            let mut slept_ms = 0u64;
            while slept_ms < total_sleep_ms {
                if shutdown_flag.load(Ordering::Relaxed) {
                    debug!("{} thread received shutdown signal", task_name);
                    return;
                }
                thread::sleep(Duration::from_millis(check_interval_ms));
                slept_ms += check_interval_ms;
            }

            // Check shutdown before starting task
            if shutdown_flag.load(Ordering::Relaxed) {
                debug!("{} thread received shutdown signal", task_name);
                return;
            }

            task();
        }
    })
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
        let shutdown = Arc::new(AtomicBool::new(false));

        let handle = spawn_periodic_task(1, Arc::clone(&shutdown), "test", move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Wait for at least one execution
        thread::sleep(Duration::from_millis(1200));
        assert!(counter.load(Ordering::SeqCst) >= 1);

        // Shutdown
        shutdown.store(true, Ordering::Relaxed);
        handle.join().unwrap();
    }

    #[test]
    fn test_periodic_task_responds_to_shutdown() {
        let shutdown = Arc::new(AtomicBool::new(false));

        let handle = spawn_periodic_task(60, Arc::clone(&shutdown), "test", || {
            // Long interval task
        });

        // Signal shutdown immediately
        shutdown.store(true, Ordering::Relaxed);

        let start = Instant::now();
        handle.join().unwrap();
        let elapsed = start.elapsed();

        // Should shutdown within ~200ms (2 check intervals), not 60 seconds
        assert!(elapsed < Duration::from_millis(500));
    }
}

