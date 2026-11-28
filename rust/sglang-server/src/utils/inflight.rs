//! Shared RAII in-flight counter. [`DrainState`](crate::api_server::core::DrainState)
//! and the P/D pipeline guard on [`Request`](crate::message::request::Request)
//! both use it so `message` does not depend on `api_server`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::Notify;

/// Count of work a drain waits for. Each [`InFlight`] increments it until drop.
#[derive(Default)]
pub struct InFlightCount {
    count: AtomicUsize,
    /// Wakes drain waiters when the count or a related flag may have changed.
    pub changed: Notify,
}

impl InFlightCount {
    /// Count one unit of work until the guard drops.
    pub fn track(self: &Arc<Self>) -> InFlight {
        self.count.fetch_add(1, Ordering::SeqCst);
        InFlight(self.clone())
    }

    pub fn get(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
}

/// One unit of in-flight work. Dropping the last one wakes [`InFlightCount::changed`].
pub struct InFlight(Arc<InFlightCount>);

impl std::fmt::Debug for InFlight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InFlight")
            .field("count", &self.0.get())
            .finish()
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        if self.0.count.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.0.changed.notify_waiters();
        }
    }
}
