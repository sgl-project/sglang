use std::sync::{Arc, Mutex};

use dashmap::DashMap;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug)]
pub struct DecodeGate {
    limit: usize,
    gates: DashMap<String, Arc<Semaphore>>,
}

impl DecodeGate {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            gates: DashMap::new(),
        }
    }

    pub async fn admit(&self, prefill_url: &str, slot: &DecodeSlot) {
        let gate = Arc::clone(
            &self
                .gates
                .entry(prefill_url.to_string())
                .or_insert_with(|| Arc::new(Semaphore::new(self.limit))),
        );
        let permit = gate.acquire_owned().await.expect("gates are never closed");
        *slot.0.lock().unwrap() = Some(permit);
    }
}

#[derive(Debug, Clone, Default)]
pub struct DecodeSlot(Arc<Mutex<Option<OwnedSemaphorePermit>>>);

impl DecodeSlot {
    pub fn release(&self) {
        self.0.lock().unwrap().take();
    }
}
