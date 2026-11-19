// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::WorkerId;
use dashmap::DashMap;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug)]
pub struct DecodeGate {
    limit: Option<NonZeroU32>,
    gates: DashMap<(WorkerId, Option<u32>), Arc<Semaphore>>,
}

impl DecodeGate {
    pub fn new(limit: Option<NonZeroU32>) -> Self {
        Self {
            limit,
            gates: DashMap::new(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.limit.is_some()
    }

    pub async fn admit(&self, worker: &WorkerId, rank: Option<u32>, slot: &DecodeSlot) {
        let Some(limit) = self.limit else { return };
        let gate = Arc::clone(
            &self
                .gates
                .entry((worker.clone(), rank))
                .or_insert_with(|| Arc::new(Semaphore::new(limit.get() as usize))),
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
