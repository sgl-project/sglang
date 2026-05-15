// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Mutex;

/// Stub for Task 6. Replaced with the full state machine then.
#[derive(Debug, Default)]
pub struct CircuitBreaker {
    inner: Mutex<()>,
}

impl CircuitBreaker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allow(&self) -> bool {
        let _g = self.inner.lock();
        true
    }

    pub fn record_success(&self) {}
    pub fn record_failure(&self) {}
}
