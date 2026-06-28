// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Lightweight process-global phase gauges for diagnosing where an admission
//! slot's lifetime is spent. Each counter is the number of requests currently
//! in a given phase between claiming a per-worker slot and releasing it. They
//! are read (sampled) into the `phase_dispatch` log so a held-but-unconnected
//! slot can be attributed to a concrete phase without per-request tracing.
//!
//! Diagnostic-only.

use std::sync::atomic::{AtomicI64, Ordering};

/// Requests that hold an admission slot and are inside the synchronous handler
/// body (post-`acquire` up to returning the response: build-body + dispatch +
/// await-upstream-headers). High while requests are stuck before streaming.
pub static HANDLER_INFLIGHT: AtomicI64 = AtomicI64::new(0);

/// Requests currently blocked in the upstream `reqwest::send().await` (connect +
/// write body + await response headers). High → stuck talking to the engine.
pub static IN_SEND: AtomicI64 = AtomicI64::new(0);

/// Active SSE pumps (post-headers streaming). ≈ live engine streams.
pub static PUMP_INFLIGHT: AtomicI64 = AtomicI64::new(0);

/// RAII: `+1` on construction, `-1` on drop, for the named gauge.
pub struct PhaseGuard(&'static AtomicI64);

impl PhaseGuard {
    pub fn handler() -> Self {
        HANDLER_INFLIGHT.fetch_add(1, Ordering::Relaxed);
        Self(&HANDLER_INFLIGHT)
    }
    pub fn in_send() -> Self {
        IN_SEND.fetch_add(1, Ordering::Relaxed);
        Self(&IN_SEND)
    }
    pub fn pump() -> Self {
        PUMP_INFLIGHT.fetch_add(1, Ordering::Relaxed);
        Self(&PUMP_INFLIGHT)
    }
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Snapshot the three gauges as `(handler, in_send, pump)`.
pub fn snapshot() -> (i64, i64, i64) {
    (
        HANDLER_INFLIGHT.load(Ordering::Relaxed),
        IN_SEND.load(Ordering::Relaxed),
        PUMP_INFLIGHT.load(Ordering::Relaxed),
    )
}
