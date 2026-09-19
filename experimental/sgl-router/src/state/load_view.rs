// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};
use std::time::Instant;

use super::load_monitor::engine_load::{EngineLoadSnapshot, EngineLoadTable};

/// One source's lazily captured snapshot, retained for a selection attempt.
#[derive(Debug)]
pub struct LoadView {
    table: Arc<EngineLoadTable>,
    snapshot: OnceLock<Arc<EngineLoadSnapshot>>,
}

impl LoadView {
    pub fn new(table: Arc<EngineLoadTable>) -> Self {
        Self {
            table,
            snapshot: OnceLock::new(),
        }
    }

    pub fn snapshot(&self) -> Arc<EngineLoadSnapshot> {
        self.snapshot
            .get_or_init(|| Arc::new(self.table.capture_snapshot(Instant::now())))
            .clone()
    }
}
