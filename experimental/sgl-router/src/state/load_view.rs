// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;
use std::time::Instant;

use super::engine_load::{EngineLoadSnapshot, EngineLoadTable};

/// One lazily captured report snapshot, shared by every pick in a stage's pass.
#[derive(Debug)]
pub struct LoadView<'a> {
    table: &'a EngineLoadTable,
    snapshot: OnceLock<EngineLoadSnapshot>,
}

impl<'a> LoadView<'a> {
    pub fn new(table: &'a EngineLoadTable) -> Self {
        Self {
            table,
            snapshot: OnceLock::new(),
        }
    }

    pub fn snapshot(&self) -> &EngineLoadSnapshot {
        self.snapshot
            .get_or_init(|| self.table.capture_snapshot(Instant::now()))
    }
}
