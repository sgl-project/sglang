// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};
use std::time::Instant;

use super::load_monitor::engine_load::{EngineLoadSnapshot, EngineLoadTable};
use crate::discovery::WorkerMode;
use crate::policies::admission::{compare_decode_pressure, compare_prefill_pressure};
use crate::workers::Worker;

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

    /// The less pressured of two engines under the stage's comparison; `left` on a tie.
    pub fn lower_pressure<'e>(
        &self,
        left: &'e Arc<Worker>,
        right: &'e Arc<Worker>,
        stage: WorkerMode,
    ) -> &'e Arc<Worker> {
        let compare = match stage {
            WorkerMode::Decode => compare_decode_pressure,
            WorkerMode::Plain | WorkerMode::Prefill => compare_prefill_pressure,
        };
        if compare(left, right, Some(self.snapshot())).is_gt() {
            right
        } else {
            left
        }
    }
}
