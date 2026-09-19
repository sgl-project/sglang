// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::state::load_monitor::engine_load::{EngineLoadSnapshot, EngineLoadTable};
use crate::state::LoadView;

/// Observations shared by one policy invocation, its admission, and nested fallback.
/// Constructed by `Policy::pick`, never stored on a policy or supplied by a bucket.
#[derive(Debug, Default)]
pub struct PickContext {
    // Each view retains its source Arc, so its address cannot be reused in this attempt.
    loads: Mutex<HashMap<usize, LoadView>>,
}

impl PickContext {
    /// Capture on first use of this source and reuse that snapshot for the attempt.
    /// Consumers supply their own injected handle; distinct sources never alias.
    pub fn load(&self, source: &Arc<EngineLoadTable>) -> Arc<EngineLoadSnapshot> {
        self.loads
            .lock()
            .expect("pick context load cache poisoned")
            .entry(Arc::as_ptr(source) as usize)
            .or_insert_with(|| LoadView::new(Arc::clone(source)))
            .snapshot()
    }
}
