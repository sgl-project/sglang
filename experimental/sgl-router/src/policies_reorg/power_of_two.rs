// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use futures::future::BoxFuture;

use crate::state::load_monitor::engine_load::EngineLoadTable;
use crate::state::LoadView;
use crate::workers::Worker;

use super::admission::Admission;
use super::{Pick, PickError, PickRequest, Policy};

/// Samples two candidates and keeps the one under lower stage load.
#[derive(Debug)]
pub struct PowerOfTwoPolicy {
    /// Shared application state; snapshots are local to each pick.
    engine_load: Arc<EngineLoadTable>,
    pub admission: Admission,
    pub fallback: Option<Arc<dyn Policy>>,
}

impl PowerOfTwoPolicy {
    pub fn new(engine_load: Arc<EngineLoadTable>) -> Self {
        Self {
            engine_load,
            admission: Admission::default(),
            fallback: None,
        }
    }
}

impl Policy for PowerOfTwoPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        Box::pin(async move {
            let _load = LoadView::new(&self.engine_load);
            let _admitted = self.admission.before(engines, request)?;
            todo!("sample two admitted engines and compare load for request.stage")
        })
    }

    fn fallback(&self) -> Option<&dyn Policy> {
        self.fallback.as_deref()
    }
}
