// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use futures::future::BoxFuture;

use crate::workers::Worker;

use super::admission::Admission;
use super::{Pick, PickError, PickRequest, Policy};

/// Samples two candidates and keeps the one under lower stage load.
#[derive(Debug, Default)]
pub struct PowerOfTwoPolicy {
    pub admission: Admission,
    pub fallback: Option<Arc<dyn Policy>>,
}

impl Policy for PowerOfTwoPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        Box::pin(async move {
            let _admitted = self.admission.before(engines, request)?;
            todo!("sample two admitted engines and compare load for request.stage")
        })
    }

    fn fallback(&self) -> Option<&dyn Policy> {
        self.fallback.as_deref()
    }
}
