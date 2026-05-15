// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod factory;
pub mod power_of_two;
pub mod random;
pub mod round_robin;

use crate::discovery::ModelId;
use crate::workers::Worker;
use dashmap::DashMap;
use std::sync::Arc;

/// Selection input — carries enough context for cache-aware (M4) without
/// reshaping the trait. M2 policies only use `workers`.
pub struct SelectionContext<'a> {
    pub model: &'a ModelId,
    pub request_body: Option<&'a [u8]>,
}

pub trait Policy: Send + Sync + std::fmt::Debug {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>>;
}

#[derive(Debug, Default)]
pub struct PolicyRegistry {
    by_model: DashMap<ModelId, Arc<dyn Policy>>,
}

impl PolicyRegistry {
    pub fn insert(&self, model: ModelId, policy: Arc<dyn Policy>) {
        self.by_model.insert(model, policy);
    }

    pub fn get(&self, model: &ModelId) -> Option<Arc<dyn Policy>> {
        self.by_model.get(model).map(|p| p.clone())
    }
}
