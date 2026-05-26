// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod active_load;
pub mod cache_aware_zmq;
pub mod factory;
pub mod kv_events;
pub mod power_of_two;
pub mod random;
pub mod registry;
pub mod round_robin;

use crate::discovery::ModelId;
use crate::workers::Worker;
use dashmap::DashMap;
use std::sync::Arc;

/// Selection input — carries the request body so that cache-aware policies
/// can hash prefix tokens without reshaping the [`Policy`] trait.  Today's
/// policies (round-robin, random, power-of-two) only read `workers`.
///
/// Constructed via [`Self::new`]; accessors expose immutable references so
/// callers cannot mutate the model id or swap in a different body without
/// going through the constructor.
pub struct SelectionContext<'a> {
    model: &'a ModelId,
    request_body: Option<&'a [u8]>,
}

impl<'a> SelectionContext<'a> {
    pub fn new(model: &'a ModelId, request_body: Option<&'a [u8]>) -> Self {
        Self {
            model,
            request_body,
        }
    }

    pub fn model(&self) -> &ModelId {
        self.model
    }

    pub fn request_body(&self) -> Option<&[u8]> {
        self.request_body
    }
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
