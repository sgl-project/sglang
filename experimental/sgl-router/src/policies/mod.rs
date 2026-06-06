// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod active_load;
pub mod cache_aware_zmq;
pub mod factory;
pub mod kv_events;
pub mod load_based;
pub mod power_of_two;
pub mod random;
pub mod registry;
pub mod round_robin;

use crate::discovery::ModelId;
use crate::server::metrics::MetricsRegistry;
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
    routing_key: Option<&'a str>,
}

impl<'a> SelectionContext<'a> {
    pub fn new(model: &'a ModelId, request_body: Option<&'a [u8]>) -> Self {
        Self {
            model,
            request_body,
            routing_key: None,
        }
    }

    pub fn with_routing_key(
        model: &'a ModelId,
        request_body: Option<&'a [u8]>,
        routing_key: Option<&'a str>,
    ) -> Self {
        Self {
            model,
            request_body,
            routing_key,
        }
    }

    pub fn model(&self) -> &ModelId {
        self.model
    }

    pub fn request_body(&self) -> Option<&[u8]> {
        self.request_body
    }

    pub fn routing_key(&self) -> Option<&str> {
        self.routing_key
    }
}

pub trait Policy: Send + Sync + std::fmt::Debug {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>>;

    /// Attach the process metrics registry after construction. Default is a
    /// no-op — only policies that emit metrics (cache-aware-zmq's
    /// `sgl_router_overlap_blocks`) override it. Mirrors
    /// `ActiveLoadRegistry::attach_metrics`: the registry is built after the
    /// policies, so it is injected here rather than passed to the constructor.
    fn attach_metrics(&self, _metrics: Arc<MetricsRegistry>) {}
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

    /// Inject the metrics registry into every registered policy. Called once
    /// at startup (after the registry is built) so metrics-emitting policies
    /// can record into the shared registry.
    pub fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        for entry in self.by_model.iter() {
            entry.value().attach_metrics(Arc::clone(&metrics));
        }
    }
}
