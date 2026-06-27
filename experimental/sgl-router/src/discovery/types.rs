// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

/// Opaque worker identifier. Wraps a string so callsites can't confuse it
/// with other string types (e.g. `ModelId`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerId(pub String);

impl std::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Opaque model identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Prefill/Decode/Plain role of a worker.
///
/// Serialises as `"plain"`, `"prefill"`, `"decode"` (snake_case).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerMode {
    Plain,
    Prefill,
    Decode,
}

/// Immutable worker description emitted by a discovery backend.
///
/// Backends emit [`DiscoveryEvent::Added`] carrying a `WorkerSpec` when a
/// new worker becomes available, and [`DiscoveryEvent::Removed`] when it
/// leaves.
///
/// `bootstrap_port` is the SGLang disagg bootstrap server port for
/// prefill workers (set via `--disaggregation-bootstrap-port` at worker
/// startup). Resolved from each worker's `/server_info` response (see
/// [`crate::workers::introspect`]); discovery backends seed it as
/// `None`. `None` for decode and plain workers — they don't own a
/// bootstrap server. The router copies the selected prefill worker's
/// `bootstrap_host`/`bootstrap_port` plus a random `bootstrap_room`
/// u64 onto every PD-disagg request body so the prefill engine can
/// match incoming KV-transfer requests from the decode peer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub id: WorkerId,
    pub url: String,
    pub mode: WorkerMode,
    pub model_ids: Vec<ModelId>,
    #[serde(default)]
    pub bootstrap_port: Option<u16>,
    /// Minimum request priority this worker will accept. `Some(N)` means
    /// the worker is eligible only for requests whose effective priority
    /// is `>= N`; `None` (the default) means it accepts any request.
    ///
    /// Used to isolate heterogeneous capacity — e.g. an RTX-6000 worker
    /// registered with `min_priority = Some(100)` only serves
    /// high-priority production traffic and never internal/long requests,
    /// which carry priority `0`. Seeded by the static-urls discovery
    /// backend (`url@min_priority=N`) today; the k8s backend currently
    /// always sets `None` (pod-label seeding is a future addition). NOT
    /// overridden by `/server_info` introspection (which only resolves
    /// mode/bootstrap) nor dropped on reconcile re-introspection.
    #[serde(default)]
    pub min_priority: Option<i64>,
}

/// Event produced by a discovery backend and consumed by `WorkerManager`.
///
/// Tagged with `"event"` for JSON clarity:
/// ```json
/// {"event":"added","id":"w1","url":"http://…","mode":"plain","model_ids":["m"]}
/// {"event":"removed","id":"w1"}
/// {"event":"mode_changed","id":"w1","mode":"decode"}
/// ```
///
/// The `Added` variant wraps the full [`WorkerSpec`]; the others carry only
/// what changed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum DiscoveryEvent {
    Added(WorkerSpec),
    Removed {
        id: WorkerId,
    },
    /// Used by the k8s backend when only the PD label flips (rare).
    ModeChanged {
        id: WorkerId,
        mode: WorkerMode,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_spec_serde_round_trip() {
        let w = WorkerSpec {
            id: WorkerId("w1".into()),
            url: "http://10.0.0.1:30000".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("qwen".into())],
            bootstrap_port: None,
            min_priority: None,
        };
        let s = serde_json::to_string(&w).unwrap();
        let d: WorkerSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(w, d);
    }

    #[test]
    fn worker_spec_with_bootstrap_port_round_trip() {
        let w = WorkerSpec {
            id: WorkerId("p1".into()),
            url: "http://10.0.0.1:30000".into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("qwen".into())],
            bootstrap_port: Some(8997),
            min_priority: None,
        };
        let s = serde_json::to_string(&w).unwrap();
        assert!(s.contains("\"bootstrap_port\":8997"));
        let d: WorkerSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(w, d);
    }

    #[test]
    fn worker_spec_with_min_priority_round_trip() {
        let w = WorkerSpec {
            id: WorkerId("rtx1".into()),
            url: "http://10.0.0.9:30000".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("glm".into())],
            bootstrap_port: None,
            min_priority: Some(100),
        };
        let s = serde_json::to_string(&w).unwrap();
        assert!(s.contains("\"min_priority\":100"));
        let d: WorkerSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(w, d);
    }

    #[test]
    fn worker_spec_deserializes_with_missing_min_priority() {
        // Older configs / hand-written JSON without the field should still
        // parse — min_priority defaults to None (worker accepts any request).
        let json = r#"{"id":"w","url":"http://x","mode":"plain","model_ids":["m"]}"#;
        let w: WorkerSpec = serde_json::from_str(json).unwrap();
        assert_eq!(w.min_priority, None);
    }

    #[test]
    fn worker_spec_deserializes_with_missing_bootstrap_port() {
        // Older configs / hand-written JSON without the field should
        // still parse — bootstrap_port defaults to None for non-PD
        // deployments.
        let json = r#"{"id":"w","url":"http://x","mode":"plain","model_ids":["m"]}"#;
        let w: WorkerSpec = serde_json::from_str(json).unwrap();
        assert_eq!(w.bootstrap_port, None);
    }

    #[test]
    fn worker_mode_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&WorkerMode::Plain).unwrap(),
            "\"plain\""
        );
        assert_eq!(
            serde_json::to_string(&WorkerMode::Prefill).unwrap(),
            "\"prefill\""
        );
        assert_eq!(
            serde_json::to_string(&WorkerMode::Decode).unwrap(),
            "\"decode\""
        );
    }

    #[test]
    fn discovery_event_round_trip() {
        let e = DiscoveryEvent::Added(WorkerSpec {
            id: WorkerId("w1".into()),
            url: "http://x:30000".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m1".into())],
            bootstrap_port: None,
            min_priority: None,
        });
        let s = serde_json::to_string(&e).unwrap();
        let d: DiscoveryEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(e, d);
    }
}
