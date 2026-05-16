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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub id: WorkerId,
    pub url: String,
    pub mode: WorkerMode,
    pub model_ids: Vec<ModelId>,
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
        };
        let s = serde_json::to_string(&w).unwrap();
        let d: WorkerSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(w, d);
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
        });
        let s = serde_json::to_string(&e).unwrap();
        let d: DiscoveryEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(e, d);
    }
}
