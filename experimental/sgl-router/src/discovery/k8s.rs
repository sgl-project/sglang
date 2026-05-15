// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes EndpointSlice discovery backend.
//!
//! Watches `EndpointSlice` resources by label selector in the configured
//! namespace, diffing against in-memory state and emitting
//! [`DiscoveryEvent`]s for the worker manager.

use crate::config::{K8sDiscoveryConfig, K8sDiscoveryMode};
use crate::discovery::{DiscoveryEvent, WorkerId, WorkerMode, WorkerSpec};
use anyhow::{Context, Result};
use futures::StreamExt;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{
    api::Api,
    runtime::{watcher, WatchStreamExt},
    Client,
};
use std::collections::{BTreeMap, HashMap};
use tokio::sync::mpsc;

/// Decide which [`WorkerMode`] an `EndpointSlice` should be assigned, based
/// on the configured discovery mode.
///
/// * `Plain` mode — every slice yields `Some(WorkerMode::Plain)`. The
///   server-side label selector has already filtered to the right set.
/// * `PD` mode — the slice's labels are matched against `prefill_selector`
///   and `decode_selector` (in that order). Returns `Some(Prefill)` /
///   `Some(Decode)` on the first match, `None` if neither matches.  The
///   server-side watch in PD mode is unfiltered (label selectors for the
///   two roles may not be mergeable into one Kubernetes selector), so this
///   client-side classification is the gate that drops irrelevant slices.
fn classify_mode(es: &EndpointSlice, mode: &K8sDiscoveryMode) -> Option<WorkerMode> {
    match mode {
        K8sDiscoveryMode::Plain { .. } => Some(WorkerMode::Plain),
        K8sDiscoveryMode::PdDisaggregation {
            prefill_selector,
            decode_selector,
        } => {
            let labels = es.metadata.labels.as_ref().cloned().unwrap_or_default();
            if labels_match_selector(&labels, prefill_selector) {
                Some(WorkerMode::Prefill)
            } else if labels_match_selector(&labels, decode_selector) {
                Some(WorkerMode::Decode)
            } else {
                None
            }
        }
    }
}

/// Match a Kubernetes label set against a comma-separated equality
/// selector like `"app=sglang,role=prefill"`.
///
/// Supports the equality-based subset of the K8s label-selector grammar:
/// `key=value` (and the alias `key==value`). Set-based operators (`in`,
/// `notin`, presence tests) are not supported — fall back to the server-
/// side selector in plain mode if you need them.
fn labels_match_selector(labels: &BTreeMap<String, String>, selector: &str) -> bool {
    for term in selector.split(',') {
        let term = term.trim();
        if term.is_empty() {
            continue;
        }
        // Accept both `=` and `==` for equality.
        let (key, expected) = if let Some((k, v)) = term.split_once("==") {
            (k.trim(), v.trim())
        } else if let Some((k, v)) = term.split_once('=') {
            (k.trim(), v.trim())
        } else {
            // Set-based or presence-only term — not supported here.
            return false;
        };
        match labels.get(key) {
            Some(v) if v == expected => {}
            _ => return false,
        }
    }
    true
}

/// Convert an `EndpointSlice` into a list of [`WorkerSpec`]s with the
/// supplied [`WorkerMode`].
///
/// Skips endpoints that are not ready (`conditions.ready != Some(true)`).
/// The worker URL is `http://<addr>:<port>` where port comes from
/// `EndpointSlice.ports[0].port`, defaulting to `30000` if absent.
///
/// `model_ids` is intentionally left empty — model membership is resolved
/// by the worker manager via `/server_info` introspection after the
/// `Added` event is emitted.
fn extract_workers(es: &EndpointSlice, mode: WorkerMode) -> Vec<WorkerSpec> {
    let port = es
        .ports
        .as_ref()
        .and_then(|p| p.first())
        .and_then(|p| p.port)
        .unwrap_or(30000);

    let mut out = Vec::new();
    for ep in es.endpoints.iter() {
        let is_ready = ep
            .conditions
            .as_ref()
            .and_then(|c| c.ready)
            .unwrap_or(false);
        if !is_ready {
            continue;
        }
        for addr in &ep.addresses {
            let url = format!("http://{addr}:{port}");
            let id = WorkerId(format!("{addr}:{port}"));
            out.push(WorkerSpec {
                id,
                url,
                mode,
                model_ids: Vec::new(),
            });
        }
    }
    out
}

/// Spawn the k8s discovery task.
///
/// Connects to the cluster via `KUBECONFIG` / in-cluster service account,
/// then watches `EndpointSlice` resources in `cfg.namespace`. Diffs against
/// in-memory state and emits [`DiscoveryEvent`]s to `tx`.
///
/// In plain mode the configured `label_selector` is pushed to the server
/// side; in PD mode the watch is unfiltered and per-slice classification
/// happens client-side via `classify_mode`.
///
/// The returned `JoinHandle` runs until the channel is closed or the watcher
/// stream ends (server restart, etc.).
pub async fn spawn(
    cfg: K8sDiscoveryConfig,
    tx: mpsc::Sender<DiscoveryEvent>,
) -> Result<tokio::task::JoinHandle<()>> {
    let mode = cfg.mode().context("validate k8s discovery selectors")?;

    let client = Client::try_default()
        .await
        .context("kube client default config")?;
    let api: Api<EndpointSlice> = Api::namespaced(client, &cfg.namespace);

    // Plain mode pushes the single selector to the server side so the LIST
    // is already filtered.  PD mode leaves the server-side selector empty
    // because the prefill/decode selectors may not be expressible as one
    // K8s label-selector — classification happens client-side per slice
    // via `classify_mode`.
    let server_side_selector = match &mode {
        K8sDiscoveryMode::Plain { label_selector } => label_selector.clone(),
        K8sDiscoveryMode::PdDisaggregation { .. } => String::new(),
    };
    let watcher_cfg = watcher::Config::default().labels(&server_side_selector);

    let handle = tokio::spawn(async move {
        let mut state: HashMap<WorkerId, WorkerSpec> = HashMap::new();
        let stream = watcher(api, watcher_cfg).applied_objects();
        tokio::pin!(stream);
        while let Some(event) = stream.next().await {
            match event {
                Ok(es) => {
                    let next: HashMap<WorkerId, WorkerSpec> = match classify_mode(&es, &mode) {
                        Some(wm) => extract_workers(&es, wm)
                            .into_iter()
                            .map(|w| (w.id.clone(), w))
                            .collect(),
                        None => HashMap::new(),
                    };
                    let prev = std::mem::take(&mut state);
                    for (id, spec) in &next {
                        if let Some(p) = prev.get(id) {
                            if p.mode != spec.mode {
                                let _ = tx
                                    .send(DiscoveryEvent::ModeChanged {
                                        id: id.clone(),
                                        mode: spec.mode,
                                    })
                                    .await;
                            }
                            if p.url != spec.url || p.model_ids != spec.model_ids {
                                let _ = tx.send(DiscoveryEvent::Removed { id: id.clone() }).await;
                                let _ = tx.send(DiscoveryEvent::Added(spec.clone())).await;
                            }
                        } else {
                            let _ = tx.send(DiscoveryEvent::Added(spec.clone())).await;
                        }
                    }
                    for id in prev.keys() {
                        if !next.contains_key(id) {
                            let _ = tx.send(DiscoveryEvent::Removed { id: id.clone() }).await;
                        }
                    }
                    state = next;
                }
                Err(e) => {
                    tracing::warn!("k8s watcher error: {e:?}");
                }
            }
        }
    });
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions, EndpointPort};
    use kube::core::ObjectMeta;

    fn make_slice(addrs: &[&str], port: i32, ready: bool) -> EndpointSlice {
        make_slice_with_labels(addrs, port, ready, &[])
    }

    fn make_slice_with_labels(
        addrs: &[&str],
        port: i32,
        ready: bool,
        labels: &[(&str, &str)],
    ) -> EndpointSlice {
        let label_map: BTreeMap<String, String> = labels
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        EndpointSlice {
            metadata: ObjectMeta {
                labels: if label_map.is_empty() {
                    None
                } else {
                    Some(label_map)
                },
                name: Some("slice".into()),
                ..Default::default()
            },
            address_type: "IPv4".into(),
            endpoints: vec![Endpoint {
                addresses: addrs.iter().map(|a| (*a).to_string()).collect(),
                conditions: Some(EndpointConditions {
                    ready: Some(ready),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ports: Some(vec![EndpointPort {
                port: Some(port),
                ..Default::default()
            }]),
        }
    }

    fn plain_mode() -> K8sDiscoveryMode {
        K8sDiscoveryMode::Plain {
            label_selector: "app=sglang".into(),
        }
    }

    fn pd_mode() -> K8sDiscoveryMode {
        K8sDiscoveryMode::PdDisaggregation {
            prefill_selector: "app=sglang,role=prefill".into(),
            decode_selector: "app=sglang,role=decode".into(),
        }
    }

    #[test]
    fn classify_mode_plain_returns_some_plain_for_any_slice() {
        let s = make_slice(&["10.0.0.1"], 30000, true);
        assert_eq!(classify_mode(&s, &plain_mode()), Some(WorkerMode::Plain));
    }

    #[test]
    fn classify_mode_pd_returns_prefill_when_labels_match_prefill_selector() {
        let s = make_slice_with_labels(
            &["10.0.0.1"],
            30000,
            true,
            &[("app", "sglang"), ("role", "prefill")],
        );
        assert_eq!(classify_mode(&s, &pd_mode()), Some(WorkerMode::Prefill));
    }

    #[test]
    fn classify_mode_pd_returns_decode_when_labels_match_decode_selector() {
        let s = make_slice_with_labels(
            &["10.0.0.1"],
            30000,
            true,
            &[("app", "sglang"), ("role", "decode")],
        );
        assert_eq!(classify_mode(&s, &pd_mode()), Some(WorkerMode::Decode));
    }

    #[test]
    fn classify_mode_pd_returns_none_when_no_selector_matches() {
        let s = make_slice_with_labels(
            &["10.0.0.1"],
            30000,
            true,
            &[("app", "sglang"), ("role", "router")],
        );
        assert!(classify_mode(&s, &pd_mode()).is_none());

        // No labels at all => still None in PD mode (every selector term
        // requires a key/value).
        let s = make_slice(&["10.0.0.1"], 30000, true);
        assert!(classify_mode(&s, &pd_mode()).is_none());
    }

    #[test]
    fn extract_workers_emits_workers_with_supplied_mode_and_empty_model_ids() {
        let s = make_slice(&["10.0.0.1"], 30000, true);
        let ws = extract_workers(&s, WorkerMode::Plain);
        assert_eq!(ws.len(), 1);
        assert_eq!(ws[0].mode, WorkerMode::Plain);
        assert_eq!(ws[0].url, "http://10.0.0.1:30000");
        assert!(
            ws[0].model_ids.is_empty(),
            "model_ids are resolved via /server_info, not at extract time"
        );

        // The mode argument flows through unchanged.
        let ws = extract_workers(&s, WorkerMode::Prefill);
        assert_eq!(ws[0].mode, WorkerMode::Prefill);
        let ws = extract_workers(&s, WorkerMode::Decode);
        assert_eq!(ws[0].mode, WorkerMode::Decode);
    }

    #[test]
    fn skips_not_ready_endpoints() {
        let s = make_slice(&["10.0.0.1"], 30000, false);
        assert!(extract_workers(&s, WorkerMode::Plain).is_empty());
    }
}
