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
/// Skips endpoints whose `conditions.ready` is explicitly `Some(false)`.
/// Per the EndpointSlice API spec, `conditions.ready = None` (absent) means
/// the endpoint **should be considered ready** (BLOCK 4 fix).
///
/// The worker URL is `http://<addr>:<port>` where port comes from
/// `EndpointSlice.ports[0].port`, defaulting to `30000` if absent.
///
/// The [`WorkerId`] includes the slice's namespace and name to avoid
/// cross-namespace IP collisions on overlay networks or with service-mesh
/// sidecars: `{ns}/{slice_name}/{addr}:{port}` (IMPORTANT-10 fix).
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

    // IMPORTANT-10: include namespace + slice name in the ID so that pods
    // with identical IPs in different namespaces (overlay nets, sidecars)
    // never share a WorkerId.
    let ns = es.metadata.namespace.as_deref().unwrap_or("");
    let slice_name = es.metadata.name.as_deref().unwrap_or("");

    let mut out = Vec::new();
    for ep in es.endpoints.iter() {
        // BLOCK 4: None → true per EndpointSlice API spec.
        let is_ready = ep.conditions.as_ref().and_then(|c| c.ready).unwrap_or(true);
        if !is_ready {
            continue;
        }
        for addr in &ep.addresses {
            let url = format!("http://{addr}:{port}");
            let id = WorkerId(format!("{ns}/{slice_name}/{addr}:{port}"));
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
/// **BLOCK 2:** if `cfg.namespace` is empty, `Api::all(client)` is used so
/// that the watcher is truly cluster-wide. `Api::namespaced(client, "")` is
/// namespace-scoped to the empty-named namespace, which is almost never what
/// callers intend.
///
/// **BLOCK 1:** state is tracked as a two-level map
/// `per_slice: HashMap<SliceUid, HashMap<WorkerId, WorkerSpec>>`.
/// K8s auto-shards Services with >100 endpoints and CNIs often shard per
/// AZ, emitting multiple `EndpointSlice` objects per Service.  Using a
/// flat state map meant each slice's event would silently drop all workers
/// from sibling slices.  Now each slice's submap is replaced independently;
/// the global union is recomputed from all submaps and diffed against
/// `prev_union`.
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

    // BLOCK 2: empty namespace → cluster-wide watch.
    let api: Api<EndpointSlice> = if cfg.namespace.is_empty() {
        Api::all(client)
    } else {
        Api::namespaced(client, &cfg.namespace)
    };

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
        // BLOCK 1: per-slice state to handle multi-slice Services correctly.
        // Key: EndpointSlice UID (String).  Value: workers in that slice.
        let mut per_slice: HashMap<String, HashMap<WorkerId, WorkerSpec>> = HashMap::new();
        let mut prev_union: HashMap<WorkerId, WorkerSpec> = HashMap::new();

        let stream = watcher(api, watcher_cfg).applied_objects();
        tokio::pin!(stream);
        while let Some(event) = stream.next().await {
            match event {
                Ok(es) => {
                    let slice_uid = es.metadata.uid.clone().unwrap_or_default();
                    let next_slice: HashMap<WorkerId, WorkerSpec> =
                        match classify_mode(&es, &mode) {
                            Some(wm) => extract_workers(&es, wm)
                                .into_iter()
                                .map(|w| (w.id.clone(), w))
                                .collect(),
                            None => HashMap::new(),
                        };
                    per_slice.insert(slice_uid, next_slice);

                    // Recompute union across all slices.
                    let union: HashMap<WorkerId, WorkerSpec> = per_slice
                        .values()
                        .flat_map(|s| s.iter().map(|(k, v)| (k.clone(), v.clone())))
                        .collect();

                    // Diff union against prev_union.
                    for (id, spec) in &union {
                        if let Some(p) = prev_union.get(id) {
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
                    for id in prev_union.keys() {
                        if !union.contains_key(id) {
                            let _ = tx.send(DiscoveryEvent::Removed { id: id.clone() }).await;
                        }
                    }
                    prev_union = union;
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

    /// Helper: build a minimal EndpointSlice with predictable metadata.
    fn make_slice(addrs: &[&str], port: i32, ready: bool) -> EndpointSlice {
        make_slice_full(addrs, port, ready, "testns", "test-slice", &[])
    }

    fn make_slice_ns(
        addrs: &[&str],
        port: i32,
        ready: bool,
        ns: &str,
        slice_name: &str,
    ) -> EndpointSlice {
        make_slice_full(addrs, port, ready, ns, slice_name, &[])
    }

    fn make_slice_with_labels(
        addrs: &[&str],
        port: i32,
        ready: bool,
        labels: &[(&str, &str)],
    ) -> EndpointSlice {
        make_slice_full(addrs, port, ready, "testns", "test-slice", labels)
    }

    fn make_slice_full(
        addrs: &[&str],
        port: i32,
        ready: bool,
        ns: &str,
        slice_name: &str,
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
                name: if slice_name.is_empty() {
                    None
                } else {
                    Some(slice_name.into())
                },
                namespace: if ns.is_empty() { None } else { Some(ns.into()) },
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
        assert_eq!(ws[0].id.0, "testns/test-slice/10.0.0.1:30000");
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

    /// BLOCK 4: conditions.ready = None must default to ready=true per
    /// EndpointSlice API spec ("undefined → endpoint should be considered
    /// ready").
    #[test]
    fn is_ready_none_defaults_to_ready() {
        let mut s = make_slice(&["10.0.0.1"], 30000, false /* overridden below */);
        s.endpoints[0].conditions.as_mut().unwrap().ready = None;
        let ws = extract_workers(&s, WorkerMode::Plain);
        assert_eq!(
            ws.len(),
            1,
            "endpoint with None ready should be treated as ready"
        );
    }

    /// IMPORTANT 10: WorkerId must include namespace + slice name to avoid
    /// cross-namespace IP collisions.
    #[test]
    fn worker_id_includes_namespace_and_slice_name() {
        let mut s = make_slice(&["10.0.0.1"], 30000, true);
        s.metadata.namespace = Some("prod".to_string());
        s.metadata.name = Some("svc-abc-xyz".to_string());
        let ws = extract_workers(&s, WorkerMode::Plain);
        assert_eq!(ws[0].id.0, "prod/svc-abc-xyz/10.0.0.1:30000");
    }

    /// IMPORTANT 10: WorkerId with no namespace (cluster-scoped) should
    /// produce a valid ID with empty-namespace prefix (leading slash).
    #[test]
    fn worker_id_handles_missing_namespace() {
        // make_slice_ns with empty ns leaves metadata.namespace = None.
        let s = make_slice_ns(&["10.0.0.1"], 30000, true, "", "my-slice");
        let ws = extract_workers(&s, WorkerMode::Plain);
        assert!(
            ws[0].id.0.contains("10.0.0.1:30000"),
            "id must contain addr:port"
        );
        assert!(
            ws[0].id.0.starts_with('/'),
            "empty ns => id starts with '/', got: {}",
            ws[0].id.0
        );
    }
}
