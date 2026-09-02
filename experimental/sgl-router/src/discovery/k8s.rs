// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes EndpointSlice discovery backend.
//!
//! Watches `EndpointSlice` resources by label selector in the configured
//! namespace, diffing against in-memory state and emitting
//! [`DiscoveryEvent`]s for the worker manager.

use crate::config::{K8sDiscoveryConfig, K8sDiscoveryMode};
use crate::discovery::{DiscoveryEvent, WorkerId, WorkerMode, WorkerSpec};
use crate::policies::kv_events::bootstrap::PeerRegistry;
use anyhow::{Context, Result};
use futures::{Stream, StreamExt};
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{api::Api, runtime::watcher, Client};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
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
/// `not in`, presence tests) are not supported — fall back to the
/// server-side selector in plain mode if you need them.
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
/// the endpoint should be considered ready.
///
/// The worker URL is `http://<addr>:<port>` where port comes from
/// `EndpointSlice.ports[0].port`, defaulting to `30000` if absent.
///
/// The [`WorkerId`] is `{ns}/{uid}` where `uid` is the pod's K8s UID
/// from `endpoint.target_ref.uid`. Pod UIDs are globally unique per
/// pod incarnation, so when a pod dies and a new pod gets the same IP
/// (kubelet IP reuse on a busy podCIDR), the router sees a fresh
/// `WorkerId` and emits a clean Removed→Added cycle — old breaker /
/// active-load state is shed instead of being mis-applied to the new
/// pod. For manually-created EndpointSlices that lack `target_ref`
/// (rare in real clusters but common in unit tests), falls back to
/// `{ns}/{slice_name}/{addr}:{port}`.
///
/// `model_ids` is intentionally left empty — model membership is resolved
/// by the worker manager via `/server_info` introspection after the
/// `Added` event is emitted.
fn extract_workers(
    es: &EndpointSlice,
    mode: WorkerMode,
    transfer_group_label: &str,
) -> Vec<WorkerSpec> {
    // PD transfer group from the slice label (one Service per group is the
    // usual LWS / RBG layout). Empty value == unlabelled == ungrouped.
    let transfer_group = es
        .metadata
        .labels
        .as_ref()
        .and_then(|l| l.get(transfer_group_label))
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .map(str::to_string);
    let port = es
        .ports
        .as_ref()
        .and_then(|p| p.first())
        .and_then(|p| p.port)
        .unwrap_or(30000);

    let ns = es.metadata.namespace.as_deref().unwrap_or("");
    let slice_name = es.metadata.name.as_deref().unwrap_or("");

    let mut out = Vec::new();
    for ep in es.endpoints.iter() {
        let is_ready = ep.conditions.as_ref().and_then(|c| c.ready).unwrap_or(true);
        if !is_ready {
            continue;
        }
        let pod_uid: Option<&str> = ep.target_ref.as_ref().and_then(|r| r.uid.as_deref());
        for addr in &ep.addresses {
            let url = format!("http://{addr}:{port}");
            let id = match pod_uid {
                Some(uid) => WorkerId(format!("{ns}/{uid}")),
                None => WorkerId(format!("{ns}/{slice_name}/{addr}:{port}")),
            };
            // bootstrap_port stays `None` here on purpose. The final
            // `WorkerMode` and the bootstrap port are both filled in by
            // the worker manager from each worker's `/server_info`
            // body (`disaggregation_mode` + `disaggregation_bootstrap_port`,
            // both fields on SGLang's ServerArgs and already surfaced
            // via `**asdict(server_args)` in the response). EndpointSlice
            // carries neither, but doesn't need to — see
            // `src/workers/introspect.rs` for the extraction and
            // `register_one` in `src/workers/manager.rs` for the override.
            out.push(WorkerSpec {
                id,
                url,
                mode,
                model_ids: Vec::new(),
                bootstrap_port: None,
                transfer_group: transfer_group.clone(),
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
/// Stable per-slice key.
///
/// Uses `metadata.uid` when present (the normal case in a real cluster), and
/// falls back to `{ns}/{name}` for slices without a UID (rare: CR shims,
/// tests, certain fake/in-memory backends).  The fallback is unique per slice
/// because EndpointSlice names are unique within a namespace.
fn slice_key(es: &EndpointSlice) -> String {
    if let Some(uid) = es.metadata.uid.as_deref() {
        if !uid.is_empty() {
            return uid.to_string();
        }
    }
    let ns = es.metadata.namespace.as_deref().unwrap_or("");
    let name = es.metadata.name.as_deref().unwrap_or("");
    format!("{ns}/{name}")
}

/// Send all `Added` / `Removed` / `ModeChanged` events that bring the
/// consumer from `prev_union` to the recomputed union of `per_slice`.
///
/// Returns `Err` on the first send failure (consumer dropped); the caller is
/// expected to exit the watcher loop.  Updates `prev_union` in place to the
/// new union on success.
async fn emit_diff(
    tx: &mpsc::Sender<DiscoveryEvent>,
    per_slice: &HashMap<String, HashMap<WorkerId, WorkerSpec>>,
    prev_union: &mut HashMap<WorkerId, WorkerSpec>,
) -> Result<(), mpsc::error::SendError<DiscoveryEvent>> {
    let union: HashMap<WorkerId, WorkerSpec> = per_slice
        .values()
        .flat_map(|s| s.iter().map(|(k, v)| (k.clone(), v.clone())))
        .collect();

    for (id, spec) in &union {
        match prev_union.get(id) {
            Some(prev) => {
                if prev.mode != spec.mode {
                    tx.send(DiscoveryEvent::ModeChanged {
                        id: id.clone(),
                        mode: spec.mode,
                    })
                    .await?;
                }
                if prev.url != spec.url
                    || prev.model_ids != spec.model_ids
                    || prev.transfer_group != spec.transfer_group
                {
                    tx.send(DiscoveryEvent::Removed { id: id.clone() }).await?;
                    tx.send(DiscoveryEvent::Added(spec.clone())).await?;
                }
            }
            None => {
                tx.send(DiscoveryEvent::Added(spec.clone())).await?;
            }
        }
    }

    let dropped: Vec<WorkerId> = prev_union
        .keys()
        .filter(|id| !union.contains_key(id))
        .cloned()
        .collect();
    for id in dropped {
        tx.send(DiscoveryEvent::Removed { id }).await?;
    }

    *prev_union = union;
    Ok(())
}

/// Drive the event-processing loop for a stream of `watcher::Event`s.
///
/// Handles the full set of `kube` watcher events:
/// * `Init` / `InitApply` / `InitDone` — full-LIST resync.  Objects are
///   buffered until `InitDone`, then swapped in atomically.  Any slices that
///   were present before but not seen during the resync are diffed out as
///   `Removed`, which catches deletions that occurred while the watcher was
///   disconnected.
/// * `Apply` — single-object upsert into `per_slice`.
/// * `Delete` — single-object removal from `per_slice`; the diff emits
///   `Removed` for every worker that lived in that slice.
///
/// On any watcher error the kube-runtime watcher auto-restarts and emits a
/// new `Init` cycle, so the resync logic above is what reconciles state
/// after transient errors — no separate state reset is required.
///
/// The loop returns when the input stream ends (logged at WARN) or when the
/// consumer drops the receiving end of `tx` (logged at INFO).
#[cfg(test)]
async fn process_events<S>(stream: S, tx: mpsc::Sender<DiscoveryEvent>, mode: K8sDiscoveryMode)
where
    S: Stream<Item = Result<watcher::Event<EndpointSlice>, watcher::Error>> + Unpin,
{
    process_events_with_group_label(
        stream,
        tx,
        mode,
        crate::config::default_transfer_group_label(),
    )
    .await
}

/// [`process_events`] with an explicit transfer-group label key.
async fn process_events_with_group_label<S>(
    mut stream: S,
    tx: mpsc::Sender<DiscoveryEvent>,
    mode: K8sDiscoveryMode,
    transfer_group_label: String,
) where
    S: Stream<Item = Result<watcher::Event<EndpointSlice>, watcher::Error>> + Unpin,
{
    let mut per_slice: HashMap<String, HashMap<WorkerId, WorkerSpec>> = HashMap::new();
    let mut prev_union: HashMap<WorkerId, WorkerSpec> = HashMap::new();
    let mut init_buffer: Option<HashMap<String, HashMap<WorkerId, WorkerSpec>>> = None;

    fn workers_for_slice(
        es: &EndpointSlice,
        mode: &K8sDiscoveryMode,
        transfer_group_label: &str,
    ) -> HashMap<WorkerId, WorkerSpec> {
        match classify_mode(es, mode) {
            Some(wm) => extract_workers(es, wm, transfer_group_label)
                .into_iter()
                .map(|w| (w.id.clone(), w))
                .collect(),
            None => HashMap::new(),
        }
    }

    while let Some(event) = stream.next().await {
        let result = match event {
            Ok(watcher::Event::Init) => {
                init_buffer = Some(HashMap::new());
                Ok(())
            }
            Ok(watcher::Event::InitApply(es)) => {
                let key = slice_key(&es);
                let workers = workers_for_slice(&es, &mode, &transfer_group_label);
                if let Some(buf) = init_buffer.as_mut() {
                    buf.insert(key, workers);
                    Ok(())
                } else {
                    // Defensive: InitApply outside an Init cycle.  Treat as Apply.
                    per_slice.insert(key, workers);
                    emit_diff(&tx, &per_slice, &mut prev_union).await
                }
            }
            Ok(watcher::Event::InitDone) => {
                if let Some(buf) = init_buffer.take() {
                    per_slice = buf;
                    emit_diff(&tx, &per_slice, &mut prev_union).await
                } else {
                    Ok(())
                }
            }
            Ok(watcher::Event::Apply(es)) => {
                let key = slice_key(&es);
                let workers = workers_for_slice(&es, &mode, &transfer_group_label);
                per_slice.insert(key, workers);
                emit_diff(&tx, &per_slice, &mut prev_union).await
            }
            Ok(watcher::Event::Delete(es)) => {
                let key = slice_key(&es);
                per_slice.remove(&key);
                emit_diff(&tx, &per_slice, &mut prev_union).await
            }
            Err(e) => {
                tracing::warn!(error = ?e, "k8s watcher error; awaiting auto-restart");
                Ok(())
            }
        };
        if result.is_err() {
            tracing::info!("k8s discovery: event channel closed; exiting watcher");
            return;
        }
    }
    tracing::warn!("k8s watcher stream ended; discovery task exiting");
}

/// Empty `cfg.namespace` triggers a cluster-wide watch via `Api::all(client)`.
/// `Api::namespaced(client, "")` is namespace-scoped to the empty-named
/// namespace, which is almost never what callers intend.
///
/// State is tracked per-slice as `HashMap<SliceKey, HashMap<WorkerId,
/// WorkerSpec>>`.  K8s auto-shards Services with >100 endpoints and CNIs
/// often shard per AZ, so multiple `EndpointSlice` objects can exist per
/// Service; a flat state map would let each slice's event silently drop all
/// workers from sibling slices.  The global union is recomputed from all
/// submaps on every event and diffed against `prev_union` to produce
/// `DiscoveryEvent`s.
///
/// The returned `JoinHandle` runs until the channel is closed or the watcher
/// stream ends (server restart, RBAC change, etc.).
pub async fn spawn(
    cfg: K8sDiscoveryConfig,
    tx: mpsc::Sender<DiscoveryEvent>,
) -> Result<tokio::task::JoinHandle<()>> {
    // The mode was resolved + validated at construction (`resolve_mode` in
    // `Cli::build_discovery`); just destructure it here.
    // `peer_selector` drives a separate watch (`spawn_peer_watch`), not the
    // worker stream this function serves.
    let K8sDiscoveryConfig {
        namespace,
        mode,
        peer_selector: _,
        transfer_group_label,
    } = cfg;

    let client = Client::try_default()
        .await
        .context("kube client default config")?;

    let api: Api<EndpointSlice> = if namespace.is_empty() {
        Api::all(client)
    } else {
        Api::namespaced(client, &namespace)
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

    // Log the resolved namespace + selector(s) at startup. We can't
    // verify the namespace exists (the router's RBAC covers
    // endpointslices/services/pods, not namespaces, and a correct
    // namespace legitimately has zero matching workers until they come
    // up), so a typo'd `--service-discovery-namespace` silently watches
    // an empty namespace. Surfacing the watch target here lets an
    // operator spot the typo in the first log lines instead of only
    // discovering it via later `no workers available` request failures.
    let namespace_display: &str = if namespace.is_empty() {
        "<all namespaces>"
    } else {
        &namespace
    };
    match &mode {
        K8sDiscoveryMode::Plain { label_selector } => tracing::info!(
            namespace = %namespace_display,
            label_selector = %label_selector,
            "k8s discovery starting (plain mode); a wrong namespace or selector matches zero EndpointSlices"
        ),
        K8sDiscoveryMode::PdDisaggregation {
            prefill_selector,
            decode_selector,
        } => tracing::info!(
            namespace = %namespace_display,
            prefill_selector = %prefill_selector,
            decode_selector = %decode_selector,
            "k8s discovery starting (PD mode); a wrong namespace or selector matches zero EndpointSlices"
        ),
    }

    let handle = tokio::spawn(async move {
        let stream = watcher(api, watcher_cfg);
        tokio::pin!(stream);
        process_events_with_group_label(stream, tx, mode, transfer_group_label).await;
    });
    Ok(handle)
}

/// Port a sibling router replica serves HTTP on, used to build peer URLs.
///
/// Taken from the EndpointSlice's first port when present, falling back to this
/// — the router's own default listen port — when the slice omits it.
const DEFAULT_ROUTER_PORT: i32 = 30000;

/// Extract sibling replica base URLs from an EndpointSlice, excluding this pod.
///
/// Only `ready` endpoints are considered: an unready sibling is either starting
/// up (cold tree, nothing worth pulling) or draining. `self_ip` is filtered out
/// because a replica bootstrapping from itself would just graft its own empty
/// tree and waste the deadline.
fn extract_peers(es: &EndpointSlice, self_ip: Option<&str>, want_ipv6: bool) -> Vec<String> {
    // Skip the address family this router does not listen on. Without this, a
    // dual-stack Service yields BOTH families per sibling: the unusable half is
    // permanently unreachable (and `Unreachable` peers are retried, so they eat
    // the bootstrap deadline), and `POD_IP` only carries the primary address, so
    // the replica fails to exclude ITSELF on the secondary family — which latches
    // `ever_had_peers` and stops a genuinely lone replica from concluding it is
    // alone.
    let family_matches = match es.address_type.as_str() {
        "IPv6" => want_ipv6,
        "IPv4" => !want_ipv6,
        // FQDN or anything unrecognised: keep it, since we cannot tell.
        _ => true,
    };
    if !family_matches {
        return Vec::new();
    }
    let port = es
        .ports
        .as_ref()
        .and_then(|ports| ports.first())
        .and_then(|p| p.port)
        .unwrap_or(DEFAULT_ROUTER_PORT);
    es.endpoints
        .iter()
        .filter(|ep| {
            // Per the EndpointSlice API, absent `ready` means ready.
            ep.conditions.as_ref().and_then(|c| c.ready).unwrap_or(true)
        })
        .flat_map(|ep| ep.addresses.iter())
        .filter(|addr| self_ip != Some(addr.as_str()))
        .map(|addr| {
            // A bare IPv6 literal is not a parseable authority, so on a
            // dual-stack cluster every peer URL from the IPv6 slice would fail
            // to connect and burn the bootstrap deadline.
            if addr.contains(':') {
                format!("http://[{addr}]:{port}")
            } else {
                format!("http://{addr}:{port}")
            }
        })
        .collect()
}

/// Watch this router's own EndpointSlices and keep `peers` current.
///
/// Separate from [`spawn`] because it answers a different question — "which
/// sibling replicas can I pull a cache snapshot from?" rather than "which
/// engines can I route to" — and because it must keep working when the worker
/// watch is in PD mode with client-side classification.
///
/// The task exits quietly when the watcher stream ends; peer bootstrap then
/// degrades to whatever peers were last seen, and eventually to cold boots.
/// That is deliberately non-fatal: routing does not depend on this watch.
pub async fn spawn_peer_watch(
    namespace: String,
    label_selector: String,
    peers: Arc<PeerRegistry>,
    want_ipv6: bool,
) -> Result<tokio::task::JoinHandle<()>> {
    let client = Client::try_default()
        .await
        .context("kube client default config for peer watch")?;
    let api: Api<EndpointSlice> = if namespace.is_empty() {
        Api::all(client)
    } else {
        Api::namespaced(client, &namespace)
    };
    // Downward-API pod IP, so a replica does not list itself as a peer. Absent
    // POD_IP is survivable: the replica may fetch a snapshot from itself, which
    // is useless but harmless (its own tree is empty, so it reports
    // `producer_ready: false` until settled and is skipped).
    let self_ip = std::env::var("POD_IP").ok();
    if self_ip.is_none() {
        tracing::warn!(
            "kv-bootstrap: POD_IP is unset, so this replica cannot exclude itself from \
             its peer list; set it from the downward API for a cleaner peer set"
        );
    }

    tracing::info!(
        namespace = %if namespace.is_empty() { "<all namespaces>" } else { &namespace },
        label_selector = %label_selector,
        self_ip = %self_ip.as_deref().unwrap_or("<unset>"),
        "kv-bootstrap: peer watch starting",
    );

    let watcher_cfg = watcher::Config::default().labels(&label_selector);
    let handle = tokio::spawn(async move {
        let stream = watcher(api, watcher_cfg);
        tokio::pin!(stream);
        process_peer_events(stream, &peers, self_ip.as_deref(), want_ipv6).await;
    });
    Ok(handle)
}

/// Drive the peer-set event loop for a stream of `watcher::Event`s.
///
/// Split out from [`spawn_peer_watch`] for the same reason [`process_events`] is:
/// the relist bookkeeping is the part that has been wrong, and it is only
/// testable without a cluster if it is a function over a stream.
async fn process_peer_events<S>(
    mut stream: S,
    peers: &PeerRegistry,
    self_ip: Option<&str>,
    want_ipv6: bool,
) where
    S: Stream<Item = Result<watcher::Event<EndpointSlice>, watcher::Error>> + Unpin,
{
    // A Service's endpoints are commonly sharded across several EndpointSlices
    // (one per AZ, one per address family), which is why this is per-slice rather
    // than a flat list.
    let mut by_slice: HashMap<String, Vec<String>> = HashMap::new();
    // Relist buffer. Publishing each `InitApply` as it arrives would expose a
    // PARTIALLY rebuilt set, and an empty partial set marks the registry synced —
    // which makes `known_to_have_no_peers()` momentarily true and can
    // short-circuit bootstrap into a cold boot.
    let mut init_buffer: HashMap<String, Vec<String>> = HashMap::new();
    let publish = |by_slice: &HashMap<String, Vec<String>>| {
        let mut all: Vec<String> = by_slice.values().flatten().cloned().collect();
        all.sort_unstable();
        all.dedup();
        peers.replace(all);
    };

    while let Some(event) = stream.next().await {
        match event {
            Ok(watcher::Event::Init) => init_buffer.clear(),
            Ok(watcher::Event::InitApply(es)) => {
                init_buffer.insert(slice_key(&es), extract_peers(&es, self_ip, want_ipv6));
            }
            Ok(watcher::Event::InitDone) => {
                // Swap atomically: the relist result fully replaces the old view,
                // so slices deleted while the watch was down disappear here.
                by_slice = std::mem::take(&mut init_buffer);
                publish(&by_slice);
            }
            Ok(watcher::Event::Apply(es)) => {
                by_slice.insert(slice_key(&es), extract_peers(&es, self_ip, want_ipv6));
                publish(&by_slice);
            }
            Ok(watcher::Event::Delete(es)) => {
                by_slice.remove(&slice_key(&es));
                publish(&by_slice);
            }
            Err(e) => {
                tracing::warn!(error = ?e, "kv-bootstrap: peer watcher error; awaiting auto-restart");
            }
        }
    }
    tracing::warn!("kv-bootstrap: peer watcher stream ended; peer set is now frozen");
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
    fn extract_workers_reads_transfer_group_from_slice_label() {
        let s = make_slice_with_labels(
            &["10.0.0.1"],
            30000,
            true,
            &[("role", "prefill"), ("sglang.ai/transfer-group", "tg-a")],
        );
        let ws = extract_workers(&s, WorkerMode::Prefill, "sglang.ai/transfer-group");
        assert_eq!(ws[0].transfer_group.as_deref(), Some("tg-a"));
        // A different label key → ungrouped.
        let ws = extract_workers(&s, WorkerMode::Prefill, "example.com/other");
        assert_eq!(ws[0].transfer_group, None);
        // Unlabelled slice → ungrouped.
        let s = make_slice(&["10.0.0.1"], 30000, true);
        let ws = extract_workers(&s, WorkerMode::Decode, "sglang.ai/transfer-group");
        assert_eq!(ws[0].transfer_group, None);
    }

    #[test]
    fn extract_workers_emits_workers_with_supplied_mode_and_empty_model_ids() {
        let s = make_slice(&["10.0.0.1"], 30000, true);
        let ws = extract_workers(&s, WorkerMode::Plain, "sglang.ai/transfer-group");
        assert_eq!(ws.len(), 1);
        assert_eq!(ws[0].mode, WorkerMode::Plain);
        assert_eq!(ws[0].url, "http://10.0.0.1:30000");
        assert_eq!(ws[0].id.0, "testns/test-slice/10.0.0.1:30000");
        assert!(
            ws[0].model_ids.is_empty(),
            "model_ids are resolved via /server_info, not at extract time"
        );

        // The mode argument flows through unchanged.
        let ws = extract_workers(&s, WorkerMode::Prefill, "sglang.ai/transfer-group");
        assert_eq!(ws[0].mode, WorkerMode::Prefill);
        let ws = extract_workers(&s, WorkerMode::Decode, "sglang.ai/transfer-group");
        assert_eq!(ws[0].mode, WorkerMode::Decode);
    }

    #[test]
    fn skips_not_ready_endpoints() {
        let s = make_slice(&["10.0.0.1"], 30000, false);
        assert!(extract_workers(&s, WorkerMode::Plain, "sglang.ai/transfer-group").is_empty());
    }

    /// `conditions.ready = None` must default to ready=true per EndpointSlice
    /// API spec ("undefined → endpoint should be considered ready").
    #[test]
    fn is_ready_none_defaults_to_ready() {
        let mut s = make_slice(&["10.0.0.1"], 30000, false /* overridden below */);
        s.endpoints[0].conditions.as_mut().unwrap().ready = None;
        let ws = extract_workers(&s, WorkerMode::Plain, "sglang.ai/transfer-group");
        assert_eq!(
            ws.len(),
            1,
            "endpoint with None ready should be treated as ready"
        );
    }

    /// WorkerId must include namespace + slice name to avoid cross-namespace
    /// IP collisions on overlay networks.
    #[test]
    fn worker_id_includes_namespace_and_slice_name() {
        let mut s = make_slice(&["10.0.0.1"], 30000, true);
        s.metadata.namespace = Some("prod".to_string());
        s.metadata.name = Some("svc-abc-xyz".to_string());
        let ws = extract_workers(&s, WorkerMode::Plain, "sglang.ai/transfer-group");
        assert_eq!(ws[0].id.0, "prod/svc-abc-xyz/10.0.0.1:30000");
    }

    /// A cluster-scoped slice (no namespace metadata) must still produce a
    /// valid `WorkerId` — empty-namespace prefix yields a leading slash.
    #[test]
    fn worker_id_handles_missing_namespace() {
        // make_slice_ns with empty ns leaves metadata.namespace = None.
        let s = make_slice_ns(&["10.0.0.1"], 30000, true, "", "my-slice");
        let ws = extract_workers(&s, WorkerMode::Plain, "sglang.ai/transfer-group");
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

    /// Two slices with no `metadata.uid` must hash to distinct per-slice
    /// keys; otherwise an event for slice B would overwrite slice A's state.
    #[test]
    fn slice_key_falls_back_to_ns_and_name_when_uid_missing() {
        let a = make_slice_ns(&["10.0.0.1"], 30000, true, "ns1", "a");
        let b = make_slice_ns(&["10.0.0.2"], 30000, true, "ns1", "b");
        assert_ne!(slice_key(&a), slice_key(&b));
        assert_eq!(slice_key(&a), "ns1/a");
    }

    #[test]
    fn slice_key_uses_uid_when_present() {
        let mut a = make_slice_ns(&["10.0.0.1"], 30000, true, "ns1", "a");
        a.metadata.uid = Some("uid-abc".into());
        assert_eq!(slice_key(&a), "uid-abc");
    }

    // -----------------------------------------------------------------------
    // Peer discovery (`extract_peers`) — the sibling-replica set that
    // cache-aware peer bootstrap pulls snapshots from.
    // -----------------------------------------------------------------------

    /// Multiple endpoints in one slice, each with its own ready condition, so the
    /// per-endpoint filtering can be exercised.
    fn peer_slice(entries: &[(&str, Option<bool>)], port: Option<i32>) -> EndpointSlice {
        EndpointSlice {
            metadata: ObjectMeta {
                name: Some("sgl-router-kv-abc".into()),
                namespace: Some("sgl-router-test".into()),
                ..Default::default()
            },
            address_type: "IPv4".into(),
            endpoints: entries
                .iter()
                .map(|(addr, ready)| Endpoint {
                    addresses: vec![(*addr).to_string()],
                    conditions: ready.map(|r| EndpointConditions {
                        ready: Some(r),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
            ports: port.map(|p| {
                vec![EndpointPort {
                    port: Some(p),
                    ..Default::default()
                }]
            }),
        }
    }

    #[test]
    fn extract_peers_excludes_self() {
        let es = peer_slice(
            &[("10.0.0.1", Some(true)), ("10.0.0.2", Some(true))],
            Some(8090),
        );
        assert_eq!(
            extract_peers(&es, Some("10.0.0.1"), false),
            vec!["http://10.0.0.2:8090"],
            "a replica must not offer itself as a bootstrap source",
        );
    }

    /// Without POD_IP there is nothing to exclude. Self-fetch is wasteful but
    /// harmless — an empty local tree reports `producer_ready: false`.
    #[test]
    fn extract_peers_keeps_self_when_pod_ip_is_unknown() {
        let es = peer_slice(&[("10.0.0.1", Some(true))], Some(8090));
        assert_eq!(
            extract_peers(&es, None, false),
            vec!["http://10.0.0.1:8090"]
        );
    }

    /// Per the EndpointSlice API an ABSENT ready condition means ready. Treating
    /// it as not-ready would empty the peer set on clusters that omit it.
    #[test]
    fn extract_peers_treats_absent_ready_as_ready() {
        let es = peer_slice(&[("10.0.0.2", None)], Some(8090));
        assert_eq!(
            extract_peers(&es, Some("10.0.0.1"), false),
            vec!["http://10.0.0.2:8090"]
        );
    }

    /// An unready sibling is either still bootstrapping (nothing worth copying)
    /// or draining. This is the primary defence against a new replica in a
    /// rolling update bootstrapping from another new replica.
    #[test]
    fn extract_peers_skips_unready_endpoints() {
        let es = peer_slice(
            &[("10.0.0.2", Some(false)), ("10.0.0.3", Some(true))],
            Some(8090),
        );
        assert_eq!(
            extract_peers(&es, Some("10.0.0.1"), false),
            vec!["http://10.0.0.3:8090"],
        );
    }

    /// Regression: a bare IPv6 literal is not a parseable authority, so on a
    /// dual-stack cluster every peer URL from the IPv6 slice failed to connect
    /// and burned the bootstrap deadline.
    #[test]
    fn extract_peers_brackets_ipv6_addresses() {
        let mut es = peer_slice(&[("fd00::2", Some(true))], Some(8090));
        es.address_type = "IPv6".into();
        assert_eq!(
            extract_peers(&es, Some("fd00::1"), true),
            vec!["http://[fd00::2]:8090"],
        );
    }

    /// A dual-stack Service yields one slice per family. Keeping the family the
    /// router does not listen on gives permanently-unreachable candidates that
    /// burn the bootstrap deadline, and breaks self-exclusion (POD_IP carries only
    /// the primary address).
    #[test]
    fn extract_peers_skips_the_other_address_family() {
        let mut v6 = peer_slice(&[("fd00::2", Some(true))], Some(8090));
        v6.address_type = "IPv6".into();
        assert!(
            extract_peers(&v6, Some("10.0.0.1"), false).is_empty(),
            "an IPv4 router must ignore the IPv6 slice",
        );

        let v4 = peer_slice(&[("10.0.0.2", Some(true))], Some(8090));
        assert!(
            extract_peers(&v4, Some("fd00::1"), true).is_empty(),
            "an IPv6 router must ignore the IPv4 slice",
        );
    }

    /// An FQDN slice cannot be classified by family, so it is kept.
    #[test]
    fn extract_peers_keeps_fqdn_slices_for_either_family() {
        let mut es = peer_slice(&[("router-1.svc", Some(true))], Some(8090));
        es.address_type = "FQDN".into();
        assert_eq!(
            extract_peers(&es, None, false),
            vec!["http://router-1.svc:8090"],
        );
    }

    #[test]
    fn extract_peers_falls_back_to_default_port() {
        let es = peer_slice(&[("10.0.0.2", Some(true))], None);
        assert_eq!(
            extract_peers(&es, None, false),
            vec![format!("http://10.0.0.2:{DEFAULT_ROUTER_PORT}")],
        );
    }

    #[test]
    fn extract_peers_on_empty_slice_is_empty() {
        assert!(extract_peers(&peer_slice(&[], Some(8090)), None, false).is_empty());
    }

    /// Same slice shape, but with a caller-chosen name so `slice_key` yields
    /// distinct keys — needed to model a multi-slice Service.
    fn named_peer_slice(
        name: &str,
        entries: &[(&str, Option<bool>)],
        port: Option<i32>,
    ) -> EndpointSlice {
        let mut es = peer_slice(entries, port);
        es.metadata.name = Some(name.into());
        es
    }

    async fn run_peer_events(
        events: Vec<Result<watcher::Event<EndpointSlice>, watcher::Error>>,
        peers: &PeerRegistry,
        self_ip: Option<&str>,
    ) {
        let stream = futures::stream::iter(events);
        tokio::pin!(stream);
        process_peer_events(stream, peers, self_ip, false).await;
    }

    /// THE regression this loop exists for: a relist must not publish a partial
    /// set. If it does, an empty partial marks the registry synced and
    /// `known_to_have_no_peers()` goes true, short-circuiting bootstrap into a
    /// cold boot even though siblings exist.
    ///
    /// Asserted MID-relist, deliberately. Checking only the settled state after
    /// `InitDone` passes even with the bug present, because the damage is a
    /// transient the bootstrap retry loop can sample and act on permanently.
    #[tokio::test]
    async fn peer_relist_does_not_publish_a_partial_set() {
        let peers = PeerRegistry::new();
        // Init plus one self-only slice, and NO InitDone: nothing may be published
        // yet, so the registry must not even be marked synced.
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.1", Some(true))], // self — yields zero peers
                    Some(8090),
                ))),
            ],
            &peers,
            Some("10.0.0.1"),
        )
        .await;

        assert!(
            !peers.synced(),
            "an incomplete relist must not mark the peer set synced",
        );
        assert!(
            !peers.known_to_have_no_peers(),
            "a partial relist must never read as 'this replica is alone'",
        );
    }

    /// And the completed relist publishes the full set.
    #[tokio::test]
    async fn peer_relist_publishes_once_complete() {
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.1", Some(true))], // self
                    Some(8090),
                ))),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-b",
                    &[("10.0.0.2", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
            ],
            &peers,
            Some("10.0.0.1"),
        )
        .await;

        assert_eq!(peers.candidates(), vec!["http://10.0.0.2:8090"]);
        assert!(!peers.known_to_have_no_peers());
    }

    /// A relist drops slices that disappeared while the watch was down.
    #[tokio::test]
    async fn peer_relist_replaces_rather_than_merges() {
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.2", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
                // Watch restarts; slice-a is gone, slice-b appears.
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-b",
                    &[("10.0.0.3", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
            ],
            &peers,
            Some("10.0.0.1"),
        )
        .await;
        assert_eq!(
            peers.candidates(),
            vec!["http://10.0.0.3:8090"],
            "a relist must replace the view, not merge into it",
        );
    }

    /// Steady-state add and remove of one slice out of several.
    #[tokio::test]
    async fn peer_apply_and_delete_update_incrementally() {
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.2", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
                Ok(watcher::Event::Apply(named_peer_slice(
                    "slice-b",
                    &[("10.0.0.3", Some(true))],
                    Some(8090),
                ))),
            ],
            &peers,
            Some("10.0.0.1"),
        )
        .await;
        let mut got = peers.candidates();
        got.sort();
        assert_eq!(got, vec!["http://10.0.0.2:8090", "http://10.0.0.3:8090"]);

        run_peer_events(
            vec![Ok(watcher::Event::Delete(named_peer_slice(
                "slice-b",
                &[("10.0.0.3", Some(true))],
                Some(8090),
            )))],
            &peers,
            Some("10.0.0.1"),
        )
        .await;
        // NOTE: a fresh call rebuilds `by_slice`, so this asserts the Delete path
        // publishes, not that state carried over.
        assert!(peers.synced());
    }

    /// A transient watcher error must preserve the peer set, not clear or
    /// republish it.
    ///
    /// Publishing an empty set on a transient error would mark the registry synced
    /// with zero peers, making `known_to_have_no_peers()` true and short-circuiting
    /// bootstrap into a cold boot. The worker loop in this file has the equivalent
    /// test; the peer loop did not.
    #[tokio::test]
    async fn peer_watcher_error_preserves_the_peer_set() {
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.2", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
                Err(watcher::Error::NoResourceVersion),
            ],
            &peers,
            Some("10.0.0.1"),
        )
        .await;

        assert_eq!(
            peers.candidates(),
            vec!["http://10.0.0.2:8090"],
            "a transient error must not disturb the peer set",
        );
        assert!(!peers.known_to_have_no_peers());
    }

    /// A genuinely single-replica deployment must be recognised as alone, so
    /// bootstrap abandons immediately instead of burning the whole deadline.
    #[tokio::test]
    async fn peer_relist_with_only_self_is_conclusively_alone() {
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.1", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
            ],
            &peers,
            Some("10.0.0.1"),
        )
        .await;
        assert!(peers.is_empty());
        assert!(
            peers.known_to_have_no_peers(),
            "synced with only self ⇒ genuinely alone",
        );
    }

    fn with_uid(mut es: EndpointSlice, uid: &str) -> EndpointSlice {
        es.metadata.uid = Some(uid.into());
        es
    }

    /// Apply → Delete on the same slice removes its workers from the union
    /// (the missing variant for `.applied_objects()` before the rewrite).
    #[tokio::test]
    async fn delete_event_emits_removed_for_workers_in_slice() {
        let s = with_uid(make_slice_ns(&["10.0.0.1"], 30000, true, "ns", "svc"), "u1");
        let events = vec![
            Ok(watcher::Event::Apply(s.clone())),
            Ok(watcher::Event::Delete(s)),
        ];
        let (tx, mut rx) = mpsc::channel(16);
        let stream = futures::stream::iter(events);
        process_events(stream, tx, plain_mode()).await;
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        assert_eq!(out.len(), 2, "{out:?}");
        assert!(matches!(out[0], DiscoveryEvent::Added(_)));
        assert!(matches!(out[1], DiscoveryEvent::Removed { .. }));
    }

    /// Init/InitDone replaces state atomically: any worker present before
    /// the Init cycle but not seen during it is diffed out as Removed.  This
    /// covers the "slice deleted while watcher was disconnected" case.
    #[tokio::test]
    async fn init_cycle_diffs_out_unseen_slices() {
        let a = with_uid(make_slice_ns(&["10.0.0.1"], 30000, true, "ns", "a"), "u-a");
        let b = with_uid(make_slice_ns(&["10.0.0.2"], 30000, true, "ns", "b"), "u-b");

        let events = vec![
            // Initial state: both slices live.
            Ok(watcher::Event::Apply(a.clone())),
            Ok(watcher::Event::Apply(b.clone())),
            // Watcher restart resyncs and only sees `a` (b was deleted offline).
            Ok(watcher::Event::Init),
            Ok(watcher::Event::InitApply(a.clone())),
            Ok(watcher::Event::InitDone),
        ];
        let (tx, mut rx) = mpsc::channel(16);
        process_events(futures::stream::iter(events), tx, plain_mode()).await;
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        let removed: Vec<_> = out
            .iter()
            .filter_map(|e| match e {
                DiscoveryEvent::Removed { id } => Some(id.0.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            removed.contains(&"ns/b/10.0.0.2:30000"),
            "init resync should diff out the deleted slice: out={out:?}"
        );
        assert!(
            !removed.contains(&"ns/a/10.0.0.1:30000"),
            "live slice must not be removed: out={out:?}"
        );
    }

    /// When the consumer drops the receiver, the watcher loop exits cleanly
    /// rather than continuing to process events into the void.
    #[tokio::test]
    async fn watcher_exits_when_consumer_drops_receiver() {
        use std::time::Duration;
        let s = with_uid(make_slice_ns(&["10.0.0.1"], 30000, true, "ns", "a"), "u-a");
        // A stream that never ends — only consumer drop should stop the loop.
        let events = futures::stream::iter(std::iter::repeat_with(move || {
            Ok(watcher::Event::Apply(s.clone()))
        }));
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let handle = tokio::spawn(process_events(events, tx, plain_mode()));
        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("process_events must exit promptly when consumer drops")
            .expect("task should not panic");
    }

    /// Watcher errors are logged but state is preserved; the next successful
    /// event continues to diff against the pre-error union.
    #[tokio::test]
    async fn watcher_error_preserves_state_and_diffs_against_pre_error_union() {
        let s = with_uid(make_slice_ns(&["10.0.0.1"], 30000, true, "ns", "a"), "u-a");
        let events = vec![
            Ok(watcher::Event::Apply(s.clone())),
            Err(watcher::Error::NoResourceVersion),
            // Same slice; nothing should be re-emitted because state survived.
            Ok(watcher::Event::Apply(s)),
        ];
        let (tx, mut rx) = mpsc::channel(16);
        process_events(futures::stream::iter(events), tx, plain_mode()).await;
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        // Exactly one Added — the second Apply is a no-op against the
        // existing union.
        let added = out
            .iter()
            .filter(|e| matches!(e, DiscoveryEvent::Added(_)))
            .count();
        assert_eq!(added, 1, "out={out:?}");
    }

    /// In PD mode, only slices whose labels match one of the role selectors
    /// produce workers; unrelated slices in the same namespace are dropped
    /// without registration.
    #[tokio::test]
    async fn pd_mode_drops_slices_whose_labels_match_no_role_selector() {
        let prefill_slice = with_uid(
            make_slice_full(
                &["10.0.0.1"],
                30000,
                true,
                "ns",
                "p",
                &[("app", "sglang"), ("role", "prefill")],
            ),
            "u-p",
        );
        let unrelated_slice = with_uid(
            make_slice_full(
                &["10.0.0.2"],
                30000,
                true,
                "ns",
                "x",
                &[("app", "sglang"), ("role", "router")],
            ),
            "u-x",
        );
        let events = vec![
            Ok(watcher::Event::Apply(prefill_slice)),
            Ok(watcher::Event::Apply(unrelated_slice)),
        ];
        let (tx, mut rx) = mpsc::channel(16);
        process_events(futures::stream::iter(events), tx, pd_mode()).await;
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        let added: Vec<_> = out
            .iter()
            .filter_map(|e| match e {
                DiscoveryEvent::Added(spec) => Some(spec),
                _ => None,
            })
            .collect();
        assert_eq!(
            added.len(),
            1,
            "only the prefill-labelled slice should be registered: out={out:?}"
        );
        assert_eq!(added[0].mode, WorkerMode::Prefill);
        assert_eq!(added[0].id.0, "ns/p/10.0.0.1:30000");
    }

    /// End-to-end K8s + PD integration: synthesize EndpointSlice events
    /// for two prefill pods and two decode pods (each backed by a real
    /// HTTP listener mounting `/server_info`), pipe them through
    /// `process_events` → DiscoveryEvent channel → manager, and assert
    /// the resulting registry has:
    ///   * two `Prefill` workers, each with `bootstrap_port` matching
    ///     what its own `/server_info` advertised (so per-worker
    ///     plumbing is verified, not just "some prefill registered"),
    ///   * two `Decode` workers with `bootstrap_port = None`.
    ///
    /// This is the load-bearing integration covering the seam this PR
    /// just opened: the K8s backend emits `bootstrap_port: None`, the
    /// PD-disaggregation classification comes from slice labels, and
    /// `WorkerMode` + `bootstrap_port` are re-resolved by the manager
    /// from each worker's `/server_info`. A regression at *any* of
    /// those three layers (k8s extract → process_events label
    /// classification → manager introspect-and-override) fails this
    /// test.
    #[tokio::test]
    async fn k8s_pd_pipeline_registers_workers_with_per_pod_bootstrap_port() {
        use crate::workers::introspect::WorkerIntrospector;
        use crate::workers::manager::run_with_introspector;
        use crate::workers::WorkerRegistry;
        use axum::{routing::get, Json, Router};
        use serde_json::{json, Value};
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::net::TcpListener;
        use tokio::sync::oneshot;

        /// Bind axum on an OS-assigned 127.0.0.1 port, mount a
        /// `/server_info` returning `body`, return the port + shutdown
        /// channel so the test can join cleanly.
        async fn spawn_fake_server_info(body: Value) -> (u16, oneshot::Sender<()>) {
            let body = Arc::new(body);
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let app = Router::new().route(
                "/server_info",
                get(move || {
                    let body = body.clone();
                    async move { Json((*body).clone()) }
                }),
            );
            let (tx, rx) = oneshot::channel::<()>();
            tokio::spawn(async move {
                let _ = axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        let _ = rx.await;
                    })
                    .await;
            });
            (port, tx)
        }

        // Four fake SGLang workers — two prefill (each with a distinct
        // bootstrap_port to verify per-pod plumbing) and two decode.
        let (port_p1, _shut_p1) = spawn_fake_server_info(json!({
            "served_model_name": "m",
            "disaggregation_mode": "prefill",
            "disaggregation_bootstrap_port": 8998,
        }))
        .await;
        let (port_p2, _shut_p2) = spawn_fake_server_info(json!({
            "served_model_name": "m",
            "disaggregation_mode": "prefill",
            "disaggregation_bootstrap_port": 8999,
        }))
        .await;
        let (port_d1, _shut_d1) = spawn_fake_server_info(json!({
            "served_model_name": "m",
            "disaggregation_mode": "decode",
        }))
        .await;
        let (port_d2, _shut_d2) = spawn_fake_server_info(json!({
            "served_model_name": "m",
            "disaggregation_mode": "decode",
        }))
        .await;

        // One EndpointSlice per worker (one address each, so the slice
        // port matches the worker's axum listener port exactly).
        // Labels match the PD selectors so `classify_mode` sends each
        // slice to the right pool.
        let prefill_labels = &[("app", "sglang"), ("role", "prefill")];
        let decode_labels = &[("app", "sglang"), ("role", "decode")];
        let p1 = with_uid(
            make_slice_full(
                &["127.0.0.1"],
                port_p1 as i32,
                true,
                "ns",
                "prefill-1",
                prefill_labels,
            ),
            "u-p1",
        );
        let p2 = with_uid(
            make_slice_full(
                &["127.0.0.1"],
                port_p2 as i32,
                true,
                "ns",
                "prefill-2",
                prefill_labels,
            ),
            "u-p2",
        );
        let d1 = with_uid(
            make_slice_full(
                &["127.0.0.1"],
                port_d1 as i32,
                true,
                "ns",
                "decode-1",
                decode_labels,
            ),
            "u-d1",
        );
        let d2 = with_uid(
            make_slice_full(
                &["127.0.0.1"],
                port_d2 as i32,
                true,
                "ns",
                "decode-2",
                decode_labels,
            ),
            "u-d2",
        );

        // Manager pipeline: DiscoveryEvent channel → run_with_introspector.
        let registry = Arc::new(WorkerRegistry::default());
        let (dtx, drx) = mpsc::channel::<DiscoveryEvent>(16);
        let introspector = Arc::new(WorkerIntrospector::new(Duration::from_millis(500)));
        let manager_handle = tokio::spawn(run_with_introspector(
            drx,
            registry.clone(),
            None,
            None,
            None,
            None,
            introspector,
        ));

        // Drive process_events with the four slice Apply events, then
        // drop dtx so the manager loop exits cleanly once it has drained.
        let events = vec![
            Ok(watcher::Event::Apply(p1)),
            Ok(watcher::Event::Apply(p2)),
            Ok(watcher::Event::Apply(d1)),
            Ok(watcher::Event::Apply(d2)),
        ];
        let producer = tokio::spawn(async move {
            let stream = futures::stream::iter(events);
            process_events(stream, dtx, pd_mode()).await;
        });

        // Poll the registry until all four workers are present with
        // their resolved mode + bootstrap_port — order isn't deterministic
        // because each worker's `/server_info` round-trip happens in a
        // separate manager task.
        let expected_p1_id = WorkerId(format!("ns/prefill-1/127.0.0.1:{port_p1}"));
        let expected_p2_id = WorkerId(format!("ns/prefill-2/127.0.0.1:{port_p2}"));
        let expected_d1_id = WorkerId(format!("ns/decode-1/127.0.0.1:{port_d1}"));
        let expected_d2_id = WorkerId(format!("ns/decode-2/127.0.0.1:{port_d2}"));

        let settled = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let p1 = registry.get(&expected_p1_id);
                let p2 = registry.get(&expected_p2_id);
                let d1 = registry.get(&expected_d1_id);
                let d2 = registry.get(&expected_d2_id);
                if let (Some(p1), Some(p2), Some(d1), Some(d2)) = (p1, p2, d1, d2) {
                    if p1.mode() == WorkerMode::Prefill
                        && p2.mode() == WorkerMode::Prefill
                        && d1.mode() == WorkerMode::Decode
                        && d2.mode() == WorkerMode::Decode
                        && p1.bootstrap_port() == Some(8998)
                        && p2.bootstrap_port() == Some(8999)
                        && d1.bootstrap_port().is_none()
                        && d2.bootstrap_port().is_none()
                    {
                        return true;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            settled.is_ok(),
            "registry did not converge to (2 prefill + 2 decode) with per-pod \
             bootstrap_port within 3s. current state: \
             p1={:?}, p2={:?}, d1={:?}, d2={:?}",
            registry
                .get(&expected_p1_id)
                .map(|w| (w.mode(), w.bootstrap_port())),
            registry
                .get(&expected_p2_id)
                .map(|w| (w.mode(), w.bootstrap_port())),
            registry
                .get(&expected_d1_id)
                .map(|w| (w.mode(), w.bootstrap_port())),
            registry
                .get(&expected_d2_id)
                .map(|w| (w.mode(), w.bootstrap_port())),
        );

        // Producer should exit when the iter stream ends; manager exits
        // when the producer drops dtx. Both should finish quickly.
        let _ = tokio::time::timeout(Duration::from_secs(1), producer).await;
        let _ = tokio::time::timeout(Duration::from_secs(1), manager_handle).await;
    }

    /// Helper: build a slice where every endpoint carries a synthetic
    /// `target_ref.uid`. The endpoint at position `i` gets `uids[i]`.
    fn make_slice_with_uids(addrs: &[&str], port: i32, uids: &[&str]) -> EndpointSlice {
        use k8s_openapi::api::core::v1::ObjectReference;
        assert_eq!(addrs.len(), uids.len());
        let endpoints = addrs
            .iter()
            .zip(uids.iter())
            .map(|(addr, uid)| Endpoint {
                addresses: vec![(*addr).to_string()],
                conditions: Some(EndpointConditions {
                    ready: Some(true),
                    ..Default::default()
                }),
                target_ref: Some(ObjectReference {
                    uid: Some((*uid).to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .collect();
        EndpointSlice {
            metadata: ObjectMeta {
                name: Some("svc".into()),
                namespace: Some("ns".into()),
                ..Default::default()
            },
            address_type: "IPv4".into(),
            endpoints,
            ports: Some(vec![EndpointPort {
                port: Some(port),
                ..Default::default()
            }]),
        }
    }

    /// Pod is replaced (same IP, different UID) — router must see this as
    /// a Removed+Added cycle so the new pod gets fresh CB/active_load
    /// state. Without UID-keyed WorkerIds, two consecutive
    /// `process_events` snapshots would dedup by `addr:port` and the
    /// new pod would inherit the dead pod's state.
    #[tokio::test]
    async fn pod_replace_with_same_ip_emits_remove_then_add() {
        let s_old = with_uid(
            make_slice_with_uids(&["10.0.0.1"], 30000, &["uid-old"]),
            "u-1",
        );
        let s_new = with_uid(
            make_slice_with_uids(&["10.0.0.1"], 30000, &["uid-new"]),
            "u-1",
        );
        let (tx, mut rx) = mpsc::channel(16);
        process_events(
            futures::stream::iter(vec![
                Ok(watcher::Event::Apply(s_old)),
                Ok(watcher::Event::Apply(s_new)),
            ]),
            tx,
            plain_mode(),
        )
        .await;
        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        // Expect: Added(uid-old) → Removed(uid-old) + Added(uid-new).
        // The order of Removed/Added within the second apply depends on
        // emit_diff's iteration; assert by counting each variant.
        assert_eq!(events.len(), 3, "got {events:?}");
        let added: Vec<&WorkerSpec> = events
            .iter()
            .filter_map(|e| match e {
                DiscoveryEvent::Added(spec) => Some(spec),
                _ => None,
            })
            .collect();
        let removed: Vec<&WorkerId> = events
            .iter()
            .filter_map(|e| match e {
                DiscoveryEvent::Removed { id } => Some(id),
                _ => None,
            })
            .collect();
        assert_eq!(added.len(), 2, "two Added (one per UID): {events:?}");
        assert_eq!(removed.len(), 1, "one Removed (for uid-old): {events:?}");
        assert_eq!(
            added[0].id.0, "ns/uid-old",
            "first Added is the original pod",
        );
        assert_eq!(
            added[1].id.0, "ns/uid-new",
            "second Added is the replacement pod with a fresh UID",
        );
        assert_eq!(
            removed[0].0, "ns/uid-old",
            "Removed targets the original pod's UID, not the IP-keyed id",
        );
        // Same URL across both, confirming the IP didn't change.
        assert_eq!(added[0].url, added[1].url);
    }

    /// End-to-end reconcile: an EndpointSlice flips `ready=true` while the
    /// engine's `/server_info` is still failing (503) — the production
    /// race where K8s readiness (cheap `/health`) leads the
    /// scheduler-backed `/server_info`. The worker registers with empty
    /// `model_ids` (invisible to routing), and discovery emits no further
    /// event. The manager's reconcile loop must re-introspect it and move
    /// it into the model pool once `/server_info` recovers — driven
    /// through the real `process_events` → manager path.
    #[tokio::test]
    async fn k8s_reconcile_recovers_worker_whose_server_info_was_initially_failing() {
        use crate::discovery::ModelId;
        use crate::workers::introspect::WorkerIntrospector;
        use crate::workers::manager::run_with_introspector_and_reconcile;
        use crate::workers::WorkerRegistry;
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        use axum::{routing::get, Json, Router};
        use serde_json::json;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::net::TcpListener;

        // Fake engine: 503 on /server_info until `ready` flips true, then
        // serves a valid body advertising model "m".
        let ready = Arc::new(AtomicBool::new(false));
        let ready_handler = ready.clone();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new().route(
            "/server_info",
            get(move || {
                let ready = ready_handler.clone();
                async move {
                    if ready.load(Ordering::SeqCst) {
                        Json(json!({"served_model_name": "m"})).into_response()
                    } else {
                        StatusCode::SERVICE_UNAVAILABLE.into_response()
                    }
                }
            }),
        );
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        });

        // EndpointSlice already marked ready=true (kubelet's /health probe
        // passed) pointing at the engine whose /server_info is still 503.
        let slice = with_uid(
            make_slice_full(&["127.0.0.1"], port as i32, true, "ns", "engine-1", &[]),
            "u-e1",
        );

        let registry = Arc::new(WorkerRegistry::default());
        let (dtx, drx) = mpsc::channel::<DiscoveryEvent>(16);
        // Hold a second sender so the channel stays open after the
        // producer's single-event stream ends — otherwise the manager
        // loop would exit before any reconcile tick fires.
        let dtx_keepalive = dtx.clone();
        let introspector = Arc::new(WorkerIntrospector::new(Duration::from_millis(300)));
        let manager_handle = tokio::spawn(run_with_introspector_and_reconcile(
            drx,
            registry.clone(),
            None,
            None,
            None,
            None,
            introspector,
            Duration::from_millis(150),
        ));

        // Drive the ready=true slice through the real discovery processor.
        let producer = tokio::spawn(async move {
            let stream = futures::stream::iter(vec![Ok(watcher::Event::Apply(slice))]);
            process_events(stream, dtx, plain_mode()).await;
        });

        // No target_ref on the endpoint => id falls back to ns/slice/addr:port.
        let id = WorkerId(format!("ns/engine-1/127.0.0.1:{port}"));
        let model = ModelId("m".into());

        // Phase 1: worker is registered but absent from the model pool
        // while /server_info keeps failing.
        let stuck = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(w) = registry.get(&id) {
                    if w.model_ids.is_empty() && registry.workers_for(&model).is_empty() {
                        return true;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            stuck.is_ok(),
            "worker should register with empty model_ids while /server_info is failing",
        );

        // Engine finishes coming up.
        ready.store(true, Ordering::SeqCst);

        // Phase 2: reconcile re-introspects and the worker joins the pool,
        // with no further discovery event.
        let recovered = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if !registry.workers_for(&model).is_empty() {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            recovered.is_ok(),
            "reconcile must re-introspect the worker and add it to the model pool once \
             /server_info recovers; registry state: {:?}",
            registry.get(&id).map(|w| w.model_ids.clone()),
        );

        drop(dtx_keepalive);
        let _ = shutdown_tx.send(());
        let _ = tokio::time::timeout(Duration::from_secs(1), producer).await;
        let _ = tokio::time::timeout(Duration::from_secs(1), manager_handle).await;
    }
}
