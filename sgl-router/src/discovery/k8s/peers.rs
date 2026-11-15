//! Watch this router's own EndpointSlices to find sibling replicas.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use futures::{Stream, StreamExt};
use k8s_openapi::api::discovery::v1::{Endpoint, EndpointSlice};
use kube::runtime::{watcher, WatchStreamExt};

use super::{endpoint_ready, endpoint_slice_api, namespace_display, slice_key};
use crate::state::kv_events::bootstrap::PeerRegistry;

// ---------------------------------------------------------------------------
// Peer discovery — sibling router replicas
// ---------------------------------------------------------------------------

/// The address family sibling URLs must use: the one this router listens on.
///
/// A dual-stack Service has one slice per family, and siblings listen as this
/// replica does, so only the family its listener accepts is reachable.
/// `0.0.0.0` opens an `AF_INET` socket, so it is IPv4-only however
/// "unspecified" it looks. Only `::` is a dual-stack listener (Linux leaves
/// `IPV6_V6ONLY` off by default), and a hostname names no family; both are
/// `Any`. Reading `::` as IPv6-only would discard every slice on a
/// single-stack IPv4 cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressFamily {
    V4,
    V6,
    Any,
}

impl AddressFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::V4 => "IPv4",
            Self::V6 => "IPv6",
            Self::Any => "any",
        }
    }
}

/// The [`AddressFamily`] of a router bound to `host`.
pub fn peer_address_family(host: &str) -> AddressFamily {
    match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(_)) => AddressFamily::V4,
        Ok(std::net::IpAddr::V6(ip)) if !ip.is_unspecified() => AddressFamily::V6,
        Ok(std::net::IpAddr::V6(_)) | Err(_) => AddressFamily::Any,
    }
}

/// How this replica recognizes itself in an EndpointSlice, so its own pod is
/// kept out of the peer set.
///
/// Built from the downward API. `pod_name` is the primary signal: it matches
/// the endpoint's `target_ref` and so excludes ALL of the pod's addresses,
/// which `ip` alone cannot do — `POD_IP` carries only the pod's primary
/// address, so on a dual-stack Service the pod's secondary-family address
/// would survive an IP-only filter. `ip` is kept as a fallback for endpoints
/// whose `target_ref` is absent (manually-created slices).
///
/// Without `POD_NAME`, `pod_name` falls back to `HOSTNAME`, as the access
/// log's pod identity does: Kubernetes defaults a pod's hostname to its name.
/// A hostname that is not the pod name (`hostNetwork`, a set `spec.hostname`)
/// just matches nothing.
#[derive(Default, Debug)]
struct SelfIdentity {
    pod_name: Option<String>,
    pod_namespace: Option<String>,
    ip: Option<String>,
}

/// A set-but-empty env var must count as unset: the downward API never yields
/// empty, but a hand-written manifest can, and `Some("")` would suppress the
/// [`SelfIdentity::is_unknown`] warning while matching nothing.
fn env_non_empty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}

impl SelfIdentity {
    fn from_env() -> Self {
        Self {
            pod_name: env_non_empty("POD_NAME").or_else(|| env_non_empty("HOSTNAME")),
            pod_namespace: env_non_empty("POD_NAMESPACE"),
            ip: env_non_empty("POD_IP"),
        }
    }

    /// True when nothing usable for self-exclusion is known (`POD_NAMESPACE`
    /// alone cannot exclude anything). This replica then stays in its own
    /// peer set.
    fn is_unknown(&self) -> bool {
        self.pod_name.is_none() && self.ip.is_none()
    }

    /// Does this endpoint belong to this replica's own pod?
    ///
    /// An endpoint is one backend and its addresses are fungible (the
    /// EndpointSlice API says so), so one address matching `POD_IP` makes the
    /// whole endpoint this replica.
    fn matches_endpoint(&self, ep: &Endpoint) -> bool {
        let self_ip = self.ip.as_deref();
        if ep.addresses.iter().any(|a| Some(a.as_str()) == self_ip) {
            return true;
        }
        let (Some(name), Some(tref)) = (self.pod_name.as_deref(), ep.target_ref.as_ref()) else {
            return false;
        };
        // Pod-backed endpoints always carry kind/name; a missing kind on a
        // manually-written slice is treated as Pod rather than never matching.
        if tref.kind.as_deref().unwrap_or("Pod") != "Pod" || tref.name.as_deref() != Some(name) {
            return false;
        }
        // When our own namespace is known, a same-named pod elsewhere is not
        // us. When it is unknown, a bare name match can only false-positive on
        // an identical pod name in another watched namespace — contrived for
        // most workloads, though deterministic for same-named StatefulSets,
        // and impossible once POD_NAMESPACE is set. A target_ref with no
        // namespace is treated as same-namespace: excluding beats listing this
        // replica as its own peer.
        match (self.pod_namespace.as_deref(), tref.namespace.as_deref()) {
            (Some(want), Some(got)) => want == got,
            _ => true,
        }
    }
}

/// One sibling as one slice lists it.
#[derive(Debug)]
struct PeerAddr {
    /// The backend behind `url`, so a sibling listed once per address family
    /// (a dual-stack Service has one slice per family) is offered once: the
    /// target pod's UID and the port, else the URL itself.
    backend: String,
    url: String,
}

/// What this replica keeps from each EndpointSlice of its own Service.
#[derive(Debug)]
struct PeerFilter {
    identity: SelfIdentity,
    family: AddressFamily,
    /// This replica's listen port.
    port: i32,
}

impl PeerFilter {
    /// Extract sibling replicas from an EndpointSlice.
    ///
    /// Only `ready` endpoints are considered: an unready sibling is either
    /// still starting (its tree is cold) or draining. This replica's own
    /// endpoint is excluded (see [`SelfIdentity`]), and so is a slice of the
    /// other [`AddressFamily`]. Each endpoint yields its FIRST address only:
    /// the API defines an endpoint's addresses as fungible, so the rest name
    /// the same backend.
    fn extract(&self, es: &EndpointSlice) -> Vec<PeerAddr> {
        // FQDN and unrecognised address types are kept.
        let family_matches = !matches!(
            (self.family, es.address_type.as_str()),
            (AddressFamily::V4, "IPv6") | (AddressFamily::V6, "IPv4")
        );
        if !family_matches {
            return Vec::new();
        }
        // Siblings are this router's own pods, so they listen where this
        // replica does: prefer an advertised port equal to our own, and use
        // our own when the slice advertises none. A `metrics` or `grpc` port
        // declared ahead of `http` must not win just by sorting first.
        let ports = es.ports.as_deref().unwrap_or_default();
        let port = ports
            .iter()
            .filter_map(|p| p.port)
            .find(|&p| p == self.port)
            .or_else(|| ports.iter().find_map(|p| p.port))
            .unwrap_or(self.port);
        es.endpoints
            .iter()
            .filter(|ep| endpoint_ready(ep))
            .filter(|ep| {
                // Otherwise silent: this line is the only trace of a sibling
                // misclassified as self.
                if self.identity.matches_endpoint(ep) {
                    tracing::debug!(
                        addrs = ?ep.addresses,
                        target_ref = ?ep.target_ref,
                        "kv-bootstrap: excluding endpoint as self (pod identity match)",
                    );
                    return false;
                }
                true
            })
            .filter_map(|ep| {
                let addr = ep.addresses.first()?;
                // IPv6 literals need brackets in a URL authority.
                let url = if addr.contains(':') {
                    format!("http://[{addr}]:{port}")
                } else {
                    format!("http://{addr}:{port}")
                };
                let backend = match ep.target_ref.as_ref().and_then(|r| r.uid.as_deref()) {
                    Some(uid) if !uid.is_empty() => format!("{uid}:{port}"),
                    _ => url.clone(),
                };
                Some(PeerAddr { backend, url })
            })
            .collect()
    }
}

/// Flatten the per-slice view into the published peer list: one URL per
/// backend, sorted.
///
/// Within a backend the lexicographically first URL wins, which keeps the
/// choice deterministic and prefers IPv4 (its digits sort before IPv6's `[`).
fn peer_urls(by_slice: &HashMap<String, Vec<PeerAddr>>) -> Vec<String> {
    let mut all: Vec<&PeerAddr> = by_slice.values().flatten().collect();
    all.sort_unstable_by(|a, b| (&a.backend, &a.url).cmp(&(&b.backend, &b.url)));
    all.dedup_by(|a, b| a.backend == b.backend);
    let mut urls: Vec<String> = all.into_iter().map(|p| p.url.clone()).collect();
    urls.sort_unstable();
    urls.dedup();
    urls
}

/// Watch this router's own EndpointSlices and keep `peers` current.
///
/// Separate from [`super::spawn`] because it tracks sibling router replicas rather
/// than engines, and must keep working when the worker watch is in PD mode
/// with client-side classification.
///
/// The task exits when the watcher stream ends; the peer set then stays at its
/// last value. Non-fatal: routing does not depend on this watch.
pub async fn spawn_peer_watch(
    namespace: String,
    label_selector: String,
    peers: Arc<PeerRegistry>,
    family: AddressFamily,
    self_port: i32,
) -> Result<tokio::task::JoinHandle<()>> {
    let api = endpoint_slice_api(&namespace)
        .await
        .context("kube client default config for peer watch")?;
    let filter = PeerFilter {
        identity: SelfIdentity::from_env(),
        family,
        port: self_port,
    };
    if filter.identity.is_unknown() {
        tracing::warn!(
            "kv-bootstrap: none of POD_NAME, HOSTNAME or POD_IP is set, so this replica \
             cannot exclude itself from its peer list; set POD_NAME from the downward API \
             for a cleaner peer set"
        );
    }

    tracing::info!(
        namespace = %namespace_display(&namespace),
        label_selector = %label_selector,
        self_pod_name = %filter.identity.pod_name.as_deref().unwrap_or("<unset>"),
        self_ip = %filter.identity.ip.as_deref().unwrap_or("<unset>"),
        self_port,
        address_family = %family.as_str(),
        "kv-bootstrap: peer watch starting",
    );

    let watcher_cfg = watcher::Config::default().labels(&label_selector);
    let handle = tokio::spawn(async move {
        // The watcher retries on the very next poll after an error, so without
        // a backoff a persistent failure — a 403 from missing RBAC, a 400 from
        // a malformed selector — becomes a tight LIST loop against the API
        // server, one WARN line per iteration, for the life of the process.
        let stream = watcher(api, watcher_cfg).default_backoff();
        tokio::pin!(stream);
        process_peer_events(stream, &peers, &filter).await;
    });
    Ok(handle)
}

/// Drive the peer-set event loop for a stream of `watcher::Event`s.
///
/// Split out from [`spawn_peer_watch`] so the relist bookkeeping is testable
/// over a plain stream, without a cluster.
async fn process_peer_events<S>(mut stream: S, peers: &PeerRegistry, filter: &PeerFilter)
where
    S: Stream<Item = Result<watcher::Event<EndpointSlice>, watcher::Error>> + Unpin,
{
    // A Service's endpoints are commonly sharded across several
    // EndpointSlices (one per AZ, one per address family), which is why this
    // is per-slice rather than a flat list.
    let mut by_slice: HashMap<String, Vec<PeerAddr>> = HashMap::new();
    // Relist buffer, `Some` only between `Init` and `InitDone`, so a relist
    // publishes its complete result at once. Publishing each `InitApply` would
    // expose a partial set, and an empty partial marks the registry synced,
    // making `known_to_have_no_peers()` briefly true. A stray `InitDone`
    // publishes nothing.
    let mut init_buffer: Option<HashMap<String, Vec<PeerAddr>>> = None;
    let publish = |by_slice: &HashMap<String, Vec<PeerAddr>>| peers.replace(peer_urls(by_slice));

    while let Some(event) = stream.next().await {
        match event {
            Ok(watcher::Event::Init) => init_buffer = Some(HashMap::new()),
            Ok(watcher::Event::InitApply(es)) => {
                let key = slice_key(&es);
                let found = filter.extract(&es);
                if let Some(buf) = init_buffer.as_mut() {
                    buf.insert(key, found);
                } else {
                    // Defensive: InitApply outside an Init cycle. Treat as Apply.
                    by_slice.insert(key, found);
                    publish(&by_slice);
                }
            }
            Ok(watcher::Event::InitDone) => {
                let Some(buf) = init_buffer.take() else {
                    continue;
                };
                // Swap atomically: the relist result fully replaces the old
                // view, so slices deleted while the watch was down disappear
                // here.
                by_slice = buf;
                if by_slice.is_empty() {
                    // A correct selector matches at least this replica's own
                    // Service's slice, so zero slices is a misconfiguration,
                    // not a fleet of one — yet it publishes as "alone".
                    tracing::warn!(
                        "kv-bootstrap: the peer selector matched no EndpointSlices; it is \
                         matched against EndpointSlice labels, which are copied from the \
                         router's Service (not its pods), in the watched namespace"
                    );
                }
                publish(&by_slice);
            }
            Ok(watcher::Event::Apply(es)) => {
                by_slice.insert(slice_key(&es), filter.extract(&es));
                publish(&by_slice);
            }
            Ok(watcher::Event::Delete(es)) => {
                by_slice.remove(&slice_key(&es));
                publish(&by_slice);
            }
            Err(e) => {
                tracing::warn!(
                    error = ?e,
                    "kv-bootstrap: peer watcher error; retrying with backoff (a persistent \
                     403 means the ServiceAccount lacks list/watch on endpointslices for \
                     the router's own Service)",
                );
            }
        }
    }
    tracing::warn!("kv-bootstrap: peer watcher stream ended; peer set is now frozen");
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::ObjectReference;
    use k8s_openapi::api::discovery::v1::{EndpointConditions, EndpointPort};
    use kube::core::ObjectMeta;

    // -----------------------------------------------------------------------
    // Peer discovery — the sibling-replica set.
    // -----------------------------------------------------------------------

    /// Multiple endpoints in one slice, each with its own ready condition, so
    /// the per-endpoint filtering can be exercised.
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

    fn identity_with_ip(ip: &str) -> SelfIdentity {
        SelfIdentity {
            ip: Some(ip.into()),
            ..Default::default()
        }
    }

    fn identity_with_pod(name: &str, namespace: Option<&str>) -> SelfIdentity {
        SelfIdentity {
            pod_name: Some(name.into()),
            pod_namespace: namespace.map(Into::into),
            ..Default::default()
        }
    }

    /// One endpoint per (addr, pod-name) entry, each pointing its `target_ref`
    /// at the named pod in `sgl-router-test`.
    fn pod_backed_peer_slice(entries: &[(&str, &str)], port: Option<i32>) -> EndpointSlice {
        let mut es = peer_slice(
            &entries
                .iter()
                .map(|(addr, _)| (*addr, Some(true)))
                .collect::<Vec<_>>(),
            port,
        );
        for (ep, (_, pod)) in es.endpoints.iter_mut().zip(entries) {
            ep.target_ref = Some(ObjectReference {
                kind: Some("Pod".into()),
                name: Some((*pod).into()),
                namespace: Some("sgl-router-test".into()),
                uid: Some(format!("uid-{pod}")),
                ..Default::default()
            });
        }
        es
    }

    /// `PeerFilter::extract`, reduced to the URLs it offers.
    fn extracted_urls(
        es: &EndpointSlice,
        identity: SelfIdentity,
        family: AddressFamily,
        port: i32,
    ) -> Vec<String> {
        PeerFilter {
            identity,
            family,
            port,
        }
        .extract(es)
        .into_iter()
        .map(|p| p.url)
        .collect()
    }

    #[test]
    fn peer_address_family_follows_the_listen_socket() {
        assert_eq!(peer_address_family("0.0.0.0"), AddressFamily::V4);
        assert_eq!(peer_address_family("10.0.0.1"), AddressFamily::V4);
        assert_eq!(peer_address_family("::"), AddressFamily::Any);
        assert_eq!(peer_address_family("fd00::1"), AddressFamily::V6);
        assert_eq!(peer_address_family("router.local"), AddressFamily::Any);
    }

    #[test]
    fn extract_peers_excludes_self() {
        let es = peer_slice(
            &[("10.0.0.1", Some(true)), ("10.0.0.2", Some(true))],
            Some(8090),
        );
        assert_eq!(
            extracted_urls(&es, identity_with_ip("10.0.0.1"), AddressFamily::V4, 8090),
            vec!["http://10.0.0.2:8090"],
            "a replica must not list itself as a peer",
        );
    }

    /// POD_NAME matches the endpoint's `target_ref` and excludes the whole
    /// pod, even when the endpoint's address is not POD_IP — which is what a
    /// dual-stack Service's secondary family looks like.
    #[test]
    fn extract_peers_excludes_self_by_pod_name() {
        let es = pod_backed_peer_slice(
            &[
                ("10.0.0.1", "sgl-router-kv-0"),
                ("10.0.0.2", "sgl-router-kv-1"),
            ],
            Some(8090),
        );
        let identity = SelfIdentity {
            ip: Some("192.168.1.5".into()), // primary addr, NOT in this slice
            ..identity_with_pod("sgl-router-kv-0", Some("sgl-router-test"))
        };
        assert_eq!(
            extracted_urls(&es, identity, AddressFamily::V4, 8090),
            vec!["http://10.0.0.2:8090"],
            "target_ref matching must exclude the pod even at an unknown address",
        );
    }

    /// With POD_NAMESPACE known, a same-named pod in another namespace is a
    /// different pod and must be kept.
    #[test]
    fn extract_peers_keeps_same_named_pod_in_another_namespace() {
        let mut es = pod_backed_peer_slice(&[("10.0.0.2", "sgl-router-kv-0")], Some(8090));
        es.endpoints[0].target_ref.as_mut().unwrap().namespace = Some("other-ns".into());
        assert_eq!(
            extracted_urls(
                &es,
                identity_with_pod("sgl-router-kv-0", Some("sgl-router-test")),
                AddressFamily::V4,
                8090
            ),
            vec!["http://10.0.0.2:8090"],
        );
    }

    /// Without POD_NAMESPACE a bare name match still excludes — EndpointSlices
    /// are namespaced, so a collision needs the same Deployment name twice.
    #[test]
    fn extract_peers_excludes_by_name_when_own_namespace_unknown() {
        let es = pod_backed_peer_slice(&[("10.0.0.1", "sgl-router-kv-0")], Some(8090));
        assert!(extracted_urls(
            &es,
            identity_with_pod("sgl-router-kv-0", None),
            AddressFamily::V4,
            8090
        )
        .is_empty());
    }

    /// Endpoints with no `target_ref` (manually-written slices) still fall
    /// back to POD_IP matching.
    #[test]
    fn extract_peers_falls_back_to_ip_without_target_ref() {
        let es = peer_slice(&[("10.0.0.1", Some(true))], Some(8090));
        let identity = SelfIdentity {
            ip: Some("10.0.0.1".into()),
            ..identity_with_pod("sgl-router-kv-0", Some("sgl-router-test"))
        };
        assert!(extracted_urls(&es, identity, AddressFamily::V4, 8090).is_empty());
    }

    /// An endpoint's addresses are fungible — one backend — so a self address
    /// anywhere in the list makes the whole endpoint this replica.
    #[test]
    fn extract_peers_ip_fallback_excludes_the_whole_endpoint() {
        let mut es = peer_slice(&[("10.0.0.9", Some(true))], Some(8090));
        es.endpoints[0].addresses = vec!["10.0.0.9".into(), "10.0.0.1".into()];
        assert!(
            extracted_urls(&es, identity_with_ip("10.0.0.1"), AddressFamily::V4, 8090).is_empty(),
            "a self address must exclude its endpoint, not just itself",
        );
    }

    /// For the same reason a sibling endpoint is one peer, at its first
    /// address, however many addresses it lists.
    #[test]
    fn extract_peers_offers_one_url_per_endpoint() {
        let mut es = peer_slice(&[("10.0.0.2", Some(true))], Some(8090));
        es.endpoints[0].addresses = vec!["10.0.0.2".into(), "10.0.0.3".into()];
        assert_eq!(
            extracted_urls(&es, identity_with_ip("10.0.0.1"), AddressFamily::V4, 8090),
            vec!["http://10.0.0.2:8090"],
        );
    }

    /// A `target_ref` pointing at a non-Pod object (a hand-written slice
    /// naming a Service, say) must not be mistaken for this replica.
    #[test]
    fn extract_peers_ignores_target_ref_of_non_pod_kind() {
        let mut es = pod_backed_peer_slice(&[("10.0.0.2", "sgl-router-kv-0")], Some(8090));
        es.endpoints[0].target_ref.as_mut().unwrap().kind = Some("Service".into());
        assert_eq!(
            extracted_urls(
                &es,
                identity_with_pod("sgl-router-kv-0", Some("sgl-router-test")),
                AddressFamily::V4,
                8090
            ),
            vec!["http://10.0.0.2:8090"],
        );
    }

    /// A `target_ref` with no kind (hand-written slice) is treated as a Pod:
    /// excluding beats listing this replica as its own peer.
    #[test]
    fn extract_peers_treats_absent_target_ref_kind_as_pod() {
        let mut es = pod_backed_peer_slice(&[("10.0.0.1", "sgl-router-kv-0")], Some(8090));
        es.endpoints[0].target_ref.as_mut().unwrap().kind = None;
        assert!(extracted_urls(
            &es,
            identity_with_pod("sgl-router-kv-0", Some("sgl-router-test")),
            AddressFamily::V4,
            8090
        )
        .is_empty());
    }

    /// A set-but-empty env var must behave as unset; otherwise it would
    /// suppress the `is_unknown` warning while matching nothing.
    #[test]
    fn env_non_empty_treats_empty_strings_as_unset() {
        // Unique names so parallel tests cannot race on these vars.
        std::env::set_var("SGL_ROUTER_TEST_ENV_EMPTY", "");
        std::env::set_var("SGL_ROUTER_TEST_ENV_SET", "x");
        assert_eq!(env_non_empty("SGL_ROUTER_TEST_ENV_EMPTY"), None);
        assert_eq!(
            env_non_empty("SGL_ROUTER_TEST_ENV_SET"),
            Some("x".to_string())
        );
        assert_eq!(env_non_empty("SGL_ROUTER_TEST_ENV_MISSING"), None);
        std::env::remove_var("SGL_ROUTER_TEST_ENV_EMPTY");
        std::env::remove_var("SGL_ROUTER_TEST_ENV_SET");
    }

    /// With neither POD_NAME nor POD_IP there is nothing to exclude, so this
    /// replica stays in its own peer set.
    #[test]
    fn extract_peers_keeps_self_when_identity_is_unknown() {
        let es = peer_slice(&[("10.0.0.1", Some(true))], Some(8090));
        assert_eq!(
            extracted_urls(&es, SelfIdentity::default(), AddressFamily::V4, 8090),
            vec!["http://10.0.0.1:8090"],
        );
        assert!(SelfIdentity::default().is_unknown());
    }

    /// Per the EndpointSlice API an ABSENT ready condition means ready.
    /// Treating it as not-ready would empty the peer set on clusters that omit
    /// it.
    #[test]
    fn extract_peers_treats_absent_ready_as_ready() {
        let es = peer_slice(&[("10.0.0.2", None)], Some(8090));
        assert_eq!(
            extracted_urls(&es, identity_with_ip("10.0.0.1"), AddressFamily::V4, 8090),
            vec!["http://10.0.0.2:8090"],
        );
    }

    /// An unready sibling is either still starting or draining, so it is left
    /// out of the peer set; in a rolling update, a new replica does not list
    /// another still-starting one.
    #[test]
    fn extract_peers_skips_unready_endpoints() {
        let es = peer_slice(
            &[("10.0.0.2", Some(false)), ("10.0.0.3", Some(true))],
            Some(8090),
        );
        assert_eq!(
            extracted_urls(&es, identity_with_ip("10.0.0.1"), AddressFamily::V4, 8090),
            vec!["http://10.0.0.3:8090"],
        );
    }

    #[test]
    fn extract_peers_brackets_ipv6_addresses() {
        let mut es = peer_slice(&[("fd00::2", Some(true))], Some(8090));
        es.address_type = "IPv6".into();
        assert_eq!(
            extracted_urls(&es, identity_with_ip("fd00::1"), AddressFamily::V6, 8090),
            vec!["http://[fd00::2]:8090"],
        );
    }

    /// A dual-stack Service yields one slice per family; only the family this
    /// router listens on is kept.
    #[test]
    fn extract_peers_skips_the_other_address_family() {
        let mut v6 = peer_slice(&[("fd00::2", Some(true))], Some(8090));
        v6.address_type = "IPv6".into();
        assert!(
            extracted_urls(&v6, identity_with_ip("10.0.0.1"), AddressFamily::V4, 8090).is_empty(),
            "an IPv4 router must ignore the IPv6 slice",
        );

        let v4 = peer_slice(&[("10.0.0.2", Some(true))], Some(8090));
        assert!(
            extracted_urls(&v4, identity_with_ip("fd00::1"), AddressFamily::V6, 8090).is_empty(),
            "an IPv6 router must ignore the IPv4 slice",
        );
    }

    /// A listener that names no family (`::`, the dual-stack listener) keeps
    /// both slices.
    #[test]
    fn an_unspecified_listener_keeps_both_address_families() {
        let v4 = peer_slice(&[("10.0.0.2", Some(true))], Some(8090));
        let mut v6 = peer_slice(&[("fd00::2", Some(true))], Some(8090));
        v6.address_type = "IPv6".into();

        assert_eq!(
            extracted_urls(&v4, SelfIdentity::default(), AddressFamily::Any, 8090),
            vec!["http://10.0.0.2:8090"],
        );
        assert_eq!(
            extracted_urls(&v6, SelfIdentity::default(), AddressFamily::Any, 8090),
            vec!["http://[fd00::2]:8090"],
        );
    }

    /// A multi-port Service lists its ports in spec order, and nothing says
    /// the one this router serves on sorts first. Siblings are this router's
    /// own pods, so its own listen port is the reliable signal — taking index
    /// 0 would address every sibling on, say, a `metrics` port.
    #[test]
    fn peer_port_prefers_this_replicas_own_listen_port() {
        let mut es = peer_slice(&[("10.0.0.2", Some(true))], None);
        es.ports = Some(vec![
            EndpointPort {
                name: Some("metrics".into()),
                port: Some(9090),
                ..Default::default()
            },
            EndpointPort {
                name: Some("http".into()),
                port: Some(8090),
                ..Default::default()
            },
        ]);
        assert_eq!(
            extracted_urls(&es, SelfIdentity::default(), AddressFamily::Any, 8090),
            vec!["http://10.0.0.2:8090"],
            "the router's own port wins over whatever the Service lists first",
        );
    }

    /// When no advertised port matches — a Service fronting the router on a
    /// different port than it listens on — fall back to the first rather than
    /// to the hard-coded default, which would be wrong more often.
    #[test]
    fn peer_port_falls_back_to_the_first_advertised_port() {
        let es = peer_slice(&[("10.0.0.2", Some(true))], Some(8080));
        assert_eq!(
            extracted_urls(&es, SelfIdentity::default(), AddressFamily::Any, 8090),
            vec!["http://10.0.0.2:8080"],
        );
    }

    /// An FQDN slice cannot be classified by family, so it is kept.
    #[test]
    fn extract_peers_keeps_fqdn_slices_for_either_family() {
        let mut es = peer_slice(&[("router-1.svc", Some(true))], Some(8090));
        es.address_type = "FQDN".into();
        assert_eq!(
            extracted_urls(&es, SelfIdentity::default(), AddressFamily::V4, 8090),
            vec!["http://router-1.svc:8090"],
        );
    }

    #[test]
    fn extract_peers_falls_back_to_own_port_when_the_slice_lists_none() {
        let es = peer_slice(&[("10.0.0.2", Some(true))], None);
        assert_eq!(
            extracted_urls(&es, SelfIdentity::default(), AddressFamily::V4, 8090),
            vec!["http://10.0.0.2:8090"],
            "siblings listen where this replica does, not on the compiled-in default",
        );
    }

    #[test]
    fn extract_peers_on_empty_slice_is_empty() {
        assert!(extracted_urls(
            &peer_slice(&[], Some(8090)),
            SelfIdentity::default(),
            AddressFamily::V4,
            8090
        )
        .is_empty());
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
        identity: SelfIdentity,
        family: AddressFamily,
    ) {
        let stream = futures::stream::iter(events);
        tokio::pin!(stream);
        let filter = PeerFilter {
            identity,
            family,
            port: 8090,
        };
        process_peer_events(stream, peers, &filter).await;
    }

    /// A partial relist must never make `known_to_have_no_peers()` true.
    ///
    /// Asserted mid-relist: the settled state after `InitDone` is correct even
    /// when a partial set was briefly published.
    #[tokio::test]
    async fn peer_relist_does_not_publish_a_partial_set() {
        let peers = PeerRegistry::new();
        // Init plus one self-only slice, and NO InitDone: nothing may be
        // published yet, so the registry must not even be marked synced.
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
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
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
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
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
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
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
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
        )
        .await;
        let mut got = peers.candidates();
        got.sort();
        assert_eq!(got, vec!["http://10.0.0.2:8090", "http://10.0.0.3:8090"]);

        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.2", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-b",
                    &[("10.0.0.3", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
                Ok(watcher::Event::Delete(named_peer_slice(
                    "slice-b",
                    &[("10.0.0.3", Some(true))],
                    Some(8090),
                ))),
            ],
            &peers,
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
        )
        .await;
        assert_eq!(
            peers.candidates(),
            vec!["http://10.0.0.2:8090"],
            "deleting a slice drops only its own endpoints",
        );
    }

    /// A transient watcher error must preserve the peer set, not clear or
    /// republish it: an empty publish would make `known_to_have_no_peers()`
    /// true.
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
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
        )
        .await;

        assert_eq!(
            peers.candidates(),
            vec!["http://10.0.0.2:8090"],
            "a transient error must not disturb the peer set",
        );
        assert!(!peers.known_to_have_no_peers());
    }

    /// A genuinely single-replica deployment reads as alone:
    /// `known_to_have_no_peers()` is true once synced.
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
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
        )
        .await;
        assert!(peers.is_empty());
        assert!(
            peers.known_to_have_no_peers(),
            "synced with only self ⇒ genuinely alone",
        );
    }

    /// A dual-stack Service lists each sibling once per family. With a
    /// listener that names no family both slices are kept, and the sibling
    /// must still be offered once, not once per family.
    #[tokio::test]
    async fn a_dual_stack_sibling_is_offered_once() {
        let v4 = pod_backed_peer_slice(&[("10.0.0.2", "sgl-router-kv-1")], Some(8090));
        let mut v6 = pod_backed_peer_slice(&[("fd00::2", "sgl-router-kv-1")], Some(8090));
        v6.metadata.name = Some("sgl-router-kv-v6".into());
        v6.address_type = "IPv6".into();
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(v4)),
                Ok(watcher::Event::InitApply(v6)),
                Ok(watcher::Event::InitDone),
            ],
            &peers,
            SelfIdentity::default(),
            AddressFamily::Any,
        )
        .await;
        assert_eq!(peers.candidates(), vec!["http://10.0.0.2:8090"]);
    }

    /// An `InitDone` outside an `Init` cycle must not publish: the empty
    /// relist buffer would mark the registry synced with zero peers.
    #[tokio::test]
    async fn a_stray_init_done_publishes_nothing() {
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![Ok(watcher::Event::InitDone)],
            &peers,
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
        )
        .await;
        assert!(!peers.synced());
        assert!(!peers.known_to_have_no_peers());
    }
}
