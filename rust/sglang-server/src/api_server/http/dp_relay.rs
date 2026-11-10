//! A generation that names another DP rank (`routed_dp_rank`) runs on that
//! rank, as Python's `DataParallelController` dispatches it.
//!
//! Under attention DP each DP group's leader runs its own listener, which
//! feeds only its own group's scheduler; no listener reaches another's
//! scheduler except over HTTP. So the listener that receives a request naming
//! another rank relays it, body and headers as received, to that rank's
//! listener (the deployment's `rust_server_endpoints`) and streams the reply
//! back. That listener serves it as its own request, from admission to its
//! request log and metrics, so the relay needs nothing else:
//!
//! * a client that hangs up drops the relayed connection, which aborts the
//!   request there as a hang-up does on any listener;
//! * `/abort_request` on DP rank 0, or on the listener that relayed the
//!   request while it is relayed, goes to every rank's listener
//!   (`CoreHandle::abort_peers`), so it reaches the request by its rid. A
//!   gRPC `Abort` stays local, as for any request another listener serves;
//! * P/D needs no relay of its own: the rank serving the request is the one
//!   that registers its bootstrap room (prefill) or fetches its KV (decode),
//!   as on Python, where that rank's scheduler gets it.
//!
//! The relay runs before any of the route's work (outside an embedding
//! package's layers, inside request decompression) and only on the routes
//! Python reads the hint on, from the sources it reads ([`HintRoute`]). A
//! request naming no rank, this listener's own, or one the relay cannot read
//! (malformed, out of range, no listener known) is served here; the core then
//! answers it as Python does or, for another rank, refuses it
//! (`CoreHandle::validate`). No request runs on a rank it did not name.
//!
//! A request naming no rank runs on the listener that received it: unlike
//! Python's `DataParallelController`, nothing balances those across the
//! ranks. Clients reach the other ranks through a router that dials each
//! listener (DP rank 0 publishes them), or by dialing them themselves.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::Response;
use serde::Deserialize;

use super::request_headers::int_header;
use crate::api_server::handlers::CoreHandle;
use crate::message::config::DeploymentInfo;
use crate::utils::response::json_error;

/// The DP rank a relayed generation is meant for, the header the control
/// relay sends for the same purpose (`x-sglang-dp-rank`, upstream
/// sgl-project/sglang#47519). A listener serves such a request only as that
/// rank and never relays it again: one serving another rank refuses it with
/// 421, so a stale listener table can neither loop a request nor run it on
/// the wrong rank.
const RELAYED_RANK_HEADER: &str = "x-sglang-dp-rank";

/// Python's router header override (`request_headers.py`), when enabled.
const OVERRIDE_HEADER: &str = "x-override-routed-dp-rank";
/// Python's OpenAI header (`OpenAIServingBase.extract_routed_dp_rank_from_header`).
const DATA_PARALLEL_RANK_HEADER: &str = "x-data-parallel-rank";

/// A listener only needs to come up; the generation itself has no deadline,
/// as on the local path.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Connection-scoped headers, which name this hop and are not relayed.
const HOP_BY_HOP: [HeaderName; 8] = [
    header::CONNECTION,
    HeaderName::from_static("keep-alive"),
    header::PROXY_AUTHENTICATE,
    header::PROXY_AUTHORIZATION,
    header::TE,
    header::TRAILER,
    header::TRANSFER_ENCODING,
    header::UPGRADE,
];

/// A generation route Python reads `routed_dp_rank` on, and where it reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HintRoute {
    /// `/generate`: the router's `x-override-routed-dp-rank` header when
    /// header overrides are on (`http_server.serve_generate_request`), then
    /// the body's `routed_dp_rank`, then its deprecated `data_parallel_rank`
    /// (`GenerateReqInput.normalize_batch_and_arguments`). One rank for the
    /// whole of a batch body.
    Generate,
    /// `/v1/chat/completions`: the override header as for `/generate`
    /// (`OpenAIServingChat`'s `apply_header_overrides`), then
    /// `X-Data-Parallel-Rank`, then the body's fields.
    ChatCompletions,
    /// `/v1/completions`: `X-Data-Parallel-Rank`, then the body's fields.
    Completions,
}

impl HintRoute {
    fn of(path: &str) -> Option<Self> {
        match path {
            "/generate" => Some(Self::Generate),
            "/v1/chat/completions" => Some(Self::ChatCompletions),
            "/v1/completions" => Some(Self::Completions),
            _ => None,
        }
    }

    /// The DP rank this route's headers name, which wins over the body's;
    /// `None` when they name none. An unparsable header is Python's 400:
    /// `X-Data-Parallel-Rank` is read first, as Python reads it before it
    /// applies the override, so a malformed one fails even under a valid
    /// override.
    pub(super) fn header_rank(
        self,
        headers: &HeaderMap,
        overrides: bool,
    ) -> Result<Option<i64>, String> {
        let data_parallel_rank = match headers.get(DATA_PARALLEL_RANK_HEADER) {
            Some(value) if self != Self::Generate => {
                let text = String::from_utf8_lossy(value.as_bytes());
                Some(python_int(&text).ok_or_else(|| {
                    format!("Invalid X-Data-Parallel-Rank header: must be an integer, got '{text}'")
                })?)
            }
            _ => None,
        };
        if overrides
            && self != Self::Completions
            && let Some(rank) = int_header(headers, OVERRIDE_HEADER)?
        {
            return Ok(Some(rank));
        }
        Ok(data_parallel_rank)
    }
}

/// Python's `int(text)` for an ASCII decimal: surrounding whitespace, a sign,
/// and single underscores between digits. A value past `i64` saturates, so
/// the range check refuses it rather than the parse.
fn python_int(text: &str) -> Option<i64> {
    let text = text.trim();
    let (negative, digits) = match text.as_bytes().first()? {
        b'-' => (true, &text[1..]),
        b'+' => (false, &text[1..]),
        _ => (false, text),
    };
    if digits
        .split('_')
        .any(|group| group.is_empty() || !group.bytes().all(|b| b.is_ascii_digit()))
    {
        return None;
    }
    let magnitude = digits
        .bytes()
        .filter(u8::is_ascii_digit)
        .fold(0i128, |value, digit| {
            value
                .saturating_mul(10)
                .saturating_add(i128::from(digit - b'0'))
        });
    let value = if negative { -magnitude } else { magnitude };
    Some(value.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64)
}

/// The body fields every hinted route reads the rank from, the last of a
/// repeated key winning, as the routes' JSON decoding keeps it. Anything that
/// does not decode (not a JSON object, a final rank that is no integer)
/// names no rank here; the route's handler answers it.
#[derive(Default)]
struct BodyRank {
    routed_dp_rank: Option<serde_json::Value>,
    data_parallel_rank: Option<serde_json::Value>,
}

impl<'de> Deserialize<'de> for BodyRank {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Fields;
        impl<'de> serde::de::Visitor<'de> for Fields {
            type Value = BodyRank;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<BodyRank, A::Error> {
                let mut ranks = BodyRank::default();
                while let Some(key) = map.next_key::<std::borrow::Cow<'de, str>>()? {
                    match key.as_ref() {
                        "routed_dp_rank" => ranks.routed_dp_rank = map.next_value()?,
                        "data_parallel_rank" => ranks.data_parallel_rank = map.next_value()?,
                        _ => {
                            map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(ranks)
            }
        }
        deserializer.deserialize_map(Fields)
    }
}

/// The rank `route`'s body names. A field set to null is unset, as Python's
/// `Optional[int]` reads it, and only the field that names the rank is
/// read: a set `routed_dp_rank` discards `data_parallel_rank`, whatever it
/// holds. The OpenAI routes read it as their handlers do ([`lax_rank`]).
fn body_rank(route: HintRoute, body: &[u8]) -> Option<i64> {
    let ranks: BodyRank = serde_json::from_slice(body).ok()?;
    let set = |value: Option<serde_json::Value>| value.filter(|value| !value.is_null());
    let value = set(ranks.routed_dp_rank).or_else(|| set(ranks.data_parallel_rank))?;
    match route {
        HintRoute::Generate => value.as_i64(),
        HintRoute::ChatCompletions | HintRoute::Completions => pydantic_int(&value),
    }
}

/// Pydantic's lax `int` (the OpenAI request models): an integer, a float or
/// numeric string with no fraction, or a bool.
fn pydantic_int(value: &serde_json::Value) -> Option<i64> {
    use serde_json::Value;
    let whole = |float: f64| {
        (float.fract() == 0.0 && float >= i64::MIN as f64 && float <= i64::MAX as f64)
            .then_some(float as i64)
    };
    match value {
        Value::Number(number) => number.as_i64().or_else(|| whole(number.as_f64()?)),
        Value::Bool(flag) => Some(i64::from(*flag)),
        // An integer, or one with a fraction of zeros (`"1.0"`).
        Value::String(text) => python_int(text).or_else(|| {
            let (int, fraction) = text.trim().split_once('.')?;
            (!fraction.is_empty() && fraction.bytes().all(|b| b == b'0'))
                .then(|| python_int(int))
                .flatten()
        }),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

/// A DP rank field of an OpenAI request, as pydantic's `Optional[int]` reads
/// it there ([`pydantic_int`]).
pub(super) fn lax_rank<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<i64>, D::Error> {
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Null => Ok(None),
        value => pydantic_int(&value).map(Some).ok_or_else(|| {
            serde::de::Error::custom(format!("invalid DP rank {value}: not an integer"))
        }),
    }
}

/// The relay of one listener: its own DP rank and every other rank's
/// listener.
struct DpRelay {
    own: u32,
    dp_size: u32,
    /// Each DP rank's listener base URL, by rank; `None` for this listener's
    /// own and for a rank the table lacks.
    listeners: Vec<Option<String>>,
    overrides: bool,
    client: reqwest::Client,
    core: CoreHandle,
}

impl DpRelay {
    /// The relay for a listener of a deployment with other DP listeners,
    /// else `None`.
    fn new(
        deployment_info: &DeploymentInfo,
        overrides: bool,
        core: CoreHandle,
    ) -> Result<Option<Self>, String> {
        let own = deployment_info.dp_rank.unwrap_or(0);
        let dp_size = deployment_info.dp_size;
        let mut listeners = vec![None; dp_size as usize];
        for endpoint in &deployment_info.rust_server_endpoints {
            if endpoint.dp_rank != own
                && let Some(slot) = listeners.get_mut(endpoint.dp_rank as usize)
            {
                *slot = Some(endpoint.endpoint.trim_end_matches('/').to_owned());
            }
        }
        if listeners.iter().all(Option::is_none) {
            return Ok(None);
        }
        // Listeners are dialed directly: an inherited HTTP proxy must not
        // carry this traffic.
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .no_proxy()
            // The peer's answer, a redirect included, is the client's.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| format!("DP relay client: {error}"))?;
        Ok(Some(Self {
            own,
            dp_size,
            listeners,
            overrides,
            client,
            core,
        }))
    }

    /// The listener serving `rank`, when it is another rank of this
    /// deployment that the table names.
    fn listener(&self, rank: i64) -> Option<(u32, &str)> {
        let rank = u32::try_from(rank).ok()?;
        if rank == self.own || rank >= self.dp_size {
            return None;
        }
        Some((rank, self.listeners[rank as usize].as_deref()?))
    }

    /// Relay `request` to DP rank `rank`'s listener at `base` and stream its
    /// response back as received.
    async fn relay(&self, rank: u32, base: &str, request: Request) -> Response {
        let (parts, body) = request.into_parts();
        let body = match axum::body::to_bytes(body, usize::MAX).await {
            Ok(body) => body,
            Err(error) => return unreadable_body(error),
        };
        let mut headers = parts.headers;
        strip_hop_by_hop(&mut headers);
        // The body is already whole: the relay sets its own length.
        for name in [header::HOST, header::CONTENT_LENGTH, header::EXPECT] {
            headers.remove(name);
        }
        headers.insert(RELAYED_RANK_HEADER, HeaderValue::from(rank));
        let path = parts
            .uri
            .path_and_query()
            .map_or(parts.uri.path(), |path| path.as_str());
        let url = format!("{base}{path}");
        tracing::debug!(rank, %url, "relaying a request naming another DP rank");
        // The drain waits for the relayed request as for one served here, and
        // an abort sent here reaches the peers meanwhile.
        let relayed = self.core.track_relayed();
        let mut upstream = match self
            .client
            .post(&url)
            .headers(headers)
            .body(body)
            .send()
            .await
        {
            Ok(upstream) => upstream,
            Err(error) => {
                let status = if error.is_connect() {
                    StatusCode::SERVICE_UNAVAILABLE
                } else {
                    StatusCode::BAD_GATEWAY
                };
                tracing::warn!(rank, %url, %error, "DP relay failed");
                return json_error(
                    status,
                    &format!(
                        "routed_dp_rank={rank}: DP rank {rank}'s listener at {base} did not \
                         answer: {error}"
                    ),
                );
            }
        };
        let status = upstream.status();
        if status == StatusCode::MISDIRECTED_REQUEST {
            // The listener table points at another rank's listener: a server
            // fault, not the client's request.
            let reply = upstream.text().await.unwrap_or_default();
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("routed_dp_rank={rank}: {base} is not DP rank {rank}'s listener: {reply}"),
            );
        }
        let mut headers = std::mem::take(upstream.headers_mut());
        strip_hop_by_hop(&mut headers);
        // Pulled chunk by chunk, so a stream reaches the client as the peer
        // writes it; dropping it (a client hang-up) closes the relayed
        // connection.
        let chunks = async_stream::stream! {
            let _relayed = relayed;
            loop {
                match upstream.chunk().await {
                    Ok(Some(chunk)) => yield Ok(chunk),
                    Ok(None) => break,
                    Err(error) => {
                        tracing::warn!(rank, %error, "DP relay response cut");
                        yield Err(error);
                        break;
                    }
                }
            }
        };
        let mut response = Response::new(Body::from_stream(chunks));
        *response.status_mut() = status;
        *response.headers_mut() = headers;
        response
    }
}

/// Remove the connection-scoped headers: the fixed set, and those
/// `Connection` names.
fn strip_hop_by_hop(headers: &mut HeaderMap) {
    let named: Vec<HeaderName> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::try_from(name.trim()).ok())
        .collect();
    for name in named.iter().chain(&HOP_BY_HOP) {
        headers.remove(name);
    }
}

fn unreadable_body(error: axum::Error) -> Response {
    json_error(
        StatusCode::BAD_REQUEST,
        &format!("Failed to buffer the request body: {error}"),
    )
}

async fn relay_requested_rank(
    State(relay): State<Arc<DpRelay>>,
    request: Request,
    next: Next,
) -> Response {
    let route = (request.method() == Method::POST)
        .then(|| HintRoute::of(request.uri().path()))
        .flatten();
    let Some(route) = route else {
        return next.run(request).await;
    };
    if let Some(target) = request.headers().get(RELAYED_RANK_HEADER) {
        let own = relay.core.dp_rank();
        return match target
            .to_str()
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
        {
            Some(target) if target == own => next.run(request).await,
            Some(target) => {
                tracing::warn!(
                    dp_rank = own,
                    requested_dp_rank = target,
                    "refusing a generation relayed for another dp rank"
                );
                json_error(
                    StatusCode::MISDIRECTED_REQUEST,
                    &format!("relayed to dp rank {own}'s listener"),
                )
            }
            None => json_error(
                StatusCode::BAD_REQUEST,
                &format!("invalid {RELAYED_RANK_HEADER} header"),
            ),
        };
    }
    let (rank, request) = match route.header_rank(request.headers(), relay.overrides) {
        Ok(Some(rank)) => (Some(rank), request),
        // The handler answers the malformed header.
        Err(_) => return next.run(request).await,
        Ok(None) => {
            let (parts, body) = request.into_parts();
            let body = match axum::body::to_bytes(body, usize::MAX).await {
                Ok(body) => body,
                Err(error) => return unreadable_body(error),
            };
            let rank = body_rank(route, &body);
            (rank, Request::from_parts(parts, Body::from(body)))
        }
    };
    match rank.and_then(|rank| relay.listener(rank)) {
        Some((rank, base)) => relay.relay(rank, base, request).await,
        None => next.run(request).await,
    }
}

/// Wrap `app` in the relay, for a listener of a deployment with other DP
/// listeners; otherwise `app` as is.
pub(super) fn apply(
    app: Router,
    deployment_info: &DeploymentInfo,
    overrides: bool,
    core: CoreHandle,
) -> Result<Router, String> {
    Ok(match DpRelay::new(deployment_info, overrides, core)? {
        Some(relay) => app.layer(middleware::from_fn_with_state(
            Arc::new(relay),
            relay_requested_rank,
        )),
        None => app,
    })
}

#[cfg(test)]
mod tests {
    use std::future::IntoFuture;
    use std::sync::Mutex;

    use axum::http::{HeaderMap, Uri};
    use axum::routing::post;
    use bytes::Bytes;
    use futures::StreamExt;
    use tower::ServiceExt;

    use super::*;
    use crate::api_server::handlers::{CoreConfig, CoreMetadata};
    use crate::message::config::RustServerEndpoint;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        pairs
            .iter()
            .map(|&(name, value)| (name.parse().unwrap(), value.parse().unwrap()))
            .collect()
    }

    /// Python's precedence per route: the override header (when enabled, not
    /// on completions), then `X-Data-Parallel-Rank` (not on `/generate`);
    /// the body's fields only when no header names a rank.
    #[test]
    fn header_rank_follows_python_per_route() {
        use HintRoute::{ChatCompletions, Completions, Generate};
        let both = headers(&[(OVERRIDE_HEADER, "1"), (DATA_PARALLEL_RANK_HEADER, "2")]);
        let dpr = headers(&[(DATA_PARALLEL_RANK_HEADER, "2")]);
        for (route, overrides, sent, expected) in [
            (Generate, true, &both, Some(1)),
            (Generate, false, &both, None),
            (Generate, true, &dpr, None),
            (ChatCompletions, true, &both, Some(1)),
            (ChatCompletions, false, &both, Some(2)),
            (Completions, true, &both, Some(2)),
            (Completions, true, &HeaderMap::new(), None),
        ] {
            assert_eq!(
                route.header_rank(sent, overrides),
                Ok(expected),
                "{route:?} overrides={overrides}"
            );
        }
        assert_eq!(
            Completions.header_rank(&headers(&[(DATA_PARALLEL_RANK_HEADER, "x")]), false),
            Err("Invalid X-Data-Parallel-Rank header: must be an integer, got 'x'".into())
        );
        // Python validates X-Data-Parallel-Rank before it applies the override,
        // and reads it with `int()`.
        let malformed = headers(&[(OVERRIDE_HEADER, "1"), (DATA_PARALLEL_RANK_HEADER, "x")]);
        assert!(ChatCompletions.header_rank(&malformed, true).is_err());
        for (text, rank) in [
            (" +0_1 ", Some(1)),
            ("-2", Some(-2)),
            ("1__0", None),
            ("_1", None),
        ] {
            assert_eq!(python_int(text), rank, "{text:?}");
        }
        assert_eq!(python_int(&"9".repeat(30)), Some(i64::MAX));
        assert_eq!(python_int("-9223372036854775808"), Some(i64::MIN));
        assert!(
            Generate
                .header_rank(&headers(&[(OVERRIDE_HEADER, "1.5")]), true)
                .is_err()
        );

        assert_eq!(
            body_rank(
                HintRoute::Generate,
                br#"{"routed_dp_rank": 1, "data_parallel_rank": 0}"#
            ),
            Some(1)
        );
        // A repeated key: the last wins, as the routes decode it.
        for repeated in [
            &br#"{"routed_dp_rank": 0, "routed_dp_rank": 1}"#[..],
            br#"{"routed_dp_rank": "discarded", "routed_dp_rank": 1}"#,
            br#"{"routed_dp_rank": 1, "data_parallel_rank": "discarded"}"#,
        ] {
            assert_eq!(body_rank(HintRoute::Generate, repeated), Some(1));
        }
        // The OpenAI routes read a string of an integer, as pydantic does.
        let text = br#"{"routed_dp_rank": null, "data_parallel_rank": "1"}"#;
        assert_eq!(body_rank(HintRoute::Completions, text), Some(1));
        for (value, rank) in [
            (serde_json::json!(1.0), Some(1)),
            (serde_json::json!(true), Some(1)),
            (serde_json::json!("1.0"), Some(1)),
            (serde_json::json!(1.5), None),
            (serde_json::json!("one"), None),
            (serde_json::json!("1e0"), None),
            (serde_json::json!("1.0000000000000001"), None),
        ] {
            assert_eq!(pydantic_int(&value), rank, "{value}");
        }
        assert_eq!(body_rank(HintRoute::Generate, text), None);
        assert_eq!(
            body_rank(
                HintRoute::Generate,
                br#"{"routed_dp_rank": null, "data_parallel_rank": 1}"#
            ),
            Some(1)
        );
        for unread in [
            &br#"{"text": "hi"}"#[..],
            br#"{"routed_dp_rank": [1]}"#,
            br#"{"routed_dp_rank": "1"}"#,
            br#"[{"routed_dp_rank": 1}]"#,
            b"not json",
        ] {
            assert_eq!(
                body_rank(HintRoute::Generate, unread),
                None,
                "{}",
                String::from_utf8_lossy(unread)
            );
        }
    }

    /// What the peer listener saw of each relayed request.
    #[derive(Clone, Debug)]
    struct Seen {
        uri: Uri,
        headers: HeaderMap,
        body: Bytes,
    }

    /// A DP rank 1 listener on loopback: records each request and answers a
    /// redirect, with a header its `Connection` names and a body in two
    /// chunks, which a relay must pass through as they are, unfollowed.
    async fn peer() -> (String, Arc<Mutex<Vec<Seen>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = seen.clone();
        let answer = move |uri: Uri, headers: HeaderMap, body: Bytes| {
            let recorded = recorded.clone();
            async move {
                recorded.lock().unwrap().push(Seen { uri, headers, body });
                let chunks = futures::stream::iter(["data: peer\n\n", "data: [DONE]\n\n"])
                    .map(|chunk| Ok::<_, std::convert::Infallible>(Bytes::from(chunk)));
                let mut response = Response::new(Body::from_stream(chunks));
                *response.status_mut() = StatusCode::TEMPORARY_REDIRECT;
                for (name, value) in [
                    ("x-peer", "1"),
                    ("location", "/elsewhere"),
                    ("connection", "x-peer-hop"),
                    ("x-peer-hop", "1"),
                ] {
                    response
                        .headers_mut()
                        .insert(name, HeaderValue::from_static(value));
                }
                response
            }
        };
        let app = Router::new()
            .route("/generate", post(answer.clone()))
            .route("/v1/completions", post(answer));
        (serve(app).await.unwrap(), seen)
    }

    async fn serve(app: Router) -> Option<String> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
        let url = format!("http://{}", listener.local_addr().ok()?);
        tokio::spawn(axum::serve(listener, app).into_future());
        Some(url)
    }

    fn core() -> CoreHandle {
        CoreHandle::new(
            flume::unbounded().0,
            flume::unbounded().0,
            CoreConfig {
                response_capacity: 1,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Vec::new(),
                multimodal: false,
                metadata: CoreMetadata::default(),
                metrics: None,
                request_log: None,
                peer_endpoints: Vec::new(),
                generate_validator: None,
            },
        )
    }

    /// DP rank 0's listener of a two-rank deployment whose rank 1 listens at
    /// `peer_url`: every hinted route answers `local` when served here.
    fn listener(peer_url: &str, overrides: bool, core: CoreHandle) -> Router {
        let info = DeploymentInfo {
            dp_size: 2,
            dp_rank: Some(0),
            rust_server_endpoints: vec![
                RustServerEndpoint {
                    dp_rank: 1,
                    endpoint: format!("{peer_url}/"),
                },
                RustServerEndpoint {
                    dp_rank: 0,
                    endpoint: "http://127.0.0.1:1".into(),
                },
            ],
            ..Default::default()
        };
        let local = || post(|body: Bytes| async move { (StatusCode::OK, body) });
        let app = Router::new()
            .route("/generate", local())
            .route("/v1/chat/completions", local())
            .route("/v1/completions", local());
        apply(app, &info, overrides, core).unwrap()
    }

    async fn send(
        app: &Router,
        path: &str,
        sent: &[(&'static str, &str)],
        body: &'static str,
    ) -> (StatusCode, HeaderMap, Bytes) {
        let mut request = Request::post(path).body(Body::from(body)).unwrap();
        *request.headers_mut() = headers(sent);
        let response = app.clone().oneshot(request).await.unwrap();
        let (parts, body) = response.into_parts();
        let body = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        (parts.status, parts.headers, body)
    }

    /// A request naming DP rank 1 reaches rank 1's listener as sent, marked
    /// for rank 1, and its response comes back as rank 1 wrote it; one naming
    /// no rank, rank 0, a rank out of range or one the body cannot carry is
    /// served here, as is one relayed here for rank 0. One relayed here for
    /// another rank is refused.
    #[tokio::test]
    async fn relays_a_request_naming_another_rank_to_its_listener() {
        let (peer_url, seen) = peer().await;
        let app = listener(&peer_url, true, core());

        let body = r#"{"input_ids": [1, 2], "routed_dp_rank": 1, "stream": true}"#;
        let (status, headers, reply) = send(
            &app,
            "/generate?trace=1",
            &[
                ("authorization", "Bearer key"),
                ("content-type", "application/json"),
                ("keep-alive", "timeout=5"),
                ("connection", "x-client-hop"),
                ("x-client-hop", "1"),
            ],
            body,
        )
        .await;
        assert_eq!(status, StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(headers["x-peer"], "1");
        assert_eq!(headers["location"], "/elsewhere");
        assert!(!headers.contains_key("x-peer-hop"), "hop-by-hop");
        assert_eq!(reply, "data: peer\n\ndata: [DONE]\n\n");
        let relayed = seen.lock().unwrap().pop().unwrap();
        assert_eq!(relayed.uri, "/generate?trace=1");
        assert_eq!(relayed.body, body);
        assert_eq!(relayed.headers[RELAYED_RANK_HEADER], "1");
        assert_eq!(relayed.headers["authorization"], "Bearer key");
        assert_eq!(relayed.headers["content-type"], "application/json");
        for name in ["keep-alive", "x-client-hop"] {
            assert!(!relayed.headers.contains_key(name), "hop-by-hop {name}");
        }

        for (path, sent, body, served_by_peer) in [
            ("/generate", vec![], r#"{"data_parallel_rank": 1}"#, true),
            (
                "/generate",
                vec![(OVERRIDE_HEADER, "1")],
                r#"{"routed_dp_rank": 0}"#,
                true,
            ),
            (
                "/generate",
                vec![(OVERRIDE_HEADER, "0")],
                r#"{"routed_dp_rank": 1}"#,
                false,
            ),
            (
                "/v1/completions",
                vec![(DATA_PARALLEL_RANK_HEADER, "1")],
                "{}",
                true,
            ),
            (
                "/generate",
                vec![(DATA_PARALLEL_RANK_HEADER, "1")],
                "{}",
                false,
            ),
            ("/generate", vec![], r#"{"text": "hi"}"#, false),
            ("/generate", vec![], r#"{"routed_dp_rank": 0}"#, false),
            ("/generate", vec![], r#"{"routed_dp_rank": 2}"#, false),
            ("/generate", vec![], r#"{"routed_dp_rank": -1}"#, false),
            ("/generate", vec![], r#"{"routed_dp_rank": [1]}"#, false),
            ("/generate", vec![(OVERRIDE_HEADER, "one")], "{}", false),
            // Relayed here for this rank: served here, never relayed again.
            (
                "/generate",
                vec![(RELAYED_RANK_HEADER, "0")],
                r#"{"routed_dp_rank": 1}"#,
                false,
            ),
        ] {
            let before = seen.lock().unwrap().len();
            let (status, _, reply) = send(&app, path, &sent, body).await;
            let relayed = seen.lock().unwrap().len() > before;
            assert_eq!(relayed, served_by_peer, "{path} {sent:?} {body}");
            if !served_by_peer {
                assert_eq!(
                    (status, reply),
                    (StatusCode::OK, Bytes::from(body)),
                    "{body}"
                );
            }
        }

        // Relayed here for another rank: refused with 421 (a malformed rank,
        // 400), neither served nor relayed again.
        for (target, status, message) in [
            (
                "1",
                StatusCode::MISDIRECTED_REQUEST,
                "relayed to dp rank 0's listener",
            ),
            (
                "one",
                StatusCode::BAD_REQUEST,
                "invalid x-sglang-dp-rank header",
            ),
        ] {
            let before = seen.lock().unwrap().len();
            let body = r#"{"routed_dp_rank": 1}"#;
            let (got, _, reply) =
                send(&app, "/generate", &[(RELAYED_RANK_HEADER, target)], body).await;
            assert_eq!(got, status, "{target}");
            let reply: serde_json::Value = serde_json::from_slice(&reply).unwrap();
            assert_eq!(reply["error"]["message"], message);
            assert_eq!(seen.lock().unwrap().len(), before, "{target}: not relayed");
        }

        // With the header overrides off, only the body names the rank.
        let app = listener(&peer_url, false, core());
        let before = seen.lock().unwrap().len();
        let (_, _, reply) = send(&app, "/generate", &[(OVERRIDE_HEADER, "1")], "{}").await;
        assert_eq!(reply, "{}");
        assert_eq!(seen.lock().unwrap().len(), before);
    }

    /// The drain waits for a relayed response until its last byte, and an
    /// unreachable listener is a 503 naming the rank, not a local serve.
    #[tokio::test]
    async fn relayed_requests_hold_the_drain_and_fail_loud() {
        let (peer_url, _) = peer().await;
        let core = core();
        let app = listener(&peer_url, false, core.clone());
        let response = app
            .clone()
            .oneshot(
                Request::post("/generate")
                    .body(Body::from(r#"{"routed_dp_rank": 1}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        core.begin_drain(false);
        let ended = core.drain_ended();
        tokio::pin!(ended);
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut ended)
                .await
                .is_err(),
            "the drain waits for the relayed response"
        );
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), ended)
            .await
            .expect("the drain ends once the relayed response is done");

        // A listener that refuses the relay as another rank's (421) is a
        // server fault: 500, naming the listener.
        let misdirected = post(|| async {
            (
                StatusCode::MISDIRECTED_REQUEST,
                "relayed to dp rank 0's listener",
            )
        });
        let wrong = serve(Router::new().route("/generate", misdirected))
            .await
            .unwrap();
        let app = listener(&wrong, false, self::core());
        let (status, _, reply) = send(&app, "/generate", &[], r#"{"routed_dp_rank": 1}"#).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        let reply: serde_json::Value = serde_json::from_slice(&reply).unwrap();
        assert_eq!(
            reply["error"]["message"],
            format!(
                "routed_dp_rank=1: {wrong} is not DP rank 1's listener: relayed to dp rank 0's listener"
            )
        );

        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let unreachable = format!("http://{}", probe.local_addr().unwrap());
        drop(probe);
        let app = listener(&unreachable, false, self::core());
        let (status, _, reply) = send(&app, "/generate", &[], r#"{"routed_dp_rank": 1}"#).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let reply: serde_json::Value = serde_json::from_slice(&reply).unwrap();
        let message = reply["error"]["message"].as_str().unwrap();
        assert!(
            message.starts_with(&format!(
                "routed_dp_rank=1: DP rank 1's listener at {unreachable}"
            )),
            "{message}"
        );
    }

    /// A client that hangs up on a relayed stream closes the relayed
    /// connection, so the listener serving it sees the hang-up (and aborts
    /// the request, as for any client).
    #[tokio::test]
    async fn a_client_hang_up_reaches_the_serving_listener() {
        struct OnDrop(Option<tokio::sync::oneshot::Sender<()>>);
        impl Drop for OnDrop {
            fn drop(&mut self) {
                let _ = self.0.take().unwrap().send(());
            }
        }
        let (dropped_tx, dropped_rx) = tokio::sync::oneshot::channel();
        let dropped_tx = Arc::new(Mutex::new(Some(dropped_tx)));
        let endless = post(move || {
            let guard = OnDrop(dropped_tx.lock().unwrap().take());
            async move {
                let chunks = futures::stream::once(async { Bytes::from_static(b"data: 1\n\n") })
                    .chain(futures::stream::pending())
                    .map(move |chunk| {
                        let _ = &guard;
                        Ok::<_, std::convert::Infallible>(chunk)
                    });
                Response::new(Body::from_stream(chunks))
            }
        });
        let peer_url = serve(Router::new().route("/generate", endless))
            .await
            .unwrap();
        let app = listener(&peer_url, false, core());
        let response = app
            .oneshot(
                Request::post("/generate")
                    .body(Body::from(r#"{"routed_dp_rank": 1, "stream": true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body = response.into_body().into_data_stream();
        assert_eq!(body.next().await.unwrap().unwrap(), "data: 1\n\n");
        drop(body);
        tokio::time::timeout(Duration::from_secs(5), dropped_rx)
            .await
            .expect("the serving listener sees the hang-up")
            .unwrap();
    }
}
